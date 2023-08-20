from copy import deepcopy
from typing import Dict, Union, Tuple, Any
import torch
from torch.nn import functional as F
from config import rorl_config
from modules import Actor, EnsembledCritic

from torch.distributions import kl_divergence


_Number = Union[float, int]


class RORL:
    def __init__(self,
                 cfg: rorl_config,
                 actor: Actor,
                 actor_optim: torch.optim.Optimizer,
                 critic: EnsembledCritic,
                 critic_optim: torch.optim.Optimizer) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.actor = actor.to(self.device)
        self.actor_optim = actor_optim
        self.actor_target = deepcopy(actor).to(self.device)
        
        self.critic = critic.to(self.device)
        self.critic_optim = critic_optim
        self.critic_target = deepcopy(critic).to(self.device)

        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=self.cfg.alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

        self.total_iterations = 0
        self.tau = cfg.tau
        self.discount = cfg.discount
        self.epsilon = cfg.epsilon
        self.tau_rorl = cfg.tau_rorl
        self.beta_smooth = cfg.beta_smooth
        self.beta_ood = cfg.beta_ood

    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, _Number]:
        self.total_iterations += 1

        # critic loss
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_states, need_log_prob=True)

            q_next = self.critic_target(next_states, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == dones.shape == rewards.shape
            q_target = rewards + self.discount * (1 - dones) * q_next.unsqueeze(-1)
        
        states_noise = torch.rand_like(states) * self.epsilon  # [0, epsilon)
        perturbed_states = states + states_noise

        perturbed_q = self.critic(perturbed_states, actions)
        current_q = self.critic(states, actions)

        smoothing_loss = self.smoothing_loss(current_q, perturbed_q)

        perturbed_action, log_prob_perturbed = self.actor(perturbed_states, need_log_prob=True)

        double_perturbed_q = self.critic(perturbed_states, perturbed_action.detach())
        ood_loss, ensemble_std = self.ood_critic_loss(double_perturbed_q)

        critic_loss = F.mse_loss(current_q, q_target.squeeze(1)) + self.beta_smooth * smoothing_loss + self.beta_ood * ood_loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor loss
        policy_action, log_prob = self.actor(states, need_log_prob=True)
        q_value = self.critic(states, policy_action).min(0).values

        jensen_divergence = self.jensen_divergence(log_prob, log_prob_perturbed)
        actor_loss = (self.alpha * log_prob - q_value + self.cfg.beta_divergence * jensen_divergence).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha loss
        alpha_loss = (-self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().detach()

        self.soft_actor_update()
        self.soft_critic_update()

        return {
            "critic/smoothing_loss": smoothing_loss.item(),
            "critic/ood_loss": ood_loss.item(),
            "critic/ensemble_std": ensemble_std.item(),
            "critic/critic_loss": critic_loss.item(),
            "critic/q_perturbed_states": perturbed_q.mean().item(),
            "critic/q_fully_perturbed": double_perturbed_q.mean().item(),
            "critic/q_values": current_q.mean().item(),
            "actor/jensen_divergence": jensen_divergence.mean().item(),
            "actor/q_values": q_value.mean().item(),
            "actor/actor_loss": actor_loss.item(),
            "actor/alpha_loss": alpha_loss.item(),
            "actor/log_prob": log_prob.mean().item(),
            "actor/log_prob_perturbed": log_prob_perturbed.mean().item(),
            "actor/alpha": self.alpha.item(),
            "actor/actor_entropy": log_prob.mean().item()
        }

    def smoothing_loss(self,
                       current_q: torch.Tensor,
                       pertutbed_q: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(pertutbed_q)
        delta = F.mse_loss(current_q, pertutbed_q, reduction="none")

        delta_plus = torch.maximum(delta, zeros)
        delta_minus = torch.minimum(delta, zeros)

        loss = (1 - self.tau_rorl) * delta_plus + self.tau_rorl * delta_minus

        return loss.max()
    
    def ood_critic_loss(self,
                        q_vlaues: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        uncertainty_term = torch.std(q_vlaues, dim=0, correction=0)
        td_target = (q_vlaues - uncertainty_term).detach()

        return F.mse_loss(q_vlaues, td_target), uncertainty_term.mean()
    
    def jensen_divergence(self,
                          log_prob: torch.Tensor,
                          log_prob_perturbed: torch.Tensor) -> torch.Tensor:
        average_log_prob = (log_prob + log_prob_perturbed) / 2

        kl_div1 = log_prob.exp() * (log_prob - average_log_prob)
        kl_div2 = log_prob_perturbed.exp() * (log_prob_perturbed - average_log_prob)

        return (kl_div1 + kl_div2) / 2

    def soft_actor_update(self):
        for param, tgt_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "alpha_optimizer": self.alpha_optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["target_critic"])
        self.actor_target.load_state_dict(state_dict["target_actor"])
        self.actor_optim.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optim.load_state_dict(state_dict["critic_optimizer"])
        self.alpha_optim.load_state_dict(state_dict["alpha_optimizer"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()
