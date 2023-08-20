import torch
from config import rorl_config
from rorl import RORL
from modules import Actor, EnsembledCritic
from dataset import ReplayBuffer
from tqdm import tqdm
import wandb


class RORLTrainer:
    def __init__(self,
                 cfg=rorl_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6

        actor = Actor(self.state_dim, self.action_dim, hidden_dim=cfg.hidden_dim, edac_init=True)
        actor_optim = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_learning_rate)
        critic = EnsembledCritic(self.state_dim,
                                 self.action_dim,
                                 cfg.hidden_dim,
                                 num_critics=cfg.num_critics)
        critic_optim = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_learning_rate)

        self.rorl = RORL(cfg,
                         actor,
                         actor_optim,
                         critic,
                         critic_optim)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.device}🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            for _ in tqdm(range(self.cfg.max_timesteps), desc="RORL steps"):
                
                batch = self.buffer.sample(self.cfg.batch_size)
                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.rorl.train(states,
                                               actions,
                                               rewards,
                                               next_states,
                                               dones)
                
                wandb.log(logging_dict, step=self.rorl.total_iterations)
