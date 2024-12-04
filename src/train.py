from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import os
from joblib import dump, load
from RF_FQI_agent import RandomForestFQI
from DQN_agent import ReplayBuffer, DQN_AGENT
from DQNNetwork import DQNNetwork

env = TimeLimit(
                env = HIVPatient(domain_randomization=False),
                max_episode_steps = 200
                )  


# ENJOY!
class ProjectAgent:
    def __init__(self, name = 'DQN'):
        self.env = TimeLimit(
                env = HIVPatient(domain_randomization=False),
                max_episode_steps = 200
                    )  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name 
        self.original_env = self.env.env
        self.nb_actions = int(self.original_env.action_space.n)
        self.model = DQNNetwork(self.env.observation_space.shape[0], 64, self.nb_actions).to(self.device)
        self.nb_neurons = 256


    def act(self, observation, use_random = False):
        if use_random:
            return self.env.action_space.sample()
        elif self.name == 'RF_FQI':
            agent = self.load()
            return agent.greedy_action(observation)
        elif self.name == 'DQN':
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        else:
            raise ValueError("Unknown model")

    def save(self):
        if self.name == 'RF_FQI':
            filename = 'src/models/RF_FQI/Qfct'
            model = {'Qfunction': self.agent.rf_model}
            dump(model, filename, compress=9)
        elif self.name == 'DQN':
            filename = "src/models/DQN/config1.pt"
            torch.save(self.model.state_dict(), filename)


    def load(self):
        if self.name == 'RF_FQI':
            loaded_data = load("src/models/RF_FQI/Qfct")
            self.Qfunctions = loaded_data['Qfunctions']
        elif self.name == 'DQN':
            device = torch.device('cpu')
            state_dim, nb_neurons, n_action = self.config['state_dim'], self.config['nb_neurons'], self.nb_actions
            self.model = self.model(state_dim, self.nb_neurons, self.nb_actions).to(device)
            self.model.load_state_dict(torch.load("src/models/DQN/config1.pt", map_location = device))
            self.model.eval()