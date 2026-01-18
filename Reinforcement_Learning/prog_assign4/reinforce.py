from typing import Iterable
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.state_dims = state_dims
        self.num_action = num_actions
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(in_features=state_dims, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            # nn.Linear(in_features=32, out_features=32),
            # nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_actions),
            nn.Softmax(dim=-1)
        )
        self.net.double()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=alpha, betas=(0.9, 0.999))


    def __call__(self,s) -> int:
        # TODO: implement this method
        self.net.eval()
        s = torch.tensor(s, dtype=torch.double)
        actions_probs = self.net(s)
        action_dist = torch.distributions.Categorical(actions_probs)
        action = action_dist.sample().item()
        return action
        

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        self.net.train()
        s = torch.tensor(s, dtype=torch.double)
        probs = self.net(s)
        pi_a = probs[a]
        log_pi = torch.log(pi_a)
        loss = -gamma_t * delta * log_pi
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.net = nn.Sequential(
            nn.Linear(in_features=state_dims, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.net.double()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.loss = torch.nn.MSELoss()

    def __call__(self,s) -> float:
        self.net.eval()
        s = torch.tensor(s).to(torch.double)
        v = self.net(s)
        return v.item()

    def update(self,s,G):
        # TODO: implement this method
        self.net.train()
        s = torch.tensor(s, dtype=torch.double)
        G = torch.tensor([G], dtype=torch.double)
        val = self.net(s)
        loss = self.loss(val, G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    Gs = []
    
    for _ in tqdm(range(num_episodes)):
        state = env.reset()
        action = pi(state)
        states = [state]
        actions = [action]
        rewards = []
        # generate an episode
        while True:
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            states.append(next_state)
            action = pi(next_state)
            actions.append(action)

        episode_len = len(rewards)
        for t in range(episode_len):
            G = np.sum([gamma ** (k-t-1) * rewards[k-1] for k in range(t+1, episode_len+1)])
            if t == 0:
                Gs.append(G)
            state = states[t]
            action = actions[t]
            gamma_t = gamma ** t
            delta = G - V(state)
            V.update(state, G)
            pi.update(state, action, gamma_t, delta)
    return Gs
