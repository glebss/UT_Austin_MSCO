from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy


class OptimalPolicy(Policy):
    
    def __init__(self, state_actions):
        super().__init__()
        self.state_actions = state_actions
    
    def action_prob(self, state, action):
        if self.state_actions[state] == action:
            return 1
        return 0
    
    def action(self, state):
        return self.state_actions[state]
    
    def update_policy(self, state_actions):
        self.state_actions = state_actions


def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    V = initV
    gamma = env_spec.gamma
    for episode in trajs:
        G = 0
        T = len(episode)
        for t, tup in enumerate(episode):
            s, a, r, snext = tup
            tau = t - n + 1
            G += gamma ** (t % n) * r
            if tau >= 0:
                if tau + n < T:
                    G = G + gamma ** n * V[snext]
                Stau = episode[tau][0]
                V[Stau] = V[Stau] + alpha * (G - V[Stau])
                G = 0

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    Q = initQ
    gamma = env_spec.gamma
    state_actions = np.argmax(Q, axis=1)
    pi = OptimalPolicy(state_actions)
    for episode in trajs:
        T = len(episode)
        G = 0
        rho = 1
        for t, tup in enumerate(episode):
            s, a, r, snext = tup
            tau = t - n + 1
            G += gamma ** (t % n) * r
            if t < T and t % n != 0:
                rho *= pi.action_prob(s, a) / bpi.action_prob(s, a)
            
            if tau >= 0:
                if tau + n < T:
                    snext, anext = episode[t+1][0:2]
                    G = G + gamma ** n * Q[snext, anext]
            stau, atau = episode[tau][0:2]
            Q[stau][atau] += alpha * rho * (G - Q[stau][atau])
            pi.update_policy(np.argmax(Q, axis=1))
            G = 0
            rho = 1 
                

    return Q, pi
