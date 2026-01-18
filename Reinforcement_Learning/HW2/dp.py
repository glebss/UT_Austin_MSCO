from typing import Tuple

import numpy as np
from env import EnvWithModel
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


def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    nS, nA = env.spec.nS, env.spec.nA
    gamma = env.spec.gamma
    V = initV
    Q = np.zeros((nS, nA))
    while True:
        delta = 0
        q_delta = 0
        for i in range(nS):
            v = V[i]
            V_new = 0
            for j in range(nA):
                q = Q[i, j]
                V_new += pi.action_prob(i, j) * np.sum([env.TD[i, j, snext] * (env.R[i, j, snext] + gamma * V[snext]) for snext in range(nS)])
                Q[i, j] = np.sum([env.TD[i, j, snext] * (env.R[i, j, snext] + gamma * V[snext]) for snext in range(nS)])
                q_delta = max(q_delta, abs(q - Q[i, j]))
            V[i] = V_new
            delta = max(delta, abs(v - V[i]))
            
        if delta < theta and q_delta < theta:
            break

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    nS, nA = env.spec.nS, env.spec.nA
    gamma = env.spec.gamma
    V = initV
    while True:
        delta = 0
        for i in range(nS):
            v = V[i]
            actions_arr = []
            for j in range(nA):
                actions_arr.append(
                    np.sum([env.TD[i, j, snext] * (env.R[i, j, snext] + gamma * V[snext]) for snext in range(nS)])
                )
            V[i] = np.max(actions_arr)
            delta = max(delta, abs(v - V[i]))
        if delta < theta:
            break
    
    state_actions = np.zeros((nS, nA))
    for i in range(nS):
        for j in range(nA):
            state_actions[i, j] = np.sum([env.TD[i, j, snext] * (env.R[i, j, snext] + gamma * V[snext]) for snext in range(nS)])
    state_actions = np.argmax(state_actions, axis=1)
    pi = OptimalPolicy(state_actions)

    return V, pi
