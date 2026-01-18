import numpy as np
from tqdm import tqdm

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = (np.ceil((state_high - state_low) / tile_width) + 1).astype(int)
        self.offsets = (tile_width / num_tilings) * np.arange(num_tilings)[:, None]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return self.num_actions * self.num_tilings * np.prod(self.num_tiles)
    
    def get_indices(self, s, a):
        indices = []
        for i in range(self.num_tilings):
            grid_position = np.floor((s - self.state_low + self.offsets[i]) / self.tile_width).astype(int)
            ind = np.ravel_multi_index(grid_position, self.num_tiles)
            offset_action = a * self.num_tilings * np.prod(self.num_tiles)
            indices.append(int(ind + i * np.prod(self.num_tiles) + offset_action))
        return indices

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        features = np.zeros(self.feature_vector_len(), dtype=np.int32)
        if done:
            return features

        indices = self.get_indices(s, a)
        features[indices] = 1
        return features

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    for episode in tqdm(range(num_episode)):
        state = env.reset()
        action = epsilon_greedy_policy(state, False, w)
        feature = X(state, False, action)
        z = np.zeros(X.feature_vector_len())
        Q_old = 0
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            action = epsilon_greedy_policy(next_state, done, w)
            feature_dash = X(next_state, done, action)
            Q = np.sum(w * feature)
            Q_dash = np.sum(w * feature_dash)
            delta = reward + gamma * Q_dash - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.sum(z * feature)) * feature
            w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * feature
            Q_old = Q_dash
            feature = feature_dash
            state = next_state

    return w
