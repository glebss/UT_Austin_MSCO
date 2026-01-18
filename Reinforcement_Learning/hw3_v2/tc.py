import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = (np.ceil((state_high - state_low) / tile_width) + 1).astype(int)
        self.total_tiles = int(np.prod(self.num_tiles) * self.num_tilings)
        self.offsets = (tile_width / num_tilings) * np.arange(num_tilings)[:, None]
        self.weights = np.zeros(self.total_tiles)
    
    def get_tile_indices(self, s):
        indices = []
        for i in range(self.num_tilings):
            grid_position = np.floor((s - self.state_low + self.offsets[i]) / self.tile_width).astype(int)
            index = np.ravel_multi_index(grid_position, self.num_tiles)
            indices.append(index + i * np.prod(self.num_tiles))
        return indices

    def __call__(self,s):
        # TODO: implement this method
        tile_indices = self.get_tile_indices(s)
        return np.sum(self.weights[tile_indices])

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        tile_indices = self.get_tile_indices(s_tau)
        estimate = np.sum(self.weights[tile_indices])
        delta = alpha * (G - estimate)
        self.weights[tile_indices] += delta / self.num_tilings

if __name__ == "__main__":
    v = ValueFunctionWithTile(0, 2, 3, 2)
    ind = v.get_tile_indices(1)
