import numpy as np

class commander():
    def __init__(self):
        self.target_position = np.array([0.0, 0.0, 100.0])
        self.n_td = np.array([0.0, 0.0, 0.0])