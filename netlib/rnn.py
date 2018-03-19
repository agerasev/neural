import numpy as np

from .layer import *
from .net import *


class RNN(Net):
    def __init__(self, sizes):
        self.sizes = sizes
        sx, sh, sy = sizes
        layers = [
            Affine(sx, sh), # W_xh
            Affine(sh, sh), # W_hh
            Affine(sh, sy), # W_hy
            Tanh()
        ]
        links = [
            ((-1, 0), 0), # x
            ((-1, 1), 1), # h_in
            (0, 3),
            (1, 3),
            (3, 2),
            (3, (-1, 1)), # h_out
            (2, (-1, 0)), # y
        ]
        super().__init__(layers, links)
    
    def newstate(self, count):
        return np.zeros((count, self.sizes[1]), dtype=np.float64)

class LSTM(Net):
    def __init__(self, sizes):
        self.sizes = sizes
        sx, sh, sy = sizes
        layers = [
            Affine(sx, sh), #  0, W_xh
            Affine(sh, sy), #  1, W_hy
            Affine(sh, sh), #  2, W_hc
            Tanh(),         #  3, a_hc
            Affine(sh, sh), #  4, W_f
            Sigmoid(),      #  5, a_f
            Affine(sh, sh), #  6, W_i
            Sigmoid(),      #  7, a_i
            Product(),      #  8, p_ih
            Affine(sh, sh), #  9, W_o
            Sigmoid(),      # 10, a_o
            Product(),      # 11, p_cf
            Tanh(),         # 12, a_ch
            Product(),      # 13, p_ho
            Uniform(),      # 14, _xh
            Uniform(),      # 15, _ch
        ]
        links = [
            ((-1, 0), 0), # x
            (0, 14),
            ((-1, 1), 14),# h_in
            ((-1, 2), (8, 0)), # c_in
            (14, 4),
            (4, 5),
            (5, (8, 1)),
            (14, 6),
            (6, 7),
            (7, (13, 0)),
            (14, 2),
            (2, 3),
            (3, (13, 1)),
            (8, 15),
            (13, 15),
            (15, (-1, 2)), # c_out
            (15, 12),
            (12, (11, 0)),
            (14, 9),
            (9, 10),
            (10, (11, 1)),
            (11, (-1, 1)), # h_out
            (11, 1),
            (1, (-1, 0)), # y
        ]
        super().__init__(layers, links)
    
    def newstate(self, count):
        return (
            np.zeros((count, self.sizes[1]), dtype=np.float64),
            np.zeros((count, self.sizes[1]), dtype=np.float64),
        )
