from Layers.Base import BaseLayer
import numpy as np


class Constant(BaseLayer):

    def __init__(self, con=0.1):
        super().__init__()
        self.con = con

    def initialize(self, weights_shape, fan_in, fan_out):
        self.initial_tensor = np.full(shape=weights_shape, fill_value=self.con) 
        return self.initial_tensor

class UniformRandom(BaseLayer):

    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out):
        self.initial_tensor = np.random.uniform(size=weights_shape)
        return self.initial_tensor
    
class Xavier(BaseLayer):

    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in+fan_out))
        self.initial_tensor = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return self.initial_tensor

class He(BaseLayer):

    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        self.initial_tensor = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return self.initial_tensor
