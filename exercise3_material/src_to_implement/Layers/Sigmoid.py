import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        self.sigmoide = 1 / (1 + np.exp(-input_tensor))
        return self.sigmoide

    def backward(self, error_tensor):
        return error_tensor * (self.sigmoide * (1-self.sigmoide))