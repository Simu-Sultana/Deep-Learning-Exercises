import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        self.tanh = np.tanh(input_tensor)
        return self.tanh

    def backward(self, error_tensor):
        return error_tensor * (1 - self.tanh**2)