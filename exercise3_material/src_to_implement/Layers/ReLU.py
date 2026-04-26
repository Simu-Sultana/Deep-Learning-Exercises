from Layers.Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):

    def __init__(self):
        super().__init__
        self.trainable = False
    
    def forward(self, input_tensor):
        output_tensor = input_tensor * (input_tensor > 0)               # max(0, input_tensor) element-wise
        self.input_tensor = input_tensor                                # Save input tensor for backward pass                          
        return output_tensor 
    
    def backward(self, error_tensor):
        previous_error_tensor =  error_tensor * (self.input_tensor > 0) # e_n if x > 0 else 0
        return previous_error_tensor
