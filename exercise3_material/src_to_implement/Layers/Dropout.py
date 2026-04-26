from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):

    def __init__(self, probability):
        super().__init__()
        self.probability = probability
    
    def forward(self, input_tensor):
        # Only use dropout in train phase
        if not self.testing_phase: 
            self.probability_mask = (np.random.random(input_tensor.shape) < self.probability)
            # (random(input_tensor shape) < prob) is equivalent to 1-prob
            return self.probability_mask * input_tensor * (1/self.probability)
        # Test phase
        else:
            return input_tensor

    def backward(self, error_tensor):
        return self.probability_mask * error_tensor * (1/self.probability)

    # Why 1/p?
    # https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
    # #Use a Larger Network part
