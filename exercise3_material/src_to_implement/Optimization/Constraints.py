import numpy as np

class L2_Regularizer:
    
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * np.linalg.norm(weights)**2             # Squared L2 norm

class L1_Regularizer:
    
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.linalg.norm(np.ravel(weights), 1)   # L1 norm along a 1D vector

        
