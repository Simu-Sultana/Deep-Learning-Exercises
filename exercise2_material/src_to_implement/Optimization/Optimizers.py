import numpy as np

class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k_1 = 0 # v^(k-1)
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        # v^(k) = mu * v^(k-1) - eta * gradient
        self.v_k_1 = self.momentum_rate * self.v_k_1 - self.learning_rate * gradient_tensor
        return weight_tensor + self.v_k_1

class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k_1 = 0
        self.r_k_1 = 0
        self.k = 1
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_k_1 = self.mu * self.v_k_1 + (1-self.mu) * gradient_tensor
        v_k_sombrerito = self.v_k_1 / (1 - self.mu**self.k)

        self.r_k_1 = self.rho * self.r_k_1 + (1-self.rho) * gradient_tensor * gradient_tensor
        r_k_sombrerito = self.r_k_1 / (1 - self.rho**self.k)

        self.k += 1

        return weight_tensor - self.learning_rate * (v_k_sombrerito / (np.sqrt(r_k_sombrerito) + np.finfo(float).eps))
