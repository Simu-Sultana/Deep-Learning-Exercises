import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
    
    def regularizer_loss(self, weights):                            # Regularizer to apply in NeuralNetwork
        if self.regularizer:
            return self.regularizer.norm(weights)
        else:
            return 0

    def regularizer_gradient(self, weights):                        # Regularizer to apply in Optimizers
        return self.regularizer.calculate_gradient(weights)

class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super(Sgd, self).__init__(learning_rate)
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        
        if self.regularizer:                                             # Apply regularization if available
            regularization = self.regularizer_gradient(weight_tensor)    # Add regularization to gradient tensor
            gradient_tensor = gradient_tensor + regularization
        
        return weight_tensor - self.learning_rate * gradient_tensor 
    
class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super(SgdWithMomentum, self).__init__(learning_rate)
        self.momentum_rate = momentum_rate
        self.v_k_1 = 0 # v^(k-1)
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        # v^(k) = mu * v^(k-1) - eta * gradient
        self.v_k_1 = self.momentum_rate * self.v_k_1 - self.learning_rate * gradient_tensor
        return_weight_tensor = weight_tensor + self.v_k_1
        
        if self.regularizer:                                                                    # Apply regularization if available
            regularization = self.regularizer_gradient(weight_tensor)                           # And update the weight tensor
            return_weight_tensor = return_weight_tensor - self.learning_rate * regularization
        
        return return_weight_tensor

class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super(Adam, self).__init__(learning_rate)
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

        return_weight_tensor = weight_tensor - self.learning_rate * (v_k_sombrerito / (np.sqrt(r_k_sombrerito) + np.finfo(float).eps))

        if self.regularizer:                                                                    # Same as with SGDMomentum
            regularization = self.regularizer_gradient(weight_tensor)                           # Apply regularizer and update weight tensor
            return_weight_tensor = return_weight_tensor - self.learning_rate * regularization

        return return_weight_tensor