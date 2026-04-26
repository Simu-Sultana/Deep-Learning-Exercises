from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import numpy as np

class BatchNormalization(BaseLayer):
    
    def __init__(self, channels, alpha=0.8):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self.channels = channels
        self.alpha = alpha                                          # Moving average decay
        self.initialize()
    
    def initialize(self, ignore_weights=None, ignore_bias=None):    # Initialize weights independently from the assigned initializer
        self.bias = np.zeros((1, self.channels))
        self.weights = np.ones((1, self.channels))
        self.first_weights = True                                   # Mark that these are the first weights and biases for computation
                                                                    # of the first mean and variance in forward pass 
    
    def forward(self, input_tensor):
        
        self.convolutional_case = (len(input_tensor.shape) == 4)    # To check faster at forward/backward pass

        if self.convolutional_case:                                 # Convolutional case 4D
            self.input_tensor = self.reformat(input_tensor)     
        else:                                                       # Fully connected case 2D
            self.input_tensor = input_tensor                    

        # First mean and variance computation
        if self.first_weights:
            self.mean = np.mean(self.input_tensor, axis=0)              # First mean and first variance
            self.variance = np.std(self.input_tensor, axis=0) ** 2      
            self.meanK = self.mean                                      # Also assign now the (k) iteration for further computation of the mean average
            self.varK = self.variance
            self.first_weights = False                                  # Unmark first weights

        # Training phase
        if not self.testing_phase:
            meanB = np.mean(self.input_tensor, axis=0)
            varB = np.std(self.input_tensor, axis=0) ** 2
            self.input_normalized = (self.input_tensor - meanB) / (np.sqrt(varB + np.finfo(float).eps))
            output_tensor = self.weights * self.input_normalized + self.bias
        
            if self.convolutional_case:
                output_tensor = self.reformat(output_tensor)
            
            # Moving average
            self.meanK = (self.alpha * self.meanK) + ((1 - self.alpha) * meanB)
            self.varK = (self.alpha * self.varK) + ((1 - self.alpha) * varB)

        # Testing phase
        else:
            self.input_normalized = (self.input_tensor - self.meanK) / (np.sqrt(self.varK - np.finfo(float).eps))   # Same as training phase but with the global mean and variance
            output_tensor = self.weights * self.input_normalized + self.bias

            if self.convolutional_case:
                output_tensor = self.reformat(output_tensor)

        return output_tensor
    
    def backward(self, error_tensor):
        
        if self.convolutional_case:
            error_tensor = self.reformat(error_tensor)     
        # otherwise fully connected case

        gradient_input = compute_bn_gradients(error_tensor, self.input_tensor, 
                                              self.weights, self.mean, self.variance)                  # Gradients w.r.t. input_tensor                    
        self.gradient_weights = np.sum(self.input_normalized * error_tensor, axis=0, keepdims=True)    # Gradient w.r.t. weights
        self.gradient_bias = np.sum(error_tensor, axis=0, keepdims=True)                               # Gradient w.r.t. bias
    
        # Compute update if optimizer is set
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        
        if self.convolutional_case:
            gradient_input = self.reformat(gradient_input)
        
        return gradient_input


    def reformat(self, input_tensor):
        
        # Image (4) to vector (2)
        if len(input_tensor.shape) == 4:
            self.b, self.h, self.m, self.n = input_tensor.shape
            reshaped = input_tensor.reshape(self.b, self.h, self.m * self.n)
            transposed = np.transpose(reshaped, (0,2,1))
            reshaped_again = transposed.reshape(self.b * self.m * self.n, self.h)
            return reshaped_again
        # Vector (2) to image (4)
        else:
            reshaped = input_tensor.reshape(self.b, self.m * self.n, self.h)
            transposed = np.transpose(reshaped, (0,2,1))
            reshaped_again = transposed.reshape(self.b, self.h, self.m, self.n)
            return reshaped_again
    
    
    ## Properties ##
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def grad_weights(self):
        return self.gradient_weights
    
    @grad_weights.setter
    def grad_weight(self, gradient_weights):
        self.gradient_weights = gradient_weights

    @property
    def grad_bias(self):
        return self.gradient_bias

    @grad_bias.setter
    def grad_bias(self, gradient_bias):
        self.gradient_bias = gradient_bias

            
    
    
    
