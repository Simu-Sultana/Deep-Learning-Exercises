import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()                    # Call the super constructor
        self.trainable = True                 # Set and create the attributes of the class
        self.weights = np.random.uniform(0,1, (input_size+1, output_size)) # Weights with the size of the bias included
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None                # Protected attribute
        self.input_tensor_with_bias = None    # Important to access it in backward pass!

    def forward(self, input_tensor):
        batch_size, _ = np.shape(input_tensor)                                 # Row size of the input tensor + 1
        bias = np.ones(batch_size)
        bias = bias[:, None]                                                   # Fill the bias with ones and make it 2-dimensional
        input_tensor_with_bias = np.concatenate((input_tensor, bias), axis=1)  # Append the bias to the input tensor
        self.input_tensor_with_bias = input_tensor_with_bias                   # Important to access it in backward pass!
        output_tensor = input_tensor_with_bias @ self.weights                  # Compute output tensor
        _, self.output_size = np.shape(output_tensor)
        return output_tensor

    # Defining the getter and setter of the optimizer attribute
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        previous_error_tensor = error_tensor @ self.weights.T               # X'*W' = Y'
        self.gradient_weight = self.input_tensor_with_bias.T @ error_tensor # Compute the gradient of the weights

        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weight) # Compute the new weights

        return previous_error_tensor[:,:-1] # Return the Y' but without the last row (column in our memory layout)

    @property
    def gradient_weights(self):
        return self.gradient_weight

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.gradient_weights = gradient_weights

