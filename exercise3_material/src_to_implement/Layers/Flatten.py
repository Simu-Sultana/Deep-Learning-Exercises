from Layers.Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor_shape = np.shape(input_tensor)                        # Save the previous shape for backward pass
        return np.ravel(input_tensor).reshape(self.input_tensor_shape[0], -1)   # (b,c,y,x) -> (b, c*y*x)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_tensor_shape)                    # Return the error tensor to previous shape
                                                                                # (b,c,y,x) <- (b, c*y*x)