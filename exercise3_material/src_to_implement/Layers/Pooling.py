import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.masks = {}     # Need this later for storing the max values positions

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.b, self.c, self.y, self.x = input_tensor.shape
        y_pool, x_pool = self.pooling_shape
        y_out = 1 + (self.y - y_pool) // self.stride_shape[0] # Times we apply the kernel vertically
        x_out = 1 + (self.x - x_pool) // self.stride_shape[1] # Times we apply the kernel horizontally
        output_tensor = np.zeros((self.b, self.c, y_out, x_out)) # Shape of output tensor filled with zeros

        for i in range(y_out):
            for j in range(x_out):
                y_start = i * self.stride_shape[0] # Y start position of kernel updated in each interation
                y_end = y_start + y_pool
                x_start = j * self.stride_shape[1] # X start position of kernel updated in each iteration
                x_end = x_start + x_pool
                input_tensor_prev_slide = self.input_tensor[:, :, y_start:y_end, x_start:x_end]     # Select area of input tensor influenced by pooling kernel
                self.maskenpflicht(slice=input_tensor_prev_slide, coords=(i,j))                     # Create mask with 1 in the position of the max value
                output_tensor[:,:,i,j] = np.max(input_tensor_prev_slide, axis=(2,3))                # Select maximum value of the input tensor's area under the kernel
        
        return output_tensor

    def backward(self, error_tensor):
        prev_error_tensor = np.zeros(self.input_tensor.shape)
        _, _, y_out, x_out = error_tensor.shape
        y_pool, x_pool = self.pooling_shape

        for i in range(y_out):
            for j in range(x_out):
                y_start = i * self.stride_shape[0]      # Y start position of kernel
                y_end = y_start + y_pool
                x_start = j * self.stride_shape[1]      # X start position of kernel
                x_end = x_start + x_pool
                prev_error_tensor[:,:,y_start:y_end, x_start:x_end] += error_tensor[:,:, i:i+1, j:j+1] * self.masks[(i,j)]  # Error accumulates in the position of the max values  
                                                                                                                            # self.masks contains 1s in the positions of the max values of input tensor
        
        return prev_error_tensor
    
    def maskenpflicht(self, slice, coords):
        mask = np.zeros(slice.shape)
        b, c, y, x = slice.shape
        slice = slice.reshape(b, c, y*x)        # Same as np.reshape(slice,(b,c,y*x))
        idslice = np.argmax(slice, axis=2)      # Indices of the max value in the slice

        b_idslice, c_idslice = np.indices((b,c))
        mask.reshape(b,c,y*x)[b_idslice, c_idslice, idslice] = 1        # Place 1 in position of max value
        self.masks[coords] = mask               # This will store in a dictionary the mask for each layer containing 1s in the position of the max values