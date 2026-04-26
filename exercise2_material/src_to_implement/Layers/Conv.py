from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom
import numpy as np
from copy import deepcopy
from scipy.signal import correlate as scorrelate
from scipy.ndimage import correlate as ncorrelate
from scipy.ndimage import convolve as convolve
from scipy.signal import convolve as sconvolve

class Conv(BaseLayer):
    
    # Assert types of stride and convolution are correct
    str_shape = int or tuple
    conv_shape = list[int,int] or list[int,int,int]

    def __init__(self, stride_shape: str_shape, convolution_shape: conv_shape, num_kernels: int):
        
        # Trainable layer
        super().__init__()
        self.trainable = True
        
        # Check the dimensions of the stride and convolution and adapt the shape according to them
        if len(stride_shape) == 1:
            self.stride_shape = (stride_shape + stride_shape)   # Both dimensions are the same
        else:
            self.stride_shape = stride_shape                       
        
        if len(convolution_shape) == 2:
            self.convolution_shape = (convolution_shape + (1,)) # 2D convolution (c,m,1)
        else:
            self.convolution_shape = convolution_shape          # 3D convolution (c,m,n)
        
        #self.kernel_channels = self.convolution_shape[0]   # Number of kernel channels (b) or (S) a.k.a depth (for inputs and kernels)
        self.kernel_m = self.convolution_shape[1]           # Kernel height (y) dimension (m)
        self.kernel_n = self.convolution_shape[2]           # Kernel width (x) dimension (n)
        
        self.num_kernels = num_kernels                      # Number of kernels (H) a.k.a number of filters
        
        self.weights_shape = ((self.num_kernels,) + self.convolution_shape)               # (b,)+(c,y,x) -> (b,c,y,x)
        self.weights = UniformRandom().initialize(self.weights_shape, None, None)         # Initialize weights using the 4-tuple
        self.bias =  UniformRandom().initialize(self.num_kernels, None, None)             # Initialize bias separetly

        self.optimizer = None
        self.optimizer_weights = None
        self.optimizer_bias = None
    
    # Properties asked to return after calculated in backward pass
    @property
    def gradient_weights(self):
        return self.grad_weights
    
    @property
    def gradient_bias(self):
        return self.grad_bias


    def forward(self, input_tensor):

        # Check the input_tensor dimension and adapt to fit the weight dimension
        if len(input_tensor.shape) == 3:
            self.input_tensor = input_tensor.reshape((input_tensor.shape + (1,)))
        else:
            self.input_tensor = input_tensor

        (self.input_b, self.input_c, self.input_y, self.input_x) = self.input_tensor.shape
                           #(b,            num_kernels,      y,            x           )
        self.output_shape = (self.input_b, self.num_kernels, self.input_y, self.input_x)

        output_tensor = []  # Output tensor is a list of the correlation between all batches and channels

        for b in range(self.input_b):
            
            output_channel_b = []   # Output of all channels in batch b
            
            for c in range(self.num_kernels):
                # Correlation using zero-padding
                # We use correlation in this step to skip the 180 degree flip to the kernel
                input_weight_corr = ncorrelate(self.input_tensor[b], self.weights[c], None, 'constant')  
                # Apply stride to the correlated image
                # Also we append the bias in the corresponding channel
                input_weight_corr_stride = input_weight_corr[self.input_c // 2][::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[c]
                output_channel_b.append(input_weight_corr_stride)
            
            output_tensor.append(np.array(output_channel_b)) # Return to numpy array
        
        output_tensor = np.array(output_tensor) # Return to numpy array

        if output_tensor.shape[3] == 1:
            output_tensor = output_tensor[:,:,:,0]  # If the convolution is 1D, we eliminate the appended dimension
        
        return output_tensor

    # Properties asked to store the optimizer for the layer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = deepcopy(optimizer)
        self.optimizer_bias = deepcopy(optimizer)
        self.optimizer_weights = deepcopy(optimizer)    
    
    def backward(self, error_tensor):
        
        # Check the input_tensor dimension and adapt to fit the weight dimension
        if len(error_tensor.shape) == 3:
            error_tensor = error_tensor.reshape((error_tensor.shape+(1,)))
        else:
            error_tensor = error_tensor
        
        # Flip the weights in backward pass
        flipped_weights = np.transpose(self.weights, (1,0,2,3)) # Swap batches and channels
        flipped_weights = flipped_weights[:, ::-1, :, :] # Make batches go backwards

        # Upsample error tensor with the stride shape as it got reduced during forward pass
        upsampled_error_tensor = np.zeros(self.output_shape)                                        # Upsampled error tensor has the shape of the output tensor in forward pass
        upsampled_error_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor # Put the elements of the error tensor in the correct place
        
        # Step 1: convolution of error tensor and its weights
        prev_error_tensor = np.zeros(self.input_tensor.shape) # Previous error tensor shape is the same as the input tensor
        # Iterate over channels and batches
        for b in range(self.input_b):
            for c in range(flipped_weights.shape[0]):
                error_weight_conv = convolve(upsampled_error_tensor[b], flipped_weights[c], None, 'constant')
                prev_error_tensor[b][c] = error_weight_conv[self.num_kernels // 2]
        
        # Step 2: compute gradient w.r.t. to weights
        
        # Padding
        # Check if even or odd to pad in y direction
        if self.kernel_m % 2 == 0:
            padding_y1 = self.kernel_m // 2
            padding_y2 = padding_y1 -1
        else:
            padding_y1 = self.kernel_m // 2
            padding_y2 = padding_y1
        # Check if even or odd to pad in x direction
        if self.kernel_n % 2 == 0:
            padding_x1 = self.kernel_n // 2
            padding_x2 = padding_x1 -1
        else:
            padding_x1 = self.kernel_n // 2
            padding_x2 = padding_x1

        # Pad the input tensor with zeros (zero-padding)
        input_tensor_padded = np.pad(self.input_tensor, ((0,0), (0,0), (padding_y1, padding_y2), (padding_x1, padding_x2)), constant_values=(0,0))
        
        gradient_weights = np.zeros(self.weights.shape)

        # Compute the new weigths "similarly" to forward pass (no rotation)
        for c in range(self.num_kernels):
            new_weight = np.zeros(self.weights.shape[1:])
            for b in range(self.input_b):
                # Also change to signal correlate as the two inputs are same N-dimensional arrays
                new_weight += scorrelate(input_tensor_padded[b], np.expand_dims(upsampled_error_tensor[b][c], axis=0), mode='valid')    # Output consists only of those elements that do not rely on the zero-padding
            gradient_weights[c] = new_weight
        self.grad_weights = gradient_weights

        # Step 3: compute gradient w.r.t. to bias

        bias_sum = np.zeros(self.num_kernels) 
        gradient_bias = np.zeros(self.num_kernels)

        for b in range(self.input_b):
            for c in range(self.num_kernels):
                bias_sum[c] = np.sum(upsampled_error_tensor[b][c])
            gradient_bias += bias_sum

        self.grad_bias = gradient_bias

        # Step 4: optimize gradient

        if self.optimizer != None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self.grad_weights)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self.grad_bias)
        
        # Step 5: Adapt shape of prev_error_tensor to correct dimensions
        
        if prev_error_tensor.shape[3] == 1:
            prev_error_tensor = prev_error_tensor[:,:,:,0]  # If the convolution is 1D, we eliminate the appended dimension
        
        return prev_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        input_size = np.prod(self.convolution_shape)  # fan_in of CNN: mult(b, y, x) <- convolution_shape
        output_size = (self.num_kernels * self.kernel_m * self.kernel_n) # fan_out of CNN: mult(H,y,x)
        self.weights = weights_initializer.initialize(self.weights_shape, input_size, output_size) # reinitialize weights and biases
        self.bias = bias_initializer.initialize(self.num_kernels, input_size, output_size)