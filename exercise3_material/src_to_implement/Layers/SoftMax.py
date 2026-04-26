from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def _init_(self):
        super().__init__

    def forward(self, input_tensor):
        xk_stable=input_tensor-np.max(input_tensor)                # shift xk to increase stability 
        yk_num=np.exp(xk_stable)                                   
        yk_den=np.sum(np.exp(xk_stable),axis=1,keepdims=True)      # axis=1 sums across the rows
        self.output_tensor=yk_num/yk_den                           # change to np.divide?
        return self.output_tensor

    def backward(self, error_tensor):
        parenthesis=error_tensor-np.sum(np.multiply(error_tensor,self.output_tensor),axis=1,keepdims=True)   # (En - sum of Enj * predj)
        return self.output_tensor*parenthesis