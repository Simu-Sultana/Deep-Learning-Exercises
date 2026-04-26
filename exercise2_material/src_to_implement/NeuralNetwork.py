import numpy as np
import copy

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None          # The object containing all the data which will be accessed in the forward pass
        self.loss_layer = None          # The object containing the final/initial loss layer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()            # Call for the current batch of data, also save label tensor to
        for layer in self.layers:                                           # access it in the backward pass
            input_tensor = layer.forward(input_tensor)                      # Iterate over the layers recursively
        return self.loss_layer.forward(input_tensor, self.label_tensor)     # Return the loss of the current forward pass iteration

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)          # Call the error tensor using the current batch labels
        for layer in self.layers[::-1]:                                     # Iterate recursively over the layers backwards
            error_tensor = layer.backward(error_tensor)
    
    def append_layer(self, layer):                              # Just append the layers to the network
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    
    def train(self, iterations):
        for _ in range(iterations):                         # Train for ? iterations
            self.loss.append(self.forward())                # Save loss of the forward pass in the list
            self.backward()                                 
    
    def test(self, input_tensor):
        for layer in self.layers:                           # If the list only contains layers until softmax we dont need to 
            input_tensor = layer.forward(input_tensor)      # explicitly go forward until that layer, thats not our business
        return input_tensor
        


