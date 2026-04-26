from Layers import Base, FullyConnected, TanH, Sigmoid
import numpy as np

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        # arguments required in Description.pdf
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(self.hidden_size)

        # arguments required to run the RNN
        self.input_node_size = self.input_size + self.hidden_size
        self.FullyConNode1 = FullyConnected.FullyConnected(self.input_node_size, self.hidden_size) # As they will form a FullyConnected layer
        self.FullyConNode2 = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.Tanh = TanH.TanH()
        self.Sig = Sigmoid.Sigmoid()

        # arguments of the @property (s)
        self._optimizer = None
        self.memory = False

    #  Memorize property to represent if the RNN regards subsequent sequences
    # as a belonging to the same long sequence
    @property
    def memorize(self):
        return self.memory

    @memorize.setter
    def memorize(self, memorize):
        self.memory = memorize

    # Forward pass        
    def forward(self, input_tensor):

        time = input_tensor.shape[0] # batch dimension is the time dimension of a sequence
                                        # over which the recurrence is performed

        # these will be stored for backward pass
        self.store_fullyCon1 = []
        self.store_tanh = []
        self.store_fullyCon2 = []
        self.store_Sig = []
        self.next_input = np.zeros((time, self.output_size)) # output of forward pass

        if self.memory is False:
            self.hidden_state = np.zeros(self.hidden_size) # hidden state for iteration is zero if memory is False

        
        for i in range(time):
            # First step in the cell: join input and hidden state
            input_node = np.concatenate((self.hidden_state.reshape(self.hidden_size,1),
                                        input_tensor[i].reshape(input_tensor.shape[1],1)))
            
            # Perform forward pass of FullyConnected layer for the input node
            fullyCon1 = self.FullyConNode1.forward(input_node.T)
            self.store_fullyCon1.append(self.FullyConNode1.input_tensor_with_bias) # Store for backward pass ONLY with bias (no weights)

            self.hidden_state = self.Tanh.forward(fullyCon1) # Perform forward pass of Tanh for the output of previous operation
            self.store_tanh.append(self.Tanh.tanh) # Store variable for backward pass

            fullyCon2 = self.FullyConNode2.forward(self.hidden_state)   # Perform forward pass of FullyConnected layer for the hidden state
            self.store_fullyCon2.append(self.FullyConNode2.input_tensor_with_bias) # Again, store ONLY with bias

            self.next_input[i] = self.Sig.forward(fullyCon2) # Perform forward pass of Sigmoid for the output of node2
            self.store_Sig.append(self.Sig.sigmoide) # Store output of forward pass
        
        return self.next_input

    # Backward pass    
    def backward(self, error_tensor):
        time = error_tensor.shape[0]
        prev_error_tensor = np.zeros((time, self.input_size))
        # initialize all gradients to 0
        gradient_hidden = np.zeros(self.hidden_size)
        self.grad2 = 0
        self.grad1 = 0

        for i in reversed(range(time)): # iterate over all the time steps starting from the end

            # Follow the cell steps backward, so use backward method for each layer
            # In each step we use previously stored variables to compute the backward pass
            # In the FullyConnected layers we compute the gradient of the weights

            self.Sig.sigmoide = self.store_Sig[i]
            Sig_back = self.Sig.backward(error_tensor[i])
            
            self.FullyConNode2.input_tensor_with_bias = self.store_fullyCon2[i]    # get input_tensor_with_bias to compute backward
            fullyCon2_back = self.FullyConNode2.backward(Sig_back)
            self.grad2 += self.FullyConNode2.gradient_weight
            
            error_node2 = gradient_hidden + fullyCon2_back                         # Accumulate error in hidden layer
            
            self.Tanh.tanh = self.store_tanh[i] 
            Tanh_back = self.Tanh.backward(error_node2)
            
            self.FullyConNode1.input_tensor_with_bias = self.store_fullyCon1[i]
            fullyCon1_back = self.FullyConNode1.backward(Tanh_back)
            self.grad1 += self.FullyConNode1.gradient_weight

            prev_error_tensor[i] = np.squeeze(fullyCon1_back.T[self.hidden_size::]) # remove bias and undo concatenate
            gradient_hidden = np.squeeze(fullyCon1_back.T[0:self.hidden_size])    

        # Optimization
        # Compute the new weights for each node
        if self._optimizer != None:
            self.FullyConNode1.weights = self._optimizer.calculate_update(self.FullyConNode1.weights, self.grad1)
            self.FullyConNode2.weights = self._optimizer.calculate_update(self.FullyConNode2.weights, self.grad2)

        return prev_error_tensor

    @property
    def gradient_weights(self):
        return self.grad1 # weights which are involved in calculating the hidden state

    # Getter and setter of the weights
    @property
    def weights(self):
        return self.FullyConNode1.weights

    @weights.setter
    def weights(self, w):
        self.FullyConNode1.weights = w

    # Add optimizer for regularization 
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    
    # To use initializers when weights and bias initializer are set
    def initialize(self, weights_initializer, bias_initializer):
        if (weights_initializer != None) and (bias_initializer != None):
            self.FullyConNode1.initialize(weights_initializer, bias_initializer)
            self.FullyConNode2.initialize(weights_initializer, bias_initializer)