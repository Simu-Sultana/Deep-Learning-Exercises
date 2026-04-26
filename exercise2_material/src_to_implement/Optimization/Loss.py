import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        return -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))
                #sum elementwise -ln(prediction + eps) only when label is 1 (correct classification)       

    def backward(self, label_tensor):
        return -label_tensor / (self.prediction_tensor+np.finfo(float).eps) #is this slower or equal to np.divide?
