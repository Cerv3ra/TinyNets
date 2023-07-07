import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
# Dense layer
class Layer_Dense:
# Layer initialization
    def __init__(self, n_inputs, n_neurons):
# Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

# Forward pass
    def forward(self, inputs):
# Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Let's see output of the first few samples:
print(dense1.output[:5])


inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []

for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)

#print(output)

#you can also do this, but it feels more.. mentally painful for me at this point

output2= []
for i in inputs:
    output2.append(max(i,0))

#print(output2)
#max of a 0 or a i value, if negative lower than 0.. a bit of mental overhead?

#also numpy?
import numpy as np
output3 = np.maximum(0,inputs) #this feels retarded without reading the docs honestly
print(output3) #we're using it! I guess it's quite more performant. 

class Activation_ReLU:
    #fwrd pass
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense1.forward(X)
activation1.forward(dense1.output)
print(activation1.forward(dense1.output[:5]))


## === SOFTMAX ===
layer_outputs = [4.8, 1.21, 2.385]
E = 2.71828182846

exp_values = []
for output in layer_outputs:
    exp_values.append(E**output)
print('exponentiated values:', exp_values)
