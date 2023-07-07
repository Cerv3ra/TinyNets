#Single Neuron with Numpy  
import numpy as np

inputs = [1,2,3,2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

outputs = np.dot(weights, inputs) + bias
#dot product --> input[1] * weight[1] + input[2] * weight [2]... 
print(outputs)

inputs = [1,2,3,2.5]
weights =[[0.2, 0.8, -0.5, 1 ],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3 , 0.5]

layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)

#--------------

inputs = [[1.0, 2.0, 3.0, 2.5],
 [2.0, 5.0, -1.0, 2.0],
 [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]


layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer_outpute)