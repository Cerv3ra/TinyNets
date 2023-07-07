import math
softmax_output = [0.7, 0.1, 0.2]
target_output =[1,0,0]

loss = -(math.log(softmax_output[0])*target_output[0]+
        math.log(softmax_output[1])*target_output[1]+
        math.log(softmax_output[2])*target_output[2])

print(loss)

loss = -math.log(softmax_output[0])
print(loss)

import numpy as np
b = 5.2
print(np.log(b))

print(math.e ** 1.64865)

##explanation of log and e and it's relationshio