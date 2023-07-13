##This file will train a CIFAR, will do it in a variety of ways, and of course, will make me
##figure out a lot of things along the way. 
##First, lets explore the cifar data. 

##Cifar-10 has 10 classes, 6k Image per class
import numpy as np
import tinygrad 
from extra.datasets import fetch_cifar
import time
import cv2
start =time.monotonic()
X_train, Y_train = fetch_cifar(train=True)

print(X_train.shape)
#(50000, 3, 32, 32) So 50k pics, RBG 32x32. 
#extract one 3,32,32
print(X_train[1,:,:,:].shape)
#img = np.transpose(X_train[4,:,:,:], (1,2,0))
#cv2.imwrite('firstpic.png',img)


##Figure out a NN
from tinygrad.nn import optim, Linear
from tinygrad.tensor import Tensor

class TinyCIFAR:
    def __init__ (self):
        self.l1 = Linear(3072,1024)
        self.l2 = Linear(1024,1024)
        self.l3 = Linear(1024,10)

    def __call__ (self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        x = x.leakyrelu()
        x = self.l3(x)
        return x.log_softmax()




        







end=time.monotonic()
print(f'{end-start:.3}s')

