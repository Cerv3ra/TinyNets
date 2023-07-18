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

# print(X_train.shape)
# #(50000, 3, 32, 32) So 50k pics, RBG 32x32. 
# #extract one 3,32,32 numpy.transpose(xtrain, (initial order in the order you want 
# # in this case my (3,32,32) matrix has to become (32,32,3) so we transpose to (1,2,0)


# print(X_train[1,:,:,:].shape)
# #img = np.transpose(X_train[4,:,:,:], (1,2,0))
# #cv2.imwrite('firstpic.png',img)


import numpy as np
import tinygrad 
from tinygrad.extra.datasets import fetch_cifar
import time
# from tinygrad.lazy import Device
# Device.DEFAULT = "CUDA"

import cv2
start =time.monotonic()
X_train, Y_train,= fetch_cifar(train=True)

from tinygrad.nn import Linear
from tinygrad.nn import optim
from tinygrad.tensor import Tensor

class TinyCIFAR:
    def __init__ (self):
        self.l1 = Linear(3072,1024, bias=True)
        self.l2 = Linear(1024,1024,bias=True)
        self.l3 = Linear(1024,1024,bias=True)
        self.l4 = Linear(1024,1024,bias=True)
        self.l5 = Linear(1024,10, bias=True)

    def __call__ (self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        x = x.leakyrelu()
        x = self.l3(x)
        x = x.leakyrelu()
        x = self.l4(x)
        return x.log_softmax()
    
net = TinyCIFAR()

print(X_train.shape)
X_trains = X_train.reshape(50000, -1)
print(X_trains.shape)
Tensor.training= True

from tinygrad.nn.optim import SGD, Adam
opt = Adam([net.l1.weight, net.l2.weight, net.l3.weight], lr=3e-4)

def cross_entropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()


BS=64

#Train the modelieren

for step in range(5000):
    samp=np.random.randint(0, X_trains.shape[0], BS)
    batch = Tensor(X_trains[0], requires_grad=True)
    labels = Y_train[samp]
    out = net(batch)
    loss = cross_entropy(out, labels)
    #print(loss.numpy())
    opt.zero_grad() #NOTE:I'm unsure why it's giving me t.grad is not None (it was array shaping)
    loss.backward()
    opt.step()
    #accuracy dealings
    pred = np.argmax(out.numpy(), axis=-1)
    acc = (pred == labels).mean()
    # if acc >= 1:
    #     break
    if step % 25 == 0:
        print(f'step{step} | loss: {loss.numpy()} | Accuracy: {acc}')


end=time.monotonic()
print(f'{end-start:.3}s')

#Validate model
X_test, Y_test= fetch_cifar(train=False)
print(X_test.shape, Y_test.shape)
av_acc = 0 #loopy fucks up without this
X_tests = X_test.reshape(10000, -1)
print(X_tests.shape)
testamount = 100
for step in range(testamount):
    samp = np.random.randint(0, X_tests.shape[0], size=BS)
    batch = Tensor(X_tests[samp], requires_grad=True)
    labels = Y_test[samp]
    out = net (batch)

    pred = np.argmax(out.numpy(), axis=-1)
    av_acc += (pred == labels).mean()
    # print(out.numpy())
    # #print(np.argmax(out.numpy()))
    # print(Y_test[samp])
    # #print(batch.numpy())

print(f"Test Accuracy: {av_acc / testamount}")


        







end=time.monotonic()
print(f'{end-start:.3}s')


