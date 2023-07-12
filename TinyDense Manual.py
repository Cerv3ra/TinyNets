#Idea is to create  a basic Dense Network of [1, 64, 64, 10] with Tinygrad
#Figure out how to do that, load data and train something, probably MNIST dataset




#In an ideal world I just write 
#Dense Network
    #layer 1 1,64
    #layer 2 64,64
    #layer 3 64,10

 #data = File
 #divide X, y {80,20}
 #Relu, Softmax
 #train
 #run 100 sample 

import tinygrad
import numpy as np
import argparse
import time
#Boilerplate
#Load Data
##Struct Network
#train
#run

#struct network
from tinygrad.nn import Linear
from tinygrad.tensor import Tensor
from tinygrad.lazy import Device
Device.DEFAULT = "CPU"



class TinyDense:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x.log_softmax()

net = TinyDense()
Tensor.training = False #boilerplatish?

from tinygrad.nn.optim import SGD
opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)

from extra.datasets import fetch_mnist
X_train, Y_train, X_test, Y_test = fetch_mnist()

from extra.training import sparse_categorical_crossentropy  
from tinygrad.state import safe_save, safe_load, get_state_dict, load_state_dict
def cross_entropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

#load model

state_dict = safe_load("TinyDense.safetensors")
load_state_dict(net, state_dict)

for step in range(8000):
    #random sample batch??
    samp = np.random.randint(0, X_train.shape[0], size=(128))
    batch = Tensor(X_train[samp], requires_grad=True)
    #labels for the same random sample? Batch of 64?
    labels = Y_train[samp]
    #fordward pass
    out = net (batch)

    #compute loss
    loss = cross_entropy(out, labels)
    #zero gradients
    opt.zero_grad()
    #backward   
    loss.backward()
    #update param
    opt.step()

    #calculate accuracy
    pred = np.argmax(out.numpy(), axis=-1)
    acc = (pred == labels).mean()
    if acc >= 1:
        break
    if step % 100 == 0:
        print(f'step{step} | Loss: {loss.numpy()} | Accuracy: {acc}')

#save model
state_dict = get_state_dict(net)
safe_save(state_dict, "TinyDense.safetensors")

import numpy as np
import tinygrad
import cv2


#Load Image
img = cv2.imread('grey3.bmp', cv2.IMREAD_GRAYSCALE)
imglabel = [3] #Output expected

#print(img.shape)


#print val result
#test model
Tensor.training = False
av_acc = 0 #reset acc
st = time.perf_counter()
print(X_test.shape)
testamount = 1
for step in range(testamount):
    #test is just fordward?
    samp = np.random.randint(0, X_test.shape[0], size=1)
    #batch = Tensor(X_test[samp])
    batch = Tensor(np.concatenate(img))
    #get labels
    #labels = Y_test[samp]
    labels = imglabel
    #forward pass
    out = net(batch)
  
    pred = np.argmax(out.numpy(), axis=-1)
    av_acc += (pred == labels).mean()
    print(batch.numpy())
    print(np.argmax(out.numpy()))
    print(labels)
 
print(f"Test Accuracy: {av_acc / testamount}")
print(f"Time: {time.perf_counter() - st}")
#94% MNIST? 

#enable continue training after the first for-loop, I dont want it to reset
#feed a imagelieren and test result? 


