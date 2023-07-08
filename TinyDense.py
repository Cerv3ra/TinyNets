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
        self.l1 = Linear(784,128   , bias=False)
        self.l2 = Linear(128,128, bias=False)
        self.l3 = Linear(128,10, bias=False)
    
    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        x = x.leakyrelu()
        x = self.l3(x)
        return x.log_softmax(x)

net = TinyDense()
Tensor.training = True #boilerplatish?

from tinygrad.nn.optim import SGD
opt = SGD([net.l1.weight, net.l2.weight, net.l3.weight], lr=3e-4)

from datasets import fetch_mnist
X_train, Y_train, X_test, Y_test = fetch_mnist()

from extra.training import sparse_categorical_crossentropy  
from tinygrad.state import safe_save, safe_load, get_state_dict, load_state_dict

state_dict = safe_load("TinyDense.safetensors")
load_state_dict(net, state_dict)

for step in range(6000):
    #random sample batch??
    
    samp = np.random.randint(0, X_train.shape[0], size=(128))
    batch = Tensor(X_train[samp], requires_grad=False)
    #labels for the same random sample? Batch of 64?
    labels = Y_train[samp]
    #fordward pass
    out = net (batch)

    #compute loss
    loss = sparse_categorical_crossentropy(out, labels)
    #zero gradients
    opt.zero_grad()
    #backward
    loss.backward()
    #update param
    opt.step()

    #calculate accuracy
    pred = np.argmax(out.numpy(), axis=-1)
    acc = (pred == labels).mean()
    if acc > .95:
        break
    if step % 100 == 0:
        print(f'step{step} | Loss: {loss.numpy()} | Accuracy: {acc}')

#save model
state_dict = get_state_dict(net)
safe_save(state_dict, "TinyDense.safetensors")


#test model
Tensor.training = False
av_acc = 0 #reset acc
st = time.perf_counter()
print(X_test.shape)

for step in range(100):
    #test is just fordward?
    samp = np.random.randint(0, X_test.shape[0], size=128)
    batch = Tensor(X_test[samp], requires_grad=False)
    #get labels
    labels = Y_test[samp]
    #forward pass
    out = net(batch)

    pred = np.argmax(out.numpy(), axis=-1)
    av_acc += (pred == labels).mean()
 
print(f"Test Accuracy: {av_acc / 1000}")
print(f"Time: {time.perf_counter() - st}")

#enable continue training after the first for-loop, I dont want it to reset
#feed a imagelieren and test result? 


