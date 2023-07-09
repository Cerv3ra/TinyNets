import numpy as np
import tinygrad
import cv2


#Load Model
from tinygrad.state import safe_save, safe_load,get_state_dict, load_state_dict
net = () #potential bug, do I have to define the whole structure again? 
state_dict = safe_load("TinyDense.safetensors")
net = get_state_dict
#Load Image
img = cv2.imread('3s.png')
print(img)
imglabel = 3 #Output expected
#print(img.shape)



#print val result