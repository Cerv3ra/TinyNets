
import numpy as np
import os
import cv2
#load file
img = cv2.imread('3.png')
assert img is not None, "File could not be read, are you pointing at it correctly brub?"
#print output stringclear
imgre = cv2.resize(img, dsize=(28,28))
cv2.imwrite("3s.png" ,imgre)