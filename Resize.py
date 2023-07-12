
import numpy as np
import os
import cv2
import gzip
import pandas as pd
import time
# %%
# #load file
# img = cv2.imread('6.png', cv2.IMREAD_GRAYSCALE)
# assert img is not None, "File could not be read, are you pointing at it correctly brub?"
# #print output stringclear
# img = cv2.resize(img, dsize=(28,28))
# #print(img.shape)

# #let's open up data from the actual MNIST. 

# #solution is to slice my [28x28x3] Tensor into a Flat 2D array. Let's try that out
# img = (255-img)
# print(img)
# cv2.imwrite('grey6.bmp',img)

# %%
start = time.monotonic()
def imgconver(input, output):
    a=cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    a=cv2.resize(a, dsize=(28,28))
    a=255 -a 
    cv2.imwrite(output, a)



imgconver('6.png','6.bmp')
end = time.monotonic()
print(f"{end-start:.2}s")