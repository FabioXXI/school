from cv2 import imread, imwrite
from numpy import float32, empty, uint8
from numba import vectorize, jit
from time import time

@vectorize(["float32(float32, float32, float32)"], target='cpu')
def t(r, g, b):
    return (r + g + b) / 3.0

@jit(cache=True)
def p(m):
    w, c, _ = m.shape
    y = empty((w, c), dtype=float32)
    
    for i in range(w):
        for j in range(c):
            r, g, b = m[i, j]
            y[i, j] = t(r, g, b)
    
    return y

s = time()
i = imread('./c.jpg')
imwrite('o.jpg',  p(i).astype(uint8))
print(time() - s)