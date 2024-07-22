from cv2 import imread, imwrite
from numpy import mean
from asyncio import run
from time import time

async def p(i):
    return mean(i, axis=2)

async def main():
    s = time()
    i = imread('./c.jpg')
    imwrite('o.jpg', await p(i))
    print(time() - s)

run(main())