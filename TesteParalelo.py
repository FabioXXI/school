import time
import numpy as np
from numba import njit
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Carrega as imagens
import numpy as np
from numba import njit
from PIL import Image

# Carrega a imagem
imagem_1 = Image.open("./c.jpg")

@njit(cache=True)
def converta_para_cinza(image_array):
    imagem_cinza = np.empty(image_array.shape[:2], dtype=np.uint8)
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            r, g, b = image_array[i, j]
            imagem_cinza[i, j] = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return imagem_cinza

def imprime_figura():
    # Converte a imagem para numpy array
    image_array = np.array(imagem_1)
    
    # Converte para escala de cinza usando Numba
    gray_image_array = converta_para_cinza(image_array)
    
    # Converte de volta para imagem PIL e mostra
    imagem_cinza = Image.fromarray(gray_image_array)
    imagem_cinza.show()

inicio = time.time()
imprime_figura()
fim = time.time()
print("Tempo de Processamento:", fim - inicio, "segundos")