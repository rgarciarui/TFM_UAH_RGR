import os
from .common_utils import *


        
def get_noisy_image(img_np, sigma):
    """Agrega ruido gaussiano a una imagen.
    
     Args:
         img_np: image, np.array con valores de 0 a 1
         sigma: std del ruido
     """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np