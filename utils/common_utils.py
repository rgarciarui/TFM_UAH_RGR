import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

def crop_image(img, d=32):
    '''Hacer dimensiones divisibles por `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Devuelve los parámetros que queremos optimizar.

     Args:
         opt_over: lista separada por comas, p. "net, input" o "net"
         net: red
         net_input: torch.Tensor que almacena la entrada `z`
     '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Crea una cuadrícula a partir de una lista de imágenes concatenándolas'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):

	"""
	Dibuja imágenes en una cuadrícula
	
	Args:
		images_np: lista de imágenes, cada imagen es np.array de tamaño 3xHxW de 1xHxW
		nrow: cuántas imágenes habrá en una fila
		factor: tamaño si la figura plt.
		interpolation: interpolación utilizada en plt.imshow
    """

	n_channels = max(x.shape[0] for x in images_np)
	
	assert (n_channels == 3) or (n_channels == 1), "las imágenes deben tener 1 o 3 canales"
	
	images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]
	
	grid = get_image_grid(images_np, nrow)
	
	plt.figure(figsize=(len(images_np) + factor, 12 + factor))
	
	if images_np[0].shape[0] == 1:
		plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
	else:
		plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
	
	plt.show()
	return grid

def load(path):
    """Leer imagen PIL."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
	"""
	Carguar una imagen y cambiar su tamaño a un tamaño específico. 
	
	Args: 
	path: ruta a la imagen 
	imresize: tupla o escalar con dimensiones; -1 para `no cambiar tamaño` 
	"""
	
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Rellena el tensor` x` con ruido del tipo `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
	"""Devuelve un pytorch. Tensor de tamaño (1 x `input_depth` x `spatial_size [0]` x `spatial_size [1]`)
     inicializado de una manera específica.
     Args:
         input_depth: número de canales en el tensor
         method: 'ruido' para llenar el tensor con ruido; `meshgrid` para np.meshgrid
         spatial_size: tamaño espacial del tensor para inicializar
         noise_type: 'u' para uniforme; 'n' para normal
         var: un factor, un ruido se multiplicará. Básicamente es un escalador de desviación estándar.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Convierte la imagen en formato PIL a np.array.
    
    Desde W x H x C [0...255] hasta C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Convierte la imagen en formato np.array a imagen PIL.
    
    Desde C x W x H [0..1] hasta  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Convierte la imagen en numpy.array en torch.

    Desde C x W x H [0..1] hasta  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Convedirte una imagen en formato torch.Tensor a un np.array.

    Desde 1 x C x W x H [0..1] hasta  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
	"""Ejecuta el bucle de optimización.

     Args:
         optimizer_type: 'LBFGS' de 'adam'
         parameters: lista de tensores para optimizar sobre
         closure: función, que devuelve variable de pérdida
         LR: tasa de aprendizaje
         num_iter: número de iteraciones
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Iniciando la optimización con LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Iniciando la optimización con ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False