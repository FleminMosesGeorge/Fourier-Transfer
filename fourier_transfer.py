import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

image = cv2.imread('lena.png', 0)
img = np.float32(image)

def iexp(n):
    '''
    Converts phase to a complex quantity.
    '''
    return complex(math.cos(n), math.sin(n))

def is_pow2(n):
    '''
    Checks if a number is a power of 2
    '''
    return False if ((n & 1) == 1 and n != 1) else (n == 1 or is_pow2(n >> 1))

def fft(x):
    '''
    Recursive implementation of FFT of a sequence x
    '''
    n = len(x)
    if is_pow2(n) == False:
        raise ValueError("Size of x must be a power of 2")
    elif n == 1:
        return [x[0]]
    else:
        # Splitting in even-odd sequences according to Tukey-cooley algoritm
        x = fft(x[::2]) + fft(x[1::2])
        for i in range(n//2):
            e = iexp(-2 * math.pi * i / n)
            x_i = x[i]
            x[i] = x_i + e * x[i + n//2]
            x[n//2 + i] = x_i - e * x[i + n//2]
        return x


def ifft(X):
    '''
    Recursive implementation of inverse FFT of a sequence X
    '''
    n = len(X)
    if is_pow2(n) == False:
        raise ValueError("Size of x must be a power of 2")
    elif n == 1:
        return [X[0]]
    else:
        # Splitting in even-odd sequences according to Tukey-cooley algoritm
        X = ifft(X[::2]) + ifft(X[1::2])
        for i in range(n//2):
            e = iexp(2 * math.pi * i / n)
            X_i = X[i]
            X[i] = X_i + e * X[i + n//2]
            X[n//2 + i] = X_i - e * X[i + n//2]
        return [val/2 for val in X]


def pad_image_2(image, pad_height, pad_width):
    '''
    Given an image, this function will pad it with zeros to 
    image.shape + (pad_height, pad_width) + (nearest power of 2)
    '''
    P = pow2_ceil(image.shape[0] + pad_height)
    Q = pow2_ceil(image.shape[1] + pad_width)
    image_padded = np.zeros((P,Q))
    image_padded[:image.shape[0], :image.shape[1]] = image
    return image_padded


def dft_2d(image):

    # Initialize DFT in matrix form of image
    # The image is also padded to the nearest power of 2 for speedy computations
    image = pad_image_2(image, 0, 0)
    imWidth = int(image.shape[1])
    imHeight = int(image.shape[0])
    image_fft = np.zeros((imHeight, imWidth), dtype=np.complex_)
    # Row-wise DFT calculation
    for x in range(imHeight):
        image_fft[x] = fft(image[x])
        
    # Column-wise DFT calculation
    image_fft_t = image_fft.transpose()
    for y in range(imWidth):
        image_fft_t[y] = fft(image_fft_t[y])
    image_fft = image_fft_t.transpose()
    
    return image_fft


def idft_2d(image_fft):

    # Initialize IDFT in matrix form of image
    imWidth = int(image_fft.shape[1])
    imHeight = int(image_fft.shape[0])
    image = np.zeros((imHeight, imWidth), dtype=np.complex_)

    # Row-wise IDFT calculation
    for x in range(imHeight):
        image[x] = ifft(image_fft[x])

    # Column-wise IDFT calculation
    image_t = image.transpose()
    for y in range(imWidth):
        image_t[y] = ifft(image_t[y])
    image = image_t.transpose()

    return image

def pow2_ceil(x):
    return 2 ** int(np.ceil(np.log2(x)))


def shift_dft(image_fft):
    '''
    Shift the fourier transform so that F(0,0) is in the center.
    '''
    # Typecasting as array
    y = np.asarray(image_fft)
    #print(y)  
    for k, n in enumerate(image_fft.shape):
        mid = (n + 1) // 2
        indices = np.concatenate((np.arange(mid, n), np.arange(mid)))
        y = np.take(y, indices, k) 
    return y

pad_height, pad_width = image.shape


dft = dft_2d(image)
dft_shift = shift_dft(dft)
magnitude_spectrum = 20*np.log(np.abs(dft))
fshift = np.fft.fftshift(dft)

rows, cols = dft.shape
crow,ccol = rows/2 , cols/2
fshift[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img_back, cmap='gray')

