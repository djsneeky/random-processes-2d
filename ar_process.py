#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np                 # Numpy is a library support computation of large, multi-dimensional arrays and matrices.
from PIL import Image              # Python Imaging Library (abbreviated as PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
import matplotlib.pyplot as plt    # Matplotlib is a plotting library for the Python programming language.

import SpecAnal as sa

def main():
    image_array = generate_ar_image()
    filtered_image = filter_image_iir(image_array)
    plot_theoretical_psd(filtered_image, 'img/ar_image_psd')
    sa.BetterSpecAnal(filtered_image, 64, 5, 'img/ar_image_better_psd')

def generate_ar_image() -> np.array(np.uint8):
    x = np.random.uniform(-0.5, 0.5, size=(512, 512))
    x_scaled = 255 * (x + 0.5)
    x_scaled_as_uint8 = x_scaled.astype(np.uint8)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(x_scaled_as_uint8, cmap=plt.cm.gray)
    ax.axis('off')
    fig.savefig('img/ar_image', bbox_inches='tight', pad_inches=0)
    
    return x

def filter_image_iir(x: np.array):
    M, N = x.shape
    y = np.zeros((M, N), dtype=float)

    for m in range(1, M):
        for n in range(1, N):
            y[m, n] = 3 * x[m, n] + 0.99 * (y[m - 1, n] + y[m, n - 1]) - 0.980 * y[m - 1, n - 1]
            
    # display the filtered image y + 127
    y_shifted = (y + 127)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(y_shifted.astype(np.uint8), cmap=plt.cm.gray)
    ax.axis('off')
    fig.savefig('img/ar_image_filtered', bbox_inches='tight', pad_inches=0)

    return y

def plot_theoretical_psd(x: np.array, output_image_path: str):
    # Compute the 2D Discrete Fourier Transform
    Y = np.fft.fft2(x)
    # Compute the power spectral density (PSD)
    Sy = np.abs(Y)**2 / np.prod(x.shape)  # Normalize by the number of pixels
    # Shift the zero frequency component to the center
    Sy = np.fft.fftshift(Sy)

    # Plot the magnitude of the PSD
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, num=x.shape[0]) # assumes square image input!
    X, Y = np.meshgrid(a, b)
    ax.plot_surface(X, Y, np.log(Sy), cmap=plt.cm.coolwarm)
    ax.set_xlabel('Frequency (cols)')
    ax.set_ylabel('Frequency (rows)')
    ax.set_zlabel('Magnitude')

    plt.savefig(output_image_path)

    
if __name__ == "__main__":
    main()