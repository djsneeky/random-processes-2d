#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np                 # Numpy is a library support computation of large, multi-dimensional arrays and matrices.
from PIL import Image              # Python Imaging Library (abbreviated as PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
import matplotlib.pyplot as plt    # Matplotlib is a plotting library for the Python programming language.

def main():
    # Read in a gray scale TIFF image.
    im = Image.open('img/img04g.tif')
    print('Read img04.tif.')
    print('Image size: ', im.size)
    
    # Display image object by PIL.
    # im.show(title='image')
    
    # Import Image Data into Numpy array.
    # The matrix x contains a 2-D array of 8-bit gray scale values. 
    x = np.array(im)
    print('Data type: ', x.dtype)
    
    # Display numpy array by matplotlib.
    # plt.imshow(x, cmap=plt.cm.gray)
    # plt.title('Image')
    # Set colorbar location. [left, bottom, width, height].
    # cax = plt.axes([0.9, 0.15, 0.04, 0.7]) 
    # plt.colorbar(cax=cax)
    # plt.show()

    BadSpecAnal(x, 64, 'img/bad_spec_anal_64')
    BadSpecAnal(x, 128, 'img/bad_spec_anal_128')
    BadSpecAnal(x, 256, 'img/bad_spec_anal_256')
    
    BetterSpecAnal(x, 64, 5, 'img/better_spec_anal_64')

def BadSpecAnal(x: np.array, N: int, output_image_path: str):
    x = np.double(x)/255.0
    
    i = 99
    j = 99
    z = x[i:N+i, j:N+j]

    # Compute the power spectrum for the NxN region.
    Z = (1/N**2)*np.abs(np.fft.fft2(z))**2

    # Use fftshift to move the zero frequencies to the center of the plot.
    Z = np.fft.fftshift(Z)

    # Compute the logarithm of the Power Spectrum.
    Zabs = np.log(Z)

    # Plot the result using a 3-D mesh plot and label the x and y axes properly.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, num = N)
    X, Y = np.meshgrid(a, b)

    surf = ax.plot_surface(X, Y, Zabs, cmap=plt.cm.coolwarm)

    ax.set_xlabel('$\mu$ axis')
    ax.set_ylabel('$\\nu$ axis')
    ax.set_zlabel('Z Label')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(output_image_path)
    
def BetterSpecAnal(x: np.array, win_size: int, num_win_per_side: int, output_image_path: str):
    x = np.double(x)/255.0
    
    # Get central windows
    center_x = x.shape[0] // 2
    center_y = x.shape[1] // 2
    start_x = (center_x - win_size // 2) - (num_win_per_side // 2) * win_size
    start_y = (center_y - win_size // 2) - (num_win_per_side // 2) * win_size
    # print(start_x, start_y)
    windows = [x[i:i+win_size, j:j+win_size] for i in range(start_x, start_x+num_win_per_side*win_size, win_size)
                                                for j in range(start_y, start_y+num_win_per_side*win_size, win_size)]
    
    # Create 2-D separable Hamming window
    hamming_window = np.outer(np.hamming(win_size), np.hamming(win_size))
    
    # Initialize power spectral density array
    psd_sum = np.zeros((win_size, win_size))
    
    # Compute squared DFT magnitude for each window
    for window in windows:
        window = window * hamming_window
        dft = np.fft.fftshift(np.fft.fft2(window))
        psd_sum += np.abs(dft)**2
        
    # Average power spectral density across windows
    psd_avg = psd_sum / (len(windows) * win_size**2)

    # Plot the result using a 3-D mesh plot and label the x and y axes properly.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, num = win_size)
    X, Y = np.meshgrid(a, b)
    surf = ax.plot_surface(X, Y, np.log(psd_avg), cmap=plt.cm.coolwarm)

    ax.set_xlabel('$\mu$ axis')
    ax.set_ylabel('$\\nu$ axis')
    ax.set_zlabel('Z Label')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(output_image_path)
    
if __name__ == "__main__":
    main()