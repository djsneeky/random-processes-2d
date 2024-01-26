#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np                 # Numpy is a library support computation of large, multi-dimensional arrays and matrices.
from PIL import Image              # Python Imaging Library (abbreviated as PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
import matplotlib.pyplot as plt    # Matplotlib is a plotting library for the Python programming language.

def main():
    generate_ar_image()

def generate_ar_image():
    x = np.random.uniform(-0.5, 0.5, size=(512, 512))
    x_scaled = 255 * (x + 0.5)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(x_scaled, cmap='gray')
    ax.axis('off')
    plt.show()
    fig.savefig('img/ar_image', bbox_inches='tight', pad_inches=0)
    
if __name__ == "__main__":
    main()