import pandas as pd
import tensorflow as tf
import numpy as np
import idx2numpy

##FUNCTIONS
def load_idx3_ubyte(filepath):
    with open(filepath, 'rb') as f:
        # Read the magic number (4 bytes)
        magic = int.from_bytes(f.read(4), 'big')
        
        # Read dimensions: number of images, rows, and columns (4 bytes each)
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read the rest of the buffer into a numpy array
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        
        # Reshape to (num_images, rows, cols)
        return data.reshape(num_images, rows, cols)

##DATA
train_images = load_idx3_ubyte('/Users/alexjackson/Documents/Personal Projects/Handwriting NN/train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('/Users/alexjackson/Documents/Personal Projects/Handwriting NN/train-labels.idx1-ubyte')
test_images = load_idx3_ubyte('/Users/alexjackson/Documents/Personal Projects/Handwriting NN/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('/Users/alexjackson/Documents/Personal Projects/Handwriting NN/t10k-labels.idx1-ubyte')

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)





