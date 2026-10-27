import pandas as pd
import tensorflow as tf
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

print(train_images.shape) #60000, 28, 28
print(train_labels.shape) #60000
print(test_images.shape)  #10000, 28, 28
print(test_labels.shape)  #10000

#intiial data testing
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for ax, img, lbl in zip(axes.ravel(), train_images[:16], train_labels[:16]):
    ax.imshow(img, cmap="gray"); ax.set_title(lbl); ax.axis("off")
#plt.show()


##DATA PREPROCESSING
train_images = train_images.astype("float32")[..., None]/255.0
test_images = test_images.astype("float32")[..., None]/255.0

##DATA SPLIT
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

#confirm data maintained structure
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for ax, img, lbl in zip(axes.ravel(), x_train[:16], y_train[:16]):
    ax.imshow(img, cmap="gray"); ax.set_title(lbl); ax.axis("off")
#plt.show()