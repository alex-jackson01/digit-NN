# digit-NN
Creating a classification neural network to recognize handwritten digits 0-9 from MNIST data

## The Model
### Augmentation Layers
- I elected to use forms of random modification (RandomRotation, RandomTranslation, RandomZoom) so that each training gets disturbed in a way that creates noise
  - The network never sees the same image twice
  - Cheap way to artificially create more varied training data, since the label doesn't change
    - It also better mimics how real handwriting varies in unconscious design
  - Reduces overfitting
- These layers are automatically disabled during inference
### Conv2D(32, 3, activation="relu")
- Convolution models share weights across positions and as writing strokes vary in the images, sharing the weights become important. A dense NN on raw pixel data risks ignoring spatial structure and requires more parameters. 
- This model style uses 32 3x3 filters across each image where each filter is charge of detecting a different pattern
  - Why 3x3 layers and not more? Given the relatively clean nature and limited range (just digits 0-9) of this data set, 3x3 layers are the minimum specs required to perform well. Two 3x3 stacked layers cover the same area as one 5x5 layer while avoiding leaning on too may parameters and avoiding extra nonlinearity
- ReLU is a tried and true method for CNN training with image data.
  - The necessary compute power is low
  - It avoids vanishing gradients that you may get with sigmoid/tanh
  - Based on my research, alternative ReLU activation functions such as LeakyReLU, ELU or GELU would make a negligible difference on clean data like this
- [Explain running it again with 64 layers]

### BatchNormalization()

### MaxPooling2D()

### Flatten()

### Dropout(0.5)

### Dense(10, activation="softmax")


    


