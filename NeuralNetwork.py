import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = data.load_data()  # Data Set with labels where labels have values e.g 9 find: https://www.tensorflow.org/tutorials/keras/classification

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255
test_images = test_images / 255

model = keras.Sequential([      # Keras.sequential - sequence of layers where you defining the layers - Flatten and 2 dense layers = typical way to create a neural network
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),  # Rectify Linear Unit - Activation Function
    keras.layers.Dense(10, activation="softmax")  # PICKS VALUES FOR EACH NEURON SO IT ADDS UP TO 1
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])  # Accyracy tests how low we can get the loss function

model.fit(train_images, train_labels, epochs=5)  # epochs - how many times model sees the information

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: ", test_acc)

# Everything from line 23 to 28 is everything needed to create a neural network

