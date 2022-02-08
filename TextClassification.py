import tensorflow as tf
from tensorflow import keras
import numpy as np

from KNN import x_train

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

print(train_data[0])

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}  # K - key = word; V - Value = Integer Value
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
# If we get values that aren't in dictionary they just get assigned these values

reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])  # Just reversing - Integer pointing to a word

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                       maxlen=250)  # Anything over 250 words won't read - replaced with <PAD>


def decode_review(text):
    return "".join([reverse_word_index.get(i, "?") for i in text])  # Returning all the human readable words


# Model Down Here:
"""""
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))  # Neuron between 0 and 1 squishes value to value we want

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # We want two outputs (binary = 0 or 1) so loss function calculates different between actual answer and output neuron

x_val = train_labels[:10000]
x_train = train_labels[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)    # Batch size = how many movie reviews each time

results = model.evaluate(test_data, test_labels)

model.save("model.h5")"""


def review_encode(s):
    encoded = [1]  # Setting a starting tag

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])    # If word in "Dictionary" it adds number associated with that word
        else:
            encoded.append(2)   # If word not in "Dictionary" it adds in that unknown tag

    return encoded


# Load in Model

model = keras.models.load_model("model.h5")

with open("Lion King Review", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"",
                                                                                                                  "").strip().split(
            " ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("prediction: " + str(predict[0])) 
print("Actual: " + str(test_labels[0]))

print(results)
'''

# In the future change something to improve accuracy as acc is very low now
