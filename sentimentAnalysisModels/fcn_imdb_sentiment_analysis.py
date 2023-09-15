from keras.datasets import imdb
import numpy as np
(train_data, y_train), (test_data, y_test) = imdb.load_data(num_words = 10000)
def vectorization(data, number_of_features = 10000): 
  res = np.zeros((len(data), number_of_features))
  for i, j in enumerate(data): 
    res[i, j] = 1 
  return res

x_train = vectorization(train_data)
x_test = vectorization(test_data)

from keras import models 
from keras import layers 
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

x_val = x_train[:10000]
x_partial_train = x_train[10000:]

y_val = y_train[:10000]
y_partial_train = y_train[10000:]
history = model.fit(x_partial_train, y_partial_train, epochs = 4, batch_size = 512, validation_data = (x_val, y_val))
model.save('IMDB_REVIEW_FCN.h5')
