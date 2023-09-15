from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import layers, models
number_of_feature = 10000
malen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = number_of_feature)
x_train = pad_sequences(x_train, maxlen = malen)
x_test = pad_sequences(x_test, maxlen = malen)
network = models.Sequential()
network.add(layers.Embedding(10000, 32))
network.add(layers.LSTM(32))
network.add(layers.Dense(1, activation = 'sigmoid'))
network.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['acc']
)
history = network.fit(x_train, y_train, batch_size = 128, epochs = 10, validation_split = 0.2)
network.save("LSTM_IMDB_REVIEW_MODEL.h5")
