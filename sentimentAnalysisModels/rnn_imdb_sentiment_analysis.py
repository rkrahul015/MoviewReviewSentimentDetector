from keras.datasets import imdb
from keras import models, layers
from keras import models, layers
from keras.utils import pad_sequences
max_number_of_features = 10000
max_len = 500
batch_size = 32
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_number_of_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
network = models.Sequential()
network.add(layers.Embedding(10000, 32))
network.add(layers.SimpleRNN(32))
network.add(layers.Dense(1, activation = 'sigmoid'))
network.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['acc']
)
network.save('RNN_IMDB_REVIEW.h5')

