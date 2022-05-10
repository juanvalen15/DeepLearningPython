from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# word_index is a dictionary mapping words to an integer index  
word_index = imdb.get_word_index()
# reverses it, mapping integer indeces to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decodes the review. The indeces are offset by 3 because 0, 1, and 2 are reserved indices for 'padding', 'start of sequence' and 'unknown'
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# encoding the integer sequences into a binary matrix
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    # creates an all-zero matrix of shape (len(sequences),dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
t_test = vectorize_sequences(test_data)

# vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# the model definition
from keras import models
from keras import layers
from keras.utils.vis_utils import plot_model

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# compiling the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# # configuring the optimizer
# from keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# # using custom losses and metrics
# from keras import losses
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss='binary_crossentropy', 
#               metrics=[metrics.binary_accuracy])


# validating your approach: 
# Idea in brief: monitor during training the accuracy of the model on data it has never seen before.
# Create a validation set by setting apart 10000 samples from the original training data.
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# train model for 20 epochs (20 iterations over all samples in x_train and y_train tensors)
# mini-batches of 512 samples
# monitor loss and accuracy on 10000 samples (taken apart)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# plotting the training and validation loss
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss') # bo => blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') # b => solid blue line
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# plotting the training and validation accuracy
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# retraining a model from scratch
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(t_test, y_test)

# use the trained network to generate predictions on new data
mpredict = model.predict(t_test)
plt.hist(mpredict, 100)
plt.show()