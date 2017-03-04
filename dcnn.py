from __future__ import print_function
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# Reproducability of initialization
np.random.seed(42)

# Going to use convnets, in part, because fully connected layers,
# for images, would result in too many weights. While our current
# set 32x32x3 is somewhat managable, larger images, say 200x200,
# would require 120k weights!

# Number of images to process at a time.
batch_size = 128

# cifar10 has 10 different kinds of objects.
nb_classes = 10

# Image dimensions.
img_rows, img_cols = 32, 32

#
# Now our adjustable hyperparameters...
#

# How long we train.
nb_epochs = 45

# Number of convnets.
nb_filters = 32

# Size of max pooling. This is how many pixels we'll inspect at a time
# to find details. Decrease if details are closer together. Remember
# That spacial relationships are lost to the convnet.
pool_size = (2, 2)

# Convent kernel size. TODO add definition.
kernel_size = (3, 3)

#
# End hyperparameters
#

# Load our data. We want to partition the data into training and test.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Tensorflow and Theano have different tensor parameter orders, so we need
# to inspect the backend and load our tensors accordingly.
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print('X_train.shape[0]:', X_train.shape[0])
print('X_test.shape[0]:', X_test.shape[0])

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Start with our layers
model = Sequential()
# Initialize our convnet with our filters, rows, and columns.
# Our convolution network is providing a means of subsampling,
# which is allowing our neurons to detect patterns without
# spacial orientation. Consider a picture of a human face. We
# might want our neurons to recognize eyes without their relationship
# to a nose, which is beneficial when an image of a face might be
# partially obstructed. This whole process is trying to get our
# neurons to ~generalize~ by detecting discrete patterns and applying
# them to a greater classification scheme.
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
# Values leaving neurons are modified via the Activation function.
# In this scenario we're using relu, which is 0 for x < 0, and
# identity thereafter.
model.add(Activation('relu'))
# Another convnet. At this point our first convent has built up some
# set of recognizers and we've clamped their values by passing them
# through an activation function. Now we'll do it all over again.
# The "shape" of this model that we're building can be thought of as
# yet another hyperparameter, which means adding another convnet, or
# changing our activation functions will impact the acurracy of our
# model and is therefore subject to change. What is the "correct"
# model paramters? Who knows. At this point we're brute-forcing a solution.
# Also notice that this convnet is the same as our first. If we had more than
# these two, we would want to wrap these up into a model generator to
# reduce repitition.
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
# And again through relu. How would our accuracy be impacted if we changed
# this to softmax?
model.add(Activation('relu'))
# Pooling just takes the output of our convnet and extracts the greatest value.
# So let's say we have a convnet that produces a 4x4 set of pixels, pooling
# would extract the greatest value from said pixel set. It's all about
# downsampling to reduce the data size.
model.add(MaxPooling2D(pool_size=pool_size))
# We're going to randomly ignore 25% of neurons during forward feeding
# in an attempt to prevent overfitting. Overfitting is where a neuron
# learns specific traits from a training set, which then makes it less
# useful for data it hasn't yet seen. Dropout forces the neurons to generalize.
model.add(Dropout(0.25))
# Now we add our "dense" layers, or fully connected layers.
# Flatten takes a shape like (64, 32, 32) and produces (65536).
model.add(Flatten())
# Dense is a fully connected network. In this example we're creating
# 256 fully connected neurons. And again, both the use of "Dense" and
# the value of "256" are hyperparameters for our model. Let's of
# guess-work here.
model.add(Dense(256))
model.add(Activation('relu'))
# Agressive dropout. We're only willing to use half the nueron's per epoch.
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
# Why softmax instead of relu? In general, relu has been found, through trial
# and error, to be better for convnets and softmax for dense networks.
model.add(Activation('softmax'))

# Ship it! Our code doesn't actually execute the model, instead, it builds a
# graph that describes our model, which is then passed off to a backend, in
# this case, Tensorflow.
# So about these parameters. Remember what we're doing here. We have a bunch
# of training and test images, and a model that is, hopefully, going to figure
# out a bunch of weights and biases that produce a high prediction accuracy
# against our test images. So we iterate over our images, crunch some numbers,
# and make a prediction. Then we compare our prediction against the actual
# known values. Then, through gradient descent, determine if our predictions
# are getting better or worse and adjust our model's values accordingly. We
# do this over and over until, again hopefully, our model begins to converge
# around our test images, meaning that our model has learned to generalize
# the necessary patterns towards a highly accurate classification.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Okay so we've loaded our training and test data, built and compiled
# our model, now let's add some metadata to our project so we
# can use Tensorflow's incredibly helpful visualizer, Tensorboard!
tb = TensorBoard(log_dir='./logs')

# Now instruct our model how to run and link its output to Tensorboard
# for those visuals.
model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tb])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print("Accuracy: %.2f%%" % (score[1]*100))
