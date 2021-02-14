import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


class DigitsNet:
    """
    A convolutional neural network (CNN) for handwritten digit classification.
    The MNIST dataset has been used here for the training step that contains 60,000
    small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

    Code example:
    digits_net = DigitsNet()
    trained_model = digits_net.train()
    digits_net.save_model(trained_model)
    """
    def __init__(self, width=28, height=28, num_classes=10,
                 learning_rate=1e-3, epochs=20, batch_size=128,
                 model_path='../models/digits_classifier.h5'):
        # image settings
        self._width = width
        self._height = height
        self._num_classes = num_classes

        # training settings
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size

        # output path for saving the model
        self._model_path = model_path

    @staticmethod
    def _evaluate(model, x_test, y_test, label_binarizer):
        """
        Evaluates the model using the test data, and print the results
        :param model: a Sequential CNN model
        :param x_test: test data
        :param y_test: test labels
        :param label_binarizer: performs the transform operation of the one-hot labels
        :return: None
        """
        predictions = model.predict(x_test)
        print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                                    target_names=[str(x) for x in label_binarizer.classes_]))

    def _build(self, debug=False):
        """
        Building the convolutional neural network (CNN) model which will classify the handwritten greyscale digits.
        The network contains 3 hidden layers: CNN and two fully connected (FC) layers.
        A MaxPooling2D (since it's a greyscale image) and a ReLU activation are been used here, since the ReLU does not
        activate all the neurons at the same time, and the results are more accurate (than the sigmoid activation).

        Class Parameters:
        width: the width of the input image (28 pixels)
        height: the height of the input image (28 pixels)
        num_classes: the number of the output classes (0-9 digits)

        :param debug: for visualization purposes
        :return: the CNN model
        """

        # initial variables
        model = Sequential()

        # define the input shape for the given image, the depth is 1 since this is a greyscale image
        input_shape = (self._width, self._height, 1)

        # build the convolutional neural network
        model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first hidden layer
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first fully connected layer
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # second fully connected layer
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(self._num_classes))
        model.add(Activation('softmax'))

        if debug:
            # visualization of the model's structure
            model.summary()

        return model

    def train(self):
        """
        Trains the CNN model using the MNIST dataset from the Keras library.
        The function encodes the classes to one-hot representations,
        which means that the digit 4 will be as [0,0,0,0,1,0,0,0,0,0].
        The function performs an evaluation and a model saving in an H5 format.
        :return: None
        """

        # load the MNIST dataset from keras library
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # scale images to the [0, 1] range
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # make sure that the images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert the labels to one-hot encode labels
        le = LabelBinarizer()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # initial variables
        optimizer = Adam(lr=self._learning_rate)
        model = self._build()
        # using a categorical_crossentropy since the are 10 digits 0-9
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # train the model
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  batch_size=self._batch_size, epochs=self._epochs, verbose=1)

        # evaluate the model
        self._evaluate(model, x_test, y_test, le)

        # save the model
        return model

    def save_model(self, model):
        """
        Saves the model to the class's argument path.
        :param model: a Sequential CNN model
        :return: None
        """
        model.save(self._model_path, save_format='h5')

    def load_model(self):
        """
        Loads the classifier model from the class's argument path.
        :return: the trained model
        """
        return load_model(self._model_path)

    def predict(self):
        pass