import os
import logging
import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


class DigitsNet:
    """
    A convolutional neural network (CNN) for handwritten digit classification.
    The MNIST dataset has been used here for the training step that contains 60,000
    small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

    Code example:
        digits_net = DigitsNet()
        trained_model = digits_net.train(reset_model=False, save_model=True)
    """
    def __init__(self, width=28, height=28, num_classes=10, epochs=100, batch_size=128,
                 model_path='../models/digits_classifier.h5'):
        # disable the tensorflow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        logging.getLogger('tensorflow').setLevel(logging.FATAL)

        # image settings
        self._width = width
        self._height = height
        self._num_classes = num_classes

        # training settings
        self._epochs = epochs
        self._batch_size = batch_size

        # output path for saving the model
        self._model_path = model_path

        # define the cnn model
        self._model = None
        if self.is_trained():
            self._model = self.load_model()

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
        The network contains 6 hidden layers: CNNs and fully connected (FC) layers.

        * A "ReLU" activation is been used here, since ot does not activate all the neurons at the same time,
        and the results are more accurate (than the sigmoid activation).
        * A "BatchNormalization" is been used here for reducing the number of training epochs required to train
        deep networks.
        * An "Adam" optimizer is been used here which combines the best properties of the AdaGrad and RMSProp algorithms
         to provide an optimization algorithm that can handle sparse gradients on noisy problems.

        Class Parameters:
        width: the width of the input image (28 pixels)
        height: the height of the input image (28 pixels)
        num_classes: the number of the output classes (0-9 digits)

        :param debug: boolean, for visualization purposes
        :return: the CNN model
        """

        # initial model
        model = Sequential()

        # define the input shape for the given image, the depth is 1 since this is a greyscale image
        input_shape = (self._width, self._height, 1)

        # the input layer
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())

        # hidden layer 1 - CNN
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())

        # hidden layer 2 - CNN
        model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # hidden layer 3 - CNN
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())

        # hidden layer 4 - CNN
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())

        # hidden layer 5 - CNN
        model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # hidden layer 6 - Fully Connected
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # the output layer
        model.add(Dense(self._num_classes, activation='softmax'))

        if debug:
            # visualization of the model's structure
            model.summary()

        return model

    def train(self, reset_model=False, save_model=True):
        """
        Trains the CNN model using the MNIST dataset from the Keras library.
        The function encodes the classes to one-hot representations,
        which means that the digit 4 will be as [0,0,0,0,1,0,0,0,0,0].
        The function performs an evaluation and a model saving in an H5 format.

        :param reset_model: boolean, if True train the initial model, otherwise, continue to train the saved model
        :param save_model: boolean, if True then save the model to the class's path
        :return: A Sequential trained model
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

        # if True, reset the model, otherwise continue to train the saved model
        if reset_model:
            model = self._build()
        else:
            model = self.load_model()

        # using a categorical_crossentropy since the are 10 digits 0-9
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train the model
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  batch_size=self._batch_size, epochs=self._epochs, verbose=1)

        # define the model's class
        self._model = model

        # evaluate the model
        self._evaluate(model, x_test, y_test, le)

        # save the model
        if save_model:
            self.save_model(model)

        return model

    def save_model(self, model):
        """
        Saves the model to the class's argument path.
        According to the directory tree, the "models" folder should be within the project.
        the prefix to the path.
        :param model: a Sequential CNN model
        :return: None
        """
        model.save(self._model_path, save_format='h5')
        print(f'{__class__.__name__} INFO: {self._model_path} was saved successfully.')

    def load_model(self):
        """
        Loads the classifier model from the class's argument path.
        :return: the trained model
        """
        model = load_model(self._model_path)
        print(f'{__class__.__name__} INFO: {self._model_path} was loaded successfully.')
        return model

    def predict(self, image):
        """
        Uses the trained model to predict the given input image.
        :param image: ndarray, the digit image
        :return: int64, return the predicted digit
        """

        # verify that the model was loaded correctly
        if self._model is None:
            raise Exception('{__class__.__name__} ERROR: cannot predict the image since the model is None.')

        # resize the image to the given cnn input shape
        image = cv2.resize(src=image, dsize=(self._width, self._height), interpolation=cv2.INTER_AREA)

        # scale the image to the [0, 1] range
        image = image.astype('float32') / 255.0

        # convert the image to array
        image_arr = img_to_array(image)

        # make sure that the images have shape (28, 28, 1)
        image_arr = np.expand_dims(image_arr, axis=0)

        return self._model.predict_classes(image_arr)[0]

    def is_trained(self):
        """
        Verify if the model was trained or not by checking the model path in the class.
        If the model path doesn't exists then return False, otherwise True.
        :return: boolean
        """
        return os.path.exists(self._model_path)