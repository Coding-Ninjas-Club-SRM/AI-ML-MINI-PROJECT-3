import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras import backend as K
import matplotlib.pyplot as plt


from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import SGD

def input_data():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

   
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    return X_test, y_test, X_train, y_train



def create_model():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate_model(X_train, y_Train, n_folds=5):

    accuracy, data = list(), list()

    
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    for x_train, x_test in kfold.split(X_train):
        
        model = create_model()
        
        trainX, trainY, testX, testY = X_train[x_train], y_Train[x_train], X_train[x_test], y_Train[x_test]
        
        data_fit = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)
        
        _, acc = model.evaluate(testX, testY, verbose=0)
        
        accuracy.append(acc)
        data.append(data_fit)
    return accuracy, data



def summarize_diagnostics(data):
    for i in range(len(data)):
       
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(data[i].history['loss'], color='red', label='green')
        pyplot.plot(data[i].history['val_loss'], color='orange', label='test')
        
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(data[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(data[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()



def summarize_performance(acc):
    
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (numpy.mean(acc) * 100, numpy.std(acc) * 100, len(acc)))

    
    pyplot.boxplot(acc)
    pyplot.show()


def test(X_train, model):
    test_images = X_train[1:5]
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    for i, test_image in enumerate(test_images, start=1):
        org_image = test_image
        test_image = test_image.reshape(1, 28, 28, 1)
        prediction = model.predict_classes(test_image, verbose=0)

        print("Predicted digit: {}".format(prediction[0]))
        plt.subplot(220 + i)
        plt.axis('off')
        plt.title("Predicted digit: {}".format(prediction[0]))
        plt.imshow(org_image, cmap=plt.get_cmap('gray'))

    plt.show()


def run():
    X_test, y_test, X_train, y_train = input_data()

    
    model = create_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    
    test(X_train, model)


    model.save("model.h5")