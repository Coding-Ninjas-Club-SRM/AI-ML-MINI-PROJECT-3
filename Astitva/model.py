import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

X_train  = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))
model.add(Dense(100,activation="relu"))
model.add(Dense(10,activation="softmax"))

model.summary()

model.compile(optimizer='adam',
     loss = tf.keras.losses.categorical_crossentropy,
      metrics=['accuracy'])

# callbacks

es =EarlyStopping(monitor = 'val_acc',
         min_delta = 0.01, patience = 4, 
        verbose= 1)

mc = ModelCheckpoint("./modelfinal.h5" ,
                    monitor = "val_acc",
                    verbose = 1,
                    save_best_only= True)


callback = [es,mc]
history = model.fit(X_train, y_train, epochs =50 , validation_split =0.3, callbacks=callback)

model.save("model.h5")
print("Saved model to disk")

# #########################################################   Model Test   #################################################################

test_images = X_test[1:5]
test_images = test_images.reshape(test_images.shape[0], 28, 28)
print ("Test images shape: {}".format(test_images.shape))
for i, test_image in enumerate(test_images, start=1):
    org_image = test_image
    test_image = test_image.reshape(1,1,28,28)
    test_image = np.float64(test_image)
    prediction = (model.predict(test_image) > 0.5)
    print ("Predicted digit: ",prediction.argmax())