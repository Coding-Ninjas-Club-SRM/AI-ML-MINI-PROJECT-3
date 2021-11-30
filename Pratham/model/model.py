import numpy as np
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

(X_train, y_train),(X_test, y_test) = mnist.load_data()

# Normaliization of Image to [0,1] range
X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

# reshape/expanding the Dimension of image to (28,28,1)
X_train  = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Converting Output Variable to one hot vectors 
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)




model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),input_shape = (28,28,1), activation='relu'))
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

mc = ModelCheckpoint("./model.h5" ,
                    monitor = "val_acc",
                    verbose = 1,
                    save_best_only= True)


callback = [es,mc]
print("[INFO] TRAINING MODEL STARTED ")
history = model.fit(X_train, y_train, epochs =50 , validation_split =0.3, callbacks = callback)

model.save("model.h5")
print("[INFO] MODEL SAVED TO DISK")
