

import cv2
from matplotlib.pyplot import imread
import tensorflow as tf
import numpy as np


mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

x_train= tf.keras.utils.normalize(x_train,axis=1)
x_test= tf.keras.utils.normalize(x_test,axis=1)

model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
loss, accuracy = model.evaluate(x_test,y_test)

print(accuracy)
print(loss)

model.save('ident_model.h5')

# test_images = x_test[1:5]
# test_images = test_images.reshape(test_images.shape[0], 28, 28)
# print ("Test images shape: {}".format(test_images.shape))
# for i, test_image in enumerate(test_images, start=1):
#     org_image = test_image
#     test_image = test_image.reshape(1,1,28,28)
#     prediction = (model.predict(x_test) > 0.5).astype("int32")
#     print ("Predicted digit: {}".format(prediction[0]))
#     plt.subplot(220+i)
#     plt.axis('off')
#     plt.title("Predicted digit: {}".format(prediction[0]))


timg=cv2.imread('21//cell4.png')
timg=timg[:,:,0]
a=np.sum(timg==255)
print(a)

new_model=tf.keras.models.load_model('ident_model.h5')
re = timg.reshape(1,28,28,1)
re=np.float64(re)
re /= 255
prediction = new_model.predict(re)
print (prediction.argmax())

