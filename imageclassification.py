import tensorflow as tf
import scipy
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
 rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
 r'C:\dogscats\dogscats\train',
 target_size=(64, 64),
 batch_size=32,
 class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
 r'C:\dogscats\dogscats\train',
 target_size=(64, 64),
 batch_size=32,
 class_mode='binary')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', 
input_shape = [64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
history = cnn.fit(x =train_generator,validation_data = validation_generator, epochs = 2)
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("Desktop/dataset/single_prediction/predict1.jpg",target_size 
=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
train_generator.class_indices
if result[0][0] == 1:
 prediction = 'dog'
else:
 prediction = 'cat'
prediction
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
