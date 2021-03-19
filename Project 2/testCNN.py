import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#get CIFAR10 training and testing images
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#sequential model is good for when you have 1 input/output per layer
model = models.Sequential()
#convolutional layer with rectified linear unit activation function
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#max pooling
model.add(layers.MaxPooling2D((2, 2)))
#convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#max pooling
model.add(layers.MaxPooling2D((2, 2)))
#convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#add dense/fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#build/complie model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train model
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
                    
#test model - note, testing data is same as validation here
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_loss, test_acc)