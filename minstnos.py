import tensorflow as tf
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_valid = tf.keras.utils.normalize(X_train, axis=1)
model = tf.keras.models.Sequential();
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  #128 neurons in the layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 3  layers
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output - no of categories(here 10 -- 0-9)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs =3)
val_loss, val_acc = model.evaluate(X_valid,y_valid)
print(val_loss, val_accuracy)


import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap = plt.cm.binary)
plt.show()
print(X_train[0])

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([X_valid])
import numpy as np

print(np.argmax(predictions[0]))
plt.imshow(X_valid[0], cmap=plt.cm.binary)
plt.show();
