# TensorFlow DigitNet - CNN-Based Handwritten Digit Classification and Recognition
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def build_sequential_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    return model

def build_functional_model(input_shape):
    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat1 = Flatten()(pool3)
    dense1 = Dense(128, activation='relu')(flat1)
    drop1 = Dropout(0.2)(dense1)
    dense2 = Dense(64, activation='relu')(drop1)
    drop2 = Dropout(0.3)(dense2)
    dense3 = Dense(32, activation='relu')(drop2)
    drop3 = Dropout(0.2)(dense3)
    output_layer = Dense(10, activation='softmax')(drop3)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build, compile, and train the Functional API model
model_func = build_functional_model((28, 28, 1))
model_func.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_func.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=1000, verbose=1)

# Save the model in the native Keras format
model_func.save('model.keras')

# Load the model in the native Keras format
model = load_model('model.keras')

# Ensure compiled metrics are available by recompiling or evaluating
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.evaluate(x_test, y_test)

# Predict and evaluate
predictions = model.predict(x_test)
labels_predicted = np.argmax(predictions, axis=1)

# Confusion Matrix and Classification Report
print(confusion_matrix(np.argmax(y_test, axis=1), labels_predicted))
print(classification_report(np.argmax(y_test, axis=1), labels_predicted))

# Visualization of a test image and its predicted label
ind = 5375
sample_image = x_test[ind, :, :, :]

plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Label: {labels_predicted[ind]}")
plt.show()
