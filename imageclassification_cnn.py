
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess the data
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the ResNet model
def resnet_block(x, filters, kernel_size=3, stride=1, activation='relu'):
    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)
    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = Activation(activation)(out)
    return out

input_tensor = Input(shape=(32, 32, 3))
x = Conv2D(64, 7, padding='same')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu')(x)

for _ in range(3):
    x = resnet_block(x, 64)

x = Conv2D(128, 3, strides=2, padding='same')(x)
for _ in range(3):
    x = resnet_block(x, 128)

x = Conv2D(256, 3, strides=2, padding='same')(x)
for _ in range(3):
    x = resnet_block(x, 256)

x = GlobalAveragePooling2D()(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Make predictions on new data
def make_prediction(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return class_idx

# Usage for making predictions
image_path = 'image1.jpg'
predicted_class_idx = make_prediction(image_path)
print(f'Predicted class index: {predicted_class_idx}')