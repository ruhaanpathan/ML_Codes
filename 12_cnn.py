# Convolutional Neural Network (CNN) for MNIST digit classification
# Requires TensorFlow. The MNIST dataset is downloaded on first run.
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixels and add the channel dimension.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

predictions = model.predict(x_test[:5])
for i in range(5):
    print(f"Actual: {y_test[i]}, Predicted: {predictions[i].argmax()}")
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title(str(predictions[i].argmax()))
    plt.axis("off")
plt.tight_layout()
plt.show()
