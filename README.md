Student Name: Veera venkata satya hara charan. Chintham
Student Id: 700756442

1. Tensor Manipulations & Reshaping
Step 1: Create a random tensor of shape (4, 6)
tensor = tf.random.normal([4, 6])
This line creates a random tensor (array) of shape 4x6 with values sampled from a normal distribution.

Step 2: Find its rank and shape
tensor_rank = tf.rank(tensor)
tensor_shape = tensor.shape
tf.rank(tensor) gives the number of dimensions of the tensor, which is 2 in this case (4 rows, 6 columns).
tensor.shape gives the shape of the tensor, which is (4, 6).

Step 3: Reshape the tensor
reshaped_tensor = tf.reshape(tensor, [2, 3, 4])
This reshapes the tensor into a new shape of (2, 3, 4). This means 2 blocks, each of size (3, 4).

Step 4: Transpose the tensor
transposed_tensor = tf.transpose(reshaped_tensor, perm=[2, 1, 0])
tf.transpose changes the order of the dimensions. In this case, the new order becomes (4, 3, 2).

Step 5: Broadcasting and Addition
small_tensor = tf.random.normal([1, 4])
broadcasted_tensor = small_tensor + reshaped_tensor
small_tensor is broadcasted (expanded) to match the shape of reshaped_tensor, and element-wise addition is performed between them.

2. Loss Functions & Hyperparameter Tuning

Step 1: Define true values and predictions
y_true = np.array([[0, 1, 0], [1, 0, 1]])
y_pred1 = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.6]])
y_pred2 = np.array([[0.3, 0.6, 0.1], [0.5, 0.3, 0.5]])
y_true is the ground truth (one-hot encoded labels).
y_pred1 and y_pred2 are the predicted values from a model.

Step 2: Compute Losses
mse = MeanSquaredError()
cce = CategoricalCrossentropy()

loss_mse1 = mse(y_true, y_pred1).numpy()
loss_cce1 = cce(y_true, y_pred1).numpy()
loss_mse2 = mse(y_true, y_pred2).numpy()
loss_cce2 = cce(y_true, y_pred2).numpy()
We use two loss functions:
Mean Squared Error (MSE): Measures the difference between predicted and true values.
Categorical Cross-Entropy (CCE): Measures the loss for classification tasks.
These losses are calculated for both predictions y_pred1 and y_pred2.

Step 3: Plot Loss Values
plt.bar(labels, values, color=['blue', 'red', 'blue', 'red'])
plt.ylabel('Loss Value')
plt.title('Comparison of MSE and Cross-Entropy Loss')
plt.show()
A bar chart is plotted to compare the MSE and CCE for both predictions.

3. Train a Model with Different Optimizers
Step 1: Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
The MNIST dataset is loaded, where x_train and x_test are the images (28x28 pixels), and y_train and y_test are the labels (digits 0-9).
The pixel values are normalized to a range [0, 1] for better model performance.

Step 2: Build a simple model
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model
A simple neural network model with one hidden layer (128 units) and an output layer (10 units for 10 classes).

Step 3: Train with Adam and SGD Optimizers
# Adam optimizer
model_adam = build_model()
model_adam.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history_adam = model_adam.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)

# SGD optimizer
model_sgd = build_model()
model_sgd.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
history_sgd = model_sgd.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)
Two models are trained with different optimizers: Adam and SGD.
Each model is trained for 5 epochs, and we track accuracy for comparison.

Step 4: Compare Accuracy Trends
plt.plot(history_adam.history['val_accuracy'], label='Adam')
plt.plot(history_sgd.history['val_accuracy'], label='SGD')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Adam vs. SGD Performance on MNIST')
plt.show()
A plot shows how validation accuracy changes over epochs for Adam and SGD optimizers.

4. Train a Neural Network and Log to TensorBoard

Step 1: Setup and Train Model with TensorBoard Logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
The model is trained with TensorBoard logging enabled. This allows us to visualize training metrics (like accuracy) in TensorBoard.

Step 2: Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
The model is evaluated on the test data, and the test accuracy is printed.
TensorBoard Visualization

After running the training , you can run the following command in your terminal to launch TensorBoard and visualize the training logs:
bash
tensorboard --logdir=logs/fit

