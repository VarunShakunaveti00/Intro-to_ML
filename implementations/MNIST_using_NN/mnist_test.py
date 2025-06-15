import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path to import two_layer_net
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from two_layer_net import TwoLayerNet

def one_hot_encode(y, num_classes):
    """
    Convert a vector of labels to one-hot encoded matrix.
    
    Parameters:
    - y: Array of class labels of shape (N,)
    - num_classes: Number of classes
    
    Returns:
    - one_hot: One-hot encoded matrix of shape (N, num_classes)
    """
    N = y.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), y.astype(int)] = 1
    return one_hot

# Load MNIST data
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy().astype(np.float32)
y = mnist.target.to_numpy().astype(np.int64)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encode the labels
num_classes = 10
y_train = one_hot_encode(y_train, num_classes)
y_val = one_hot_encode(y_val, num_classes)
y_test = one_hot_encode(y_test, num_classes)

# Print the shapes of our datasets
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Initialize the model
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.001
reg = 0.01

# Create the model
model = TwoLayerNet(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    learning_rate=learning_rate,
    regularization=reg
)

print("Training the model...")
history = model.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    num_epochs=10,
    batch_size=128,
    verbose=True
)

test_pred = model.forward(X_test)
test_accuracy = model.accuracy(y_test, test_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()