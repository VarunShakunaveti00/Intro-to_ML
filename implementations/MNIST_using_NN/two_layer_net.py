import numpy as np
def relu(x):
    """
    Rectified Linear Unit activation function.
    """
    return np.maximum(0, x)

def grad_relu(x):
    """
    Gradient of the ReLU function.
    """
    return np.where(x > 0, 1, 0)

def softmax(x):
    """
    Softmax activation function.
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def grad_softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

"""def adam(w, dw, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):

    m = np.zeros_like(w)
    v = np.zeros_like(w)
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * dw ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w -= learning_rate * m_hat / (np.sqrt(v_hat + epsilon))
    return w"""

class TwoLayerNet(object):
    """
    Two-layer neural network for classification.
    Models a simple feedforward network with one hidden layer.
    Attributes:
    - input_size: Number of input features
    - hidden_size: Number of neurons in the hidden layer
    - output_size: Number of classes
    - learning_rate: Learning rate for weight updates
    - W1: Weights for the input to hidden layer
    - b1: Biases for the hidden layer
    - W2: Weights for the hidden to output layer
    - b2: Biases for the output layer
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, regularization=0.0, epochs=15):
        """
        Initialize the neural network.
        Parameters:
        - input_size: Number of input features
        - hidden_size: Number of neurons in the hidden layer
        - output_size: Number of classes
        - learning_rate: Learning rate for weight updates
        """
        self.params = {}
        self.reg = regularization
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Initialize weights and biases
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.params['b2'] = np.zeros(output_size)
        
        # Initialize Adam optimizer momentum variables
        self.adam_cache = {
            'W1': {'m': np.zeros((input_size, hidden_size)), 'v': np.zeros((input_size, hidden_size)), 't': 1},
            'b1': {'m': np.zeros(hidden_size), 'v': np.zeros(hidden_size), 't': 1},
            'W2': {'m': np.zeros((hidden_size, output_size)), 'v': np.zeros((hidden_size, output_size)), 't': 1},
            'b2': {'m': np.zeros(output_size), 'v': np.zeros(output_size), 't': 1}
        }
    
    def forward(self, X):
        params = self.params
        #layer 1
        params['X_0'] = X
        params['Z1'] = params['X_0'] @ params['W1'] + params['b1']
        params['A1'] = relu(params['Z1'])
        #layer 2
        params['Z2'] = params['A1'] @ params['W2'] + params['b2']
        params['y_out'] = softmax(params['Z2'])
        return params['y_out']
    
    def backward(self, y, y_out):
        params = self.params
        change_w = {}
        batch_size = y.shape[0]

        error = y_out - y 
        change_w['W2'] = params['A1'].T @ error
        change_w['b2'] = np.sum(error, axis=0)
        change_w['W2'] += self.reg * params['W2']

        error_1 = error @ params['W2'].T * grad_relu(params['Z1'])
        change_w['W1'] = params['X_0'].T @ error_1
        change_w['b1'] = np.sum(error_1, axis=0)
        change_w['W1'] += self.reg * params['W1']
        return change_w
    
    def update_weights(self, change_w):
        """Update weights using Adam optimization"""
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            
            param = self.params[param_name]
            grad = change_w[param_name]
            cache = self.adam_cache[param_name]
            cache['t'] += 1
            cache['m'] = beta1 * cache['m'] + (1 - beta1) * grad
            cache['v'] = beta2 * cache['v'] + (1 - beta2) * np.square(grad)
            m_hat = cache['m'] / (1 - beta1 ** cache['t'])
            v_hat = cache['v'] / (1 - beta2 ** cache['t'])
            self.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon) 
            
       
    def accuracy(self, y, y_out):
        return np.mean(np.argmax(y, axis=1) == np.argmax(y_out, axis=1))
        
    def predict(self, X):
        y_out = self.forward(X)
        return np.argmax(y_out, axis=1)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, num_epochs=10, verbose=False):
        """
        Train the neural network.
        
        Parameters:
        - X_train: Training data
        - y_train: Training labels
        - X_val: Validation data
        - y_val: Validation labels
        - batch_size: Mini-batch size
        - num_epochs: Number of epochs to train for
        - verbose: Whether to print progress
        
        Returns:
        - history: Dictionary containing training/validation loss and accuracy
        """
        num_train = X_train.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Shuffle data at the start of each epoch
            idx = np.random.permutation(num_train)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(iterations_per_epoch):
                # Get mini-batch
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_train)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                batch_loss = -np.mean(np.sum(y_batch * np.log(y_pred + 1e-10), axis=1))
                epoch_loss += batch_loss
                
                # Backward pass
                grads = self.backward(y_batch, y_pred)
                self.update_weights(grads)
                
            epoch_loss /= iterations_per_epoch
            history['train_loss'].append(epoch_loss)
            y_train_pred = self.forward(X_train)
            train_acc = self.accuracy(y_train, y_train_pred)
            history['train_acc'].append(train_acc)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = -np.mean(np.sum(y_val * np.log(y_val_pred + 1e-10), axis=1))
                val_acc = self.accuracy(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f}")
        
        return history







