import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# 1. Load data
df = pd.read_csv("../Sources/heart.csv")

numerical_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
target_col     = "HeartDisease"

X = df[numerical_cols].values.astype(float)
y = df[target_col].values.astype(int)

# 2. Normalise to [-1, 1]
X_min  = X.min(axis=0)
X_max  = X.max(axis=0)

def normalise(data, x_min, x_max):
    return 2 * (data - x_min) / (x_max - x_min) - 1

X_norm = normalise(X, X_min, X_max)

# 3. Train / Test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)
X_train_raw, X_test_raw = train_test_split(
    X, test_size=0.2, random_state=42, stratify=y
)
# 4. One-hot encode labels
# e.g. label 1 → [0, 1],  label 0 → [1, 0]
n_classes = len(set(y))

def one_hot(labels, n):
    oh = np.zeros((len(labels), n))
    oh[np.arange(len(labels)), labels] = 1
    return oh

y_train_oh = one_hot(y_train, n_classes)
y_test_oh  = one_hot(y_test,  n_classes)

# 5. Network architecture: Input:6 | Hidden1:3 | Hidden2:4 | Output:2
n_input  = X_train.shape[1]  # 6
n_h1     = 3
n_h2     = 4
n_output = n_classes          # 2

# ReLU: negatives → 0, positives stay unchanged
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)
# Leaky ReLU: negatives get small slope instead of 0
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)
# Softmax: converts raw scores to probabilities that sum to 1
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# 7. Training accepts any activation function and its derivative
def train_network(activation, activation_deriv):
    np.random.seed(42)

    # weights initialised randomly, biases start at zero
    W1 = np.random.randn(n_input, n_h1)     * 0.5
    b1 = np.zeros((1, n_h1))
    W2 = np.random.randn(n_h1,   n_h2)     * 0.5
    b2 = np.zeros((1, n_h2))
    W3 = np.random.randn(n_h2,   n_output) * 0.5
    b3 = np.zeros((1, n_output))

    lr     = 0.1
    epochs = 5000

    train_loss_history, test_loss_history, train_acc_history, test_acc_history = [], [], [], []

    for epoch in range(epochs):

        # Forward pass
        z1 = X_train @ W1 + b1
        a1 = activation(z1)

        z2 = a1 @ W2 + b2
        a2 = activation(z2)

        z3 = a2 @ W3 + b3
        a3 = softmax(z3)

        # Loss
        loss = -np.mean(np.sum(y_train_oh * np.log(a3 + 1e-9), axis=1))

        # Record history
        ta1 = activation(X_test @ W1 + b1)
        ta2 = activation(ta1    @ W2 + b2)
        ta3 = softmax(ta2       @ W3 + b3)

        train_loss_history.append(loss)
        test_loss_history.append(-np.mean(np.sum(y_test_oh * np.log(ta3 + 1e-9), axis=1)))
        train_acc_history.append(np.mean(np.argmax(a3,  axis=1) == y_train) * 100)
        test_acc_history.append( np.mean(np.argmax(ta3, axis=1) == y_test)  * 100)

        # Backpropagation
        d3  = a3 - y_train_oh
        dW3 = a2.T @ d3 / len(X_train)
        db3 = d3.mean(axis=0, keepdims=True)

        d2  = (d3 @ W3.T) * activation_deriv(z2)
        dW2 = a1.T @ d2 / len(X_train)
        db2 = d2.mean(axis=0, keepdims=True)

        d1  = (d2 @ W2.T) * activation_deriv(z1)
        dW1 = X_train.T @ d1 / len(X_train)
        db1 = d1.mean(axis=0, keepdims=True)

        # Update weights
        W1 -= lr * dW1;  b1 -= lr * db1
        W2 -= lr * dW2;  b2 -= lr * db2
        W3 -= lr * dW3;  b3 -= lr * db3

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:4d}  loss={loss:.4f}")

    # Evaluate
    def predict(data):
        a1 = activation(data @ W1 + b1)
        a2 = activation(a1   @ W2 + b2)
        a3 = softmax(a2      @ W3 + b3)
        return np.argmax(a3, axis=1)

    train_acc = np.mean(predict(X_train) == y_train) * 100
    test_acc  = np.mean(predict(X_test)  == y_test)  * 100

    print(f"  Training accuracy: {train_acc:.2f}%")
    print(f"  Testing  accuracy: {test_acc:.2f}%")

    print(f"\nW1 (Input→Hidden1)  shape={W1.shape}:\n{np.round(W1, 6)}")
    print(f"\nb1 (bias Hidden1):\n{np.round(b1, 6)}")
    print(f"\nW2 (Hidden1→Hidden2) shape={W2.shape}:\n{np.round(W2, 6)}")
    print(f"\nb2 (bias Hidden2):\n{np.round(b2, 6)}")
    print(f"\nW3 (Hidden2→Output)  shape={W3.shape}:\n{np.round(W3, 6)}")
    print(f"\nb3 (bias Output):\n{np.round(b3, 6)}")

    return W1, b1, W2, b2, W3, b3, train_acc, test_acc, \
           train_loss_history, test_loss_history, train_acc_history, test_acc_history

# 8. Train using ReLU swap relu/relu_deriv for any other activation pair
print("Training with leaky_ReLU...")
W1, b1, W2, b2, W3, b3, train_acc, test_acc, \
    train_loss_history, test_loss_history, train_acc_history, test_acc_history = train_network(leaky_relu, leaky_relu_deriv)

# 9. Save weights for Excel (K.2 spreadsheet)
weights = {
    "W1": W1.tolist(), "b1": b1.tolist(),
    "W2": W2.tolist(), "b2": b2.tolist(),
    "W3": W3.tolist(), "b3": b3.tolist(),
    "X_min":      X_min.tolist(),
    "X_max":      X_max.tolist(),
    "features":   numerical_cols,
    "train_acc":  round(train_acc, 4),
    "test_acc":   round(test_acc,  4),
    "X_test_raw":  X_test_raw.tolist(),
    "X_test_norm": X_test.tolist(),
    "y_test":     y_test.tolist(),
}
with open("../Sources/weights.json", "w") as f:
    json.dump(weights, f, indent=2)