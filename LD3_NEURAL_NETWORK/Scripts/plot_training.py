import matplotlib.pyplot as plt
from neural_network import train_network, relu, relu_deriv, leaky_relu, leaky_relu_deriv

W1, b1, W2, b2, W3, b3, train_acc, test_acc, \
    train_loss_h, test_loss_h, train_acc_h, test_acc_h = train_network(leaky_relu, leaky_relu_deriv)

epochs = range(1, 5001)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_loss_h, label="Train Loss")
ax1.plot(epochs, test_loss_h,  label="Test Loss", linestyle="--")
ax1.set_title("Loss over Epochs"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, train_acc_h, label="Train Acc")
ax2.plot(epochs, test_acc_h,  label="Test Acc", linestyle="--")
ax2.set_title("Accuracy over Epochs"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../Sources/training_curves.png", dpi=150)
plt.show()