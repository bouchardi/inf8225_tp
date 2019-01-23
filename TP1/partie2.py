import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()
X = digits.data  # 8x8 image of a digit
y = digits.target  # int representing a target digit

y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test , test_size=0.5, random_state=42)

W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))  # weights of shape KxL

best_W = None
best_accuracy = 0
lr = 0.001
nb_epochs = 50
minibatch_size = len(y_train) // 20

losses = []
accuracies = []

def softmax(X):
    e_x = np.exp(X - np.max(X))
    return e_x / e_x.sum(axis=0)

def get_accuracy(X, y, W):
    res = 0
    # Validation
    for model_input, target in zip(X, y):
        pred = get_prediction(model_input, W)
        res += int(np.argmax(pred) == np.argmax(target))
    return res / len(X)


def get_grads(target, pred, model_input):
    return np.dot(np.expand_dims(model_input, 1), np.expand_dims((pred - target), 1).T).T

def get_loss(target, pred):
    return (-target * np.log(pred) - (1 - target) * np.log(1 - pred)).mean()

def get_prediction(model_input, W):
    return softmax(np.dot(W, model_input))

print(f'len(X_train) {len(X_train)}')
print(f'len(y_train) {len(y_train)}')

# For each epoch
for epoch in range(nb_epochs):

    loss = 0
    accuracy = 0

    # For each batch
    for i in range(0 , minibatch_size * 19, minibatch_size):
        grads = 0

        # For each example
        for index in range(minibatch_size):

            # Model input, target
            model_input = X_train[i+index]
            target = y_train[i+index]

            # Forward pass, get prediction
            pred = get_prediction(model_input, W)

            # Compute the loss
            loss += get_loss(target, pred)

            # Get the gradient of the loss
            grads += get_grads(target, pred, model_input)

        # Get a step in the opposite direction
        delta = - lr * grads/minibatch_size

        # update the weights
        W = W + delta

    print(f'Loss {loss / minibatch_size} at {epoch} epochs')
    losses.append(loss / minibatch_size)

    accuracy = get_accuracy(X_validation, y_validation, W)
    accuracies.append(accuracy)

    if accuracy > best_accuracy:
        best_W = W.copy()

accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
print(accuracy_on_unseen_data)

plt.plot(losses)
plt.imshow(best_W[4, :].reshape(8, 8 ))
