import numpy as np
from tqdm import tqdm

def softmax(z):
    return (np.exp(z).T / np.sum(np.exp(z), 1)).T

def fCE(X, Y, w, b, alpha):     # if alpha == 0 then not using regularization
    Y_hat = softmax(X.dot(w) + b)
    return - 1 / (len(X)) * np.sum((Y * np.log(Y_hat))) + (alpha / 2) * np.sum(w.T.dot(w) * np.eye(10))

def one_hot(y):
    return np.eye(10)[y]

def accuracy(Y, Y_hat):
    Y_hat_label = one_hot(np.argmax(Y_hat, 1))
    return np.sum(np.sum(Y == Y_hat_label, 1) == 10) / len(Y)

def softmax_regression (X_train, Y_train, X_valid, Y_valid):
    alphas = [1e-4, 1e-5, 1e-6, 1e-7]
    epsilons = [5e-3, 3e-3, 1e-3, 5e-4]
    batch_sizes = [10, 30, 100, 300]
    epoch_nums = [50, 100, 150, 200]
    validate_cases = []
    # validate_cases = [[1e-5, 5e-3, 20, 100]]
    for a in alphas:
        for e in epsilons:
            for bat in batch_sizes:
                for ep in epoch_nums:
                    validate_cases.append([a, e, bat, ep])
    best_sets = []
    best_fce = float("inf")

    def gradient(X, Y):
        Y_hat = softmax(X.dot(w) + b)
        w_ = (1 / batch_size) * X.T.dot(Y_hat - Y) + alpha / (batch_size) * w
        b_ = ((1 / batch_size) * (Y_hat - Y))[0]
        return w_, b_

    for alpha, epsilon, batch_size, epoch_num in tqdm(validate_cases):
        w = np.random.random((28 * 28, 10))
        b = 0
        batch_num = len(X_train) // batch_size
        for epoch_idx in range(epoch_num):
            for batch_idx in range(batch_num):
                X_batch = X_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                Y_batch = Y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                w_grad, b_grad = gradient(X_batch, Y_batch)
                w = w - epsilon * w_grad
                b = b - epsilon * b_grad
        validate_fce = fCE(X_valid, Y_valid, w, b, alpha)
        validate_acc = accuracy(Y_valid, softmax(X_valid.dot(w) + b))
        if validate_fce < best_fce:
            best_fce = validate_fce
            best_sets = [alpha, epsilon, batch_size, epoch_num, w, b]
        # print('alpha: %e, epsilon: %e, batch size: %d, epoch nums: %d : loss = %f, acc = %f'%(alpha, epsilon, batch_size, epoch_num, validate_fce, validate_acc))

    print(" * Best parameter set:")
    print('alpha = %e, epsilon = %e, batch size = %d, epoch nums = %d' % (best_sets[0], best_sets[1], best_sets[2], best_sets[3]))
    return best_sets[-2], best_sets[-1]

def train_softmax_regressor():

    X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28))
    ytr = np.load("fashion_mnist_train_labels.npy")
    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("fashion_mnist_test_labels.npy")

    X_tr, X_te = X_tr / 255, X_te / 255
    ytr, yte = one_hot(ytr), one_hot(yte)

    X_train, X_valid = X_tr[:int(0.8*len(X_tr))], X_tr[int(0.8*len(X_tr)):]
    Y_train, Y_valid = ytr[:int(0.8*len(X_tr))], ytr[int(0.8*len(X_tr)):]

    w, b = softmax_regression(X_train,Y_train,X_valid,Y_valid)

    fce_te = fCE(X_te, yte, w, b, 0)
    acc_te = accuracy(yte, softmax(X_te.dot(w) + b))
    print(" * Testing set result:")
    print("fCE = ", fce_te, " accuracy = ", acc_te)

train_softmax_regressor()
