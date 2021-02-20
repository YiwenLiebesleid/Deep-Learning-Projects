import numpy as np
from tqdm import tqdm

def lr_1 (X_tr, y_tr):
    X_T = X_tr.T
    I = np.eye(X_T.dot(X_tr).shape[0])
    XT_X_inv = np.linalg.solve(X_T.dot(X_tr), I)

    return XT_X_inv.dot(X_T.dot(y_tr))

def train_1():
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = lr_1(X_tr, ytr)

    yhat_tr = X_tr.dot(w)
    fmse_tr = 1/(2*len(X_tr)) * np.sum(np.power(yhat_tr - ytr, 2))
    yhat_te = X_te.dot(w)
    fmse_te = 1/(2*len(X_te)) * np.sum(np.power(yhat_te - yte, 2))
    print("Fmse of training set: " + str(fmse_tr))
    print("Fmse of testing set: " + str(fmse_te))

def fmse(X, Y, w, b):
    return 1 / (2 * len(X)) * np.sum(((X.dot(w) + b) - Y) ** 2)

def linear_regression (X_train,Y_train,X_valid,Y_valid):
    alphas = [1e-3, 1e-4, 1e-5, 1e-6]
    epsilons = [1e-3, 5e-4, 3e-4, 1e-4]
    batch_sizes = [100, 80, 40, 20]
    epoch_nums = [100, 120, 150, 180]
    validate_cases = []
    for a in alphas:
        for e in epsilons:
            for bat in batch_sizes:
                for ep in epoch_nums:
                    validate_cases.append([a, e, bat, ep])
    best_sets = []
    best_fmse = float("inf")
    def f(X,Y):
        return 1/(2*len(X))*np.sum(((X.dot(w) + b) - Y)**2) + alpha/(2*len(X))*np.sum(np.power(w, 2))
    def w_gradient(X,Y):
        return 1 / (batch_size) * (((X.dot(w) + b) - Y).T.dot(X)).reshape(w.shape) + alpha / (batch_size) * w
    def b_gradient(X,Y):
        return (1 / (batch_size) * (((X.dot(w) + b) - Y)))[0]

    for alpha, epsilon, batch_size, epoch_num in tqdm(validate_cases):
        # print('alpha = %e, epsilon = %e, batch size = %d, epoch nums = %d'%(alpha, epsilon, batch_size, epoch_num))
        w = np.random.random((48*48))*2 - 1
        b = 0
        batch_num = len(X_train)//batch_size
        for epoch_idx in range(epoch_num):
            for batch_idx in range(batch_num):
                X_batch = X_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                Y_batch = Y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                w_grad = w_gradient(X_batch, Y_batch)
                b_grad = b_gradient(X_batch, Y_batch)
                w = w - epsilon*w_grad
                b = b - epsilon*b_grad
        validate_fmse = fmse(X_valid,Y_valid, w, b)
        if validate_fmse < best_fmse:
            # print("validation_loss = ", validate_fmse)
            best_fmse = validate_fmse
            best_sets = [alpha, epsilon, batch_size, epoch_num, w, b]

    print('alpha = %e, epsilon = %e, batch size = %d, epoch nums = %d' % (best_sets[0], best_sets[1], best_sets[2], best_sets[3]))
    return best_sets[-2], best_sets[-1]

def train_age_regressor():
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_train,X_valid = X_tr[:int(0.8*len(X_tr))], X_tr[int(0.8*len(X_tr)):]
    Y_train,Y_valid = ytr[:int(0.8*len(X_tr))], ytr[int(0.8*len(X_tr)):]
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w, b = linear_regression(X_train,Y_train,X_valid,Y_valid)

    fmse_te = fmse(X_te, yte, w, b)
    print("Fmse of testing set: " + str(fmse_te))

print(" * L2 regularized via SGD:")
train_age_regressor()
print(" * Unregularized:")
train_1()
