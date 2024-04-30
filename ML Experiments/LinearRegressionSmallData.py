import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
n_samples = 100
forest_air_temp_c = np.random.normal(20, 5, n_samples)
forest_soil_temp_c = np.random.normal(15, 5, n_samples)
forest_rh = np.random.uniform(30, 100, n_samples)
forest_soil_wc = np.random.uniform(0, 1, n_samples)
pond_air_temp_c = forest_air_temp_c + np.random.normal(0, 0.5, n_samples)  # Target variable

data = pd.DataFrame({
    'forest_air_temp_c': forest_air_temp_c,
    'forest_soil_temp_c': forest_soil_temp_c,
    'forest_rh': forest_rh,
    'forest_soil_wc': forest_soil_wc,
    'pond_air_temp_c': pond_air_temp_c
})

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, valid_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


def get_input_targets(data):
    t = np.array(data["pond_air_temp_c"])
    X_fets = np.array(data[["forest_air_temp_c", "forest_soil_temp_c", "forest_rh", "forest_soil_wc"]])
    n = len(data)
    X = np.concatenate((np.ones((n, 1)), X_fets), axis=1)
    return (X, t)


X_train, t_train = get_input_targets(train_data)
X_valid, t_valid = get_input_targets(valid_data)
X_test, t_test = get_input_targets(test_data)


def pred(w, X):
    return X @ w


def mse(w, X, t):
    n = X.shape[0]
    y = pred(w, X)
    err = y - t
    return np.sum(err ** 2) / (2 * n)


def grad(w, X, t):
    n = X.shape[0]
    gradient = (X.T @ (pred(w, X) - t)) / n
    return gradient


def solve_via_gradient_descent(alpha=0.0025, niter=1000, X_train=X_train, t_train=t_train, X_valid=X_valid,
                               t_valid=t_valid, w_init=None):
    w = np.zeros(X_train.shape[1]) if w_init is None else w_init
    train_mses = []
    valid_mses = []

    for it in range(niter):
        w -= alpha * grad(w, X_train, t_train)
        train_mses.append(mse(w, X_train, t_train))
        valid_mses.append(mse(w, X_valid, t_valid))

    plt.title("Training Curve (Training and Validation MSE)")
    plt.plot(train_mses, label="Training MSE")
    plt.plot(valid_mses, label="Validation MSE")
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


solve_via_gradient_descent(alpha=0.0001, niter=100)
