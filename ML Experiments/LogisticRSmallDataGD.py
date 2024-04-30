import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate some random data
np.random.seed(0)
n_samples = 1000
gender = np.random.binomial(1, 0.5, n_samples)
race_ethnicity = np.random.randint(1, 7, n_samples)
chest_pain_ever = np.random.binomial(1, 0.25, n_samples)
drink_alcohol = np.random.binomial(1, 0.6, n_samples)
age = np.random.normal(50, 12, n_samples)
blood_cholesterol = np.random.normal(200, 30, n_samples)
BMI = np.random.uniform(18, 35, n_samples)
blood_pressure_sys = np.random.normal(120, 15, n_samples)
diastolic_bp = np.random.normal(80, 10, n_samples)
calories = np.random.normal(2000, 500, n_samples)
family_income = np.random.randint(1, 10, n_samples)
target_heart = np.random.binomial(1, 0.5, n_samples)  # Binary target for logistic regression

data = pd.DataFrame({
    'gender': gender,
    'race_ethnicity': race_ethnicity,
    'chest_pain_ever': chest_pain_ever,
    'drink_alcohol': drink_alcohol,
    'age': age,
    'blood_cholesterol': blood_cholesterol,
    'BMI': BMI,
    'blood_pressure_sys': blood_pressure_sys,
    'diastolic_bp': diastolic_bp,
    'calories': calories,
    'family_income': family_income,
    'target_heart': target_heart
})

feature_names = [
    "intercept",
    "gender_female",
    "re_hispanic",
    "re_white",
    "re_black",
    "re_asian",
    "chest_pain",
    "drink_alcohol",
    "age",
    "blood_cholesterol",
    "BMI",
    "blood_pressure_sys",
    "diastolic_bp",
    "calories",
    "family_income"]

data_fets = np.stack([
    np.ones(data.shape[0]),
    data["gender"] == 2,
    (data["race_ethnicity"] == 1) + (data["race_ethnicity"] == 2),
    data["race_ethnicity"] == 3,
    data["race_ethnicity"] == 4,
    data["race_ethnicity"] == 6,
    data["chest_pain_ever"] == 1,
    data["drink_alcohol"] == 1,
    data["age"],
    data["blood_cholesterol"],
    data["BMI"],
    data["blood_pressure_sys"],
    data["diastolic_bp"],
    data["calories"],
    data["family_income"]
], axis=1)

X_train, X_test, t_train, t_test = train_test_split(data_fets, data["target_heart"], test_size=0.2, random_state=1)
X_train, X_valid, t_train, t_valid = train_test_split(X_train, t_train, test_size=0.25, random_state=1)

numerical_value_start = 8
mean = X_train[:, numerical_value_start:].mean(axis=0)
std = X_train[:, numerical_value_start:].std(axis=0)

X_train_norm = X_train.copy()
X_valid_norm = X_valid.copy()
X_test_norm = X_test.copy()
X_train_norm[:, numerical_value_start:] = (X_train[:, numerical_value_start:] - mean) / std
X_valid_norm[:, numerical_value_start:] = (X_valid[:, numerical_value_start:] - mean) / std
X_test_norm[:, numerical_value_start:] = (X_test[:, numerical_value_start:] - mean) / std


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pred(w, X):
    z = np.dot(X, w)
    return sigmoid(z)


# Uses cross-entropy loss
def loss(w, X, t):
    z = np.dot(X, w)
    return np.mean(t * np.logaddexp(0, -z) + (1 - t) * np.logaddexp(0, z))


def accuracy(w, X, t, thres=0.5):
    y = pred(w, X)
    predictions = (y >= thres).astype(int)
    return np.mean(predictions == t)


def grad(w, X, t):
    y = pred(w, X)
    error = y - t
    return np.dot(X.T, error) / X.shape[0]


def solve_via_gradient_descent(alpha=0.0025, niter=1000,
                               X_train=X_train_norm, t_train=t_train,
                               X_valid=X_valid_norm, t_valid=t_valid,
                               w_init=None, plot=True):
    w = np.zeros(X_train.shape[1]) if w_init is None else w_init

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    for it in range(niter):
        dw = grad(w, X_train, t_train)
        w = w - alpha * dw

        if plot:
            train_loss.append(loss(w, X_train, t_train))
            valid_loss.append(loss(w, X_valid, t_valid))
            train_acc.append(accuracy(w, X_train, t_train))
            valid_acc.append(accuracy(w, X_valid, t_valid))

    if plot:
        plt.title("Training Curve Showing Training and Validation Loss at each Iteration")
        plt.plot(train_loss, label="Training Loss")
        plt.plot(valid_loss, label="Validation Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

        plt.title("Training Curve Showing Training and Validation Accuracy at each Iteration")
        plt.plot(train_acc, label="Training Accuracy")
        plt.plot(valid_acc, label="Validation Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.show()

        print("Final Training Loss:", train_loss[-1])
        print("Final Validation Loss:", valid_loss[-1])
        print("Final Training Accuracy:", train_acc[-1])
        print("Final Validation Accuracy:", valid_acc[-1])
    return w


sol = solve_via_gradient_descent(alpha=0.01, niter=500, X_train=X_train_norm, t_train=t_train, X_valid=X_valid_norm,
                                 t_valid=t_valid)
