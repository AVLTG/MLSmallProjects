import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Synthetic dataset generation
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


def mse(w, X, t):
    n = X.shape[0]
    y = X @ w
    err = y - t
    return np.sum(err ** 2) / (2 * n)


# Using sklearn for linear regression
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, t_train)

# Print weights and MSE
print("Weights:", lr.coef_)
print("Training MSE:", mse(lr.coef_, X_train, t_train))
print("Validation MSE:", mse(lr.coef_, X_valid, t_valid))

# Visualization of MSE values
training_mse = mse(lr.coef_, X_train, t_train)
validation_mse = mse(lr.coef_, X_valid, t_valid)
plt.bar(['Training MSE', 'Validation MSE'], [training_mse, validation_mse], color=['blue', 'red'])
plt.title('Training and Validation MSE')
plt.ylabel('MSE')
plt.show()
