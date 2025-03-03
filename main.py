!pip install openpyxl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import convolve
from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]
data = pd.read_excel(filename)

X = data[['SMART_5_NORMALIZED', 'SMART_187_NORMALIZED', 'SMART_188_NORMALIZED', 'SMART_197_NORMALIZED', 'SMART_198_NORMALIZED']]
y = data['SMART_5_NORMALIZED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("TRAINING SET SHAPE:", X_train.shape, y_train.shape)
print("TESTING SET SHAPE:", X_test.shape, y_test.shape)

def weibull_component(t, alpha1, beta1):
    return (beta1 / alpha1) * (t / alpha1) * (beta1 - 1) * np.exp(-(t / alpha1) * beta1)

def log_logistic_component(t, alpha2, beta2):
    return (beta2 / alpha2) * (t / alpha2) * (beta2 - 1) / ((1 + (t / alpha2) * beta2) ** 2)

def wll_accumulation_operator_vectorized(data, p=0.3, alpha1=1.5, alpha2=0.5, beta1=2.0, beta2=2.5):
    n = len(data)
    t = np.arange(1, n + 1)
    h = p * weibull_component(t, alpha1, beta1) + (1 - p) * log_logistic_component(t, alpha2, beta2)
    x1 = np.convolve(data, h, mode='full')[:n]
    return x1

X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_train_wll = X_train_numeric.apply(lambda col: wll_accumulation_operator_vectorized(col.values), axis=0)
print("TRANSFORMED TRAINING SET WITH WLL OPERATOR APPLIED:\n", X_train_wll.head())

p = 0.3
alpha1 = 1.5
alpha2 = 0.5
beta1 = 2.0
beta2 = 2.5
n = len(data)
t = np.arange(1, n + 1)
h = p * weibull_component(t, alpha1, beta1) + (1 - p) * log_logistic_component(t, alpha2, beta2)
mean_h = np.mean(h)

def reliability_function(t, alpha1, beta1, alpha2, beta2, p):
    r1 = np.exp(-(t / alpha1) ** beta1)
    r2 = 1 / (1 + (t / alpha2) ** beta2)
    return p * r1 + (1 - p) * r2

reliability = reliability_function(t, alpha1, beta1, alpha2, beta2, p)

def failure_rate(t):
    f1 = weibull_component(t, alpha1, beta1)
    f2 = log_logistic_component(t, alpha2, beta2)
    return (p * f1 + (1 - p) * f2) / reliability

print(failure_rate(t))

def compute_x1_sequence(x0_sequence):
    n = len(x0_sequence)
    x1_sequence = np.zeros(n)
    for k in range(n):
        x1_sequence[k] = sum(failure_rate(k - j + 1) * x0_sequence[j] for j in range(k + 1))
    return x1_sequence

X_original_sequences = [X_train_numeric.iloc[:, i].values for i in range(X_train_numeric.shape[1])]
X_other_sequences = [compute_x1_sequence(x0) for x0 in X_original_sequences]

import random

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def wllgm_predict(X, mu, gamma, v, smart_coefficients):
    return np.dot(X, smart_coefficients) * mu + gamma - v

def whale_optimization_algorithm(X_train, y_train, search_space, iterations=20, whales_count=10):
    whales = [np.array([random.uniform(low, high) for low, high in search_space]) for _ in range(whales_count)]
    best_whale = None
    best_mape = float('inf')
    smart_coefficients = np.random.rand(X_train.shape[1])
    for iteration in range(iterations):
        for whale in whales:
            mu, gamma, v = whale
            y_pred = wllgm_predict(X_train, mu, gamma, v, smart_coefficients)
            mape = calculate_mape(y_train, y_pred)
            if mape < best_mape:
                best_mape = mape
                best_whale = whale
        for i in range(whales_count):
            if random.random() < 0.5:
                whales[i] = best_whale + (random.random() - 0.5) * 2 * best_whale
            else:
                whales[i] = np.array([random.uniform(low, high) for low, high in search_space])
        print(f"Iteration {iteration + 1}/{iterations}, BEST MAPE: {best_mape}")
    return best_whale, best_mape, smart_coefficients

search_space = [(0.01, 1), (0.01, 5), (0.01, 2)]
best_params, best_mape, smart_coefficients = whale_optimization_algorithm(X_train_wll, y_train, search_space)

print("OPTIMAL PARAMETERS FOUND BY WOA:")
print("MU:", best_params[0])
print("GAMMA:", best_params[1])
print("V:", best_params[2])
model = wllgm_predict(X_train_wll, best_params[0], best_params[1], best_params[2], smart_coefficients)
print(np.mean(model))

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

mu_opt, gamma_opt, v_opt = best_params

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_wll)

ls_svm = SVR(kernel='linear', C=1.0)
ls_svm.fit(X_train_scaled, y_train)

support_vector_coeffs = ls_svm.coef_
print("ESTIMATED COEFFICIENTS (B_I) FOR EACH FEATURE:", support_vector_coeffs)

X_test_scaled = scaler.transform(X_test)
y_pred = ls_svm.predict(X_test_scaled)

test_mape = calculate_mape(y_test, y_pred)
print("TEST MAPE:", test_mape)

X1_sequence = np.array(X_train_wll)
X_other_sequences = [np.array([X1_sequence])]

def compute_z1(k, X1_sequence):
    return 0.5 * (X1_sequence[k-1] + X1_sequence[k])

def compute_wllgm_1_n(k, X1_sequence, X_other_sequences, a, gamma, v, b_i, b_n_plus_1):
    sum_term = sum(b_i[i] * X_other_sequences[i][k] for i in range(len(b_i)))
    interaction_term = (sum_term) ** v
    z1_k = compute_z1(k, X1_sequence)
    left_term = X1_sequence[k] - X1_sequence[k-1] + a * z1_k
    right_term = gamma * interaction_term + b_n_plus_1
    wllgm_1_n_value = left_term - right_term
    return wllgm_1_n_value

b_i = support_vector_coeffs
b_n_plus_1 = 0
k = 2
wllgm_value = compute_wllgm_1_n(k, X1_sequence, X_other_sequences, a=0.5, gamma=gamma_opt, v=v_opt, b_i=b_i, b_n_plus_1=b_n_plus_1)
print(f"WLLGM(1, N) VALUE AT K={k}: {wllgm_value}")

X_test_wll = X_test.apply(lambda col: wll_accumulation_operator_vectorized(col.values), axis=0)
X_test_scaled = scaler.transform(X_test_wll)

from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

lasso_model = LassoCV(alphas=np.logspace(-3, 3, 10), cv=5)
lasso_model.fit(X_train_scaled, y_train)

cv_mape_scores_lasso = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_percentage_error')
print("CROSS-VALIDATED MAPE WITH LASSO:", np.mean(cv_mape_scores_lasso))

y_pred_lasso = lasso_model.predict(X_test_scaled)
test_mape_lasso = calculate_mape(y_test, y_pred_lasso)
print("TEST MAPE AFTER REGULARIZATION WITH LASSO:", test_mape_lasso)

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae = mean_absolute_error(y_test, y_pred_lasso)
ape = np.abs((y_test - y_pred_lasso) / y_test) * 100
std_ape = np.std(ape)

print("EVALUATION METRICS ON TEST SET")
print(f"MAPE: {test_mape_lasso} %")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"STD OF APE: {std_ape}")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100
std_ape = np.std(ape)

print("EVALUATION METRICS ON TEST SET.")
print(f"MAPE: {test_mape} %")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"STD OF APE: {std_ape}")

influence_strength = np.abs(smart_coefficients)
most_influential_index = np.argmax(influence_strength)
print(f"THE MOST INFLUENTIAL SMART ATTRIBUTE IS SMART ATTRIBUTE {most_influential_index + 1}")
print("SMART ATTRIBUTE INFLUENCES:", influence_strength)

import matplotlib.pyplot as plt
smart_attribute_names = ['SMART_5_NORMALIZED', 'SMART_187_NORMALIZED', 'SMART_188_NORMALIZED', 'SMART_197_NORMALIZED', 'SMART_198_NORMALIZED']
plt.bar(smart_attribute_names, influence_strength)
plt.xlabel('SMART ATTRIBUTES')
plt.ylabel('INFLUENCE (COEFFICIENT MAGNITUDE)')
plt.title('INFLUENCE OF SMART ATTRIBUTES ON HDD FAILURE')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = data.dropna()
independent_vars = ['SMART_198_NORMALIZED', 'SMART_188_NORMALIZED', 'SMART_187_NORMALIZED', 'SMART_197_NORMALIZED']
dependent_var = 'SMART_5_NORMALIZED'
X = data[independent_vars]
y = data[dependent_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

coefficients = model.coef_
intercept = model.intercept_
print("INTERCEPT", intercept)
print("COEFFICIENTS:", coefficients)

print("ESTIMATED PARAMETERS (MLR):")
for var, coef in zip(independent_vars, coefficients):
    print(f"{var}: {coef:.4f}")

def absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / y_true) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def std_ape(y_true, y_pred):
    ape = absolute_percentage_error(y_true, y_pred)
    return np.std(ape)

fitted_values = model.predict(X_train)
predicted_values = model.predict(X_test)

mape_fitted = np.mean(absolute_percentage_error(y_train, fitted_values))
rmse_fitted = root_mean_squared_error(y_train, fitted_values)
std_ape_fitted = std_ape(y_train, fitted_values)
mae_fitted = mean_absolute_error(y_train, fitted_values)

mape_predicted = np.mean(absolute_percentage_error(y_test, predicted_values))
rmse_predicted = root_mean_squared_error(y_test, predicted_values)
std_ape_predicted = std_ape(y_test, predicted_values)
mae_predicted = mean_absolute_error(y_test, predicted_values)

print("METRICS FOR FITTED VALUES (TRAINING SET):")
print(f"MAPE: {mape_fitted:.2f} %")
print(f"RMSE: {rmse_fitted:.2f}")
print(f"STD OF APE: {std_ape_fitted:.2f}")
print(f"MAE: {mae_fitted:.2f}")

print("METRICS FOR PREDICTED VALUES (TEST SET):")
print(f"MAPE: {mape_predicted:.2f} %")
print(f"RMSE: {rmse_predicted:.2f}")
print(f"STD OF APE: {std_ape_predicted:.2f}")
print(f"MAE: {mae_predicted:.2f}")