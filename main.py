# IMPORTING THE REQUIRED LIBRARIES
!pip install openpyxl  # Install openpyxl for Excel file handling
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import convolve
# FOR GOOGLE COLAB:
from google.colab import files

# Load the data and split into training and testing sets where X are the independent SMART attributes
uploaded = files.upload()  # Upload file in Google Colab
filename = list(uploaded.keys())[0]  # Get the filename
data = pd.read_excel(filename)  # Read the Excel file using pandas

# Select the SMART attributes as features
X = data[['SMART_5_NORMALIZED', 'SMART_187_NORMALIZED', 'SMART_188_NORMALIZED', 'SMART_197_NORMALIZED', 'SMART_198_NORMALIZED']]
# Using 'SMART_5_NORMALIZED' as target variable
y = data['SMART_5_NORMALIZED']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output shapes of the training and test sets
print("TRAINING SET SHAPE:", X_train.shape, y_train.shape)
print("TESTING SET SHAPE:", X_test.shape, y_test.shape)

# "DEFINING THE PDF OF EACH DISTRIBUTION SUCH AS WEIBULL AND LOG LOGISTIC AND THEN USING IT TO DEFINE THE WEIBULL LOG LOGISTIC MODEL AND THEN CALCULATE THE WLL ACCUMULATION GENERATION OPERATOR."

def weibull_component(t, alpha1, beta1):
    """WEIBULL PROBABILITY DENSITY FUNCTION (PDF)"""
    return (beta1 / alpha1) * (t / alpha1) ** (beta1 - 1) * np.exp(-(t / alpha1) ** beta1)

def log_logistic_component(t, alpha2, beta2):
    """LOG-LOGISTIC PROBABILITY DENSITY FUNCTION (PDF)"""
    return (beta2 / alpha2) * (t / alpha2) ** (beta2 - 1) / ((1 + (t / alpha2) ** beta2) ** 2)

def wll_accumulation_operator_vectorized(data, p=0.3, alpha1=1.5, alpha2=0.5, beta1=2.0, beta2=2.5):
    """APPlies THE WEIBULL LOG-LOGISTIC (WLL) ACCUMULATION GENERATION OPERATOR ON A DATA SEQUENCE WITHOUT USING FOR LOOPS."""
    n = len(data)
    t = np.arange(1, n + 1)  # Time steps from 1 to n
    # Compute H(t) for all t using vectorized operations
    h = p * weibull_component(t, alpha1, beta1) + (1 - p) * log_logistic_component(t, alpha2, beta2)
    # Perform convolution between data and H(t)
    x1 = np.convolve(data, h, mode='full')[:n]
    return x1

# Ensure X_train contains only numeric data
X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
# Apply WLL accumulation generation operator to each SMART attribute in the training set
X_train_wll = X_train_numeric.apply(lambda col: wll_accumulation_operator_vectorized(col.values), axis=0)
# Display the transformed training set
print("TRANSFORMED TRAINING SET WITH WLL OPERATOR APPLIED:\n", X_train_wll.head())

# "TO CALCULATE THE HAZARD FUNCTION I.E THE VALUE OF FAILURE RATE OF WLLGM MODEL"
p = 0.3  # MIXING PROPORTION
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

# Calculate X(1)_i sequence using the accumulation operator
def compute_x1_sequence(x0_sequence):
    n = len(x0_sequence)
    x1_sequence = np.zeros(n)
    for k in range(n):
        x1_sequence[k] = sum(failure_rate(k - j + 1) * x0_sequence[j] for j in range(k + 1))
    return x1_sequence

# Placeholder for X_ORIGINAL_SEQUENCES (not provided in document)
X_original_sequences = [X_train_numeric.iloc[:, i].values for i in range(X_train_numeric.shape[1])]
X_other_sequences = [compute_x1_sequence(x0) for x0 in X_original_sequences]

# "USING THE WHALE OPTIMIZATION TECHNIQUE TO CALCULATE THE META-HEURISTIC PARAMTERS AND CALCULATING THE BEST MAPE I.E THE FITTED MAPE FOR THE MODEL."
import random

# Define the objective function (MAPE)
def calculate_mape(y_true, y_pred):
    """CALCULATE MAPE BETWEEN ACTUAL AND PREDICTED VALUES."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to calculate WLLGM predictions, considering SMART attribute coefficients
def wllgm_predict(X, mu, gamma, v, smart_coefficients):
    return np.dot(X, smart_coefficients) * mu + gamma - v

# Whale Optimization Algorithm
def whale_optimization_algorithm(X_train, y_train, search_space, iterations=20, whales_count=10):
    # Define parameters and initialize whales randomly within the search space
    whales = [np.array([random.uniform(low, high) for low, high in search_space]) for _ in range(whales_count)]
    best_whale = None
    best_mape = float('inf')
    # Initialize random SMART attribute coefficients
    smart_coefficients = np.random.rand(X_train.shape[1])  # One coefficient per SMART attribute

    for iteration in range(iterations):
        for whale in whales:
            mu, gamma, v = whale
            y_pred = wllgm_predict(X_train, mu, gamma, v, smart_coefficients)
            mape = calculate_mape(y_train, y_pred)
            # Update best whale
            if mape < best_mape:
                best_mape = mape
                best_whale = whale
        # Whale position update based on encircling behavior and random exploration
        for i in range(whales_count):
            if random.random() < 0.5:
                whales[i] = best_whale + (random.random() - 0.5) * 2 * best_whale  # Encircle
            else:
                whales[i] = np.array([random.uniform(low, high) for low, high in search_space])  # Explore
        print(f"Iteration {iteration + 1}/{iterations}, BEST MAPE: {best_mape}")
    return best_whale, best_mape, smart_coefficients

# Define search space for each parameter
search_space = [(0.01, 1),  # RANGE FOR MU
                (0.01, 5),  # RANGE FOR GAMMA
                (0.01, 2)]  # RANGE FOR V

# Run WOA to find the optimal parameters
best_params, best_mape, smart_coefficients = whale_optimization_algorithm(X_train_wll, y_train, search_space)

print("OPTIMAL PARAMETERS FOUND BY WOA:")
print("MU:", best_params[0])
print("GAMMA:", best_params[1])
print("V:", best_params[2])
model = wllgm_predict(X_train_wll, best_params[0], best_params[1], best_params[2], smart_coefficients)
print(np.mean(model))

# "USING THE LEAST SQUARE SUPPORT VECTOR MACHINE (LS-SVM) TO MODEL THE NON LINEAR PARAMETERS AND TO CALCULATE THE TEST MAPE I,E FOR PREDICTED VALUES."
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Parameters obtained from WOA
mu_opt, gamma_opt, v_opt = best_params

# Define the LS-SVM
scaler = StandardScaler()  # Achieved the data to improve SVM performance
X_train_scaled = scaler.fit_transform(X_train_wll)  # Scale transformed data

# Instantiate and train the Support Vector Regressor (SVR) as LS-SVM substitute
ls_svm = SVR(kernel='linear', C=1.0)  # Linear kernel for linear relationship
ls_svm.fit(X_train_scaled, y_train)

# Output the support vector coefficients (equivalent to the weights BI in the paper)
support_vector_coeffs = ls_svm.coef_
print("ESTIMATED COEFFICIENTS (B_I) FOR EACH FEATURE:", support_vector_coeffs)

# Calculate predictions on the test set for evaluation
X_test_scaled = scaler.transform(X_test)  # Scale the test set with the training scaler
y_pred = ls_svm.predict(X_test_scaled)

# Evaluate model performance on the test set
test_mape = calculate_mape(y_test, y_pred)
print("TEST MAPE:", test_mape)

# "TO COMPUTE THE WLLGM VALUE OF THE MODEL"
X1_sequence = np.array(X_train_wll)
X_other_sequences = [np.array([X1_sequence])]

# Define Z(1)(k) based on X1_sequence
def compute_z1(k, X1_sequence):
    return 0.5 * (X1_sequence[k-1] + X1_sequence[k])

# Compute the WLLGM(1, N) value
def compute_wllgm_1_n(k, X1_sequence, X_other_sequences, a, gamma, v, b_i, b_n_plus_1):
    # Compute the sum term (SUM_I=2^N B_I * X(1)_i(k))^V
    sum_term = sum(b_i[i] * X_other_sequences[i][k] for i in range(len(b_i)))
    interaction_term = (sum_term) ** v
    # Compute Z(1)(k)
    z1_k = compute_z1(k, X1_sequence)
    # Apply equation (14)
    left_term = X1_sequence[k] - X1_sequence[k-1] + a * z1_k
    right_term = gamma * interaction_term + b_n_plus_1
    # Calculate WLLGM(1, N) at index k
    wllgm_1_n_value = left_term - right_term
    return wllgm_1_n_value

# Placeholder values for b_i and b_n_plus_1 (not provided in document)
b_i = support_vector_coeffs  # Assuming b_i from LS-SVM coefficients
b_n_plus_1 = 0  # Placeholder
k = 2
wllgm_value = compute_wllgm_1_n(k, X1_sequence, X_other_sequences, a=0.5, gamma=gamma_opt, v=v_opt, b_i=b_i, b_n_plus_1=b_n_plus_1)
print(f"WLLGM(1, N) VALUE AT K={k}: {wllgm_value}")

# "LASSO REGRESSION IS USED WHEN THE MAPE IS LARGER AND WE REDUCE IT BY TAKING THE CROSS VALIDATION TERMS."
# Apply the WLL accumulation generation operator to the test set as we did with the training set
X_test_wll = X_test.apply(lambda col: wll_accumulation_operator_vectorized(col.values), axis=0)
# Now, scale the transformed test set with the same scaler used for the training set
X_test_scaled = scaler.transform(X_test_wll)

from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

# Apply LassoCV for regularization to minimize overfitting
lasso_model = LassoCV(alphas=np.logspace(-3, 3, 10), cv=5)  # Adjust alpha range if needed
lasso_model.fit(X_train_scaled, y_train)  # Train with scaled WLL-transformed data

# Evaluate with cross-validation
cv_mape_scores_lasso = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_percentage_error')
print("CROSS-VALIDATED MAPE WITH LASSO:", np.mean(cv_mape_scores_lasso))

# Predictions and evaluation on the test set with consistent scaling
y_pred_lasso = lasso_model.predict(X_test_scaled)
test_mape_lasso = calculate_mape(y_test, y_pred_lasso)
print("TEST MAPE AFTER REGULARIZATION WITH LASSO:", test_mape_lasso)

# "USED TO CALCULATE THE EVALUATION METRICS IN CASE OF LASSO REGRESSION"
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
# Calculate MAE
mae = mean_absolute_error(y_test, y_pred_lasso)
# Calculate standard deviation of absolute percentage error
ape = np.abs((y_test - y_pred_lasso) / y_test) * 100
std_ape = np.std(ape)

# Display metrics
print("EVALUATION METRICS ON TEST SET")
print(f"MAPE: {test_mape_lasso} %")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"STD OF APE: {std_ape}")

# "USED TO CALCULATE THE EVALUATION METRICS SUCH AS RMSE,MAE,STD FOR THE MODEL."
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
# Calculate standard deviation of absolute percentage error
ape = np.abs((y_test - y_pred) / y_test) * 100
std_ape = np.std(ape)

# Display metrics
print("EVALUATION METRICS ON TEST SET.")
print(f"MAPE: {test_mape} %")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"STD OF APE: {std_ape}")

# "TO DETERMINE THE SMART ATTRIBUTE THAT AFFECTS THE FAILURE RATE OF THE MODEL THE MOST BY COMPARING ALL THE SMART ATTRIBUTES."
# Determine the most influential SMART attribute
influence_strength = np.abs(smart_coefficients)  # Absolute values of the coefficients
# Get the index of the most influential SMART attribute
most_influential_index = np.argmax(influence_strength)
# Print the SMART attribute with the greatest influence
print(f"THE MOST INFLUENTIAL SMART ATTRIBUTE IS SMART ATTRIBUTE {most_influential_index + 1}")
# Display the influence of all SMART attributes
print("SMART ATTRIBUTE INFLUENCES:", influence_strength)

# Optional: Plot the influence of SMART attributes
import matplotlib.pyplot as plt
smart_attribute_names = ['SMART_5_NORMALIZED', 'SMART_187_NORMALIZED', 'SMART_188_NORMALIZED', 'SMART_197_NORMALIZED', 'SMART_198_NORMALIZED']
plt.bar(smart_attribute_names, influence_strength)
plt.xlabel('SMART ATTRIBUTES')
plt.ylabel('INFLUENCE (COEFFICIENT MAGNITUDE)')
plt.title('INFLUENCE OF SMART ATTRIBUTES ON HDD FAILURE')
plt.show()

# "TO COMPARE THE WLLGM MODEL WITH OTHER MODEL WE USE THE BELOW MLR MODEL"
# WHERE WE CHECK FOR NULL OR MISSING VALUES AND DROP IF ANY. THEN WE SET THE INDEPENDENT AND DEPENDENT VARIABLES AND SPLIT THE DATA INTO TEST AND TRAIN DATASETS.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Remove rows with missing values if any
data = data.dropna()
# Select SMART attributes as independent variables (based on the research paper)
independent_vars = ['SMART_198_NORMALIZED', 'SMART_188_NORMALIZED', 'SMART_187_NORMALIZED', 'SMART_197_NORMALIZED']
dependent_var = 'SMART_5_NORMALIZED'
X = data[independent_vars]
y = data[dependent_var]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Multiple Linear Regression to establish the linear relation
model = LinearRegression()
model.fit(X_train, y_train)

# Retrieve the estimated parameters (coefficients and intercept)
coefficients = model.coef_
intercept = model.intercept_
print("INTERCEPT", intercept)
print("COEFFICIENTS:", coefficients)

# Display the results
print("ESTIMATED PARAMETERS (MLR):")
for var, coef in zip(independent_vars, coefficients):
    print(f"{var}: {coef:.4f}")

# "CALCULATING THE EVALUATION METRICS FOR BOTH FITTED AND PREDICTED VALUES."
# Function to calculate APE (Absolute Percentage Error)
def absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / y_true) * 100

# Function to calculate RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to calculate STD of absolute percentage error
def std_ape(y_true, y_pred):
    ape = absolute_percentage_error(y_true, y_pred)
    return np.std(ape)

# Get fitted and predicted values
fitted_values = model.predict(X_train)  # Fitted values for the training set
predicted_values = model.predict(X_test)  # Predicted values for the test set

# Calculate metrics for fitted values (training set)
mape_fitted = np.mean(absolute_percentage_error(y_train, fitted_values))
rmse_fitted = root_mean_squared_error(y_train, fitted_values)
std_ape_fitted = std_ape(y_train, fitted_values)
mae_fitted = mean_absolute_error(y_train, fitted_values)

# Calculate metrics for predicted values (test set)
mape_predicted = np.mean(absolute_percentage_error(y_test, predicted_values))
rmse_predicted = root_mean_squared_error(y_test, predicted_values)
std_ape_predicted = std_ape(y_test, predicted_values)
mae_predicted = mean_absolute_error(y_test, predicted_values)

# Display results
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