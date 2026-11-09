import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt


#train modifieds i truncated the dataset so it would run faster
#Train_processed is the original
#Use the larger training datasets for better accuracy
df = pd.read_csv("studentplacement/train_modified.csv")
#print(df.head())
y = df["label"]
X = df.drop("label", axis=1)

X = X.to_numpy()
y = y.to_numpy()

# Class labelse3


print("Training data:")
print(X)
print("Labels:")
print(y)
print(X.shape)
print(y.shape)

    



# Number of training examples
n = len(X)
#m = len(y)

# Soft-margin hyperparameter C
C_val = 1.0

# Compute the kernel matrix K (linear kernel: K = X @ X.T)
#EDIT THIS FOR KERNEL TRICK?
K = X @ X.T

# P matrix for quadratic form: P_ij = y_i * y_j * K_ij
Y_matrix = np.outer(y, y)
P = cp.psd_wrap(Y_matrix * K)

# Define dual variable alpha
alpha = cp.Variable(n)

# Objective: minimize (1/2) * alpha^T P alpha - sum(alpha)
objective = cp.Minimize(0.5 * cp.quad_form(alpha, P) - cp.sum(alpha))

# Constraints: 0 <= alpha <= C, and sum(alpha * Y) == 0
constraints = [
    alpha >= 0,
    alpha <= C_val,
    cp.sum(cp.multiply(alpha, y)) == 0
]

# Solve the problem
problem = cp.Problem(objective, constraints)
print("Finding alpha values...")
problem.solve()

# Retrieve optimal dual variables
alpha_opt = alpha.value
print("Optimal Alpha Values")
print(alpha_opt)

print("y")
print(y)
print("Alpha * y")
print(alpha.value.shape)
print(y.shape)
print(X.shape)

w = X.T @ (alpha.value * y)

print(w.shape)



print("Weight vector:")
print(w)

# Identify support vectors
support_vector_indices = np.where(
    alpha.value > 1e-5
)[0]

# Calculate bias
b_values = (
    y[support_vector_indices]
    - X[support_vector_indices] @ w
)

b = np.mean(b_values)

print("Bias:")
print(b)

print("Support vector indices in train dataset:")
print(support_vector_indices)

def predict(X_new):
    scores = X_new @ w + b
    return np.sign(scores)


print("Now predicting Test Dataset")

#Test values are the testing data
test_values = pd.read_csv("studentplacement/test_modified.csv")

#Test signs are the labels binary clasifiers -1, 1 of the data
test_signs = pd.read_csv("studentplacement/test_processed.csv")["label"]
predictions = predict(test_values)



print("Predicted labels:")
print(predictions)

print("Shape of test labels, shape of predictions")
print(test_signs.shape, predictions.shape)

#making sure they are 1's and -1s
test_signs = np.sign(test_signs)
predictions = np.sign(predictions)

correct_predictions = np.sum(test_signs == predictions)
print("Correct predictions:")
print(correct_predictions)

print("Accuracy:")
print(correct_predictions/predictions.size)

print("Use the larger training datasets or change kernel for better predictions")
