"""
Iris Classification using K-Nearest Neighbors (KNN)

This script demonstrates a complete basic Machine Learning pipeline:
1. Load dataset
2. Explore dataset
3. Split into training and testing sets
4. Visualize data
5. Train KNN model
6. Make predictions
7. Evaluate performance
"""

# -----------------------------
# Step 1: Import required libraries
# -----------------------------

import numpy as np
import pandas as pd
import mglearn  # Used only for visualization (educational purpose)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix


# -----------------------------
# Step 2: Load the Iris dataset
# -----------------------------

# The Iris dataset is a classic classification dataset
# It contains 150 samples of iris flowers with 4 features each
iris_dataset = load_iris()


# -----------------------------
# Step 3: Explore dataset structure
# -----------------------------

print("Keys of iris_dataset:\n", iris_dataset.keys())

# Description of the dataset
print("\nDataset description:\n")
print(iris_dataset['DESCR'][:200] + "...\n")

# Target classes (labels)
print("Target names:", iris_dataset['target_names'])

# Feature names
print("\nFeature names:")
print(iris_dataset['feature_names'])

# Data type and shape
print("\nType of data:", type(iris_dataset['data']))
print("Shape of data:", iris_dataset['data'].shape)

# First five samples
print("\nFirst five rows of data:")
print(iris_dataset['data'][:5])

# Target information
print("\nTarget type:", type(iris_dataset['target']))
print("Target shape:", iris_dataset['target'].shape)
print("Targets:\n", iris_dataset['target'])


# -----------------------------
# Step 4: Split data into training and testing sets
# -----------------------------

# Training data is used to build the model
# Testing data is used to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    random_state=0
)

print("\nTraining data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)

print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)


# -----------------------------
# Step 5: Data visualization
# -----------------------------

# Convert training data into pandas DataFrame for visualization
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# Scatter plot matrix helps visualize relationships between features
scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(12, 12),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=0.8,
    cmap=mglearn.cm3
)


# -----------------------------
# Step 6: Create and train KNN model
# -----------------------------

# KNN is a simple algorithm that classifies a sample
# based on its nearest neighbors in the training set
knn = KNeighborsClassifier(n_neighbors=1)

# Train (fit) the model
knn.fit(X_train, y_train)


# -----------------------------
# Step 7: Predict a new sample
# -----------------------------

# Example new flower measurement
X_new = np.array([[5, 2.9, 1, 0.2]])

print("\nNew sample shape:", X_new.shape)

# Predict class
prediction = knn.predict(X_new)

print("Prediction (numeric):", prediction)
print("Predicted class:",
      iris_dataset['target_names'][prediction])


# -----------------------------
# Step 8: Evaluate model performance
# -----------------------------

# Predict on test set
y_pred = knn.predict(X_test)

print("\nTest set predictions:\n", y_pred)

# Manual accuracy calculation
accuracy_manual = np.mean(y_pred == y_test)
print("\nTest set accuracy (manual): {:.2f}".format(accuracy_manual))

# Built-in scoring method
accuracy_builtin = knn.score(X_test, y_test)
print("Test set accuracy (built-in): {:.2f}".format(accuracy_builtin))