# Logistic Regression Classifier with Gradient Descent

This project implements a **binary classification model** using **logistic regression** with **gradient descent**, applied to the Iris dataset. It includes a custom-built classifier and visualizes decision boundaries.

## 🔍 Overview

Logistic regression is a widely used classification algorithm that models the probability that a given input belongs to a particular category.

### Mathematical Background

- **Sigmoid Function**:  
  Maps any real-valued number into the range (0, 1):  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

- **Log-Odds (Logit)**:  
  Converts probabilities into real-valued scores:  
  \[
  \text{logit}(p) = \log\left(\frac{p}{1 - p}\right)
  \]

- **Cost Function (Binary Cross-Entropy)**:  
  The loss function used in logistic regression:  
  \[
  J(w) = -\sum_{i} [y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)}))]
  \]

- **Optimization**:  
  We use gradient descent to minimize the cost function and find the optimal weights.

## 📦 Features

- Custom implementation of `LogisticRegressionGD` class
- Gradient descent for weight optimization
- Visualization of the sigmoid and cost functions
- Plotting decision regions
- Evaluation using `classification_report` from `sklearn`

## 📁 Project Structure

- `logisticregressionGD`: Class implementing logistic regression
- `sigmoid`: Function to compute the sigmoid activation
- `plot_decision_regions`: Helper function to visualize decision boundaries
- Model training and evaluation using the Iris dataset
