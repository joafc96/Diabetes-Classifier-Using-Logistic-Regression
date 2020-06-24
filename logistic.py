"""
logistic.py
~~~~~~~~~~

Module is for building a classic Logistic Regression model from scratch using only python and numpy.
The data-set used is pima-indians-diabetes about diabetes risk from kaggle.
link: "https://www.kaggle.com/uciml/pima-indians-diabetes-database"
This module is a vectorized implementation of Logistic Regression with gradient descent (very fast :D )
Note:
  - This is one implementation that worked for me but might be far from optimum.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


class LogisticRegression:
    """Logistic Regression Class"""

    def __init__(self, n_features, alpha=0.0001, iterations=1_00_000):
        """
        :param n_features(int): No of features
        :param alpha_learning_rate(int): Learning rate for the gradient descent
        :param iterations(int): No of iterations to be performed
        """
        self.alpha = alpha
        self.iterations = iterations
        self.w = np.array([0 for _ in range(n_features)])
        self.b = 0

    def cost_gradient_derivative(self, x, y):
        """
        Method calculating the cost and the gradients

        :param x: in-dependant variables
        :param y: dependant variables
        :return: calculates the grads and cost
        """
        m = x.shape[0]

        # Prediction
        h_theta_of_x = self._sigmoid_activation(np.dot(self.w, x.T) + self.b)

        cost = (-1 / m) * (np.sum((y * np.log(h_theta_of_x)) + ((1 - y) * (np.log(1 - h_theta_of_x + np.finfo(np.float32).eps)))))

        # Gradient calculation
        dw, db = (1 / m) * (np.dot(x.T, (h_theta_of_x - y))), (1 / m) * (np.sum(h_theta_of_x - y))

        grads = {"dw": dw, "db": db}

        return grads, cost

    def model_train(self, train_predictors, train_targets, verbose=True):
        """
        Main function operating the Logistic Regression training

        Steps at each iteration:
        1- Calculation of gradient and cost
        2- Perform gradient descent (theta(weight) - alpha * derivative_of_theta)
        3- The cost is provided if needed

        :return: Performs sigmoid function on the new trained weights and intercept
        """
        costs = []
        for i in range(self.iterations):

            grads, cost = self.cost_gradient_derivative(train_predictors, train_targets)
            #
            dw = grads["dw"]
            db = grads["db"]
            #
            self.w = self.w - (self.alpha * dw)  # weight update
            #
            self.b = self.b - (self.alpha * db)  # intercept update
            #
            if i % 100 == 0 and verbose:
                costs.append(cost)
                print("Cost after %i iteration is %f" % (i, cost))

        return self.model_train_test(train_predictors, train_targets, train_test='Train')

    def model_train_test(self, predictors, targets, train_test):
        final_result_normalised = list(map(lambda x: 1 if x > 0.5 else 0,
                                           self._sigmoid_activation(np.dot(self.w,
                                                                           predictors.T) + self.b)))  # normalizes the \
        # final result after sigmoid activation to ones and zeros
        print(f'{train_test} accuracy is', accuracy_score(final_result_normalised, targets))

    @staticmethod
    def _sigmoid_activation(z):
        """
         The sigmoid function, classic activation function
        """
        return 1.0 / (1.0 + np.exp(-z))


if __name__ == '__main__':
    df = pd.read_csv('pima-indians-diabetes.csv')
    features = np.array(df.iloc[:, :-1])
    labels = np.array(df.iloc[:, -1])

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,
                                                        shuffle=True)

    lg = LogisticRegression(n_features=features.shape[1])
    lg.model_train(train_predictors=x_train, train_targets=y_train)  # trains the model
    lg.model_train_test(predictors=x_test, targets=y_test, train_test="Test")  # model is tested on new data
