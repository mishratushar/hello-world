import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test,y_test = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train,y_train)
    y = clf.predict(x_test)
    # Get MSE value on the validation set
    MSE = np.mean(np.linalg.norm(y-y_test)**2)
    
    plt.plot(x_train,y_train,'bx')
    plt.plot(x_test,y,'ro')
    
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        W = np.ones_like(x)
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        dist = np.linalg.norm((self.x[None]-x[:,None]),axis = 2)
        w = np.exp(-(dist**2)/(2*self.tau**2))
        y_pred = np.zeros(m)
        for i, W in enumerate(w):
            W = np.diag(W)
            theta = np.linalg.pinv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)
            # make prediction
            y_pred[i] = x[i].dot(theta)
        return y_pred
        # *** END CODE HERE ***
