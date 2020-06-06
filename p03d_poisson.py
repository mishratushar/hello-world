import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path,add_intercept = False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression()
    clf.fit(x_train,y_train,lr)
    h = clf.predict(x_eval)
    np.savetxt(pred_path,h,delimiter = ',')
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y,lr):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        i = 0
        m = x.shape[0]
        if (self.theta == None):
            self.theta = np.zeros(x.shape[1])
        while(True):
            h = np.exp(x.dot(self.theta))
            update = (1/m)*((y - h).dot(x))
            theta = self.theta
            self.theta = self.theta + lr*update      
            i += 1
            if(np.linalg.norm(self.theta - theta)<self.eps):
                break
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        h = np.exp(x.dot(self.theta))
        return h
        # *** END CODE HERE ***
