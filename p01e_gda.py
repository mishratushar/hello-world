import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path, k = 0):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval,y_eval = util.load_dataset(eval_path,add_intercept = True)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    clf = GDA()
    theta = clf.fit(x_train,y_train)
    p = clf.predict(x_eval)
    if(k==0):
        np.savetxt(pred_path,p,delimiter = ',')
        util.plot(x_eval, y_eval, theta, '{}.png'.format(pred_path))
    if(k==1):
        return theta,p
    
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        phi = np.mean(y)
        m = y.shape[0]
        
        ind_0 = y == 0
        mu_0 = np.mean(x[ind_0],axis = 0)
        
        ind_1 = y == 1
        mu_1 = np.mean(x[ind_1],axis = 0)
        
        
        xa = x.copy()
        xa[ind_0] -= mu_0
        xa[ind_1] -= mu_1
        sigma = (xa.T.dot(xa))/m
        
        sig_inv = np.linalg.pinv(sigma)
        
        theta0 = 0.5*((mu_0.T).dot(sig_inv).dot(mu_0) - (mu_1.T).dot(sig_inv).dot(mu_1) - np.log((1-phi)/phi))
        theta0 = float(theta0)
        
        theta = ((mu_1 - mu_0).T).dot(sig_inv)
        theta = theta.reshape(2,)
        theta = list(theta)
        theta.append(theta0)
        theta = np.array(theta)

        self.theta = np.array([theta[2],theta[0],theta[1]])
        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        p = 1/(1+np.exp(-(x.dot(self.theta))))
        ind = p>=0.5
        p[ind] = 1
        index = p<0.5
        p[index] = 0
        return p

        # *** END CODE HERE