import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path, k = 0):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    clf = LogisticRegression()
    theta = clf.fit(x_train,y_train)
    p = clf.predict(x_eval)
    if(k==0):
        np.savetxt(pred_path,p,delimiter = ',')
        sp = 'output/p01b_plot'
        util.plot(x_eval,y_eval,theta,sp)
    elif(k==1):
        ind = p < 0.5
        p[ind] = 0
        index = p >= 0.5
        p[index] = 1
        return theta,p
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        if(self.theta == None):
            self.theta=np.zeros(x.shape[1])
        theta = np.ones_like(self.theta)
        k = 0
        while(np.linalg.norm(self.theta - theta) > self.eps):

            T = x.dot(self.theta)
            h = 1/(1+np.exp(-T))
            g = y - h
            
            gradJ = np.zeros(x.shape[1])
            gradJ = (-1/x.shape[1])*(g.dot(x))
            
            H = np.zeros((x.shape[1],x.shape[1]))
            H = (1 / x.shape[1]) * h.dot(1-h) * (x.T).dot(x)
             
            H_inv = np.linalg.pinv(H)
            u = H_inv.dot(gradJ)
            theta = self.theta.copy()
            self.theta = theta - u
            k = k+1
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
        
        T = x.dot(self.theta)
        P = 1/(1+np.exp(-T))
        return P
        
        # *** END CODE HERE ***
