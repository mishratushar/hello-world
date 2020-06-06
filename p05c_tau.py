import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid,y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test,y_test =  util.load_dataset(test_path, add_intercept=True)
    # *** START CODE HERE ***
    MSE = np.zeros_like(tau_values)
    i = 0
    for tau in tau_values:
    # Search tau_values for the best tau (lowest MSE on the validation set)
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train,y_train)
        y = clf.predict(x_valid)
        MSE[i] = np.mean(np.linalg.norm(y-y_valid)**2)
        plt.figure()
        plt.plot(x_train,y_train,'bx')
        plt.plot(x_valid,y,'ro')
        plt.title(tau)
        i += 1
    tau = tau_values[np.argmin(MSE)]
    print(min(MSE))
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train,y_train)
    y1 = clf.predict(x_test)
    mse = np.mean(np.linalg.norm(y1-y_test)**2)
    print(mse)
    np.savetxt(pred_path,y1)
    plt.figure()
    plt.plot(x_train,y_train,'bx')
    plt.plot(x_test,y,'ro')
    
        
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
