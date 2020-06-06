import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path,k=0):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    clf = LogisticRegression()
    x_train,y_train = util.load_dataset(train_path,label_col = 't',add_intercept = True)
    theta = clf.fit(x_train,y_train)
    x_test,y_test = util.load_dataset(test_path, label_col = 't', add_intercept = True)
    p = clf.predict(x_test)
    
    if(k == 1):
        ind = p < 0.5
        p[ind] = 0
        index = p >= 0.5
        p[index] = 1
        return p
    
    if(k==0):
        np.savetxt(pred_path_c, p, delimiter = ',')
        sp = 'output/p02_plot'
        util.plot(x_test,y_test,theta,sp)
    # Make sure to save outputs to pred_path_c
    
    
    # Part (d): Train on y-labels and test on true labels
    clf.theta = None
    x_train,y_train = util.load_dataset(train_path,label_col = 'y',add_intercept = True)
    theta = clf.fit(x_train,y_train)
    x_test,y_test = util.load_dataset(test_path,label_col = 't',add_intercept = True)
    p = clf.predict(x_test)
    
    if(k == 2):
        ind = p < 0.5
        p[ind] = 0
        index = p >= 0.5
        p[index] = 1
        return p
    
    if(k==0):
        np.savetxt(pred_path_d,p,delimiter = ',')
        sp = 'output/p02d_plot'
        util.plot(x_test,y_test,theta,sp)
    # Make sure to save outputs to pred_path_d
    
    
    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid,y_valid = util.load_dataset(valid_path,label_col = 'y',add_intercept = True)
    a = y_valid == 1
    p1 = clf.predict(x_valid)
#     alpha = p1[y_valid == 1].sum() / (y_valid == 1).sum()
    alpha = np.sum(p1[a])/(np.sum(y_valid[a]))
    # print(alpha)
    
    correction = 1 + (np.log(2 / alpha - 1) / clf.theta[0])
    
    x_test,y_test = util.load_dataset(test_path,label_col = 't',add_intercept = True)
    P = clf.predict(x_test)
    P = P/alpha
    if(k==3):
        ind = P < 0.5
        P[ind] = 0
        index = P >= 0.5
        P[index] = 1
        return P
    if(k==0):
        np.savetxt(pred_path_e,p,delimiter = ',')
        sp = 'output/p02e_plot'
        util.plot(x_test,y_test,theta,sp,correction = correction)

    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE



