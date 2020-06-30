#!/usr/bin/env python
__all__ = ['SearchFair']

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
import sklearn.metrics.pairwise as kernels
from sklearn.metrics import confusion_matrix
import numpy as np
import cvxpy as cp
import random


class SearchFair(BaseEstimator):
    """SearchFair

    Parameters
    ----------
    fairness_notions: string
        The name of the fairness notion that the classifier should respect. 'DDP' or 'DEO' can be used.
    fairness_regularizer: string
        The name of the fairness relaxation that is used as a regularizer. It can be 'linear', or 'wu'. For 'wu', the 'wu_bound' can be chosen.
    wu_bound: string
        The name of the function that is used in the bounds of Wu et al. It can be 'hinge', 'logistic', 'squared', 'exponential'
    reg_beta: float
        Regularization parameter Beta for the l2 regularization.
    kernel: string
        The kind of kernel that is used. It can be 'linear', 'rbf' or 'poly'. For 'rbf' and 'poly', the parameter gamma can be used.
    gamma: float
        For kernel='rbf', gamma is the kernel width, for kernel='poly', gamma is the degree.
    loss_name: string
        The name of the loss used. Possible values: 'hinge', 'logistic', 'squared', 'exponential'
    lambda_max: float
        The value of lambda_max for the start of the binary search.
    max_iter: int
        The number of iterations of the solver chosen.
    reason_points: float
        The ratio of points used as reasonable points for the similarity-based approach of SearchFair.
    stop_criterion: float
        If SearchFair finds a classifier that is at least as fair as 'stop_criterion', than it stops the search.
    max_search_iter: int
        The number of iterations for the binary search.
    solver: string
        The solver that is used by cvxpy. It can be 'SCS' or 'ECOS'.
    verbose: boolean

    Attributes
    ----------
    coef_: numpy array
        An array containing the trained weights for each reasonable point.
    reason_pts_index: numpy array
        An array containing the indices of the reasonable points in the training data.

    Notes
    ----------

    """

    def __init__(self, fairness_notion='DDP', fairness_regularizer='wu', wu_bound='hinge', reg_beta=0.001, kernel='linear', gamma=None, loss_name='hinge', lambda_max=1, max_iter=3000, reason_points=0.5, stop_criterion=0.01, max_search_iter=10, solver='SCS', verbose=False):

        self.reg_beta = reg_beta
        self.fairness_notion = fairness_notion
        self.max_iter = max_iter
        self.max_search_iter = max_search_iter
        self.solver = solver
        self.verbose = verbose
        self.stop_criterion = stop_criterion
        self.reason_points = reason_points
        self.lambda_max = lambda_max
        self.wu_bound = wu_bound
        self.fairness_regularizer = fairness_regularizer
        self.wu_bound = wu_bound
        self.gamma = gamma
        self.loss_name = loss_name
        self.kernel = kernel

    def fit(self, x_train, y_train, s_train=None):
        """Fits SearchFair on the given training data.

        Parameters
        ----------
        x_train: numpy array
            The features of the training data with shape=(number_points,number_features).
        y_train: numpy array
            The class labels of the training data with shape=(number_points,).
        s_train: numpy array
            The binary sensitive attributes of the training data with shape=(number_points,).

        Returns
        ----------
        self: object
        """

        self.x_train = x_train
        self.y_train = y_train
        self.s_train = s_train

        if self.verbose:
            print("Preprocessing...")
        self._preprocess()

        lbda_min, lbda_max = 0, self.lambda_max

        def learn(reg, bound='upper'):
            # If bound is None, we have decided which one to use, and we are in the middle of the binary search

            self.fairness_lambda = reg
            if bound is not None:
                self._construct_problem(bound=bound)
            self._optimize()
            DDP, DEO = self.compute_fairness_measures(self.predict(x_train), y_train, s_train)
            if self.fairness_notion == 'DDP':
                fair_value = DDP
            else:
                fair_value = DEO
            if self.verbose: print("Obtained:",self.fairness_notion, "= %0.4f with lambda = %0.4f" % (fair_value, reg))
            return fair_value, self.coef_.copy()

        criterion = False

        bound = 'upper' # even though an upper bound is specified, since lambda_min is 0, it falls away
        if self.verbose: print("Testing lambda_min: %0.2f" % lbda_min)
        min_fair_measure, min_alpha = learn(lbda_min, bound=bound)
        if np.sign(min_fair_measure) < 0: bound = 'lower'
        if self.verbose: print("Testing lambda_max: %0.2f" % lbda_max)
        max_fair_measure, max_alpha = learn(lbda_max, bound)

        if np.abs(min_fair_measure) < np.abs(max_fair_measure):
            best_lbda, best_fair_measure = lbda_min, min_fair_measure
            best_alpha = min_alpha
        else:
            best_lbda, best_fair_measure = lbda_max, max_fair_measure
            best_alpha = max_alpha
        if  np.abs(best_fair_measure) < self.stop_criterion:
            print("Classifier is fair enough with lambda = {:.4f}".format(best_lbda))
        elif np.sign(min_fair_measure) == np.sign(max_fair_measure):
            print('Fairness value has the same sign for lambda_min and lambda_max.')
            print('Either try a different fairness regularizer or change the values of lambda_min and lambda_max') # Possibly, there could be a few more tries by reducing lambda.
        else:
            search_iter = 0
            if self.verbose: print("Starting Binary Search...")
            while not criterion and search_iter < self.max_search_iter:
                lbda_new = (lbda_min + lbda_max) / 2

                if self.verbose:
                    print(10*'-'+"Iteration #%0.0f" % search_iter + 10*'-')
                    print("Testing new Lambda: %0.4f" % lbda_new)

                new_rd, new_alpha = learn(lbda_new, None)
                if np.abs(new_rd) < np.abs(best_fair_measure):
                    best_fair_measure = new_rd
                    best_lbda = lbda_new
                    best_alpha = new_alpha.copy()

                if np.sign(new_rd) == np.sign(min_fair_measure):
                    min_fair_measure = new_rd
                    lbda_min = lbda_new
                else:
                    max_fair_measure = new_rd
                    lbda_max = lbda_new
                if np.abs(new_rd) < self.stop_criterion:
                    criterion = True

                search_iter += 1
            if search_iter==self.max_search_iter and self.verbose:
                print("Hit maximum iterations of Binary Search.")
            elif self.verbose:
                print("Sufficient fairness obtained before maximum iterations were reached.")

        if self.verbose: print(10*'-'+"Found Lambda %0.4f with fairness %0.4f" % (best_lbda, best_fair_measure)+10*'-')
        self.coef_ = best_alpha.copy()

        return self

    def predict(self, x_test):
        """Predict the label of test data.

        Parameters
        ----------
        x_test: numpy array
            The features of the test data with shape=(number_points,number_features).

        Returns
        ----------
        y_hat: numpy array
            The predicted class labels with shape=(number_points,).
        """
        kernel_matr = self.kernel_function(x_test, self.x_train[self.reason_pts_index])
        y_hat = np.dot(self.coef_, np.transpose(kernel_matr))
        return np.sign(y_hat)

    def _preprocess(self):
        """Setting the attributes loss_func, kernel_function, and weight_vector,
        which depends on the fairness notion, and is used in fairness related objects.
        """
        self.coef_ = None
        self.fairness_lambda = 0
        if self.loss_name == 'logistic':
            self.loss_func = lambda z: cp.logistic(-z)
        elif self.loss_name == 'hinge':
            self.loss_func = lambda z: cp.pos(1.0 - z)
        elif self.loss_name == 'squared':
            self.loss_func = lambda z: cp.square(-z)
        elif self.loss_name == 'exponential':
            self.loss_func = lambda z: cp.exp(-z)
        else:
            print('Using default loss: hinge loss.')
            self.loss_func = lambda z: cp.pos(1.0 - z)

        if self.kernel == 'rbf':
            self.kernel_function = lambda X, Y: kernels.rbf_kernel(X, Y, self.gamma)
        elif self.kernel == 'poly':
            self.kernel_function = lambda X, Y: kernels.polynomial_kernel(X, Y, degree=self.gamma)
        elif self.kernel == 'linear':
            self.kernel_function = lambda X, Y: kernels.linear_kernel(X, Y) + 1
        else:
            self.kernel_function = kernel

        if self.wu_bound == 'logistic':
            self.cvx_kappa = lambda z: cp.logistic(z)
            self.cvx_delta = lambda z: 1 - cp.logistic(-z)
        elif self.wu_bound == 'hinge':
            self.cvx_kappa = lambda z: cp.pos(1 + z)
            self.cvx_delta = lambda z: 1 - cp.pos(1 - z)
        elif self.wu_bound == 'squared':
            self.cvx_kappa = lambda z: cp.square(1 + z)
            self.cvx_delta = lambda z: 1 - cp.square(1 - z)
        elif self.wu_bound == 'exponential':
            self.cvx_kappa = lambda z: cp.exp(z)
            self.cvx_delta = lambda z: 1 - cp.exp(-z)
        else:
            print('Using default bound with hinge.')
            self.cvx_kappa = lambda z: cp.pos(1 + z)
            self.cvx_delta = lambda z: 1 - cp.pos(1 - z)

        self.nmb_pts = len(self.s_train)
        self.nmb_unprotected = np.sum(self.s_train == 1)
        self.prob_unprot = self.nmb_unprotected / self.nmb_pts
        self.prob_prot = 1 - self.prob_unprot

        self.nmb_pos = np.sum(self.y_train == 1)
        self.nmb_prot_pos = np.sum(self.y_train[self.s_train == -1] == 1)
        self.prob_prot_pos = self.nmb_prot_pos / self.nmb_pos
        self.prob_unprot_pos = 1 - self.prob_prot_pos

        # Create weights that are necessary for the fairness constraint
        if self.fairness_notion == 'DDP':
            normalizer = self.nmb_pts
            self.weight_vector = np.array(
                [1.0 / self.prob_prot if self.s_train[i] == -1 else 1.0 / self.prob_unprot for i in range(len(self.s_train))]).reshape(-1,1)
            self.weight_vector = (1 / normalizer) * self.weight_vector
        elif self.fairness_notion == 'DEO':
            normalizer = self.nmb_pos
            self.weight_vector = np.array(
                [1.0 / self.prob_prot_pos if self.s_train[i] == -1 else 1.0 / self.prob_unprot_pos for i in range(len(self.s_train))]).reshape(-1, 1)
            self.weight_vector = 0.5 * (self.y_train.reshape(-1, 1) + 1) * self.weight_vector
            self.weight_vector = (1 / normalizer) * self.weight_vector

        # Choose random reasonable points
        if self.reason_points <= 1:
            self.reason_pts_index = list(range(int(self.nmb_pts * self.reason_points)))
        else:
            self.reason_pts_index = list(range(self.reason_points))
        self.nmb_reason_pts = len(self.reason_pts_index)

    def _construct_problem(self, bound='upper'):
        """ Construct the cvxpy minimization problem.
        It depends on the fairness regularizer chosen.
        """

        # Variable to optimize
        self.alpha_var = cp.Variable((len(self.reason_pts_index), 1))
        # Parameter for Kernel Matrix
        self.kernel_matrix = cp.Parameter(shape=(self.x_train.shape[0], len(self.reason_pts_index)))
        self.fair_reg_cparam = cp.Parameter(nonneg=True)


        # Form SVM with L2 regularization
        if self.fairness_lambda == 0:
            self.loss = cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var))) + self.reg_beta * self.nmb_pts * cp.square(
                cp.norm(self.alpha_var, 2))
        else:
            sy_hat = cp.multiply(self.s_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var)

            if self.fairness_regularizer == 'wu':
                if bound == 'upper':
                    fairness_relaxation = cp.sum(cp.multiply(self.weight_vector, self.cvx_kappa(sy_hat))) - 1
                else:
                    fairness_relaxation = -1 * cp.sum(cp.multiply(self.weight_vector, self.cvx_delta(sy_hat))) - 1


            elif self.fairness_regularizer == 'linear':
                if bound == 'upper':
                    fairness_relaxation = cp.sum(cp.multiply(self.weight_vector, self.kernel_matrix @ self.alpha_var))
                else:
                    fairness_relaxation = -1 * cp.sum(cp.multiply(self.weight_vector, self.kernel_matrix @ self.alpha_var))

            if self.reg_beta == 0:
                self.loss = (1/self.nmb_pts) * cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var))) + \
                                self.fair_reg_cparam * fairness_relaxation
            else:
                self.loss = (1 / self.nmb_pts) * cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var))) + \
                            self.fair_reg_cparam * fairness_relaxation + self.reg_beta * cp.square(cp.norm(self.alpha_var, 2))

        self.prob = cp.Problem(cp.Minimize(self.loss))

    def _optimize(self):
        """Conduct the optimization of the created problem by using ECOS or SCS
        with cvxpy. 
        """

        # Compute and initialize kernel matrix
        self.K_sim = self.kernel_function(self.x_train, self.x_train[self.reason_pts_index])
        self.kernel_matrix.value = self.K_sim
        self.fair_reg_cparam.value = self.fairness_lambda

        if self.verbose == 2:
            verbose = True
        else:
            verbose = False
        if self.solver == 'SCS':
            self.prob.solve(solver=cp.SCS, max_iters=self.max_iter, verbose=verbose, warm_start=True)
        elif self.solver == 'ECOS':
            try:
                self.prob.solve(solver=cp.ECOS, max_iters=self.max_iter, verbose=verbose, warm_start=True)
            except Exception as e:
                self.prob.solve(solver=cp.SCS, max_iters=self.max_iter, verbose=verbose, warm_start=True)
        if verbose:
            print('status %s ' % self.prob.status)
            print('value %s ' % self.prob.value)
        self.coef_ = self.alpha_var.value.squeeze()

    def compute_fairness_measures(self, y_predicted, y_true, sens_attr):
        """Compute value of demographic parity and equality of opportunity for given predictions.

        Parameters
        ----------
        y_predicted: numpy array
            The predicted class labels of shape=(number_points,).
        y_true: numpy array
            The true class labels of shape=(number_points,).
        sens_attr: numpy array
            The sensitive labels of shape=(number_points,).

        Returns
        ----------
        DDP: float
            The difference of demographic parity.
        DEO: float
            The difference of equality of opportunity.
        """
        positive_rate_prot = self.get_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
        positive_rate_unprot = self.get_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
        true_positive_rate_prot = self.get_true_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
        true_positive_rate_unprot = self.get_true_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
        DDP = positive_rate_unprot - positive_rate_prot
        DEO = true_positive_rate_unprot - true_positive_rate_prot

        return DDP, DEO

    def get_positive_rate(self, y_predicted, y_true):
        """Compute the positive rate for given predictions of the class label.

        Parameters
        ----------
        y_predicted: numpy array
            The predicted class labels of shape=(number_points,).
        y_true: numpy array
            The true class labels of shape=(number_points,).

        Returns
        ---------
        pr: float
            The positive rate.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
        pr = (tp+fp) / (tp+fp+tn+fn)
        return pr

    def get_true_positive_rate(self, y_predicted, y_true):
        """Compute the true positive rate for given predictions of the class label.

        Parameters
        ----------
        y_predicted: numpy array
            The predicted class labels of shape=(number_points,).
        y_true: numpy array
            The true class labels of shape=(number_points,).

        Returns
        ---------
        tpr: float
            The true positive rate.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
        tpr = tp / (tp+fn)
        return tpr
