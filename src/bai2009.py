# coding: utf-8
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import preprocessing
import patsy

class InteractiveFixedEffects():
    r"""An implementation of the Interactive Fixed Effects estimator by Bai (2009)."""
    
    n_factors = 0

    log_betas = []
    log_errors = []

    i_encoder = None 
    t_encoder = None

    def __init__(self, n_factors=3):
        self.n_factors = n_factors
    
    def reset_estimation(self):
        self.log_betas = []
        self.log_errors = []
        self.i_encoder = None 
        self.t_encoder = None

    def fit(self, Y, X, dim1, dim2, atol=10**-14):
        r""" Cleans and defines the data to be used and then calls for the estimation by passing the cleaned data to the `_fit` method.
            * Y: Outputs
            * X: Observable regressors
            * dim1: Indexes for the first dimension (i)
            * dim2: Indexes for the second dimension (t)    
        """
        # Reset previous estimation results
        self.reset_estimation()

        # Normalize factor dimensions labels into indexes
        self.i_encoder = preprocessing.LabelEncoder()
        self.t_encoder = preprocessing.LabelEncoder()
        self.i_encoder.fit(dim1)
        self.t_encoder.fit(dim2)

        _dim1_idx = self.i_encoder.transform(dim1).astype(int)[:,np.newaxis]
        _dim2_idx = self.t_encoder.transform(dim2).astype(int)[:,np.newaxis]

        return self._fit(Y, X, _dim1_idx, _dim2_idx, atol)

    def fit_from_formula(self, formula, data, atol=10**-14):
        """ Recovers the data for estimation from a Patsy formula and then 
            calls the `fit()` method to define the data and complete the estimation.
        """

        # Parse custom formula
        formulas = formula.split('~')
        if len(formulas) != 3:
            raise("Invalid formula. Format is Y~X~ife(I,T)")

        formula = formulas[0] + '~' + formulas[1]
        factors_formula = formulas[2] + '-1'

        def ife(dim1, dim2):
            output = np.hstack([
                        dim1.values[:, np.newaxis], 
                        dim2.values[:, np.newaxis]
                    ])
            return output
        
        Y, X = patsy.dmatrices(formula, data) 
        factors_idx = patsy.dmatrix(factors_formula, data)

        # Rename factor columns
        factors_idx.design_info = patsy.DesignInfo(['dim1', 'dim2'])

        # Call the estimation procedure
        return self.fit(Y, X, factors_idx[:,0], factors_idx[:,1], atol)

    def _fit(self, Y, X, dim1_idx, dim2_idx, atol=10**-14):
        r"""Computes the estimates for the Fixed Effects Estimator."""
        n_factors = self.n_factors

        # Initialize the PCA calculator
        PCA_calc = PCA(n_components=n_factors)

        # Initialize the betas with an initial (bad) guess
        new_betas = np.zeros((X.shape[1], 1))
        old_betas = np.ones((X.shape[1], 1))

        # Count the number of different items in each direction
        N = len(self.i_encoder.classes_)
        T = len(self.t_encoder.classes_)

        # Make an initial guess of Interactive parameters
        Lambda = np.random.random((N,n_factors))
        F = np.random.random((T,n_factors))

        # Compute the inverse of (X'X) to be shared across iterations
        XX_inv = np.linalg.inv(X.T @ X)

        # Logging variables for post-estimation analysis
        self.log_Fs = []
        self.log_lambdas = []
        self.log_betas.append(new_betas)

        #
        # Find the solution by iteration
        #

        ## Iterate until all the estimates for beta converge
        while not np.all(np.isclose(new_betas, old_betas, atol=atol)):
            # Step 0 - Assign the corresponding factors to each observation
            lambdas_t = Lambda[dim1_idx[:,0]]
            F_i = F[dim2_idx[:,0]]

            # Step 1 - Given F and Lambda get Beta
            old_betas = new_betas
            new_betas = XX_inv @ X.T @ (Y - np.multiply(F_i, lambdas_t).sum(axis=1)[:,np.newaxis])

            # Step 2 - Given Beta get F and Lambda using PCA of (Y_i-X_i*Beta)
            W = Y - X @ new_betas

            # This pivot table follows the role of W in the original paper
            pivot_table = np.zeros((T, N))
            pivot_table[dim2_idx[:,0], dim1_idx[:,0]] = W[:,0]

            # Our estimate of F are the Eigenvectors of WW' multiplied by sqrt(T) due to the normalization
            new_F = PCA_calc.fit( pivot_table @ pivot_table.T / (N) ).components_.T * np.sqrt(T)

            # Step 3 - Given the new F, compute Lambdas_t as the OLS estimator of W_t = Lambda_t * F_i 
            new_Lambda = (pivot_table.T @ new_F) / T

            # Store values for convergence analysis
            self.log_Fs.append(new_F)
            self.log_lambdas.append(new_Lambda)
            self.log_betas.append(new_betas)
            self.log_errors.append(Y - np.multiply(F_i, lambdas_t).sum(axis=1)[:,np.newaxis])

            # Update parameter values for the next iteration
            F = new_F
            Lambda = new_Lambda
        
        return new_betas, F, Lambda
