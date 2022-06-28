import bai2009
import numpy as np
import pandas as pd


# Read example data (N=100, T=50)
df = pd.read_csv('./example_data.csv')

# Load the  Interactive Fixed Effects estimator with r=3 factors
ife = bai2009.InteractiveFixedEffects(3)

# Estimate the model using a Patsy formula
betas, F, Lambda = ife.fit_from_formula('Y~0+X1+X2~ife(I,T)', df)

print(betas.T)
print(F.shape)
print(Lambda.shape)

# Alternatively, estimate the model specifying every term explicitly
betas, F, Lambda = ife.fit(
                        df['Y'].values[:,np.newaxis], # Outcome
                        df[['X1', 'X2']].values, # Observable regressors
                        df['I'].values[:,np.newaxis], # First level of the factor model (I)
                        df['T'].values[:,np.newaxis] # Second level of the factor model (T)
                    )
print(betas.T)
print(F.shape)
print(Lambda.shape)
