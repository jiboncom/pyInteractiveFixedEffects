# Panel Data Models with Interactive Fixed Effects
*Python implementation of the **Interactive Fixed Effects Estimator** for panel data presented in [Bai (2009)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA6135).*

## Installation
```pip install pyInteractiveFixedEffects```

## Usage
First, we need to import the module into our script. 
```python
import bai2009
```

Then we need the initiate the estimator by specifying the number of factors in the model. 
```python
# Load the  Interactive Fixed Effects estimator with r=3 factors
ife = bai2009.InteractiveFixedEffects(3)
```
Finally, there are two ways to estimate a model with interactive fixed effects (ife).

### Estimation from a Patsy formula
The easiest way to get an estimate is using a [Patsy](https://github.com/pydata/patsy) formula to specify the model. The regression is specified as normal and we add an additional term `~ife(I,T)`at the end to specify the columns of the data with the $N$ and $T$ index of each observation. 
```python
# Estimate the model using a Patsy formula
betas, F, Lambda = ife.fit_from_formula('Y~0+X1+X2~ife(I,T)', df)
```

### Estimation from explicit definition of terms
If you prefer to specify each term explicitly in your code, you can use the code below.

```python
# Alternatively, estimate the model specifying every term explicitly
betas, F, Lambda = ife.fit(
                        df['Y'].values[:,np.newaxis], # Outcome
                        df[['X1', 'X2']].values, # Observable regressors
                        df['I'].values[:,np.newaxis], # First level of the factor model (I)
                        df['T'].values[:,np.newaxis] # Second level of the factor model (T)
                    )
```
## Estimation results
The estimator returns:
* $\beta$ a $p\times 1$ vector of coefficients associated with the observables.
* $F$ a $T\times r$ matrix of the factors on the $T$ dimension.
* $\Lambda$ a $N\times r$ matrix of the loadings on the $N$ dimension.