#' # Models for Continous Data with Constant Variance
#' 
#' ### 3.1 Introduction
#' 
#' This is just classic linear regression:
#' 
#' \[
#' Y_i \sim N(\mu_i, \sigma ^2), \mu = \eta, \eta = \sum_{1}^{p}{x_j \beta_j}
#' \]
#' 
#' Again these are our Random, Link, and Systemic components
#' 
#' ### 3.2 Error Structure
#' 
#' The $Y$ observations are assumed to have equal variance and be independent.
#' Normality isn't necessary for large samples. Constant variance is important
#' enough to where you basically have to check for it, but we will cover that
#' later. 
#' 
#' If your predictions need to be restricted to positive values $\log{Y}$ is 
#' often used. 
#' 
#' ### 3.3 Systematic Component
#' 
#' #### Continuous Covariates
#' 
#' Can use transformations as long as those transformations are linear in their 
#' impact. 
#' 
#' #### Qualitative Covariates
#' 
#' Use dummy variables, you can't use all levels of a factor. 
#' 
#' ### 3.5 Aliasing
#' 
#' This deals with the possible ways to deal with including all levels of the 
#' factor in the model. We have three options:
#' 
#' 1. Make the intercept = 0 and include all levels
#' 2. Make the first level = 0 and use that as a reference
#' 3. Force the intercept = the group mean and the coefficients for all the 
#' levels are new the deviation from the group mean. This can also be weighted
#' to account for different group sizes. 
#' 
#' #### Functional Relations Among Covariates
#' 
#' Basically it doesn't make sense to include higher order terms without the 
#' lower ones (e.g. don't have $x$ and $x^3$ in a model without $x^2$). You are
#' just implicitly setting the coefficient of the lower terms = 0 for no reason.
#' 
#' ### 3.6 Estimation
#' 
#' #### The Maximum-liklihood Equations
#' 
#' For Normal errors the log liklihood of a model is:
#' 
#' \[
#' -2l = n \log{(2\pi\sigma ^2)} + \sum_{i=1}^{n}(y_i - \mu_i)^{2}/\sigma ^2
#' \]
#' 
#' When $\sigma ^2$ is fixed we are just minimizing the sum of square errors. 
#' We know 
#' 
#' \[
#' \mu_i = \sum_{j=1}^{p}x_{ij}\beta_j
#' \]
#' 
#' And if we differentiate the sum of square errors with respect to $\beta_j$ 
#' and set to 0 we get:
#' 
#' \[
#' \sum_{i}x_{ij}(y_i - \hat{\mu_i}) = 0 for j = 1,...,p
#' \]
#' 
#' Basically we are saying the linear combinations of $x$s and $Y$ is equal to
#' the linear combination of $x$s and the fitted values, $\mu$. Somehow this 
#' means that the vector of residuals, $y_i - \hat{\mu_i}$, is orthogonal to 
#' the columns of the model matrix $X$, which means:
#' 
#' \[
#' \mathbf{X}^T(\mathbf{y} - \mathbf{\hat{\mu}})
#' \]
#'
#' 
#' 