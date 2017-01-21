#' # An Outline of Generalized Linear Models
#' 
#' ### 2.1 Processes in Model Fitting
#' 
#' Three steps in model fitting:
#' 
#' 1. Model Selection 
#' 2. Parameter Estimation 
#' 3. Prediction of Future Values
#' 
#' #### Model Selection
#' 
#' Generalized Linear Models (GLM) assume independent, or at least uncorrelated,
#' observations. Can also be independent within blocks of fixed or known size.
#' 
#' Building a model means we want it to be relevant to the data under study.
#' Picking a scale for $Y$ should be relavent to the data and also should, if
#' possible, provide 
#' 
#' * Constancy of Variance
#' * Approximate Normality of Errors
#' * Additivity of Systematic Effects
#' 
#' There sometimes *isn't* a scale that provides all three. 
#' 
#' In GLMs we can relax the Normality and Variance assumptions as long as we
#' know how the variance depends on the mean. 
#' 
#' #### Parameter Estimation
#' 
#' Use a goodness of fit measure to find the parameter values that maximize it.
#' Usually this in terms of some liklihood function. Sometimes the *scaled 
#' deviance* is used instead which measures how close the liklihood comes to the
#' maximum possible liklihood for an exact fit (you are still maximizing the
#' liklihood function). 
#' 
#' \[
#' D^*(y; u) = 2l(y;y) - 2l(u;y)
#' \]
#' 
#' #### Prediction
#' 
#' Statements about the likely values of unobserved events. *Calibration* is 
#' sometimes used when when the response is fixed and we are required to make
#' statements about the likely values of $x$. 
#' 
#' Predictions also need to include measures of precision. 
#' 
#' ### 2.2 Components of a GLM
#' 
#' There are common components to all GLMs
#' 
#' 1. Random Component
#'   + Any exponential family of distributions works here
#' 2. Systematic Component (terms add up)
#' 3. The Link between the two previous components
#'   + any monotonic differentiable function
#' 
#' The random component we assume independence and constant variance of errors.
#' These are very important.
#' 
#' #### Liklihood Functions for GLMs
#' 
#' The exponential family has the following form:
#' 
#' \[
#' f_{Y}(y; \theta, \phi) = \exp\{(y * \theta - b(\theta) )  /  (a(\phi) + c(y, \phi))\}
#' \]
#' 
#' for specific functions of $a(.)$, $b(.)$, and $c(.)$. 
#' 
#' To give an example for the normal distribution:
#' 
#' \[
#' f_{Y}(y; \theta, \phi) = \frac{1}{\sqrt{2\pi \sigma ^{2}}}\exp\{-(y - \mu)^{2}/2\sigma ^{2}\}
#' \]
#' \[
#' = \exp\{(y\mu - \mu ^{2}/2) / \sigma ^2 - \frac{1}{2}(y^{2}/\sigma ^{2} + log(2\pi\sigma ^{2}))\}
#' \]
#' 
#' So $\theta = \mu$, $\phi = \sigma ^2$, $a(\phi) = \phi$, $b(\theta) = \theta ^{2}/2$, and $c(y, \phi) = -\frac{1}{2}(y^{2}/\sigma ^{2} + log(2\pi\sigma ^{2}))\}$
#' 
#' We can find $\theta$ and $\phi$ from finding theta that maximizes the 
#' liklihood function by taking partial derivatives and setting equal to 0. 
#' 
#' \[
#' E(Y) = b'(\theta)
#' \]
#' 
#' (which for normal distributions is just $\mu$) and
#' 
#' \[
#' var(Y) = b'\'(\theta)a(\phi)
#' \]
#' 
#' The variance consists of two functions:
#' 
#' * $b'\'(\theta)$ only depends on the mean, usually called the *variance 
#' function* ($V(\mu)$). 
#' * $a(\phi)$ is usually of the form $\phi / w$. $\phi$
#' is called the dispersion parameter and is constant over observations. $w$ is
#' a prior weight that varies between observations.
#' 
#' #### Link Functions
#' 
#' The link function relates the linear predictors ($\eta$) to the expected
#' value of the response variable ($\mu$). The link function must map all
#' possible values of $\eta$ to the domain of the response variable.
#' 
#' There are famous canonical links for different distributions.
#' 
#' * Normal:           $\eta = \mu$
#' * Poisson:          $\eta = \log{\mu}$
#' * Binomial:         $\eta = \log{\pi / (1 - \pi)}$
#' * Gamma:            $\eta = \mu^{-1}$
#' * Inverse Guassian: $\eta = \mu^{-2}$
#' 
#' ### 2.3 Measuring The Goodness of Fit
#' 
#' The *null model* assigns all variation to the random component of the model 
#' and hence predicts the same value for all $y$s. The *full model* has a 
#' parameter for each observation and assigns all the random variation to $y$ 
#' and none to the random component. This obviously does not generalize well to 
#' new data.
#' 
#' The discrepency of a fit is sometimes measured as twice the difference
#' between the maximum log liklihood achievable and the model under
#' investigation, which is just the scaled deviance.
#' 
#' ### 2.4 Residuals
#' 
#' residuals are whats left over after a prediction; datum = fitted value +
#' residual.
#' 
#' #### Pearson Residual
#' 
#' The normal residual scaled by the estimated standard deviation of $Y$.\:
#' 
#' \[ r_p = \frac{y - \mu}{\sqrt{V(\mu)}} \]
#' 
#' This can be skewed for non-normal distributions. You can use the deviance 
#' residuals for those
#' 
#' ### 2.5 An Algorithm for Fitting Generalized Linear Models
#' 
#' The coefficients $\beta$ can be fit using iteratively weighted least squares.
#' The dependent variable is now $z$, the linearized form of the link function
#' that has been applied to $Y$. 
#' 
#' 1. Start with an initial estimate of $\eta$ and $\mu$.
#' 2. find
#' \[
#' z_0 = \hat{\eta}_0 + (y - \hat{\mu}_0)\left(\frac{d\eta}{d\mu}\right)_0
#' \]
#' by evaluating the derivative at $\hat{\mu}_0$
#' 3. The weights for each iteration are found by
#' \[
#' W_{0}^{-1} = \left(\frac{d\eta}{d\mu}\right)_{0}^{2} V_0
#' \]
#' . $V_0$ is the variance function evaluated at $\hat{\mu}_0$. 
#' 4. Regress $z_0$ on the variables with the weights to find $\beta_1$.
#' 5. Repeat until changes converge. 
#' 
#' ### R code for algorithms
#' 
#' #### Iteratively Reweighted Least Squares
#' 
#+

#' 
#' 
