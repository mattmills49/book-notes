#' # Bayesian Data Analysis
#' # Chapter 2 Examples
#' 
#' ### Binomial Birth Model
#' 
#' *theta* = proportion of female births
#' *y* = number of female births
#' *n* = number of total births
#' 
#' *p(y | theta) = Bin(y | n, theta)*
#' * = (n y) * theta ^ y * (1 - theta) ^ (n - y)
#' 
#' First we assume a uniform prior distribution on theta and see how our 
#' posterior distribution is affected by different amounts of data.
#' 
#' We can remove the factorial when we calculate our posterior since it is the
#' same for all fixed *n* and *y*:
#' 
#' *p(theta | y) ~ theta ^ y * (1 - theta) ^ (n - y)
#' 
#+
library(purrr)
library(ggplot2)
library(dplyr)

n <- c(5, 20, 100, 1000)
y <- n * .6

theta_prob <- function(theta, n, y){
  (theta ^ y) * ((1 - theta) ^ (n - y))
}

sample_posterior <- function(num_samples, n, y){
  thetas <- runif(n = num_samples)
  probs <- theta_prob(thetas, n, y)
  return(data.frame(theta = thetas, prob = probs, prob_normal = probs / sum(probs), n = n, y = y))
}

posterior_samples <- map2_df(n, y, sample_posterior, num_samples = 5000)

posterior_samples %>%
  mutate(n = paste("n =", n),
         n = factor(n, levels = rev(c("n = 1000", "n = 100", "n = 20", "n = 5")))) %>%
  ggplot(aes(x = theta, y = prob_normal)) +
  geom_line(aes(group = n)) +
  facet_wrap(~n, nrow = 2, scales = "free") +
  ylab("P(theta | y)") +
  xlab("theta") +
  ggtitle("Posterior Distributions of Different Sample Sizes", "Uniform Prior")

#' ### Placenta Previa Example
#' 
#' Study found 437 out of 980 placenta previa births were female. How much
#' evidence does this provide for the claim that the proportions of females
#' is less than the .485 accepted proportion?
#' 
#' If we assume a uniform prior we can just use a *Beta(438, 544)* posterior
#' distribution
#+

ggplot(aes(x = probs), data = data_frame(probs = rbeta(n = 10000, 438, 544))) +
  geom_density() +
  geom_vline(xintercept = .485, linetype = "dashed", color = "red") +
  ggtitle("Posterior of Female Placenta Previa Births with Uniform Prior", "Accepted Population Proportion in Red") +
  xlab("Theta")

#' We can replicate the technique for the first example with different sample sizes but, I'm good for now. 
#' 
#' ### Informative Prior Distribution for Cancer Rates
#' 
#' Estimate kidney cancer deats for each county in the US
#' 
#' For each county we have 
#' 
#' *y ~ Poisson(10 * n * theta)*
#' 
#' where
#' *y* = number of deaths in county
#' *n* = number of people in county
#' *theta* = county specific parameter for rate of deaths
#' 
#' We can use the same prior distribution for all counties and let the posterior
#' be dependent on how many people are in each county. Our prior will be
#' *Gamma(20 + y, 430000 + 10 * n)*
#' 
#+

num_counties <- 10000
njs <- round(10 ^ runif(n = num_counties, min = 3, max = 7))
yjs <- rbinom(n = num_counties, size = njs, prob = 4.65e-5)

qplot(x = yjs / (10 * njs), geom = "histogram", binwidth = 5e-6) +
  xlab("Raw Death Rates for Counties") +
  ggtitle("Raw Death Rates for Counties")

qplot(x = (20 + yjs) / (430000 + 10 * njs), geom = "histogram", binwidth = 5e-6)




