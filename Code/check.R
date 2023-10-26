# Load the necessary library
library(tidyverse)

# Create the data frame
data <- tibble(
  year = rep(1984:1988, each = 4),
  quarter = rep(1:4, times = 5),
  cases = c(1, 6, 16, 23, 27, 39, 31, 30, 43, 51, 63, 70, 88, 97, 91, 104, 110, 113, 149, 159)
)

# Add a new column for the time period
data <- data %>% mutate(time_period = row_number())

# Calculate ln(y_i) and ln(i)
data <- data %>% mutate(ln_cases = log(cases), ln_time_period = log(time_period))

# Plot ln(y_i) against ln(i)
ggplot(data, aes(x = ln_time_period, y = ln_cases)) +
  geom_point() +
  labs(x = "ln(Time Period)", y = "ln(Number of Cases)", title = "ln(Number of Cases) vs. ln(Time Period)")

# Initialize the beta coefficients
beta <- matrix(c(0, 0), nrow = 2)

# Define the maximum number of iterations and the convergence threshold
max_iter <- 100
epsilon <- 1e-8

# Perform Fisher's scoring method
for (i in 1:max_iter) {
  # Calculate the expected values and the variance
  mu <- exp(beta[1] + beta[2] * data$ln_time_period)
  V <- mu
  
  # Calculate the working response and the working weights
  z <- beta[1] + beta[2] * data$ln_time_period + (data$cases - mu) / mu
  W <- mu
  
  # Calculate the design matrix
  X <- cbind(1, data$ln_time_period)
  
  # Update the beta coefficients
  beta_new <- solve(t(X) %*% diag(W) %*% X) %*% (t(X) %*% diag(W) %*% z)
  
  # Check for convergence
  if (max(abs(beta_new - beta)) < epsilon) {
    beta <- beta_new
    break
  }
  
  # Update the beta coefficients for the next iteration
  beta <- beta_new
}

# Print the estimated beta coefficients
print(beta)

# Fit the GLM
model <- glm(cases ~ ln_time_period, family = poisson(link = "log"), data = data)

# Print the summary of the model
summary(model)