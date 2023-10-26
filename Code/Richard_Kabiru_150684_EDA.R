#Install r libraries
install.packages("tidyverse")
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("ggthemes")
install.packages("ggpubr")
install.packages("ggrepel")
install.packages('GGally')
install.packages('caret')
install.packages('MASS')
install.packages('broom')
install.packages('car')

# import r libraries for data analysis
library(tidyverse)
library(dplyr)
library(tidyr)


# import r libraries for data visualization
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(ggrepel)

# Load libraries
library(GGally)
library(caret)
library(MASS)
library(broom)
library(car)

# Load csv file
sal <- read.csv('C:\\Users\\Rick-Royal\\Documents\\Strathmore University Data Science and Analytics\\Module 3\\Computational Techniques in Data Science\\table_203b.csv')

# View the first 6 rows of the data
head(sal)

# View the last 6 rows of the data
tail(sal)

# View the structure of the data
str(sal)

# View the summary of the data
summary(sal)

# View the dimensions of the data
dim(sal)

# View the column names of the data
colnames(sal)

# plot a correlation matrix
cor(sal)

# plot a correlation matrix
cor(sal, method = "spearman")

# plot a histogram of LCURRENT with 10 bins and title Current Salary and x label Current Salary and y label Frequency
hist(sal$LCURRENT, breaks = 10, main = "Current Salary", xlab = "Current Salary", ylab = "Frequency")

# Plot a histogram of LSTART with 10 bins and title Starting Salary and x label Starting Salary and y label Frequency
hist(sal$LSTART, breaks = 10, main = "Starting Salary", xlab = "Starting Salary", ylab = "Frequency")

# Using ggplot2, plot a scatter plot for lstart and lcurrent
ggplot(sal, aes(x = LSTART, y = LCURRENT)) + geom_point()

# Using ggplot2, plot a scatter plot for lstart and lcurrent with a linear regression line
ggplot(sal, aes(x = LSTART, y = LCURRENT)) + 
  geom_point() + 
  geom_smooth(method = "lm")

# Using ggplot2, plot a scatter plot for CURRENT and START columns with a linear regression line
ggplot(sal, aes(x = START, y = CURRENT)) + 
  geom_point() + geom_smooth(method = "lm")+ 
  xlab("Starting Salary") + ylab("Current Salary")

# using ggplot2 and geom_col, plot x = SEX and y = CURRENT with xlabel as Sex and ylabel as Current Salary
ggplot(sal, aes(x = SEX, y = LCURRENT)) + 
  geom_col() + xlab("Sex") + 
  ylab("LCurrent Salary")

# using ggplot2 and geom_col, plot x = JOBCAT and y = CURRENT with xlabel as Job Category and ylabel as Current Salary
ggplot(sal, aes(x = JOBCAT, y = LCURRENT)) + 
  geom_col() + 
  xlab("Job Category") + ylab("LCurrent Salary")

# using ggplot2 and geom_col, plot x = JOBCAT and y = CURRENT with xlabel as Job Category and ylabel as Current Salary
ggplot(sal, aes(x = JOBCAT, y = START)) + 
  geom_col() + 
  xlab("Job Category") + ylab("LStarting Salary")

# using ggplot2 and geom_point, plot x = START and y = current with xlabel as Starting Salary and ylabel as Current Salary, hue = SEX
ggplot(sal, aes(x = LSTART, y = LCURRENT, color = SEX)) + 
  geom_point() + 
  xlab("LStarting Salary") + ylab("LCurrent Salary")

# using ggplot2 and geom_point, plot x = START and y = current with xlabel as Starting Salary and ylabel as Current Salary, hue = SEX
ggplot(sal, aes(x = LSTART, y = LCURRENT, color = JOBCAT)) + 
  geom_point() + 
  xlab("LStarting Salary") + ylab("LCurrent Salary")

# using ggplot2 and geom_point, plot x = START and y = current with xlabel as Starting Salary and ylabel as Current Salary, hue = SEX
ggplot(sal, aes(x = LSTART, y = LCURRENT, color = RACE)) + 
  geom_point() + 
  xlab("LStarting Salary") + ylab("LCurrent Salary")

# Pairwise scatter plots and correlations
ggpairs(sal, columns = 1:9, ggplot2::aes(color = as.factor(SEX)))

# Boxplots for categorical predictors
sal %>%
  gather(key = "predictor", value = "value", SEX, JOBCAT, RACE) %>%
  ggplot(aes(x = value, y = LCURRENT, fill = value)) +
  geom_boxplot() +
  facet_wrap(~ predictor, scales = "free") +
  theme_minimal() +
  labs(title = "LCURRENT vs Categorical Predictors")

# Fit the model
model <- lm(LCURRENT ~ LSTART + SEX + JOBCAT + RACE + EDUC + SENIOR + AGE + EXPER, data = sal)

# Model summary
summary(model)

# Check assumptions: randomness, fixed location, fixed scale, and fixed distribution
par(mfrow = c(2, 2))
plot(model, which = 1:4)

# Leverage, influence, and outliers
influence_measures <- influence.measures(model)
influence_plot(model)

# Train-test split
set.seed(42)
trainIndex <- createDataPartition(sal$LCURRENT, p = 0.8, list = FALSE)
train <- sal[trainIndex, ]
test <- sal[-trainIndex, ]

# Full model
full_model <- lm(LCURRENT ~ LSTART + SEX + JOBCAT + RACE + EDUC + SENIOR + AGE + EXPER, data = train)

# Best model: use stepwise selection based on AIC
best_model <- stepAIC(full_model, direction = "both", trace = FALSE)

# Out-of-sample performance comparison
# Make predictions on the test set
test$pred_full <- predict(full_model, newdata = test)
test$pred_best <- predict(best_model, newdata = test)

# Calculate RMSE for the full and best models
rmse_full <- sqrt(mean((test$LCURRENT - test$pred_full)^2))
rmse_best <- sqrt(mean((test$LCURRENT - test$pred_best)^2))

cat("RMSE for the full model:", rmse_full, "\n")
cat("RMSE for the best model:", rmse_best, "\n")

# Compare the models using R-squared
r_squared_full <- 1 - (sum((test$LCURRENT - test$pred_full)^2) / sum((test$LCURRENT - mean(test$LCURRENT))^2))
r_squared_best <- 1 - (sum((test$LCURRENT - test$pred_best)^2) / sum((test$LCURRENT - mean(test$LCURRENT))^2))

cat("R-squared for the full model:", r_squared_full, "\n")
cat("R-squared for the best model:", r_squared_best, "\n")


