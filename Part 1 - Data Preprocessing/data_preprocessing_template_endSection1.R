# Data Preprocessing

# Importing the dataset
dataset = read.csv('Data.csv')
  # Datasets start at index 1 in R

# Taking care of missing data
# Replace missing data in Age column
dataset$Age = ifelse(
                is.na(dataset$Age), # Condition
                ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), # If condition is true
                dataset$Age # If condition is false
              )
#Replace missing data in Salary column
dataset$Salary = ifelse(
  is.na(dataset$Salary), # Condition
  ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), # If condition is true
  dataset$Salary # If condition is false
)

# Encoding categorical data
# Replace country names with numerical values
dataset$Country = factor(
                    dataset$Country, # Data that we want to transform into factors
                    levels = c('France', 'Spain', 'Germany'), # names of categories in our column
                    labels = c(1, 2, 3) # Labels - Which number to give to the levels, non-order related
                  )
# Replace purchased labels with numerical values
dataset$Purchased = factor(
  dataset$Purchased, # Data that we want to transform into factors
  levels = c('No', 'Yes'), # names of categories in our column
  labels = c(0, 1) # Labels - Which number to give to the levels, non-order related
)

# Splitting the dataset into the training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])