# Decision Tree Classification

# Importing the dataset
dataset = read.csv('iris.csv', 
                   col.names = c('sepal length','sepal width','petal length','petal width','class'),
                   header = FALSE)
# Encoding the target feature as factor
dataset$class = factor(dataset$class, levels = c('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$class, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = class ~ .,
                   data = training_set)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-5], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 5], y_pred)
#                     Iris-setosa Iris-versicolor Iris-virginica
# Iris-setosa              12               0              0
# Iris-versicolor           0               9              3
# Iris-virginica            0               1             11

#Getting the accuracy of the model
ac= sum(diag(cm))/sum(cm)
# ac = 0.8888889

# visualizing the tree
plot(classifier)
text(classifier)