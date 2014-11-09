
# clean workspace
rm(list = ls())

# Data Import -------------------------------------------------------------

wine <- read.csv("http://www.nd.edu/~mclark19/learn/data/goodwine.csv")
summary(wine)


# Load R packages ---------------------------------------------------------

library(doSNOW)    # parallel processing (in order to allow caret to allot tasks to more cores simultaneously)
library(corrplot)  # graphical display of the correlation matrix
library(caret)     # classification and regression training
library(e1071)
library(randomForest)  # Random Forest, also for recursive feature elimination
library(pROC)
library(nnet)      # Neural Networks (nnet and avNNet)
library(kernlab)   # Support Vector Machines


# setup parallel processing on 3 cores ------------------------------------

#  allot tasks to three cores simultaneously (leave at least one core free)
registerDoSNOW(makeCluster(3, type = "SOCK"))


# Feature Selection -------------------------------------------------------

# We don't want to include highly correlated features in our models.
# Since we definied a new variable, "good", if the wine quality>=6, we know that these 2 variables
# are associated, so we don't want to include "quality" in our models.
# There are several ways to perform a featur selection: look at the correlation matrix, evaluate the
# relative importance of the features, do a backwards selection (aka recursive feature elimination),
# do a forward selection, and so on...

# Option 1: feature selection by looking at the correlation matrix
# Since we want to avoid collinearity in the data, we can take a look at correlation matrix.
# Of course we can't include "color" and "white" in the correlation matrix because they are Factor variables
# ("white" is of type int, but it should be converted to Factor).
correlationMatrix <- cor(wine[, -c(12, 13, 14, 15)])  # we can't include quality, color, white
corrplot(correlationMatrix, method = "number", tl.cex = 0.5)
# if we set a threshold (e.g 0.6), we can see which features are highly correlated
# (we want to remove "quality", "color", "white" as well)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.6)
colnames(correlationMatrix)[highlyCorrelated]

# Option 2: feature selection by relative importance
# This approach produces a different result, and we can have a visual representation
# of the features according to their relative importance
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- train(good ~., data = wine[, -c(12, 13, 14)], method = "glm",
               preProcess = "scale",  # normalization
               trControl = control)   # (repeated) cross validation
# scale the importance into the 0-100 range
importance <- varImp(model, scale = TRUE)
print(importance)
plot(importance)

# Option 3: feature selection by recursive feature elimination (aka backward elimination)
# With this approach we can see how the accuracy of the classifier changes by including more features.
# According to this approach we should exclude "density" but include "total.sulfur.dioxide"
# Note: this approach is really slow (~3 minutes with parallel processing, 3 cores)
# define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
# run the RFE algorithm
results <- rfe(x = wine[, -c(12, 15)], y = wine$good, sizes = c(1:11), rfeControl = control)
print(results)
# list the chosen features and plot the results
predictors(results)  # same command as results$optVariables
# accuracy of the best model (model with the selected features)
max(results$results$Accuracy) # 0.84469
plot(results, type = c("g", "o"), main = "Recursive Feature Elimination")


# Data Partition ----------------------------------------------------------

# Caret has its own partitioning function we can use here to split the data into training set
# and test set. There are 6497 total observations, of which we will put 80% into the training set.

# set the seed, so that the indices will be the same when re-run
set.seed(1234) 
trainIndices <- createDataPartition(wine$good, p = 0.8, list = FALSE)

# The three approaches for feature selection reported different results. Let's stick to the correlation matrix
# approach (Option 1) and remove the highly correlated features from the training set and the test set
wanted <- !colnames(wine) %in% c("free.sulfur.dioxide", "density", "quality", "color", "white")
wine_train <- wine[trainIndices, wanted]
wine_test <- wine[-trainIndices, wanted]


# Training Set Normalization ----------------------------------------------

# Normalize the continuous variables to the [0,1] range
normalized_wine_train <- preProcess(wine_train[, -10], method = "range")
wine_trainplot <- predict(normalized_wine_train, wine_train[, -10])
# Let’s take an initial peek at how the predictors separate on the target
featurePlot(wine_trainplot, wine_train$good, "box")

# For the training set, it looks like alcohol content, volatile acidity and chlorides separate most
# with regard to good classification. While this might give us some food for thought, note that
# the figure does not give insight into interaction effects, which methods such as trees will get at.


# K-nearest neighbors classifier ------------------------------------------

# Let's try to classify wine with a KNN classifier. But how many neighbors would work best?
# In order to choose the best number of neighbours, we can perform a 10 fold cross validation.
# Note: For whatever tuning parameters are sought, the train function will expect a dataframe with a ’.’
# before the parameter name as the column name.

# train the KNN classifier(s)
set.seed(1234)
# setup the cross validation (here a repeated 10-fold cross validation perform slightly better than a
# 10-fold cross validation)
cv_opts <- trainControl(method = "repeatedcv", number = 10)
# train several KNN classifiers (set k as an odd number, in order to avoid ties of the neighbours)
knn_opts <- data.frame(.k = c(3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 75, 101)) 
results_knn <- train(good ~., data = wine_train, method = "knn",
                    preProcess = "range",  # normalization
                    trControl = cv_opts,   # cross validation
                    tuneGrid = knn_opts)   # grid search to find the best k
results_knn
# In this case it looks like choosing the nearest three neighbors (k = 31) works best in terms of accuracy

# predict with the chosen KNN classifier
preds_knn <- predict(results_knn, newdata = wine_test[, -10])
# show the classification results in a confusion matrix (aka contingency table) 
confusionMatrix(preds_knn, wine_test[, 10], positive = "Good")

# The KNN classifier is robust to outliers, but it is susceptible to irrelevant features
# and to correlated inputs. Let's plot the relative importance of the features.
# Note: this is not the same plot as the one in the Feature Selection section.
dotPlot(varImp(results_knn, scale = TRUE))
# It seems that 3 features ("alcohol", "volatile.acidity", "chlorides") are far more important
# than the other ones.

# Let's try building a new KNN classifier with only the 3 most important features
results_knn3 <- train(good ~., data = wine_train[, c(2, 5, 9, 10)], method = "knn",
                     preProcess = "range",  # normalization
                     trControl = cv_opts,   # cross validation
                     tuneGrid = knn_opts)   # grid search to find the best k
results_knn3

# predict with the refined KNN classifier
preds_knn3 <- predict(results_knn3, newdata = wine_test[, -10])
# show the classification results in a confusion matrix (aka contingency table) 
confusionMatrix(preds_knn3, wine_test[, 10], positive = "Good")
# The KNN classifier with only the 3 most important features performs slightly better than the previuos one


# Neural networks classifier ----------------------------------------------

# Let's try to classify wine with a neural networks classifier
# Note: the training can be really slow
results_nnet <- train(good ~., data = wine_train, method = "nnet",
                      preProcess = "range",  # normalization
                      trControl = cv_opts,   # cross validation
                      # we could find the best performing set of parameters with grid search, but it would
                      # be very time consuming. We could find the best set of:
                      # .size = number of units in the hidden layer
                      # .decay = weight decay of the neural network (regularization parameter)
                      tuneGrid = expand.grid(.size = c(1, 5, 10),
                                             .decay = c(0, 0.001, 0.1)),
                      tuneLength = 5,  # number of levels for each tuning parameters
                      trace = FALSE,   # switch for tracing optimization. Default TRUE
                      maxit = 1000)    # maximum number of iterations. Default 100

results_nnet

# predict with the Neural Networks classifier
preds_nnet <- predict(results_nnet, newdata = wine_test[, -10], type = "raw")
confusionMatrix(preds_nnet, wine_test[, 10], positive = "Good")
# The Neural Networks classifier performs slightly better than the KNN classifier, but nothing special


# Model Averaged Neural Network -------------------------------------------

# The avNNet method in the caret package train several neural networks classifiers,
# aggregate them and average them. It is an ensemble method.

results_avgNnet <- train(good ~., data = wine_train, method = "avNNet",
                         preProcess = "range",  # normalization
                         trControl = cv_opts,   # cross validation
                         tuneGrid = expand.grid(.size = c(1, 5, 10),
                                                .decay = c(0, 0.001, 0.1),
                                                .bag = FALSE),
                         tuneLength = 5,
                         allowParallel = TRUE,  # use parallel processing if loaded and available (DoSNOW)
                         trace = FALSE,
                         maxit = 1000)
results_avgNnet 

# predict with the Model Averaged Neural Networks classifier
preds_avgNnet <- predict(results_avgNnet, newdata = wine_test[, -10])
confusionMatrix(preds_avgNnet, wine_test[, 10], positive = "Good")
# The Model Averaged Neural Networks classifier performs slightly better than the single Neural Networks


# Random Forest classifier ----------------------------------------------

set.seed(1234)
# The only tunable parameter in a randomForest classifier is mtry
# mtry = Number of variables randomly sampled as candidates at each split
results_rf <- train(good ~., data = wine_train, method = "rf",
                    preProcess = "range",  # normalization
                    trControl = cv_opts,   # cross validation
                    tuneGrid = expand.grid(.mtry = c(2:6)),
                    n.tree = 1000)
# NEVER prune trees in a Random Forests classifier! (from a talk by Jeremy Howard)
results_rf 

# predict with the Random Forest classifier
preds_rf <- predict(results_rf, wine_test[, -10])
confusionMatrix(preds_rf, wine_test[, 10], positive="Good")
# The Random Forest classifier performs quite better than the Model Averaged Neural Networks,
# and it is significantly faster to train.
# Random forest classifier does not suffer of irrelevant and/or correlated predictors, and since it is an
# ensemble method, it contains the combined information of many models, so it is lesser prone to overfitting.


# Support Vector Machines (SVM) classifier --------------------------------

set.seed(1234)
results_svm <- train(good ~., data = wine_train, method = "svmLinear",
                    preProcess = "range",  # normalization
                    trControl = cv_opts,   # cross validation
                    tuneGrid = expand.grid(.C = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)),
                    tuneLength = 5)
results_svm

# predict with the SVM classifier
preds_svm <- predict(results_svm, wine_test[, -10])
confusionMatrix(preds_svm, wine_test[, 10], positive = "Good")
# This SVM classifier does not perform particularly well. Maybe we could try with a different kernel function.





