
# Data Import -------------------------------------------------------------

wine = read.csv("http://www.nd.edu/~mclark19/learn/data/goodwine.csv")
summary(wine)


# Load R packages ---------------------------------------------------------

library(doSNOW)    # parallel processing (in order to allow caret to allot tasks to more cores simultaneously)
library(corrplot)  # graphical display of the correlation matrix
library(caret)     # classification and regression training
library(e1071)
library(randomForest)


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
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.6)
colnames(correlationMatrix)[highlyCorrelated]

# Option 2: feature selection by relative importance
# This approach produces a different result, and we can have a visual representation
# of the features according to their relative importance
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(good~., data=wine[, -c(12, 13, 14)], method="glm", preProcess="scale", trControl=control)
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

# Option 3: feature selection by recursive feature elimination (aka backward elimination)
# With this approach we can see how the accuracy of the classifier changes by including more features.
# According to this approach we should exclude "density" but include "total.sulfur.dioxide"
# Note: this approach is really slow (~3 minutes with parallel processing, 3 cores)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(x = wine[, -c(12, 15)], y = wine$good, sizes=c(1:11), rfeControl=control)
print(results)
# list the chosen features and plot the results
predictors(results)  # same command as results$optVariables
# accuracy of the best model (model with the selected features)
max(results$results$Accuracy) # 0.84469
plot(results, type=c("g", "o"), main ="Recursive Feature Elimination")


# Data Partition ----------------------------------------------------------

# Caret has its own partitioning function we can use here to split the data into training set
# and test set. There are 6497 total observations, of which we will put 80% into the training set.

# set the seed, so that the indices will be the same when re-run
set.seed(1234) 
trainIndices = createDataPartition(wine$good, p = 0.8, list = FALSE)

# The three approaches for feature selection reported different results. Let's stick to the correlation matrix
# approach (Option 1) and remove the highly correlated features from the training set and the test set
wanted = !colnames(wine) %in% c("free.sulfur.dioxide", "density", "quality", "color", "white")
wine_train = wine[trainIndices, wanted]
wine_test = wine[-trainIndices, wanted]




