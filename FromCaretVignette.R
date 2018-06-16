#First, we split the data into two groups: a training set and a test set. 
#To do this, the createDataPartition function is used:

library(caret)
library(mlbench)
data(Sonar)

set.seed(107)
inTrain <- createDataPartition(
	y = Sonar$Class,
	## the outcome data are needed
	p = 0.75,
	# the percentage of data in the training set
	list = FALSE
)

# The format of the results

# the output is a set of integers for the rows of Sonar that
# belong in the training set
str(inTrain)

training <- Sonar[inTrain,]
testing <- Sonar[-inTrain,]

nrow(training)
nrow(testing)

# To tune a model using the algorithm above, the train function can be used.
# Here, a partial least squares discriminant analysis (PLSDA) model 
# will be tuned over the number of PLS components that should be retained.

# to choose different measures of performance
# we make use of the trainControl() function

#   - method controls the type of resampling
#   - number is K in the K-fold crossvalidation
#   - repeats controls the number of repetitions
#   - summaryFunction = twoClassSummary -> compute 
#     measures specific to two-class problems (AUC, sensitivity, specificity)
#   - classProbs is used to include theese calculations
ctrl <- trainControl(
  method = "repeatedcv",
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# The following function sets up a grid of 
# tuning parameters for a number of 
# classification and regression routines, 
# fits each model and calculates a resampling 
# based performance measure

# the list of methods can be found running
names(getModelInfo())

plsFit <- train(
	Class ~ .,
	data = training,
	method = 'pls',
	# center and scale the predictors for the training set and all
	# future samples
	preProc = c("center", "scale"),
	tuneLength = 15,
	trControl = ctrl,
	metric = "ROC"
)

plsFit

# Accuracy = (TP+TN)/(TP+FP+TN+FN)
# Kappa = (accuracy - agreement_prob)/(1 - agreement_prob)
? trainControl
? train

# To predict new samples

plsClasses = predict(plsFit, newdata = testing)
str(plsClasses)

plsProbs = predict(plsFit, newdata = testing, 
                   type = "prob")
head(plsProbs)

confusionMatrix(data = plsClasses, testing$Class)




# From the Documentation --------------------------------------------------

library(caret)
data(iris)
TrainData <- iris[,1:4]
TrainClasses <- iris[,5]

knnFit1 <- train(TrainData, TrainClasses,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "cv"))

knnFit2 <- train(TrainData, TrainClasses,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "boot"))
knnFit1
# The chosen model
knnFit1$method
# An identifier of the model type.
knnFit1$modelType
# Info about the choosen model
knnFit1$modelInfo
# A data frame the training error rate and values of the tuning parameters.
knnFit1$results
# A data frame with the final parameters.
knnFit1$bestTune
# The (matched) function call with dots expanded
knnFit1$call
# A list containing any ... values passed to the original call
knnFit1$dots
# A string that specifies what summary metric will be used to select the optimal model.
knnFit1$metric
# The list of control parameters.
knnFit1$control
# Either NULL or an object of class preProcess
knnFit1$preProcess
# A fit object using the best parameters
knnFit1$finalModel
# A data frame
knnFit1$trainingData
# A data frame with columns for each performance metric. Each row corresponds to each resample
knnFit1$resample
# A character vector of performance metrics that are produced by the summary function
knnFit1$perfNames
# A logical recycled from the function arguments.
knnFit1$maximize
# The range of the training set outcomes.
knnFit1$yLimits
# A list of execution times: everything is for the entire call to train, final for the final model fit
knnFit1$times


knnFit2
# The chosen model
knnFit2$method
# An identifier of the model type.
knnFit2$modelType
# Info about the choosen model
knnFit2$modelInfo
# A data frame the training error rate and values of the tuning parameters.
knnFit2$results
# A data frame with the final parameters.
knnFit2$bestTune
# The (matched) function call with dots expanded
knnFit2$call
# A list containing any ... values passed to the original call
knnFit2$dots
# A string that specifies what summary metric will be used to select the optimal model.
knnFit2$metric
# The list of control parameters.
knnFit2$control
# Either NULL or an object of class preProcess
knnFit2$preProcess
# A fit object using the best parameters
knnFit2$finalModel
# A data frame
knnFit2$trainingData
# A data frame with columns for each performance metric. Each row corresponds to each resample
knnFit2$resample
# A character vector of performance metrics that are produced by the summary function
knnFit2$perfNames
# A logical recycled from the function arguments.
knnFit2$maximize
# The range of the training set outcomes.
knnFit2$yLimits
# A list of execution times: everything is for the entire call to train, final for the final model fit
knnFit2$times

knnFit2$pred

knnFit3 <- train(TrainData, TrainClasses,
                 method = "lda",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "LOOCV"))
knnFit3

knnFit4 <- train(TrainData, TrainClasses,
                 method = "svmLinear",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "cv"))
knnFit4$modelInfo

svmGrid <-  expand.grid(sigma = seq(0.01, 1, 0.1), C = seq(0.25, 1, 0.25))

knnFit5 <- train(TrainData, TrainClasses,
                 method = "svmRadial",
                 preProcess = c("center", "scale"),
                 tuneGrid = svmGrid,
                 trControl = trainControl(method = "cv"))
? kernlab::ksvm
