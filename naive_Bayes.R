# This function solves the binary classification using naive bayes algorithm

joint = function(x, den, i){
  # this function computes the joint probability density function
  # assuming independance above the variables. In other words,
  # it is assumed that the joint pdf factorizes into the product
  # of the one dimensional pdfs.
  
  # x is a vector
  # den is a list of functions - the 1-d density functions
  # i is an additional integer parameter: is the index of the 
  # class variable to be dropped
  
  
  x = x[-i]
  # init a vector that will contain the values of the 1d pdfs
  joint = rep(0, ncol(x))
  # for each feature in the vector, store the 1d pdf
  for(j in 1:ncol(x)){
    joint[j] = apply(x[j],1, den[[j]])
  }
  # return the product of the pdfs
  return(prod(joint))
}

# The r function computes the regression function
r = function(l1, l2, p1, p2) return(p1*l1/(p1*l1+p2*l2))


na.bay = function(test, train, i){
  # test is a dataframe with the new observations
  # data is the training set
  # i is the index of the label column in the training set
  
  # save the column with the true classes in the train set
  trueclass = train[i]
  
  # split the train set into two subsest according the class
  # the data belong to
  X <- split(train, train[i])
  
  # save the names of the two classes
  a = as.character(unique(X[[1]][,i]))
  b = as.character(unique(X[[2]][,i]))
  
  # compute the empirical probability density functions for the
  # two classes
  pi1 = nrow(X[[1]])/nrow(train)
  pi2 = nrow(X[[2]])/nrow(train)
  # estimate the conditional density functions and interpolate
  # them in order to evaluate them for new datapoints
  den1<-apply(X[[1]][,-i], 2, density)
  den2<-apply(X[[2]][,-i], 2, density)
  den1 = lapply(den1, approxfun, rule = 2)
  den2 = lapply(den2, approxfun, rule = 2)
  
  # split the test set into a list containig each separate rows
  # this step is necessary to use the lapply function
  list <- split(test, seq(nrow(test)))
  # for each item (datapoint) in the list, compute the joint
  # conditional probability density functions using the 
  # Naive Bayes rule.
  # ...see joint() function for more details
  f1 = unlist(lapply(list, joint, den1, i))
  f2 = unlist(lapply(list, joint, den2, i))
  
  r = r(f1, f2, pi1, pi2)

  # for each row in the test set, use the value of the regression
  # function to establish to which class the datapoint belongs to
  Prediction = rep(NA, nrow(test))
  for (i in 1:nrow(test)){
    if(r[i]>1/2) {
      Prediction[i] <- a
      }
    else {
      Prediction[i] <- b
    }
  }
  return(factor(Prediction))
}

dataA = data.frame(x = rnorm(1000, -2, 1), y = rnorm(1000, 2, 1), class = "classA")
dataB = data.frame(x = rnorm(1000, 2, 1), y = rnorm(1000, -2, 1), class = "classB")
data = rbind(dataB, dataA)
plot(data$x, data$y, col = data$class, pch = 19)

# Split Train set and Test set
intr = createDataPartition(data$class, p = 0.7, list = F)
train = data[intr,]
test = data[-intr,]

naba = na.bay(test = test, train = train, i = 3)
cat("Accuracy = ",sum(naba == test$class)/length(naba))
print(table(test$class, naba, dnn = c("True", "Predicted")))
