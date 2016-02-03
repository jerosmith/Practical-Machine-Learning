# Practical Machine Learning Course Project
2nd February, 2016  


### Executive Summary

Nowadays there is increasing awareness on the health benefits of exercise, and increasingly people want to know if they do their physical fitness exercises correctly. However, it is expensive to hire a personal trainer every day to correct them. Machine learning provides a cheap solution. Various wearable devices record a person's body movements, and this data can be used to predict if he/she is performing the exercise correctly.

In this study we created three machine learning models, using the data set provided by the HAR (Human Activity Recognition) research team at the Pontifical Catholic University of Rio de Janeiro.

We used R statistical programming language to perform the analysis. The R code is included, in compliance with the principles of reproducible research.


### Cleaning and Preparation of Data

We used the following R libraries:


```r
library(plyr)
library(gtools)
library(ggplot2)
library(caret)
library(rpart)
library(randomForest)
library(gbm)
```

We downloaded the training and validation data sets from the URL's provided: files pml-training.csv and pml-testing.csv respectively. We then loaded these two files into two data frames: train and test. We performed the following cleansing operations:

1. We eliminated variables assumed to be irrelevant as predictors. Upon inspection, variables such as user_name and raw_timestamp_part_1, containing user names and times respectively, are clearly not extendible to a new scenario with a new user, and hence irrelevant.
2. We converted factor variables that clearly contained numeric data to the numeric data type.
3. We eliminated variables with very little data in either the training or testing data sets, defined as those with more than 80% of NA's.

The following R code performs these operations:


```r
# Load training and test data:
setwd("C:/A/2 University/Coursera/8 Practical Machine Learning/Course Project")

# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

train = read.csv(file="pml-training.csv") # Load training data into train data frame.
test = read.csv(file="pml-testing.csv") # Load validation data into test data frame.

# Eliminate variables assumed to be irrelevant:
x1 = c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window") # Vector of names of irrelevant variables.
x1 = match(x1, names(train)) # Vector x1 converted to vector of indices.
train = train[, -x1] # Eliminate irrelevant variables.
test = test[, -x1] # Eliminate irrelevant variables.
test = test[, -match("problem_id", names(test))] # Eliminate problem_id from test, assumed irrelevant.

# Convert factor variables to numeric:
x1 = as.logical(lapply(train, is.factor)) # Obtain logical vector indicating factor variables.
names(x1) = names(train) # Name the elements of x1 by the variable names of train.
x2 = x1[x1] # Get vector of factor variable names.
x3 = c("classe") # Vector of factor variable names not to be converted.
x4 = match(x3, names(x2)) # Vector x3 converted to vector of indices.
x2 = x2[-x4] # Remove from x2 variables not to be converted.
x2 = names(x2) # Names of variables to be converted.
x5 = match(x2, names(train)) # Vector x2 converted to vector of indices.
train[, x5] = lapply(train[, x5], as.numeric) ## Convert factor variables to numeric.
test[, x5] = lapply(test[, x5], as.numeric) ## Convert factor variables to numeric.

# Remove variables in train with very little data (above a threshold of 80% NA's):
x1 = nrow(train)
x2 = apply(train, 2, is.na)
x3 = apply(x2, 2, sum)
x4 = (x3 < 0.8)
x5 = x3[x4]
train = train[, names(x5)]

# Remove variables in test with very little data (above a threshold of 80% NA's):
x1 = nrow(test)
x2 = apply(test, 2, is.na)
x3 = apply(x2, 2, sum)
x4 = (x3 < 0.8)
x5 = x3[x4]
test = test[, names(x5)]

# Keep only variables in train that are in test, apart from outcome variable classe:
train = train[, append(names(test), "classe")]
```

After cleaning we were left with a training data set with 52 numeric predictor variables, and the outcome variable "classe". The validation data set had the same 52 predictor variables and no outcome variable.


### Exploratory Data Analysis

It is difficult and time-consuming to graphically explore and plot all 52 predictor variables. Above all, it is most likely not worthwhile to explore them all. Therefore we needed a criterion to select the two most relevant variables. The correlation with the outcome variable appears to be an obvious choice. However, there is a difficulty. The outcome variable "classe" is categorical, and the correlation coefficient is only defined for numeric variables. To overcome this we created a numeric proxy variable, "classeNum"", assigning arbitrary numeric values to the values "A", "B", "C", etc. of classe. The values used were simply A=0, B=1, C=2, etc. We then calculated the correlation between all the predictor variables and classeNum, and for the plots chose the two with the highest correlation.

Note that computing this numeric variable classeNum in this way is not a rigorous statistical test. It is simply a method for selecting some predictor variables for exploration and in graphical plots. Therefore in the table we call it the pseudo-correlation between the predictor variables and the outcome.

After the correlations had been calculated we eliminated classeNum from the training data set.


```r
# Create classeNum variable for exploratory data analysis:
train$classeNum = asc(as.character(train$classe))-65

# Calculate correlations and put in data frame in descending order:
x1 = match("classe", names(train))
x2 = match("classeNum", names(train))
corr = abs(cor(train[,1:x1-1], train[,x2]))
x3 = order(corr[,1], decreasing=T)
corr = as.data.frame(corr[x3,])
corr$Variable = rownames(corr)
names(corr)[1] = "Correlation"
corr = corr[, c(2, 1)]
rownames(corr) = NULL

# Cleanup: eliminate classeNum from train:
x4 = match("classeNum", names(train))
train = train[,-x4]
```

**Table 1: Pseudo-correlation of Predictor Variables with Outcome**

```
##                Variable  Correlation
## 1         pitch_forearm 0.3438258280
## 2          magnet_arm_x 0.2959635721
## 3         magnet_belt_y 0.2903490682
## 4          magnet_arm_y 0.2566701750
## 5           accel_arm_x 0.2425926172
## 6       accel_forearm_x 0.1886853999
## 7      magnet_forearm_x 0.1821333559
## 8         magnet_belt_z 0.1800306214
## 9             pitch_arm 0.1776848826
## 10  total_accel_forearm 0.1545383505
## 11    magnet_dumbbell_z 0.1498691916
## 12         magnet_arm_z 0.1498429260
## 13      total_accel_arm 0.1258251832
## 14     accel_dumbbell_x 0.1186463203
## 15     magnet_forearm_y 0.1078377426
## 16             roll_arm 0.0876958401
## 17          accel_arm_y 0.0870538987
## 18       pitch_dumbbell 0.0862350775
## 19         accel_belt_z 0.0793872759
## 20     total_accel_belt 0.0771561662
## 21     accel_dumbbell_z 0.0727393323
## 22    magnet_dumbbell_x 0.0666360945
## 23            roll_belt 0.0621513426
## 24 total_accel_dumbbell 0.0519633122
## 25              yaw_arm 0.0492630395
## 26          yaw_forearm 0.0455123108
## 27     magnet_forearm_z 0.0452970115
## 28          accel_arm_z 0.0445529804
## 29        roll_dumbbell 0.0425073033
## 30     gyros_dumbbell_y 0.0382605676
## 31         roll_forearm 0.0252010973
## 32      accel_forearm_y 0.0225533252
## 33      gyros_forearm_x 0.0202074822
## 34         gyros_belt_y 0.0187492178
## 35        magnet_belt_x 0.0183995965
## 36          gyros_arm_y 0.0179227064
## 37     accel_dumbbell_y 0.0158570644
## 38         gyros_belt_z 0.0151227826
## 39             yaw_belt 0.0136011048
## 40         gyros_belt_x 0.0115895498
## 41           pitch_belt 0.0107516010
## 42      gyros_forearm_y 0.0100783439
## 43         yaw_dumbbell 0.0093144427
## 44          gyros_arm_z 0.0084477066
## 45     gyros_dumbbell_z 0.0064802386
## 46         accel_belt_x 0.0063069314
## 47     gyros_dumbbell_x 0.0058488248
## 48      gyros_forearm_z 0.0057230541
## 49          gyros_arm_x 0.0033195422
## 50    magnet_dumbbell_y 0.0026623264
## 51         accel_belt_y 0.0015312437
## 52      accel_forearm_z 0.0009977051
```

As can be seen from Table 1, pitch_forearm and magnet_arm_x have the highest pseudo-correlation with the outcome. Therefore we include them in the following plots.


```r
plot1 = ggplot(train, aes(y=pitch_forearm, x=magnet_arm_x, colour=classe)) + geom_point()
plot2 = ggplot(train, aes(y=classe, x=pitch_forearm)) + geom_point()
```


**Plot 1. pitch_forearm, magnet_arm_x and outcome classe as colour**
![](Practical-Machine-Learning-Course-Project_files/figure-html/plot1-1.png)

In plot 1 we observe the highest concentration of points for outcome "A" for the lowest values of pitch_forearm and magnet_arm_x. For outcome "B" there are many points where magnet_arm_x is low and pitch_forearm is positive. Outcome "D" occurs at both high and low values of magnet_arm_x. There is a large quantity of points with outcome "E" at high values of magnet_arm_x and pitch_forearm > 0.


**Plot 2. Outcome classe plotted against pitch_forearm**
![](Practical-Machine-Learning-Course-Project_files/figure-html/plot2-1.png)

In plot 2 we observe a slight, barely discernible, correlation between classe and pitch_forearm.

Not much can be inferred from the plots. No doubt the relation (if any) between classe and the 52 predictor variables is complex and not immediately evident through observation. A relation can be confirmed if complex predictive models are built which exhibit a high degree of accuracy. This is explained in the next section.


### Choice of Model: Classification Tree, Random Forest, Boosting with Trees

We tested models built with three algorithms: classification trees, random forests and boosting with trees.

In order to increase accuracy in the selection and reduce overfitting we used k-fold cross validation. We used a value of k=10. For each fold we computed the accuracy of each model. We then took the average accuracy across all folds of each model and chose the model with greatest accuracy.

Moreover we calculated the standard deviation across all folds of the accuracy to ensure that the averages were significant.

Following is the code:


```r
# Use k-fold cross-validation, for 3 methods: tree, random forest, boosting with trees:

set.seed(1)
k=10
Accuracies = data.frame(Tree=1:k, RandomForest=1:k, Boosting=1:k)
folds = createFolds(y=train$classe, k=k, list=T, returnTrain=T)

for (i in 1:k){

  # Show progress:
  # Sys.sleep(0.1)
  # print(i)
  # flush.console() 
    
  # Get training and testing data from fold i:
  train1 = train[folds[[i]],]
  test1 = train[-folds[[i]],]
  
  # Tree:
  model = rpart(data=train1, formula = classe ~ ., method="class")
  prediction = predict(model, newdata=test1, type="class")
  cm = confusionMatrix(prediction, test1$classe)
  Accuracies$Tree[i] = cm[[3]][1]
  
  # Random Forest:
  model = randomForest(x=train1[,-match(c("classe"), names(train1))], y=train1$classe, ntree=200)
  prediction = predict(model, newdata=test1, type="class")
  cm = confusionMatrix(prediction, test1$classe)
  Accuracies$RandomForest[i] = cm[[3]][1]
  
  # Boosting:
  model = gbm(data=train1, formula = classe ~ ., n.trees=100, distribution="multinomial")
  predictionprob = predict(model, newdata=test1, n.trees=100, type="response")
  predictionprob = as.data.frame(predictionprob)
  y = substr(names(predictionprob),1,1)
  predictionindex = apply(predictionprob, 1, which.max)  
  prediction = y[predictionindex]
  cm = confusionMatrix(prediction, test1$classe)
  Accuracies$Boosting[i] = cm[[3]][1]
  
}

meanAccuracy = apply(Accuracies, 2, mean)
sdAccuracy = apply(Accuracies, 2, sd)
```


**Table 2: Accuracies of the three Algorithms**

```
##         Tree RandomForest  Boosting
## 1  0.7559857    0.9969435 0.4987264
## 2  0.7334353    0.9969419 0.4663609
## 3  0.7573904    0.9969419 0.4648318
## 4  0.7069317    0.9959225 0.5214067
## 5  0.7447784    0.9969435 0.4982170
## 6  0.7356088    0.9959246 0.4956699
## 7  0.7354740    0.9959225 0.4785933
## 8  0.7584098    0.9943935 0.4872579
## 9  0.7543323    0.9974516 0.4796126
## 10 0.7440082    0.9964304 0.5104539
```

Average Accuracies:  0.7426355, 0.9963816, 0.490113

Standard Deviations: 0.0158048, 8.8125139\times 10^{-4}, 0.0183127

Clearly the random forests model exhibits a much greater accuracy than the other two. In addition, its standard deviation is very low, making the average statistically significant.

Therefore we chose the model built with random forests.

The average accuracy of the chosen model allows us to estimate the out-of-sample error: 0.36%.

### Validation of Chosen Model

To perform the final validation of the chosen Random Forests model, we used it to generate predictions from the test data set.


```r
# Train Random Forest:
model = randomForest(x=train[,-match(c("classe"), names(train))], y=train$classe, ntree=200)

# Validate Random Forest:
prediction = predict(model, newdata=test, type="class")
prediction
```

All 20 predictions generated by the final model were correct.


### Final Conclusion

The random forests model built with the 52 predictor variables in table 1 has been tested and validated satisfactorily, and can therefore be used to make predictions on new data sets with an accuracy of 99.64%.
