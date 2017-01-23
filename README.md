# Coursera: Practical Machine Learning Course Project
Paul Harrington  

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

  
<br>
<br>


## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The project, and the data used therein, are described further in the following paper:


Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4VBNAhMMs


<br>
<br>


## Data Loading

For the purpose of this exercise, our working data set is going to be the training data (pml-training.csv), We load this as follows. Note that we can take care of some inconsistency in the format of "N/A" type data in this step:


```r
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl,destfile="./pml-training.csv")
pml_data <- read.csv("./pml-training.csv", na.strings=c("#DIV/0!","NA",""))
```


## Data Cleansing

We will remove data we don't need for this analysis: 

- The time-series data in columns up to 7, as these are not a factor in predicting whether or not an exercise is performed correctly

- The training data set contains many columns for which the vast majority of the values are N/A. Due to time constraints, we will simply remove these columns from the analysis, Decision rule is: if the column has more than 75% NA values, exclude



```r
pml_data <- pml_data[-(1:7)]
na.col_list <- sapply(colnames(pml_data), function(x) if(sum(is.na(pml_data[, x])) > 0.75 * nrow(pml_data)){return(TRUE)}else{return(FALSE)})
pml_data <- pml_data[, !na.col_list]
```


## Data Partitioning

We now split the working data set into training and testing partitions:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.2
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.3.2
```

```r
pml_data_partition <- createDataPartition(y=pml_data$classe, p=0.7, list=FALSE)
pml_data_training <- pml_data[pml_data_partition, ]
pml_data_testing <- pml_data[-pml_data_partition, ]
```

<br>
(Note that the partitioning of our working data set into training and testing partitions is not to be confused with the existence of separate training and test data sets at the HAR site; we will also be using our prediction model to predict values for the test data set, later in the exercise)
<br>


## Generation of Prediction Model

We will generate Prediction Models using the following methods:

1. Linear Discriminant Analysis
2. Random Forests

We are using the classe variable to train the prediction models. For both methods, we train using the Training partition (pml_data_training). We then perform **Cross-Validation** using the Testing partition (pml_data_testing).

### 1. Linear Discriminant Analysis


```r
set.seed(12345)
training_lda <- train(classe ~ .,data=pml_data_training,method="lda")
```

```
## Loading required package: MASS
```

```r
prediction_lda <- predict(training_lda, pml_data_testing)
confusionMatrix(prediction_lda,pml_data_testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1374  161  101   46   44
##          B   42  736  107   46  194
##          C  128  149  670  114   98
##          D  123   45  123  721   93
##          E    7   48   25   37  653
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7059         
##                  95% CI : (0.694, 0.7175)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6279         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8208   0.6462   0.6530   0.7479   0.6035
## Specificity            0.9164   0.9180   0.8994   0.9220   0.9756
## Pos Pred Value         0.7961   0.6542   0.5781   0.6525   0.8481
## Neg Pred Value         0.9279   0.9153   0.9247   0.9492   0.9161
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2335   0.1251   0.1138   0.1225   0.1110
## Detection Prevalence   0.2933   0.1912   0.1969   0.1878   0.1308
## Balanced Accuracy      0.8686   0.7821   0.7762   0.8349   0.7896
```

### 2. Random Forests


```r
set.seed(12345)
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.3.2
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
training_rf <- randomForest(classe ~ .,data=pml_data_training)
prediction_rf <- predict(training_rf, pml_data_testing)
confusionMatrix(prediction_rf,pml_data_testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    4    0    0    0
##          B    3 1135    4    0    0
##          C    0    0 1017    5    0
##          D    0    0    5  959    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9978)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9965   0.9912   0.9948   1.0000
## Specificity            0.9991   0.9985   0.9990   0.9990   1.0000
## Pos Pred Value         0.9976   0.9939   0.9951   0.9948   1.0000
## Neg Pred Value         0.9993   0.9992   0.9981   0.9990   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1929   0.1728   0.1630   0.1839
## Detection Prevalence   0.2846   0.1941   0.1737   0.1638   0.1839
## Balanced Accuracy      0.9986   0.9975   0.9951   0.9969   1.0000
```

As we can see, the Random Forest method provides a very accurate prediction. We will therefore not test any other methods, but will proceed with the Random Forest method.
<br>
<br>
We can calculate Accuracy and Out of Sample error from the above results, as follows:


## Accuracy, and Out of Sample Error


```r
cmatrix_rf <- confusionMatrix(prediction_rf,pml_data_testing$classe)
accuracy_cmatrix_rf <- cmatrix_rf$overall['Accuracy']
accuracy_cmatrix_rf
```

```
##  Accuracy 
## 0.9964316
```

<br>
**Accuracy of the Random Forests predictor is 0.9964316 . Out of Sample Error is  0.0035684 **
<br>
<br>
<br>

## Submission of HAR Test Data Set

Download and submit the Testing data set to our predictor:


```r
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download.file(fileUrl,destfile="./pml-testing.csv")
pml_test_data <- read.csv("./pml-testing.csv", na.strings=c("#DIV/0!","NA",""))
prediction_rf <- predict(training_rf, pml_test_data)
prediction_rf
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```




