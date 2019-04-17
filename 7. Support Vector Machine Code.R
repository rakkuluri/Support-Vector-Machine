
rm(list=ls())

# Load Library
library(leaps)
library(sqldf)
library(car)
library(ROSE)
library(caret)
library(dplyr)          
library(pROC)  
library(e1071)        # support vector machine 
library(ROCR)


setwd("C:/Users/Ravinder/Desktop/SVM")
mydata<-read.csv("reg data2.5.csv")

#Exploring data
str(mydata)

# user function for creating descriptive statistics
mystats <- function(x) {
  nmiss<-sum(is.na(x))
  a <- x[!is.na(x)]
  m <- mean(a)
  n <- length(a)
  s <- sd(a)
  min <- min(a)
  p1<-quantile(a,0.01)
  p5<-quantile(a,0.05)
  p10<-quantile(a,0.10)
  q1<-quantile(a,0.25)
  q2<-quantile(a,0.5)
  q3<-quantile(a,0.75)
  p90<-quantile(a,0.90)
  p95<-quantile(a,0.95)
  p99<-quantile(a,0.99)
  max <- max(a)
  UC <- m+3*s
  LC <- m-3*s
  outlier_flag<- max>UC | min<LC
  return(c(n=n, nmiss=nmiss, outlier_flag=outlier_flag, mean=m, stdev=s,min = min, p1=p1,p5=p5,p10=p10,q1=q1,q2=q2,q3=q3,p90=p90,p95=p95,p99=p99,max=max, UC=UC, LC=LC ))
}

vars <- c( "SeriousDlqin2yrs" , "RevolvingUtilizationOfUnsecuredLines" ,  "age",   
           "NumberOfTime30.59DaysPastDueNotWorse" , "DebtRatio", "MonthlyIncome" , "NumberOfOpenCreditLinesAndLoans" ,
           "NumberOfTimes90DaysLate" , "NumberRealEstateLoansOrLines" , "NumberOfTime60.89DaysPastDueNotWorse" , 
           "NumberOfDependents")

diag_stats<-t(data.frame(apply(mydata[vars], 2, mystats)))

# Writing Summary stats to external file
write.csv(diag_stats, file = "diag_stats.csv")

#Outliers
mydata$age[mydata$age>87]<- 87
mydata$RevolvingUtilizationOfUnsecuredLines[mydata$RevolvingUtilizationOfUnsecuredLines>1]<- 1
mydata$NumberOfTime30.59DaysPastDueNotWorse[mydata$NumberOfTime30.59DaysPastDueNotWorse>12.9993772]<- 13
mydata$DebtRatio[mydata$DebtRatio>6466.4650758]<- 6466.4650758
mydata$NumberOfOpenCreditLinesAndLoans[mydata$NumberOfOpenCreditLinesAndLoans>23.890613]<- 24
mydata$NumberOfTimes90DaysLate[mydata$NumberOfTimes90DaysLate>12.7738847]<- 13
mydata$NumberRealEstateLoansOrLines[mydata$NumberRealEstateLoansOrLines>4.407553]<- 4
mydata$NumberOfTime60.89DaysPastDueNotWorse[mydata$NumberOfTime60.89DaysPastDueNotWorse>12.7059249]<- 13
mydata$MonthlyIncome[mydata$MonthlyIncome>45311.57]<- 45311.57
mydata$NumberOfDependents[mydata$NumberOfDependents>4]<- 4

summary(mydata)

#missing values
mydata$MonthlyIncome[is.na(mydata$MonthlyIncome ==  TRUE)] <- 6460
mydata$NumberOfDependents[is.na(mydata$NumberOfDependents ==  TRUE)] <- 1
mydata$age[mydata$age==0]<- 52

cor_matrix<-data.frame(cor(mydata))
write.csv(cor_matrix, file = "cor_matrix.csv")

# Creating Training & Validation Data sets(70:30)
set.seed(125)
smp_size <- floor(0.70 * nrow(mydata))
train_ind <- sample(seq_len(nrow(mydata)), size = smp_size)
training <- mydata[train_ind, ]
testing <- mydata[-train_ind, ]

#See how much balanced the training data is!
table(training$SeriousDlqin2yrs)    #by Counts
prop.table(table(training$SeriousDlqin2yrs))  #by %

############################# 2B. SUB-SET of DATA #####################################
#SVM usually takes a long time for computation
#Hence, Considering only a subset of the data

trainset <- training[1:2500, ]
testset <- testing[1:1500, ]

#Convert 2 Factors
trainset$SeriousDlqin2yrs<-as.factor(trainset$SeriousDlqin2yrs)
testset$SeriousDlqin2yrs<-as.factor(testset$SeriousDlqin2yrs)

############################# 3. SVM MODEL ###################################################

#M-1
mysvm1 = svm(SeriousDlqin2yrs~., data = trainset, type='C', kernel='linear')

pred1 = predict (mysvm1, testset[,-1])
table(pred1, testset[,1])
mean(pred1==testset[,1])


#M-2
mysvm2 = svm(SeriousDlqin2yrs~., data = trainset, type='C', kernel='polynomial', degree=3)
pred2 = predict(mysvm2, testset[,-1])
table(pred2, testset[,1])
mean(pred2==testset[,1])

#M-3
mysvm3 = svm(SeriousDlqin2yrs~., data = trainset, type='C', kernel='radial',gamma=2^(-4:1))
pred3 = predict(mysvm3, testset[,-1])
table(pred3, testset[,1])
mean(pred3==testset[,1])

summary(mysvm3)
mysvm3$gamma

#M-4 with TUNING
mysvm4 = tune.svm(SeriousDlqin2yrs~., data=trainset, type='C', kernel='radial', gamma = 2^(-4:1), cost = 2^(0:3))
pred4 <- predict(mysvm4$best.model, testset[,-1],type="class")
table(pred = pred4, true = testset[,1])
mean(pred4==testset[,1])

#M-5 with TUNING
mysvm5 = tune.svm(SeriousDlqin2yrs~., data=trainset, type='C', kernel='radial', gamma = 2^(-4:1), cost = 2^(0:3))
pred5 <- predict(mysvm5$best.model, testset[,-1],type="class")
table(pred = pred5, true = testset[,1])
mean(pred5==testset[,1])

#M-4 with TUNING
mysvm5 = tune.svm(SeriousDlqin2yrs~., data=trainset, type='C', kernel='radial', gamma = 2^(-4:1), cost = 2^(0:3))
pred5 <- predict(mysvm5$best.model, testset[,-1],type="class")
table(pred = pred5, true = testset[,1])
mean(pred5==testset[,1])

