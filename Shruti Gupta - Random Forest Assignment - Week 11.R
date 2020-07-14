#Assignment: R_CASE_STUDY

#loading required pacakges
library(randomForest)
library(bit64)
library(data.table)
library(sqldf)
library(caret)


setwd("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datascienceusingr\\class\\week11\\assignment")
getwd()

#read the csv file

data<- read.csv('Titanic.csv',header = TRUE)
head(data)

data$Age[is.na(data$Age)] = mean(data$Age, na.rm = TRUE)
data$Sex = as.numeric(data$Sex)

data$Sex<-factor(data$Sex)
data$Survived<-factor(data$Survived)
data$SexCode<-factor(data$SexCode)

train <- data[1:891,]
test <- data[892:1309,]

set.seed(754)

# Build the model (note: not all possible variables are used)

titanic_model <- randomForest(Survived ~ Sex + Age,data = train)

prediction <- predict(titanic_model, test)

prediction

