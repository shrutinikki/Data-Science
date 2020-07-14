#Assignment:
  
# There are two tasks for students:
  
#1. Using same old titanic dataset create models with new techniques learnt (SVM and Naive bayes) 
#and compare performance of all models developed on this dataset

#2. Use Ionsphere data set present in "mlbench" library to create 
#classification model for class as dependent variable.

#Installing and loading required pacakges
install.packages("bit64")
install.packages("randomForest")
library(randomForest)
library(bit64)
library(data.table)
library(sqldf)
library(caret)

setwd("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datascienceusingr\\class\\week12\\Assignment")
getwd()

#read the csv file

titaics<- read.csv('Titanic.csv',header = TRUE)
head(titaics)

