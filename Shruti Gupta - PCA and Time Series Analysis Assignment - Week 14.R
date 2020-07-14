#Task: Use R inbuilt dataset for time series forecasting of presidential ratings. 
#Predict ratings for next 5 years


#Dataset name: presidents


#Description:
  
  
#  Quarterly Approval Ratings of US Presidents
#Description
#The (approximately) quarterly approval rating for the President of 
#the United States from the first quarter of 1945 to the last quarter of 1974.



#Installing and loading required pacakges
install.packages("bit64")
install.packages("randomForest")
library(randomForest)
library(bit64)
library(data.table)
library(sqldf)
library(caret)

pres<-datasets::presidents

