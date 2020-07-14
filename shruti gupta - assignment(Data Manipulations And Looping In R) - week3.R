#author: Shruti Gupta
#file name: Assignment (Data Manipulations And Looping In R)
#date: 14/03/2019

#Assignment

#Objective: Intent of this assignment is to get hands-on experience with functions in R

#Load the Boston dataset (Hint - install. Packages(MASS); library(MASS); data(Boston))

#Q1. Create a custom function for calculating the square root of medv variable in Boston
#Q2. Find Random samples from all variables of dataset
#Q3. Find average of numeric variables using relevant apply function


#installing MASS Package/Library
install.packages("MASS")

#loading Mass Library
library("MASS")

#loading Boston data
data(Boston,"Boston")

#Q1. Create a custom function for calculating the square root of medv variable in Boston

sq_root<- function(x)
{
  #this is to return the square root as 1 if x value is 0 or 1
  if (x == 0 || x == 1)
  {
      x
  }
     
  inital = 1
  final = x
  sp=0
  #in this loop the camparision if between the starting value and x value
  while (inital<=final)  
  {         
    midval=(inital+final)/2; #creating a mid value for checking if the square is eqaul to x
    
    
    if (midval*midval==x)
    {
      midval
    }
    #this if condtion is for checking for the square root is less than x and it to anx
    if (midval*midval<x)  
    { 
      inital = midval + 1; 
      sq = midval; 
    }  
    else
    final = midval-1;         
  } 
  sq #returning the values in the sq variable
}
sq_root(Boston$medv)

#Q2. Find Random samples from all variables of dataset

set.seed(120)
retrive <- sample(1:nrow(Boston))
Boston[retrive, ]

#Q3. Find average of numeric variables using relevant apply function

sapply(Boston,FUN = mean) #using sapply to get the mean in a vector output fortmat
lapply(Boston,FUN = mean) #using lapply to get the mean in a list output fortmat
