#Author: Shruti Nikhila Gupta

#Assignment 1: Introduction to R Programming 
#Assignment Objective: Hands-on practice on diamond dataset using functions from dplyr package

#Questions
#Q1. Which color has maximum price/carat
#Q2. Which clarity has minimum price/carat
#Q3. Which cut has more missing values
#Q4. Which color has minimum median price
#Q5. What conclusion can you draw from the variables color, cut-price, and carat
#data set: BigDiamonds.cvs

#including libary
library("dplyr")
#set working directory
setwd ("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datascienceusingr\\class\\week2")

#retrieve the working directory name
getwd()

#retriving file
data_set = read.csv("BigDiamonds.csv",header = TRUE,stringsAsFactors = FALSE,na.strings = "NA")
data_set

#Q1. Which color has maximum price/carat
maxcarat<-max(data_set$carat,na.rm = FALSE)

colorfind<-data_set$color
finalcolor<-setdiff(colorfind, maxcarat)
finalcolor

#Q2. Which clarity has minimum price/carat
minprice<-min(data_set$price,na.rm = TRUE)
minprice

clarityfind<-data_set$clarity
finalclarity<-setdiff(clarityfind, minprice)
finalclarity


#Q3. Which cut has more missing values

napresent<- function() {
  
  price=filter_all(data_set,all_vars(is.na(data_set$price)))
  x=filter_all(data_set,all_vars(is.na(data_set$x)))
  y=filter_all(data_set,all_vars(is.na(data_set$y)))
  z=filter_all(data_set,all_vars(is.na(data_set$z)))
  
}
data_set%>%group_by(carat)%>%summarise_all(funs(filter_all(data_set,all_vars(is.na(data_set$price)))))
select_all(data_set,funs(filter_all(data_set,all_vars(is.na(data_set$price)))))


#Q4. Which color has minimum median price





#Q5. What conclusion can you draw from the variables color, cut-price, and carat




