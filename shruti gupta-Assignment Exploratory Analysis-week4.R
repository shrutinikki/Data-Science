#Author:Shruti Gupta
#File Name: shruti gupta-Assignment Exploratory Analysis-week4
#Date: 26/03/2019

#The intent of this assignment is to introduce you to a new dataset- IPL data so that you can do exploratory analysis and summaries information from it.

# Q1. Find measures of central tendencies and dispersion for features: batsman score an extra run
# Q2. Also, check for normality of these variables
# Q3. Summarise these variables (batsman score an extra run) over Match_id,inning_id and over_id
# Q4. Make your interpretations basis above exploration


#loading libraries modeest and dplyr
library("modeest")

library("dplyr")

#setting and getting directory
setwd("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datasets")

getwd()

#getting file 
ipl<-read.csv("Ball_by_Ball.csv",header = TRUE,stringsAsFactors = FALSE,na.strings = "NA")
ipl
# Q1. Find measures of central tendencies and dispersion for features: 
#batsman score an extra run

#central tendencies
mean(ipl$Extra_Runs, na.rm = TRUE)
median(ipl$Extra_Runs, na.rm = TRUE)
mlv(ipl$Extra_Runs, method = "mfv", na.rm = TRUE)

ipl$Batsman_Scored = as.numeric(as.character(ipl$Batsman_Scored))
mean(ipl$Batsman_Scored, na.rm = TRUE)
median(ipl$Batsman_Scored, na.rm = TRUE)
mlv(ipl$Batsman_Scored, method = "mfv", na.rm = TRUE)

#dispension
range(ipl$Extra_Runs, na.rm = TRUE)
IQR(ipl$Extra_Runs, na.rm = TRUE)
var(ipl$Extra_Runs, na.rm = TRUE)
sd(ipl$Extra_Runs, na.rm = TRUE)

range(ipl$Batsman_Scored, na.rm = TRUE)
IQR(ipl$Batsman_Scored, na.rm = TRUE)
var(ipl$Batsman_Scored, na.rm = TRUE)
sd(ipl$Batsman_Scored, na.rm = TRUE)


# Q2. Also, check for normality of these variables

hist(ipl$Extra_Runs, na.rm = TRUE)
qqnorm(ipl$Extra_Runs, na.rm = TRUE)
ks.test(ipl$Extra_Runs, rnorm(length(ipl$Over_Id)))
shapiro.test(ipl$Extra_Runs, na.rm = TRUE)

hist(ipl$Batsman_Scored, na.rm = TRUE)
qqnorm(ipl$Batsman_Scored, na.rm = TRUE)
ks.test(ipl$Batsman_Scored, rnorm(length(ipl$Over_Id)))
shapiro.test(ipl$Batsman_Scored)


# Q3. Summarise these variables (batsman score an extra run) 
#over Match_id,inning_id and over_id

get_details_basedon_id<-ipl%>%group_by(Match_Id,Innings_Id,Over_Id)

get_details_basedon_id%>%summarise(sum(ipl$Batsman_Scored, na.rm = TRUE))

get_details_basedon_id%>%summarise(sum(ipl$Extra_Runs, na.rm = TRUE))


# Q4. Make your interpretations basis above exploration

the observation that can be made on the output from the calculation is that 
the Batsman_Scored field and extra_run field does not have a major positive colloration
and the based on the summary of the field is that the data it does not change the expected 
based on the match_id, inning_id and over_id.