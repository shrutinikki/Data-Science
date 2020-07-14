#Author: Shruti Gupta
#File Name: Shruti Gupta- Assignment Visualisation In R- Week 5
#Date: 02/04/2019

#installing and loading library corrgram 
install.packages("corrgram")
library("corrgram")

#setting and getting directory
setwd("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datasets")

getwd()

#getting file 
ipl<-read.csv("Ball_by_Ball.csv",header = TRUE,stringsAsFactors = FALSE,na.strings = "NA")
head(ipl)
ipl$Batsman_Scored = as.numeric(as.character(ipl$Batsman_Scored))
ipl$Extra_Runs = as.numeric(as.character(ipl$Extra_Runs))
#Objective: To explore different visualizations learned on IPL dataset


# Q1. Create univariate charts for batsman scored and an extra run. 
#Also, create frequency chart for other categorical variables

par(mfrow=c(2,3))
hist(ipl$Extra_Runs,main="Extra Runs",col=topo.colors(3,0.3))
hist(ipl$Batsman_Scored,main="Batsman Scored",col=topo.colors(5,0.5))
plot(ipl$Ball_Id,ipl$Team_Bowling_Id,main="balls used by bowlers",xlab = "ball id",ylab = "bowler id",col=topo.colors(3,0.4))
boxplot(ipl$Player_dissimal_Id~ipl$Dissimal_Type,main="dissimal type by players")
boxplot(ipl$Striker_Id~ipl$Striker_Batting_Position,main="Batting position used by batsman")
pie(table(ipl$Extra_Type),labels = names(table(ipl$Extra_Type)),col=topo.colors(3,0.6),main = "the extra run types")


# Q2. Create correlogram for batsman scored an extra run

#creating a vector varliable to pass to data set for the requried position
vars1<-c("Batsman_Scored","Extra_Runs")
corrgram(ipl[vars1],order=TRUE, main="Correlogram between Batsman Scored and Extra Run",
         lower.panel=panel.shade, upper.panel=panel.shade)




