#Assignment Week: 7
#Date: 23/06/2019
#Author: Shruti Gupta

#loading library
library(BSDA)

#Q1. 
Data<-c(5000,2000,3000,3456,3623,5200,3400,1200,4500,3500)

z.test(Data_2,alternative = 'two.sided',mu=3500,sigma.x = 3)
z.test(Data_2,alternative = 'greater',mu=3500,sigma.x = 3)
z.test(Data_2,alternative = 'less',mu=3500,sigma.x = 3)

#q2
Data_2<-c(5000,2000,3000,3456,3623,5200,3400,1200,4500,3500)

t.test(Data,alternative = 'two.sided',mu=3500)
t.test(Data,alternative = 'greater',mu=3500)
t.test(Data,alternative = 'less',mu=3500)
