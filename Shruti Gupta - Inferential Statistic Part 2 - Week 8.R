#Assignment Week: 7
#Date: 23/06/2019
#Author: Shruti Gupta

library('xlsx')

#set working directory
setwd ("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datascienceusingr\\class\\week8\\Assignment")

#retrieve the working directory name
getwd()

win_loss<- read.xlsx('Win-Loss.xlsx', as.data.frame=TRUE, header = TRUE, sheetName='Win - Loss')

female_data<- read.xlsx('Female Heights and Weights.xlsx', as.data.frame=TRUE, header = TRUE, sheetName='women')

prices<- read.xlsx('Prices in Rupees.xlsx', as.data.frame=TRUE, header = TRUE, sheetName='Win - Loss')

ks_testing<- read.xlsx('Two Sample KS Test.xlsx', as.data.frame=TRUE, header = TRUE, sheetName='Sheet1')

#q1

library('PASWR')
female_data
#1
SIGN.test(female_data$Height,md=64)

#2
SIGN.test(female_data$Weight,md=135)

#q2

win_loss
SIGN.test(win_loss$Outcome.2,win_loss$Outcome.1)

#q3

prices
wilcox.test(prices$City.A..prices.in.rupees.,prices$City.B..prices.in.rupees.,paired = TRUE)

#q4

ks_testing

ks.test(ks_testing$Sample.1,ks_testing$Sample.2)

#q5

M <- as.table(rbind(c(8000, 800, 7200), c(1600, 120, 1480)))
dimnames(M) <- list(div = c("p1", "p2"),
                    party = c("Total","Female", "Male"))
(Xsq <- chisq.test(M))  # Prints test summary
Xsq$observed   # observed counts (same as M)
Xsq$expected   # expected counts under the null
Xsq$residuals  # Pearson residuals
Xsq$stdres    

