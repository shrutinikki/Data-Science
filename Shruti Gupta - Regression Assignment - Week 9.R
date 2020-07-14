#Assignment
#Problem statement: Create a linear regression model on 75% data and predict reading scores of students. (reading score is dependent variable)

#Dataset
#Please access the â KCF Dataset â from the Common Dataset section.

setwd("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datascienceusingr\\class\\week9\\Assignment")
getwd()

#read the csv file

kcf<- read.csv('kcf.csv',header = TRUE)
head(kcf)

#loading package
library('lmtest')

#creating the model
models<-lm(Consumption~Income,data = kcf)
dwtest(Consumption~Income,data = kcf)

plot(models$fit,(models$residuals))

splits<-sample(seq_len(nrow(kcf)),size = floor(0.75*nrow(kcf)))

train<-kcf[splits,]
test<-kcf[-splits,]
#head(train)
#head(test)

pre_models<-lm(Consumption~Income,data = kcf)
prediction<-predict(pre_models,newdata=test)

SSE<-sum((test$Consumption-prediction)^2) 
SST<-sum((test$Consumption-mean(test$Consumption))^2) 
1-SSE/SST

