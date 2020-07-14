#Assignment
#Problem statement: In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, 
#we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

#Dataset
#Please access the “Titanic Dataset “from the Common Dataset section.
datasets::Titanic
titanics<-datasets::Titanic
# fill in missing values
titanics$Age[is.na(titanics$Age)] = mean(titanics$Age, na.rm = TRUE)
titanics$Sex = as.numeric(titanics$Sex)


#splitting the data
set.seed(222)
t= sample(1:nrow(titanics), 0.7*nrow(titanics))
train = titanics[t,]
test = titanics[-t,]
#creating model
model = glm(Survived ~ ., data = train, family = binomial)
summary(model)

predictTest = predict(model, type = "response", newdata = test)

# no preference over error t = 0.5
test$Survived = as.numeric(predictTest >= 0.5)
table(test$Survived)

