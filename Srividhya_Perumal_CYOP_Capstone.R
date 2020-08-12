# SRIVIDHYA PERUMAL BINARY CLASSIFICATION OF SURVIVAL OF BREAST CANCER SURGERY PATIENTS
# Download, install and load the packages required for constructing our model
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")

library("caret")
library("e1071")
library("gbm")

# set.seed() so our results are reproducible
set.seed(1)

# Load and assign data that we downloaded to variable patients
patients <- read.csv("haberman-data.txt", header = FALSE)
# View structure of haberman dataset
str(patients)
# Add respective headings to columns
names(patients)<- c('Operation Age', 'Operation Year', '+ve Axillary Nodes', 'Survival')
# Assign 0 as dead and 1 as alive
patients$Survival <- patients$Survival - 1
# View structure of haberman dataset
str(patients)
# Check for NAs in the 4 columns
sapply(patients, function(x) sum(is.na(x)))
# Create density plots by attribute
par(mfrow=c(2,3))
for(i in c(1:4)) {
  plot(density(patients[,i]), main=names(patients)[i])
}
# Check proportion of dead and alive patients
# Here 1 represents a dead patient
prop.table(table(patients$Survival))
# Convert values in Survival column to factors
patients$Survival <- ifelse(patients$Survival==1,'yes','no')
patients$Survival <- as.factor(patients$Survival)
# Create training and testing datasets
test <- createDataPartition(patients$Survival, p = .3, times = 1, list = FALSE)
train_set <- patients[-test,]
test_set <- patients[test,]
# Control resampling of data
control <- trainControl(method='cv', number=2, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
# Train model
model <- train(patients[,names(patients)[names(patients) != 'Survival']], patients[,'Survival'], method='gbm', trControl=control,  metric = "ROC", preProc = c("center", "scale"))
summary(model)
model
# Create predictions with model
predictions <- predict(object=model, test_set[,names(patients)[names(patients) != 'Survival']], type='raw')
# Compute accuracy
print(postResample(pred=predictions, obs=as.factor(test_set[,'Survival'])))
