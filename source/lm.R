#todo:
#D-Read: https://www.kaggle.com/skirmer/house-prices-advanced-regression-techniques/fun-with-real-estate-data/discussion
#D-Read: https://www.kaggle.com/notaapple/house-prices-advanced-regression-techniques/detailed-exploratory-data-analysis-using-r/discussion
#D-make simple submission: lm_initial: Trn/CV errors=0.403522/0.7154108, Train error=0.4576414, Score=0.26634
#-Use 1 feature (GrLivArea): lm_GrLivArea: 0.2760749/0.2736812, 0.2755768, 0.28918
#-Print the other thing that shows significance of variable
#-Group OverallQual into low(1-4), med(5-7), high(8-10)
#-Experiment with more features
#-Read more forum posts
#-handle negative values better somehow
#-make new features from the interaction b/n highly-correlated features (eg. train$year_qual = train$YearBuilt*train$OverallQual)


#Remove all objects from the current workspace
rm(list = ls())
setwd('/Users/dan/Desktop/Kaggle/Housing')

library(caret) #createDataPartition
library(hydroGOF) #rmse
source('source/_plot.R')


#============== Functions ===============

createModel = function(data) {
  set.seed(754)
  return(lm(SalePrice ~
              # OverallQual + #0.79098160
              # GarageCars + #0.64040920 #highly correlated with GarageArea
              # GarageArea + #0.62343144
              # TotalBsmtSF + #0.61358055 #highly correlated with X1stFlrSF
              # X1stFlrSF + #0.60585218
              # FullBath + #0.56066376
              # TotRmsAbvGrd + #0.53372316
              # YearBuilt + #0.52289733
              # YearRemodAdd + #0.50710097
              # MasVnrArea + #0.47261450
              # GarageYrBlt + #0.46905608
              # Fireplaces + #0.46692884
              GrLivArea, #0.70862448 #keep this one last since I know it will be in there
            data=data))
}

createPrediction = function(model, newData, verbose=F) {
  prediction = predict(model, newData, type='response')
  minY = min(prediction)
  if (minY <= 0) {
    amountToAdd = (-minY) + 1
    if (verbose) {
      cat('    Prediction included negative value: ', minY,'.  Adding ', amountToAdd, ' to all values.\n', sep='')
    }
    prediction = prediction + amountToAdd
  }
  return(prediction)
}

computeError = function(y, yhat) {
  return(rmse(log(y), log(yhat)))
}

#============= Main ================

#Globals
FILENAME = 'lm_GrLivArea'
PROD_RUN = T
Y_NAME = 'SalePrice'

source('source/_getData.R')
data = getData(Y_NAME)
train = data$train
test = data$test

#plot learning curve
plotLearningCurve(train, Y_NAME, createModel, createPrediction, computeError, save=PROD_RUN)

cat('Creating Linear Model...\n')
model = createModel(train)
cat('Getting train error...\n')
cat('    Train Error=', computeError(train$SalePrice, createPrediction(model, train, verbose=T)), '\n', sep='')

if (PROD_RUN) {
  #Output solution
  cat('Creating test prediction...\n')
  prediction = createPrediction(model, test, verbose=T)
  solution = data.frame(Id = test$Id, SalePrice = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  cat('Writing solution to file: ', outputFilename, '...\n', sep='')
  write.csv(solution, file=outputFilename, row.names=F)
}

cat('Done!\n')
