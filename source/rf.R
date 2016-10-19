#todo:
#Read: https://www.kaggle.com/endintears/house-prices-advanced-regression-techniques/simple-random-forest/code
#-Use all features: rf_all: Trn/Cv Error=0.06576678/0.1449745, Train Error=0.06394271, Score=0.15308

#Remove all objects from the current workspace
rm(list = ls())
setwd('/Users/dan/Desktop/Kaggle/Housing')

library(randomForest) #randomForest
library(hydroGOF) #rmse
library(ggplot2) #visualization
library(ggthemes) # visualization
source('source/_getData.R')
source('source/_plot.R')
source('source/_util.R')

#============== Functions ===============

createModel = function(data, yName, xNames='.') {
  set.seed(754)
  return(randomForest(as.formula(paste(yName, '~', paste(xNames, collapse='+'))), data=data))
}

createPrediction = function(model, newData) {
  return(predict(model, newData))
}

computeError = function(y, yhat) {
  return(rmse(log(y), log(yhat)))
}

#I do not understand any of this code, I borrowed it from a kaggler
plotImportances = function(model, save=FALSE) {
  cat('Plotting Feature Importances...\n')

  # Get importance
  importances = importance(model)
  varImportance = data.frame(Variables = row.names(importances),
                             Importance = round(importances[, 1], 2))

  # Create a rank variable based on importance
  rankImportance = varImportance %>%
    mutate(Rank = paste0('#',dense_rank(desc(Importance))))

  if (save) png(paste0('Importances_', FILENAME, '.png'), width=500, height=350)
  print(ggplot(rankImportance, aes(x = reorder(Variables, Importance),
                                   y = Importance, fill = Importance)) +
          geom_bar(stat='identity') +
          geom_text(aes(x = Variables, y = 0.5, label = Rank),
                    hjust=0, vjust=0.55, size = 4, colour = 'red') +
          labs(title='Feature Importances', x='Features') +
          coord_flip() +
          theme_few())
  if (save) dev.off()
}

#This returns the top 10 features based on feature importances
findBestSetOfFeatures = function(data, yName, createModel) {
  cat('Finding best set of features to use...\n')

  featuresToUse = '.'

  # numTopFeatures = 10
  #
  # model = createModel(data, yName)
  # sortedImportances = sort(importance(model)[,1], decreasing=T)
  # featuresToUse = names(sortedImportances[1:numTopFeatures])

  cat('   Features to use:', paste(featuresToUse, collapse=', '), '\n')
  return(featuresToUse)
}

#============= Main ================

#Globals
ID_NAME = 'Id'
Y_NAME = 'SalePrice'
FILENAME = 'rf_all'
PROD_RUN = T
PLOT = 'lc' #lc=learning curve, fi=feature importances

data = getData(Y_NAME, oneHotEncode=F)
train = data$train
test = data$test

# #find best set of features to use based on cv error
featuresToUse = findBestSetOfFeatures(train, Y_NAME, createModel)

cat('Creating Random Forest...\n')
model = createModel(train, Y_NAME, featuresToUse)

#plots
if (PROD_RUN || PLOT=='lc') plotLearningCurve(train, Y_NAME, featuresToUse, createModel, createPrediction, computeError, increment=100, save=PROD_RUN)
if (PROD_RUN || PLOT=='fi') plotImportances(model, save=PROD_RUN)

#print trn/cv, train error
cat('Computing Errors...\n')
trnCvErrors = computeTrainCVErrors(train, Y_NAME, featuresToUse, createModel, createPrediction, computeError)
trnError = trnCvErrors$train
cvError = trnCvErrors$cv
trainError = computeError(train[[Y_NAME]], createPrediction(model, train))
cat('    Trn/CV, Train: ', trnError, '/', cvError, ', ', trainError, '\n', sep='')

if (PROD_RUN) outputSolution(model, test, ID_NAME, Y_NAME, paste0(FILENAME, '.csv'))

cat('Done!\n')
