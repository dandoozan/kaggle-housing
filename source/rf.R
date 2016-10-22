#todo:
#Read: https://www.kaggle.com/endintears/house-prices-advanced-regression-techniques/simple-random-forest/code
#D-Use all features: rf_all: NumFeaturesUsed=80/80, Trn/Cv Error=0.06576678/0.1449745, Train Error=0.06394271, Score=0.15308
#D-Try using different values for ntree (nope: it made trn and cv errors worse)
#D-Use one hot encoded features (nope: it didn't make anything better: 0.06895515/0.1489397, 0.06599618)
#D-Try using top X features based on feature importances (nothin: it just makes higher trn and cv errors (although they are closer together sometimes))
#D-Remove Id as a feature: rf_-Id: 79/79, 0.06545684/0.1441382, 0.06373601, 0.15344
#D-Use Boruta confirmed features: rf_borutaConfirmed: 48, 0.06497725/0.1444829, 0.06406024, 0.15359
#D-Use Boruta confirmed+tentative features: rf_borutaConfirmedTentative: 60, 0.06428625/0.1440843, 0.06317321, 0.15280
#D-Predict log(y) instead of y: rf_logy: 60, 0.05691862/0.1380704, 0.05604239, 0.14729
#-Figure out why i'm overfitting


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
  #transform y -> log(y) because rf uses mse as internal error metric, and I'm wanting
  #ultimately to have the smallest msle
  data[[yName]] = log(data[[yName]])

  set.seed(754)
  return(randomForest(getFormula(yName, xNames),
                      data=data,
                      ntree=500))
}

createPrediction = function(model, newData) {
  #'exp' the prediction to convert it back to regular y since I'm using log(y) in the model
  return(exp(predict(model, newData)))
}

computeError = function(y, yhat) {
  return(rmse(log(y), log(yhat)))
}

#I do not understand any of this code, I borrowed it from a kaggler
plotImportances = function(model, save=FALSE) {
  cat('Plotting Feature Importances...\n')

  # Get importance
  importances = randomForest::importance(model)
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

findBestSetOfFeatures = function(data, possibleFeatures) {
  cat('Finding best set of features to use...\n')

  #boruta features
  borutaConfirmed = c('MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'LandContour', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', 'X1stFlrSF', 'X2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF')
  borutaTentative = c('Alley', 'LotShape', 'LandSlope', 'Condition1', 'RoofStyle', 'MasVnrType', 'BsmtCond', 'Electrical', 'EnclosedPorch', 'ScreenPorch', 'Fence', 'SaleCondition')
  borutaRejected = c('Street', 'Utilities', 'LotConfig', 'Condition2', 'RoofMatl', 'ExterCond', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtHalfBath', 'X3SsnPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType')

  featuresToUse = c(borutaConfirmed, borutaTentative)

  cat('    Number of features to use: ', length(featuresToUse), '/', length(possibleFeatures), '\n')
  #cat('    Features to use:', paste(featuresToUse, collapse=', '), '\n')
  return(featuresToUse)
}

#============= Main ================

#Globals
ID_NAME = 'Id'
Y_NAME = 'SalePrice'
FILENAME = 'rf_logy'
PROD_RUN = T
PLOT = 'lc' #lc=learning curve, fi=feature importances

data = getData(Y_NAME, oneHotEncode=F)
train = data$train
test = data$test
possibleFeatures = setdiff(names(train), c(ID_NAME, Y_NAME))

#find best set of features to use based on cv error
featuresToUse = findBestSetOfFeatures(train, possibleFeatures)

cat('Creating Random Forest...\n')
model = createModel(train, Y_NAME, featuresToUse)

#plots
if (PROD_RUN || PLOT=='lc') plotLearningCurve(train, Y_NAME, featuresToUse, createModel, createPrediction, computeError, increment=100, ylim=c(0, 0.3), save=PROD_RUN)
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
