#todo:
#D-Read: https://www.kaggle.com/skirmer/house-prices-advanced-regression-techniques/fun-with-real-estate-data/discussion
#D-Read: https://www.kaggle.com/notaapple/house-prices-advanced-regression-techniques/detailed-exploratory-data-analysis-using-r/discussion
#D-make simple submission: lm_initial: Trn/CV errors=0.403522/0.7154108, Train error=0.4576414, Score=0.26634
#D-Use 1 feature (GrLivArea): lm_GrLivArea: 0.2760749/0.2736812, 0.2755768, 0.28918
#D-Print significance using summary(lm(SalePrice ~ ., data=train))
#D-Use all significant features that don't cause a warning/error: lm_allSig: 0.142278/0.1758096, 0.1338661, 0.87067
#D-Use 2 features (GrLivArea, X1stFlrSF): lm_X1stFlrSF: 0.2563019/0.2487681, 0.2547064, 0.26421
#D-Add YearBuilt feature: lm_YearBuilt: 0.2231406/0.2048988, 0.2229374, 0.23230
#-Experiment with more features
#-Group OverallQual into low(1-4), med(5-7), high(8-10) so that it doesnt cause an error
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
              #These are from the correlation plot (they include numeric features only)
              #I.e. [Feature] + #[correlation]
              # OverallQual + #0.79098160 #<--in both
              # GarageCars + #0.64040920 #<--in both
              # GarageArea + #0.62343144 #<--in both
              # TotalBsmtSF + #0.61358055
              # X1stFlrSF + #0.60585218 #<--in both
              # FullBath + #0.56066376 #<--in both
              # TotRmsAbvGrd + #0.53372316 #<--in both
              # YearBuilt + #0.52289733 #<--in both
              # YearRemodAdd + #0.50710097 #<--in both
              # MasVnrArea + #0.47261450 #<--in both
              # GarageYrBlt + #0.46905608
              # Fireplaces + #0.46692884 #<--in both

              #These are from the summary(model) output (they include both numeric and factor features)
              #I.e. [Feature] + #[p-value] [Signif. code]
              # X2ndFlrSF + #< 2e-16 ***
              # RoofMatlCompShg + #< 2e-16 ***
              # RoofMatlMembran + #< 2e-16 ***
              # RoofMatlMetal + #< 2e-16 ***
              # RoofMatlRoll + #< 2e-16 ***
              # RoofMatlTar.Grv + #< 2e-16 ***
              # RoofMatlWdShake + #< 2e-16 ***
              # RoofMatlWdShngl + #< 2e-16 ***
              # Condition2PosN + #6.03e-15 ***
              X1stFlrSF + #2.05e-14 *** #<--in both
              # BsmtFinSF1 + #3.76e-13 ***
              # KitchenQualGd + #3.61e-11 ***
              # KitchenQualTA + #1.64e-08 ***
              # OverallCond + #4.88e-11 ***
              # OverallQual + #2.41e-10 *** #<--in both
              # LotArea + #3.37e-10 ***
              # BsmtQualGd + #2.84e-07 ***
              # NeighborhoodStoneBr + #3.75e-06 ***
              # BsmtExposureGd + #4.56e-06 ***
              # ExterQualGd + #1.11e-05 ***
              # ExterQualTA + #0.000118 ***
              # BsmtUnfSF + #1.24e-05 ***
              # GarageQualFa + #1.30e-05 ***
              # GarageQualGd + #3.65e-05 ***
              # GarageQualPo + #0.000293 ***
              # GarageQualTA + #2.64e-05 ***
              YearBuilt + #6.75e-05 *** #<--in both
              # PoolQCFa + #8.59e-05 ***
              # PoolQCGd + #0.000339 ***
              # Condition1Norm + #9.67e-05 ***
              # MasVnrArea + #0.000204 *** #<--in both
              # LandSlopeSev + #0.000260 ***
              # BsmtFinSF2 + #0.000338 ***
              # GarageCondTA + #0.000451 ***
              # GarageCondFa + #0.000667 ***
              # GarageCondGd,# + #0.000872 ***
              #
              # ScreenPorch + #0.003963 **
              # PoolArea + #0.004163 **
              # MSZoning + #0.004912 **
              # RoofStyle + #.007314 **
              # LotConfig + #0.007919 **
              # WoodDeckSF + #0.008967 **
              # Street + #0.009917 **
              #
              # Fireplaces + #0.014309 * #<--in both
              # Fence + #0.015386 *
              # BedroomAbvGr + #0.019804 *
              # SaleCondition + #0.021169 *
              # GarageArea + #0.021749 * #<--in both
              # Functional + #0.024499 *
              # #BsmtFinType1 + #0.026342 * #WARN: prediction from a rank-deficient fit may be misleading
              # #BsmtCond + #0.029894 * #WARN: prediction from a rank-deficient fit may be misleading
              # #GarageType + #0.048381 * #WARN: prediction from a rank-deficient fit may be misleading
              #
              # MoSold + #0.050353 .
              # YearRemodAdd + #0.065138 . #<--in both
              # Foundation + #0.070998 .
              # KitchenAbvGr + #0.075688 .
              # TotRmsAbvGrd + #0.076535 . #<--in both
              # FullBath + #0.080060 . #<--in both
              # BsmtFinType2 + #0.080396 .
              # Heating + #0.083424 .
              # GarageCars + #0.093282 . #<--in both
              # HouseStyle + #0.096509 .

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
Y_NAME = 'SalePrice'
FILENAME = 'lm_YearBuilt'
PROD_RUN = T

source('source/_getData.R')
data = getData(Y_NAME)
train = data$train
test = data$test

#one hot encode factors
#train = oneHotEncode(train)

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
