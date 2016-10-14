#todo:
#D-Read: https://www.kaggle.com/skirmer/house-prices-advanced-regression-techniques/fun-with-real-estate-data/discussion
#D-Read: https://www.kaggle.com/notaapple/house-prices-advanced-regression-techniques/detailed-exploratory-data-analysis-using-r/discussion
#D-make simple submission: lm_initial: Trn/CV errors=0.403522/0.7154108, Train error=0.4576414, Score=0.26634
#D-Use 1 feature (GrLivArea): lm_GrLivArea: 0.2760749/0.2736812, 0.2755768, 0.28918
#D-Print significance using summary(lm(SalePrice ~ ., data=train))
#D-Use all significant features that don't cause a warning/error: lm_allSig: 0.142278/0.1758096, 0.1338661, 0.87067
#D-Use 2 features (GrLivArea, X1stFlrSF): lm_X1stFlrSF: 0.2563019/0.2487681, 0.2547064, 0.26421
#D-Add YearBuilt feature: lm_YearBuilt: 0.2231406/0.2048988, 0.2229374, 0.23230
#D-Add MasVnrArea feature: lm_MasVnrArea: 0.2186844/0.2048588, 0.2188392, 0.22588
#D-Add sig numeric cols that lower cvError: lm_sigNum: 0.1855814/0.1673258, 0.1826243, 0.18767
#D-Add sig one-hot-encoded factors: lm_sigFac: 0.1806693/0.1611016, 0.1776368, 0.18913
#D-Add OverallQual as factor: lm_+OverallQual: 0.1671927/0.1550466, 0.1625823, 0.17642
#-Experiment with more features
#-Read more forum posts
#-handle negative values better somehow
#-make new features from the interaction b/n highly-correlated features (eg. train$year_qual = train$YearBuilt*train$OverallQual)


#Remove all objects from the current workspace
rm(list = ls())
setwd('/Users/dan/Desktop/Kaggle/Housing')


library(hydroGOF) #rmse
source('source/_plot.R')
source('source/_util.R')


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
              X2ndFlrSF + #< 2e-16 ***
              # RoofMatlCompShg + #< 2e-16 ***
              # RoofMatlMembran + #< 2e-16 ***
              # RoofMatlMetal + #< 2e-16 ***
              # RoofMatlRoll + #< 2e-16 ***
              # RoofMatlTar.Grv + #< 2e-16 ***
              # RoofMatlWdShake + #< 2e-16 ***
              # RoofMatlWdShngl + #< 2e-16 ***
              # Condition2PosN + #6.03e-15 ***
              X1stFlrSF + #2.05e-14 *** #<--in both
              BsmtFinSF1 + #3.76e-13 ***
              # KitchenQualGd + #3.61e-11 ***
              # KitchenQualTA + #1.64e-08 ***
              # OverallCond + #4.88e-11 ***
              OverallQual10 + #0.000194 *** #<--in both
              LotArea + #3.37e-10 ***
              # BsmtQualGd + #2.84e-07 ***
              NeighborhoodStoneBr + #3.75e-06 ***
              # BsmtExposureGd + #4.56e-06 ***
              ExterQualGd + #1.11e-05 ***
              # ExterQualTA + #0.000118 ***
              BsmtUnfSF + #1.24e-05 ***
              # GarageQualFa + #1.30e-05 ***
              # GarageQualGd + #3.65e-05 ***
              # GarageQualPo + #0.000293 ***
              GarageQualTA + #2.64e-05 ***
              YearBuilt + #6.75e-05 *** #<--in both
              # PoolQCFa + #8.59e-05 ***
              # PoolQCGd + #0.000339 ***
              # Condition1Norm + #9.67e-05 ***
              MasVnrArea + #0.000204 *** #<--in both
              LandSlopeSev + #0.000260 ***
              BsmtFinSF2 + #0.000338 ***
              # GarageCondTA + #0.000451 ***
              # GarageCondFa + #0.000667 ***
              # GarageCondGd + #0.000872 ***
              #
              ScreenPorch + #0.003963 **
              # PoolArea + #0.004163 **
              MSZoningFV + #0.004912 **
              # RoofStyleShed + #.007314 **
              LotConfigCulDSac + #0.007919 **
              WoodDeckSF + #0.008967 **
              StreetPave + #0.009917 **

              OverallQual9 + #0.012977 *
              Fireplaces + #0.014309 * #<--in both
              # FenceNA + #0.015386 *
              # FenceMnPrv + #0.016233 *
              BedroomAbvGr + #0.019804 *
              # SaleConditionNormal + #0.021169 *
              # GarageArea + #0.021749 * #<--in both
              # FunctionalTyp + #0.024499 *
              BsmtFinType1GLQ + #0.026342 *
              # BsmtCondPo + #0.029894 *
              GarageTypeDetchd + #0.048381 *

              MoSold + #0.050353 .
              # BsmtFinType2LwQ + #0.064031 .
              YearRemodAdd + #0.065138 . #<--in both
              FoundationWood + #0.070998 .
              KitchenAbvGr + #0.075688 .
              # TotRmsAbvGrd + #0.076535 . #<--in both
              FullBath + #0.080060 . #<--in both
              BsmtFinType2BLQ + #0.080396 .
              # HeatingQCGd + #0.083424 .
              # GarageCars + #0.093282 . #<--in both
              HouseStyle2Story + #0.096509 .

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
FILENAME = 'lm_+OverallQual'
PROD_RUN = T

source('source/_getData.R')
data = getData(Y_NAME)
train = data$train
test = data$test

#one hot encode factors
train = oneHotEncode(train)
test = oneHotEncode(test)

#plot learning curve, and suppress those pesky "prediction from a rank-deficient fit may be misleading" warnings
suppressWarnings(plotLearningCurve(train, Y_NAME, createModel, createPrediction, computeError, ylim=c(0, 0.4), save=PROD_RUN))

cat('Creating Linear Model...\n')
model = createModel(train)

#print trn/cv, train error
cat('Computing Errors...\n')
trnCvErrors = computeTrainCVErrors(train, Y_NAME, createModel, createPrediction, computeError)
trnError = trnCvErrors$train
cvError = trnCvErrors$cv
trainError = computeError(train[[Y_NAME]], createPrediction(model, train, verbose=T))
cat('    Trn/CV, Train: ', trnError, '/', cvError, ', ', trainError, '\n', sep='')

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
