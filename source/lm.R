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
#D-Make another pass adding sig features: lm_addSig: 0.1659369/0.1486695, 0.1619065, 0.17769
#D-Made 2 passes with new sig rankings: lm_newSig: 0.1543123/0.1343682, 0.1496322, 0.18712
#-Maybe convert year features (eg. YearBuilt) to factors
#-Maybe convert count features (eg. FullBath) to factors
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
              #RoofMatlCompShg + #2e-16 ***
              #RoofMatlMembran + #2e-16 ***
              #RoofMatlMetal + #2e-16 ***
              #RoofMatlRoll + #2e-16 ***
              #RoofMatlTar.Grv + #2e-16 ***
              #RoofMatlWdShake + #2e-16 ***
              #RoofMatlWdShngl + #2e-16 ***
              X2ndFlrSF + #2e-16 ***
              X1stFlrSF + #5.8e-15 ***
              BsmtFinSF1 + #1.35e-12 ***
              LotArea + #1.56e-09 ***
              KitchenQualGd + #1.92e-06 ***
              YearBuilt + #1.52e-05 ***
              BsmtUnfSF + #2.42e-05 ***
              KitchenQualTA + #2.57e-05 ***
              BsmtExposureGd + #4.46e-05 ***
              NeighborhoodStoneBr + #0.000105 ***
              OverallQual10 + #0.000194 ***
              BsmtFinSF2 + #0.000212 ***
              Condition1Norm + #0.000222 ***
              ScreenPorch + #0.000412 ***
              LandSlopeSev + #0.00049 ***
              MSZoningFV + #0.000614 ***
              #MSZoningRL + #0.000622 ***
              BsmtQualGd + #0.000803 ***
              #MSZoningRM + #0.000813 ***
              #PoolQCFa + #0.000921 ***
              #MasVnrArea + #0.002084 **
              RoofStyleShed + #0.005024 **
              MSZoningRH + #0.005106 **
              #PoolQCGd + #0.005459 **
              #NeighborhoodEdwards + #0.006674 **
              GarageArea + #0.008497 **
              #PoolArea + #0.009049 **
              LotConfigCulDSac + #0.009108 **
              KitchenQualFa + #0.011639 *
              Fireplaces + #0.011664 *
              OverallQual9 + #0.012977 *
              SaleConditionNormal + #0.015183 *
              FoundationWood + #0.017417 *
              GarageTypeDetchd + #0.019292 *
              NeighborhoodNoRidge + #0.021048 *
              NeighborhoodMitchel + #0.023648 *
              FireplaceQuNA + #0.025691 *
              #GarageTypeBasment + #0.027843 *
              KitchenAbvGr + #0.028201 *
              BsmtFinType1GLQ + #0.030652 *
              #GarageTypeAttchd + #0.03158 *
              X3SsnPorch + #0.032375 *
              StreetPave + #0.036155 *
              FunctionalTyp + #0.036698 *
              NeighborhoodNAmes + #0.037629 *
              #LandSlopeMod + #0.039676 *
              MoSold + #0.039938 *
              #FireplaceQuPo + #0.040948 *
              BsmtQualTA + #0.041567 *
              #GarageQualFa + #0.042238 *
              GarageTypeBuiltIn + #0.043626 *
              NeighborhoodNWAmes + #0.044775 *
              Condition1RRAn + #0.050091 .
              WoodDeckSF + #0.053896 .
              #Condition2RRAe + #0.055468 . #ERROR: object 'Condition2RRAe' not found
              BsmtExposureNo + #0.05789 .
              GarageQualTA + #0.060856 .
              FullBath + #0.065037 .
              GarageQualGd + #0.065551 .
              GarageQualPo + #0.066676 .
              #GarageTypeCarPort + #0.067736 .
              #LandContourLvl + #0.072964 .
              LotShapeIR2 + #0.075697 .
              #PoolQCNA + #0.07675 .
              #NeighborhoodNridgHt + #0.077902 .
              FireplaceQuTA + #0.078805 .
              LotConfigFR2 + #0.079863 .
              #UtilitiesNoSeWa + #0.080384 .
              HouseStyle2Story + #0.082891 .
              Condition1PosN + #0.083065 .
              BsmtFinType2BLQ + #0.092605 .
              FireplaceQuGd + #0.092895 .
              FenceNA + #0.094496 .
              FenceMnPrv + #0.095795 .
              YearRemodAdd + #0.096592 .
              BldgTypeDuplex + #NA
              BsmtCondNA + #NA
              #BsmtFinType1NA + #NA #WARN: prediction from a rank-deficient fit may be misleading
              #ElectricalMix + #NA #ERROR: object 'ElectricalMix' not found
              Exterior2ndCBlock + #NA
              #GarageCondNA + #NA
              #GarageFinishNA + #NA
              #GarageQualNA + #NA
              GrLivArea + #NA
              #TotalBsmtSF + #NA #WARN: prediction from a rank-deficient fit may be misleading

              Condition2PosN,# + #2e-16 *** #the first shall be last
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
FILENAME = 'lm_newSig'
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
