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
#D-Add high-correlation features: lm_addCorr, 0.1551667/0.1333876, 0.1500612, 0.18435
#D-Find set of features to use automatically: lm_autoFeatures: numFeatures=39, 0.1656297/0.1459447, 0.1602873, 0.17735
#D-Use mice to impute missing values: lm_mice: 41, 0.1619088/0.1451876, 0.1562768, 0.17645
#D-Change trn/cv ratio to same as train/test (0.5 instead of 0.8): lm_ratio05: 34, 0.1689338/0.1658276, 0.1610147, 0.18286
#D-Change ratio back to 0.8: lm_ratio08: 41, 0.1619088/0.1451876, 0.1562768, 0.17645
#-Maybe make new features from interactions b/n highly-correlated features
#-Perhaps do multiple rounds in findBestSetOfFeatures
#-Try Kernel Ridge Regression, whatever that is
#-Try lasso
#-Try ridge
#-handle negative values better somehow


#Remove all objects from the current workspace
rm(list = ls())
setwd('/Users/dan/Desktop/Kaggle/Housing')


library(hydroGOF) #rmse
source('source/_getData.R')
source('source/_plot.R')
source('source/_util.R')


#============== Functions ===============

createModel = function(data, yName, xNames='.') {
  set.seed(754)
  return(lm(getFormula(yName, xNames), data=data))
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

#This finds the best set of features using the following technique:
#1. Get p-value of all features
#2. Start with the most significant feature (the one with the lowest p-value)
#3. Iterate through the rest of the features in order of increasing p-value
#4. For each feature, check if adding it results in a lower trnError, cvError, and trainError.
#5. If it lowers all 3 errors, then add it to the list of features to use, else, discard it
findBestSetOfFeatures = function(data, possibleFeatures, yName, createModel, createPrediction, computeError, verbose=T) {
  cat('Finding best set of features to use...\n')

  sortedPValues = sort(summary(createModel(data, yName, possibleFeatures))$coefficients[,4])
  naFeatureNames = setdiff(possibleFeatures, names(sortedPValues))

  featuresNamesToTry = c(names(Filter(function(x) x < 0.1, sortedPValues)), naFeatureNames)

  featuresToUse = c()
  prevTrnError = Inf
  prevCvError = Inf
  prevTrainError = Inf
  for (name in featuresNamesToTry) {
    if (verbose) cat(name, '\n')
    tempFeatureNames = c(featuresToUse, name)
    model = createModel(data, yName, tempFeatureNames)
    tempTrainError = tryCatch(computeError(data[[yName]], createPrediction(model, data)),
                              warning=function(w) w)
    if (!is(tempTrainError, 'warning')) {
      trainError = tempTrainError
      trnCvErrors = suppressWarnings(computeTrainCVErrors(data, yName, tempFeatureNames, createModel, createPrediction, computeError))
      trnError = trnCvErrors$train
      cvError = trnCvErrors$cv
      if (verbose) cat('   errors:', trnError, cvError, trainError)

      if (trnError < prevTrnError && cvError < prevCvError && trainError < prevTrainError) {
        #keep the new feature
        featuresToUse = c(featuresToUse, name)
        prevTrnError = trnError
        prevCvError = cvError
        prevTrainError = trainError
      } else {
        if (verbose) cat('         discarding')
      }
    } else {
      if (verbose) cat('    got warning')
    }
    if (verbose) cat('\n')
  }


  cat('    Number of features to use: ', length(featuresToUse), '/', length(possibleFeatures), '\n')
  cat('    Final Errors (Trn/CV, Train): ', prevTrnError, '/', prevCvError, ', ', prevTrainError, '\n', sep='')
  return(featuresToUse)
}

#============= Main ================

#Globals
ID_NAME = 'Id'
Y_NAME = 'SalePrice'
FILENAME = 'lm_ratio08'
PROD_RUN = F

data = getData(Y_NAME, oneHotEncode=T)
train = data$train
test = data$test
possibleFeatures = setdiff(names(train), c(ID_NAME, Y_NAME))

#find best set of features to use based on cv error
featuresToUse = findBestSetOfFeatures(train, possibleFeatures, Y_NAME, createModel, createPrediction, computeError, verbose=F)

#plot learning curve, and suppress those pesky "prediction from a rank-deficient fit may be misleading" warnings
suppressWarnings(plotLearningCurve(train, Y_NAME, featuresToUse, createModel, createPrediction, computeError, ylim=c(0, 0.4), save=PROD_RUN))

cat('Creating Linear Model...\n')
model = createModel(train, Y_NAME, featuresToUse)

#print trn/cv, train error
cat('Computing Errors...\n')
trnCvErrors = computeTrainCVErrors(train, Y_NAME, featuresToUse, createModel, createPrediction, computeError)
trnError = trnCvErrors$train
cvError = trnCvErrors$cv
trainError = computeError(train[[Y_NAME]], createPrediction(model, train, verbose=T))
cat('    Trn/CV, Train: ', trnError, '/', cvError, ', ', trainError, '\n', sep='')

if (PROD_RUN) outputSolution(model, test, ID_NAME, Y_NAME, paste0(FILENAME, '.csv'))

cat('Done!\n')
