#todo:
#D-use all features: xgb_all: numSeedsTried=1, avgTrainError=2782.016, avgCvError=29970.59, bestSeed=266, nrounds=67, bestTrainError=2782.016, bestCvError=29970.59, score=0.16427
#-Predict log(y) instead of y: xgb_logy:
  #NumFeaturesUsed=?/?
  #1, 0.008649, 0.146446, 266, 103, 0.008649, 0.146446
  #Trn/CV Error=0.009038904/0.1433992, Train Error=0.01297175, Score=0.14155 <- New best!
#D-Use caret::dummyVars to one-hot-encode: xgb_dummyVars:
  #NumFeaturesUsed=295/295, 1, 0.008649, 0.146446, 266, 103, 0.008649, 0.146446
  #0.009038904/0.1433992, 0.01297175, 0.14155
#-Try one hot encoding train and test together
#-Recompute boruta now that im predicting log(y) instead of y
#-Tune earlyStopRound in findBestSeedAndNrounds
#-Tune hyperparams
#-Try Boruta features
#-Read (for finding best params): https://www.kaggle.com/jiashenliu/house-prices-advanced-regression-techniques/updated-xgboost-with-parameter-tuning/run/362252/comments#135758
#-Try dummyVars fullRank=F
#-Move print trn/cv, train errors to util function

#Remove all objects from the current workspace
rm(list = ls())
setwd('/Users/dan/Desktop/Kaggle/Housing')

library(xgboost) #xgb.train, xgb.cv
library(caret) #dummyVars
library(Ckmeans.1d.dp) #xgb.plot.importance
library(hydroGOF) #rmse
source('source/_getData.R')
source('source/_plot.R')
source('source/_util.R')

#================= Functions ===================

createModel = function(data, yName, xNames) {
  set.seed(SEED)
  return(xgb.train(data=getDMatrix(data, yName, xNames),
                    params=getHyperParams(),
                    nrounds=NROUNDS,
                    verbose=0))
}
createPrediction = function(model, newData, xNames) {
  return(exp(predict(model, newData[, xNames])))
}
computeError = function(y, yhat) {
  return(rmse(log(y), log(yhat)))
}
getHyperParams = function() {
  return(list(
    #range=[0,1], default=0.3, toTry=0.01,0.015,0.025,0.05,0.1
    #eta = 0.005, #learning rate. Lower value=less overfitting, but increase nrounds when lowering eta

    #range=[0,∞], default=0, toTry=?
    #gamma = 0, #Larger value=less overfitting

    #range=[1,∞], default=6, toTry=3,5,7,9,12,15,17,25
    #max_depth = 5, #Lower value=less overfitting

    #range=[0,∞], default=1, toTry=1,3,5,7
    #min_child_weight = 1, #Larger value=less overfitting

    #range=(0,1], default=1, toTry=0.6,0.7,0.8,0.9,1.0
    #subsample = 0.8, #ratio of sample of data to use for each instance (eg. 0.5=50% of data). Lower value=less overfitting

    #range=(0,1], default=1, toTry=0.6,0.7,0.8,0.9,1.0
    #colsample_bytree = 0.6, #ratio of cols (features) to use in each tree. Lower value=less overfitting

    #values=gbtree|gblinear|dart, default=gbtree, toTry=gbtree,gblinear
    #booster = 'gbtree', #gbtree/dart=tree based, gblinear=linear function. Remove eta when using gblinear

    objective = 'reg:linear'
  ))
}

plotCVErrorRates = function(data, yName, xNames, ylim=NULL, save=FALSE) {
  cat('Plotting CV error rates...\n')

  dataAsDMatrix = getDMatrix(data, yName, xNames)

  set.seed(SEED)
  cvRes = xgb.cv(data=dataAsDMatrix,
                 params=getHyperParams(),
                 nfold=5,
                 nrounds=(NROUNDS * 1.5), #times by 1.5 to plot a little extra
                 verbose=0)
  trainErrors = cvRes[[1]]
  cvErrors = cvRes[[3]]

  if (is.null(ylim)) {
    ylim = c(0, max(cvErrors, trainErrors))
  }

  if (save) png(paste0('ErrorRates_', FILENAME, '.png'), width=500, height=350)
  plot(trainErrors, type='l', col='blue', ylim=ylim, main='Train Error vs. CV Error', xlab='Num Rounds', ylab='Error')
  lines(cvErrors, col='red')
  legend(x='topright', legend=c('train', 'cv'), fill=c('blue', 'red'), inset=0.02, text.width=15)
  if (save) dev.off()
}

plotFeatureImportances = function(model, xNames, save=FALSE) {
  cat('Plotting feature importances...\n')

  importances = xgb.importance(feature_names=xNames, model=model)
  if (save) png(paste0('Importances_', FILENAME, '.png'), width=500, height=350)
  print(xgb.plot.importance(importance_matrix=importances))
  if (save) dev.off()
}

findBestSeedAndNrounds = function(data, yName, xNames, earlyStopRound=100, numSeedsToTry=1) {
  cat('Finding best seed and nrounds.  Trying ', numSeedsToTry, ' seeds...\n', sep='')

  dataAsDMatrix = getDMatrix(data, yName, xNames)

  initialNrounds = 1000
  maximize = FALSE
  bestSeed = 1
  bestNrounds = 0
  bestTrainError = Inf
  bestCvError = Inf
  trainErrors = numeric(numSeedsToTry)
  cvErrors = numeric(numSeedsToTry)
  set.seed(1) #set seed at the start here so that we generate the same following seeds every time
  for (i in 1:numSeedsToTry) {
    seed = sample(1:1000, 1)
    cat('    ', i, '. Seed ', seed, ': ', sep='')
    set.seed(seed)
    output = capture.output(cvRes <- xgb.cv(data=dataAsDMatrix,
                                            params=getHyperParams(),
                                            nfold=5,
                                            nrounds=initialNrounds,
                                            early.stop.round=earlyStopRound,
                                            maximize=maximize,
                                            verbose=0))
    nrounds = if (length(output) > 0) strtoi(substr(output, 27, nchar(output))) else initialNrounds
    trainErrors[i] = cvRes[[1]][nrounds] #mean train error
    cvErrors[i] = cvRes[[3]][nrounds] #mean test error
    cat('nrounds=', nrounds, ', trainError=', trainErrors[i], ', cvError=', cvErrors[i], sep='')
    if (cvErrors[i] < bestCvError) {
      bestSeed = seed
      bestNrounds = nrounds
      bestTrainError = trainErrors[i]
      bestCvError = cvErrors[i]
      cat(' <- New best!')
    }
    cat('\n')
  }

  cat('    Average errors: train=', mean(trainErrors), ', cv=', mean(cvErrors), '\n', sep='')
  cat('    Best seed=', bestSeed, ', nrounds=', bestNrounds, ', trainError=', bestTrainError, ', cvError=', bestCvError, '\n', sep='')

  return(list(seed=bestSeed, nrounds=bestNrounds))
}

findBestSetOfFeatures = function(data, possibleFeatures) {
  cat('Finding best set of features to use...\n')

  featuresToUse = possibleFeatures

  cat('    Number of features to use: ', length(featuresToUse), '/', length(possibleFeatures), '\n')
  return(featuresToUse)
}

getSparseMatrix = function(data) {
  dmy = caret::dummyVars('~.', data, fullRank=T)
  return(predict(dmy, data))
}

getDMatrix = function(data, yName, xNames) {
  set.seed(634)
  return(xgb.DMatrix(data=data[, xNames], label=log(data[, yName])))
}

#============= Main ================

#Globals
ID_NAME = 'Id'
Y_NAME = 'SalePrice'
FILENAME = 'xgb_dummyVars'
PROD_RUN = F
PLOT = 'cv' #cv=cv errors, lc=learning curve, fi=feature importances

data = getData(Y_NAME, oneHotEncode=F)
train = data$train
test = data$test

#todo: remove this and just use the oneHotEncode param in getData
#convert data to sparse matrix
train = getSparseMatrix(train)
test = getSparseMatrix(test)

possibleFeatures = setdiff(colnames(train), c(ID_NAME, Y_NAME))

#find best set of features to use based on cv error
featuresToUse = findBestSetOfFeatures(train, possibleFeatures)

#find best seed and nrounds
sn = findBestSeedAndNrounds(train, Y_NAME, featuresToUse, numSeedsToTry=1)
SEED = sn$seed
NROUNDS = sn$nrounds

#create model
cat('Creating Model...\n')
model = createModel(train, Y_NAME, featuresToUse)

#plots
if (PROD_RUN || PLOT=='cv') plotCVErrorRates(train, Y_NAME, featuresToUse, ylim=c(0, 0.2), save=PROD_RUN)
if (PROD_RUN || PLOT=='lc') plotLearningCurve(train, Y_NAME, featuresToUse, createModel, createPrediction, computeError, save=PROD_RUN)
if (PROD_RUN || PLOT=='fi') plotFeatureImportances(model, featuresToUse, save=PROD_RUN)

#print trn/cv, train error
printTrnCvTrainErrors(model, train, Y_NAME, featuresToUse, createModel, createPrediction, computeError)

if (PROD_RUN) outputSolution(createPrediction, model, test, ID_NAME, Y_NAME, featuresToUse, paste0(FILENAME, '.csv'))

cat('Done!\n')
