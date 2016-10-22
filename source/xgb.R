#todo:
#D-use all features: xgb_all: numSeedsTried=1, avgTrainError=2782.016, avgCvError=29970.59, bestSeed=266, nrounds=67, bestTrainError=2782.016, bestCvError=29970.59, score=0.16427
#-Predict log(y) instead of y:
  #1, 0.008649, 0.146446, 266, 103, 0.008649, 0.146446
  #Trn/CV Error=0.009038904/0.1433992, Train Error=0.01297175, Score=0.14155 <- New best!
#-Try one hot encode using dummyVars
#-Try one hot encoding train and test together
#-Tune earlyStopRound in findBestSeedAndNrounds
#-Tune hyperparams
#-Try Boruta features
#-Read (for finding best params): https://www.kaggle.com/jiashenliu/house-prices-advanced-regression-techniques/updated-xgboost-with-parameter-tuning/run/362252/comments#135758

#Remove all objects from the current workspace
rm(list = ls())
setwd('/Users/dan/Desktop/Kaggle/Housing')

library(xgboost)
library(Matrix) #sparse.model.matrix
library(caret) #createDataPartition
library(Ckmeans.1d.dp) #xgb.plot.importance
library(hydroGOF) #rmse
source('source/_getData.R')
source('source/_util.R')

#================= Functions ===================

createModel = function(data, yName, xNames, params, nrounds, seed) {
  set.seed(seed)
  return(xgb.train(data=getDMatrix(data, yName, xNames),
                    params=params,
                    nrounds=nrounds,
                    verbose=0))
}
createPrediction = function(model, newData, xNames) {
  return(exp(predict(model, getSparseMatrix(newData, xNames))))
}
computeError = function(y, yhat) {
  return(rmse(log(y), log(yhat)))
}

plotCVErrorRates = function(data, yName, xNames, params, nrounds, seed, ylim=NULL, save=FALSE) {
  cat('Plotting CV error rates...\n')

  dataAsDMatrix = getDMatrix(data, yName, xNames)

  set.seed(seed)
  cvRes = xgb.cv(data=dataAsDMatrix,
                 params=params,
                 nfold=5,
                 nrounds=(nrounds * 1.5), #times by 1.5 to plot a little extra
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

plotLearningCurve = function(data, yName, xNames, params, nrounds, seed, save=FALSE) {
  cat('Plotting learning curve...\n')

  #split data into train and cv
  split = splitData(data, yName)
  train = split$train
  cv = split$cv

  incrementSize = 5
  increments = seq(incrementSize, nrow(train), incrementSize)
  numIterations = length(increments)
  trainErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)

  count = 1
  for (i in increments) {
    if (i %% 100 == 0) cat('    On training example', i, '\n')
    trainSubset = train[1:i,]

    model = createModel(trainSubset, yName, xNames, params, nrounds, seed)

    trainErrors[count] = computeError(trainSubset[[yName]], createPrediction(model, trainSubset, xNames))
    cvErrors[count] = computeError(cv[[yName]], createPrediction(model, cv, xNames))

    count = count + 1
  }

  if (save) png(paste0('LearningCurve_', FILENAME, '.png'), width=500, height=350)
  plot(increments, trainErrors, col='blue', type='l', ylim = c(0, max(cvErrors)), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors, col='red')
  legend('topright', legend=c('train', 'cv'), fill=c('blue', 'red'), inset=.02, text.width=100)
  if (save) dev.off()
}

plotFeatureImportances = function(model, data, xNames, save=FALSE) {
  cat('Plotting feature importances...\n')

  dataAsSparseMatrix = getSparseMatrix(data, xNames)
  importances = xgb.importance(feature_names=dataAsSparseMatrix@Dimnames[[2]], model=model)
  if (save) png(paste0('Importances_', FILENAME, '.png'), width=500, height=350)
  print(xgb.plot.importance(importance_matrix=importances))
  if (save) dev.off()
}

findBestSeedAndNrounds = function(data, yName, xNames, params, earlyStopRound=100, numSeedsToTry=1) {
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
                                            params=params,
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

computeTrainCVErrors = function(data, yName, xNames, params=NULL, nrounds=NULL, seed=NULL) {
  #split data into train and cv
  split = splitData(data, yName)
  train = split$train
  cv = split$cv

  model = createModel(train, yName, xNames, params, nrounds, seed)

  trainError = computeError(train[[yName]], createPrediction(model, train, xNames))
  cvError = computeError(cv[[yName]], createPrediction(model, cv, xNames))

  return(list(train=trainError, cv=cvError))
}

getSparseMatrix = function(data, xNames) {
  return(sparse.model.matrix(~., data=data[xNames]))
}

getDMatrix = function(data, yName, xNames) {
  set.seed(634)
  return(xgb.DMatrix(data=getSparseMatrix(data, xNames), label=log(data[[yName]])))
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

#============= Main ================

#Globals
ID_NAME = 'Id'
Y_NAME = 'SalePrice'
FILENAME = 'xgb_logy'
PROD_RUN = T
PLOT = 'fi' #cv=cv errors, lc=learning curve, fi=feature importances

data = getData(Y_NAME)
train = data$train
test = data$test
possibleFeatures = setdiff(names(train), c(ID_NAME, Y_NAME))

#find best set of features to use based on cv error
featuresToUse = findBestSetOfFeatures(train, possibleFeatures)

#set hyper params
params = getHyperParams()

#find best seed and nrounds
sn = findBestSeedAndNrounds(train, Y_NAME, featuresToUse, params, numSeedsToTry=1)
seed = sn$seed
nrounds = sn$nrounds

#create model
cat('Creating Model...\n')
model = createModel(train, Y_NAME, featuresToUse, params, nrounds, seed)

#plots
if (PROD_RUN || PLOT=='cv') plotCVErrorRates(train, Y_NAME, featuresToUse, params, nrounds, seed, ylim=c(0, 0.2), save=PROD_RUN)
if (PROD_RUN || PLOT=='lc') plotLearningCurve(train, Y_NAME, featuresToUse, params, nrounds, seed, save=PROD_RUN)
if (PROD_RUN || PLOT=='fi') plotFeatureImportances(model, train, featuresToUse, save=PROD_RUN)

#print trn/cv, train error
cat('Computing Errors...\n')
trnCvErrors = computeTrainCVErrors(train, Y_NAME, featuresToUse, params, nrounds, seed)
trnError = trnCvErrors$train
cvError = trnCvErrors$cv
trainError = computeError(train[[Y_NAME]], createPrediction(model, train, featuresToUse))
cat('    Trn/CV, Train: ', trnError, '/', cvError, ', ', trainError, '\n', sep='')

if (PROD_RUN) {
  #Output solution
  cat('Outputing solution...\n')
  cat('    Creating prediction...\n')
  prediction = createPrediction(model, test, featuresToUse)
  solution = data.frame(test[[ID_NAME]], prediction)
  colnames(solution) = c(ID_NAME, Y_NAME)
  outputFilename = paste0(FILENAME, '.csv')
  cat('    Writing solution to file: ', outputFilename, '...\n', sep='')
  write.csv(solution, file=outputFilename, row.names=F)
}

cat('Done!\n')
