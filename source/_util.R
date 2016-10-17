#Remove cols from a dataframe
removeCols = function(data, colNames) {
  colsToKeep = setdiff(names(data), colNames)
  return(data[, colsToKeep])
}

#Get the names of X columns
getXNames = function(data, yName) {
  allNames = names(data)
  return(allNames[allNames != yName])
}


splitData = function(data, yName) {
  require(caret) #createDataPartition

  #split data into train and cv
  set.seed(837)
  partitionIndices = caret::createDataPartition(data[[yName]], p=0.8, list=FALSE)
  train = data[partitionIndices,]
  cv = data[-partitionIndices,]
  return(list(train=train, cv=cv))
}

computeTrainCVErrors = function(data, yName, xNames, createModel, createPrediction, computeError) {
  #split data into train, cv
  split = splitData(data, yName)
  train = split$train
  cv = split$cv

  #compute train and cv errors
  model = createModel(train, yName, xNames)
  trainError = computeError(train[[yName]], createPrediction(model, train))
  cvError = computeError(cv[[yName]], createPrediction(model, cv))

  return(list(train=trainError, cv=cvError))
}

findBestSetOfFeatures = function(data, yName, createModel, createPrediction, computeError, verbose=T) {
  cat('Finding best set of features to use...\n')

  sortedPValues = sort(summary(createModel(data, yName))$coefficients[,4])
  allFeatureNames = getXNames(data, yName)
  naFeatureNames = setdiff(allFeatureNames, names(sortedPValues))

  featuresNamesToTry = c(names(Filter(function(x) x < 0.1, sortedPValues)), naFeatureNames)

  featuresToUse = c()
  prevTrnError = Inf
  prevCvError = Inf
  prevTrainError = Inf
  for (name in featuresNamesToTry) {
    if (verbose) cat(name, '\n')
    tempFeatureNames = c(featuresToUse, name)
    trnCvErrors = suppressWarnings(computeTrainCVErrors(data, yName, tempFeatureNames, createModel, createPrediction, computeError))
    trnError = trnCvErrors$train
    cvError = trnCvErrors$cv

    model = createModel(data, yName, tempFeatureNames)
    tempTrainError = tryCatch(computeError(data[[yName]], createPrediction(model, data)),
                          warning=function(w) w)
    if (!is(tempTrainError, 'warning')) {
      trainError = tempTrainError
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
      if (verbose) cat('       got warning')
    }
    if (verbose) cat('\n')
  }


  cat('    Number of features to use: ', length(featuresToUse), '/', length(allFeatureNames), '\n')
  cat('    Final Errors (Trn/CV, Train): ', trnError, '/', cvError, ', ', trainError, '\n', sep='')
  return(featuresToUse)
}