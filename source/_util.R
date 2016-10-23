#Remove cols from a dataframe
removeCols = function(data, colNames) {
  colsToKeep = setdiff(names(data), colNames)
  return(data[, colsToKeep])
}

splitData = function(data, yName) {
  require(caret) #createDataPartition

  #split data into train and cv
  set.seed(837)
  partitionIndices = caret::createDataPartition(data[, yName], p=0.8, list=FALSE)
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
  trainError = computeError(train[, yName], createPrediction(model, train, xNames))
  cvError = computeError(cv[, yName], createPrediction(model, cv, xNames))

  return(list(train=trainError, cv=cvError))
}

outputSolution = function(createPrediction, model, testData, idName, yName, xNames, filename) {
  cat('Outputing solution...\n')
  cat('    Creating prediction...\n')
  prediction = createPrediction(model, testData, xNames)
  solution = data.frame(testData[, idName], prediction)
  colnames(solution) = c(idName, yName)
  cat('    Writing solution to file: ', filename, '...\n', sep='')
  write.csv(solution, file=filename, row.names=F)
}

getFormula = function(yName, xNames) {
  return(as.formula(paste(yName, '~', paste(xNames, collapse='+'))))
}
