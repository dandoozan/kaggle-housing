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

outputSolution = function(model, testData, idName, yName, filename) {
  cat('Outputing solution...\n')
  cat('    Creating prediction...\n')
  prediction = createPrediction(model, testData)
  solution = data.frame(testData[[idName]], prediction)
  colnames(solution) = c(idName, yName)
  cat('    Writing solution to file: ', filename, '...\n', sep='')
  write.csv(solution, file=filename, row.names=F)
}
