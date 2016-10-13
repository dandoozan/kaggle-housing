splitData = function(data, y) {
  require(caret) #createDataPartition

  #split data into train and cv
  set.seed(837)
  partitionIndices = caret::createDataPartition(data[[y]], p=0.8, list=FALSE)
  train = data[partitionIndices,]
  cv = data[-partitionIndices,]
  return(list(train=train, cv=cv))
}

computeTrainCVErrors = function(data, y, createModel, createPrediction, computeError) {
  #split data into train, cv
  split = splitData(data, y)
  train = split$train
  cv = split$cv

  #compute train and cv errors
  model = createModel(train)
  trainError = computeError(train[[y]], createPrediction(model, train))
  cvError = computeError(cv[[y]], createPrediction(model, cv))

  return(list(train=trainError, cv=cvError))
}