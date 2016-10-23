#This is a utility file to hold functions that plot many different things


#Plot learning curve
#E.g. plotLearningCurve(train, 'SalePrice', createModel, createPrediction, computeError, save=PROD_RUN)
plotLearningCurve = function(data, yName, xNames, createModel, createPrediction, computeError, increment=5, startIndex=increment, ylim=NULL, save=FALSE) {
  cat('Plotting learning curve...\n')

  #split data into train and cv
  split = splitData(data, yName)
  train = split$train
  cv = split$cv

  increments = seq(startIndex, nrow(train), increment)
  numIterations = length(increments)
  trainErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)

  count = 1
  found=F #tbx
  for (i in increments) {
    if (i %% 200 == 0) cat('    On training example ', i, '\n', sep='')
    trainSubset = train[1:i,]

    #tbx
    #print(names(Filter(function(x) (is.factor(x) && length(levels(x)) < 2), trainSubset)))

    model = createModel(trainSubset, yName, xNames)
    trainErrors[count] = computeError(trainSubset[, yName], createPrediction(model, trainSubset, xNames))
    cvErrors[count] = computeError(cv[, yName], createPrediction(model, cv, xNames))

    #tbx
    # if (cvErrors[count] > 0.5 && !found) {
    #   cat('i=', i, ', count=', count, 'error=', cvErrors[count], '\n')
    #   found = T
    # }

    count = count + 1
  }

  if (is.null(ylim)) {
    ylim = c(0, max(cvErrors, trainErrors))
  }

  if (save) png(paste0('LearningCurve_', FILENAME, '.png'), width=500, height=350)
  plot(increments, trainErrors, col='blue', type='l', ylim=ylim, main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors, col='red')
  legend('topright', legend=c('train', 'cv'), fill=c('blue', 'red'), inset=.02, text.width=150)
  if (save) dev.off()
}



#This will plot a histogram of all the cols in data, so be careful
#to pass in data that only contains the columns you want plotted
#Note: All cols must be factors, and the number of cols should be <= 8
#E.g. plotTrainAndTestHistograms(Filter(is.factor, train), Filter(is.factor, test))
#E.g. plotTrainAndTestHistograms(Filter(is.factor, train), Filter(is.factor, test), 5)
#E.g. plotTrainAndTestHistograms(Filter(is.factor, train), Filter(is.factor, test), 10, 1)
plotTrainAndTestHistograms = function(trainData, testData, startIndex=1, numColsToPlot=4, numGridCols=2) {
  numCols = ncol(trainData)
  numTestCols = ncol(testData)

  #error checking
  if (numCols != numTestCols) stop(paste('Train and test data have a different number of columns (test has', numCols, 'columns, test has', numTestCols, 'columns). They must be the same.'))
  if (numCols != ncol(Filter(is.factor, trainData))) stop('Train contains non-factor columns.  All columns must be factors.')
  if (numCols != ncol(Filter(is.factor, testData))) stop('Test contains non-factor columns.  All columns must be factors.')

  require(gridExtra) #grid.arrange
  plots = list()
  endIndex = startIndex + numColsToPlot - 1
  for (i in startIndex:endIndex) {
    #add train plot
    trainDf = data.frame(x=trainData[[i]])
    trainPlot = ggplot(data=trainDf, aes(x=factor(x))) + stat_count() + xlab(colnames(trainData)[i]) + theme_light() +
      theme(axis.text.x = element_text(angle = 90, hjust =1))

    #add test plot
    testDf = data.frame(x=testData[[i]])
    testPlot = ggplot(data=testDf, aes(x=factor(x))) + stat_count() + xlab(colnames(testData)[i]) + theme_light() +
      theme(axis.text.x = element_text(angle = 90, hjust =1))
    plots = c(plots, list(trainPlot, testPlot))
  }

  do.call('grid.arrange', c(plots, ncol=numGridCols))
}

#Plot the correlations between all numeric cols in data
#E.g. plotCorrelations(train, 'SalePrice')
plotCorrelations = function(data, yName) {
  require(corrplot) #cor, corrplot
  correlationsMatrix = cor(data[,names(Filter(is.numeric, data))])
  sortedCorrelations = sort(correlationsMatrix[yName, ], decreasing=T)
  cat('Correlations with ', yName, ' (in order):\n', sep='')
  print(stack(sortedCorrelations))
  corrplot(correlationsMatrix)
}