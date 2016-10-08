#This is a utility file to hold functions that plot many different things


#Plot learning curve
#E.g. plotLearningCurve(train, 'SalePrice', createModel, createPrediction, computeError, save=PROD_RUN)
plotLearningCurve = function(data, y, createModel, createPrediction, computeError, ylim=NULL, save=FALSE) {
  cat('Plotting learning curve...\n')

  #split data into train and cv
  set.seed(837)
  partitionIndices = createDataPartition(data[[y]], p=0.8, list=FALSE)
  train = data[partitionIndices,]
  cv = data[-partitionIndices,]

  incrementSize = 5
  increments = seq(incrementSize, nrow(train), incrementSize)
  numIterations = length(increments)
  trainErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)

  count = 1
  found=F #tbx
  for (i in increments) {
    if (i %% 100 == 0) cat('    On training example ', i, '\n', sep='')
    trainSubset = train[1:i,]
    model = createModel(trainSubset)
    trainErrors[count] = computeError(trainSubset[[y]], createPrediction(model, trainSubset))

    #tbx
    if (trainErrors[count] > 0.5 && !found) {
      cat('i=', i, ', count=', count, 'error=', trainErrors[count], '\n')
      found = T
    }

    cvErrors[count] = computeError(cv[[y]], createPrediction(model, cv))

    count = count + 1
  }

  #print final train and cv errors
  model = createModel(train)
  cat('    Final Train/CV Error=', computeError(train[[y]], createPrediction(model, train)), '/', computeError(cv[[y]], createPrediction(model, cv)), '\n', sep='')

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
#E.g. plotAllHistograms(Filter(is.factor, train)[, 3:6])
#E.g. plotAllHistograms(Filter(is.factor, train)[, 1:9], 3)
plotAllHistograms = function(data, ncol=2) {
  maxCols = 9
  numCols = ncol(data)
  numFactorCols = ncol(Filter(is.factor, data))

  #error checking
  if (numCols != numFactorCols) stop('Data contains non-factor columns.  All columns must be factors.')
  if (numCols > maxCols) stop(paste('Data has', numCols, 'columns.  It should have <=', maxCols, 'columns'))

  require(gridExtra) #grid.arrange
  plots = list()
  for (i in 1:numCols) {
    df = data.frame(x=data[[i]])
    plot = ggplot(data=df, aes(x=factor(x))) + stat_count() + xlab(colnames(data)[i]) + theme_light() +
      theme(axis.text.x = element_text(angle = 90, hjust =1))
    plots = c(plots, list(plot))
  }
  do.call('grid.arrange', c(plots, ncol=ncol))
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