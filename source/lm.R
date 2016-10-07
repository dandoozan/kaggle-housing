#todo:
#D-select features based on correlation plot: OverallQual, GrLivArea, GarageArea
#D-plot learning curve
#D-make simple submission: lm_initial: Trn/CV errors=0.403522/0.7154108, Train error=0.4576414, Score=0.26634
#-handle negative values better
#-make new features from the interaction b/n highly-correlated features (eg. train$year_qual = train$YearBuilt*train$OverallQual)


#Remove all objects from the current workspace
rm(list = ls())
setwd('/Users/dan/Desktop/Kaggle/Housing')

library(caret) #createDataPartition
library(ggplot2) #visualization
library(ggthemes) # visualization
library(corrplot) #cor, cor rplot
library(car) #scatterplot
library(hydroGOF) #rmse


#============== Functions ===============

createLm = function(data) {
  set.seed(754)
  return(lm(SalePrice ~ OverallQual + GrLivArea + GarageArea, data=data))
}

createPrediction = function(model, newData, verbose=T) {
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

getError = function(y, yhat) {
  return(rmse(log(y), log(yhat)))
}

plotLearningCurve = function(data, save=FALSE) {
  cat('Plotting learning curve...\n')

  #split data into train and cv
  set.seed(837)
  partitionIndex = createDataPartition(data$SalePrice, p=0.8, list=FALSE)
  trn = data[partitionIndex,]
  cv = data[-partitionIndex,]

  incrementSize = 5
  increments = seq(incrementSize, nrow(trn), incrementSize)
  numIterations = length(increments)
  trnErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)

  count = 1
  for (i in increments) {
    if (i %% 100 == 0) cat('    On training example ', i, '\n', sep='')
    trnSubset = trn[1:i,]
    model = createLm(trnSubset)
    trnErrors[count] = getError(trnSubset$SalePrice, createPrediction(model, trnSubset, verbose=F))
    cvErrors[count] = getError(cv$SalePrice, createPrediction(model, cv, verbose=F))

    count = count + 1
  }

  #print final trn and cv errors
  model = createLm(trn)
  cat('    Final Train/CV Error=', getError(trn$SalePrice, createPrediction(model, trn, verbose=F)), '/', getError(cv$SalePrice, createPrediction(model, cv, verbose=F)), '\n', sep='')

  if (save) png(paste0('LearningCurve_', FILENAME, '.png'), width=500, height=350)
  plot(increments, trnErrors, col='blue', type='l', ylim = c(0, max(cvErrors)), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors, col='red')
  legend('topright', legend=c('train', 'cv'), fill=c('blue', 'red'), inset=.02, text.width=100)
  if (save) dev.off()
}

plotCorrelations = function(data) {
  correlations = cor(data[,names(Filter(is.numeric, data))])
  corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")
}

#I'm not sure what to use this for, but I've seen several other people
#use it, so maybe it'll come in handy in the future
plotScatterplotMatrix = function(data) {
  pairs(~YearBuilt+OverallQual+TotalBsmtSF+GrLivArea,
        data=data,
        main="Scatterplot Matrix")
}

plotScatterplot = function(data) {
  scatterplot(SalePrice ~ GrLivArea,
              data=data,
              #grid=FALSE,
              ylab="Sale Price")
}



#============= Main ================

#Globals
FILENAME = 'lm_initial'
PROD_RUN = T

source('source/_getData.R')
data = getData()
train = data$train
test = data$test

#plot correlations
#plotCorrelations(train)

#plot scatterplot matrix
#plotScatterplotMatrix(train)

#plot scatterplot
#plotScatterplot(train)

#plot learning curve
plotLearningCurve(train, save=PROD_RUN)


cat('Creating Linear Model...\n')
model = createLm(train)
cat('Getting train error...\n')
cat('    Train Error=', getError(train$SalePrice, createPrediction(model, train)), '\n', sep='')


if (PROD_RUN) {
  #Output solution
  cat('Creating test prediction...\n')
  prediction = createPrediction(model, test)
  solution = data.frame(Id = test$Id, SalePrice = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  cat('Writing solution to file: ', outputFilename, '...\n', sep='')
  write.csv(solution, file=outputFilename, row.names=F)
}

cat('Done!\n')
