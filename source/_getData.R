#todo:
#D-fill NAs using roughfix
#D-combine rare values in all factor cols
#D-change NAs to -1s for GarageYrBlt before roughfix b/c the NAs represent no garage
#-convert cols that have ratings as strings (eg. "Ex"/"Gd"/"TA"/"Fa"/"Po") to ints (5:1)

setwd('/Users/dan/Desktop/Kaggle/Housing')

library(dplyr) #bind_rows
library(randomForest) #na.roughfix
library(caret) #dummyVars

#cols=Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,
#     LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,
#     Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,
#     YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,
#     MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,
#     BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,
#     Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,
#     GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,
#     KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,
#     GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,
#     GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,
#     ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,
#     SaleType,SaleCondition,SalePrice

NUMERIC_COLS = c("Id", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
                 "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "X1stFlrSF", "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
                 "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
                 "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                 "X3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold", "SalePrice")

imputeMissingValues = function(data) {
  cat('Imputing missing values...\n')

  #to investigate:
    #-LotFrontage: should be numbers, but has 259 NAs in train, 227 NAs in test
    #-MasVnrArea: should be numbers, but has 8 NAs in train, 15 NAs in test
    #-MasVnrType: should be factor without 'NA', but has 8 'NA's in train, 16 'NA's in test (the number of NAs is the same as the number of NAs in MasVnrArea)
    #-Electrical: should be factor without 'NA', but has 1 'NA' in train - leave it be since it probably indicates weird electrical issue with the house
    #-BsmtFinSF1: should be numbers, but has 1 NA in test
    #-BsmtFinSF2: should be numbers, but has 1 NA in test
    #-BsmtUnfSF: should be numbers, but has 1 NA in test
    #-TotalBsmtSF: should be numbers, but has 1 NA in test
    #-BsmtFullBath: should be numbers, but has 2 NAs in test
    #-BsmtHalfBath: should be numbers, but has 2 NAs in test
    #-GarageYrBlt: should be numbers, but has 81 NAs in train, 78 NAs in test - the NAs mean there is no garage
    #-GarageCars: should be numbers, but has 1 NA in test
    #-GarageArea: should be numbers, but has 1 NA in test
    #-KitchenQual: should be factor without 'NA', but has 1 'NA' in test - set it to TA

  #Replace NAs with -1s in GarageYrBlt
  data[is.na(data$GarageYrBlt), 'GarageYrBlt'] = -1

  #use roughfix to impute the other missing values
  for (col in NUMERIC_COLS) {
    data[[col]] = na.roughfix(data[[col]])
  }

  # #Fill LotFrontage using mice
  # data$LotFrontage = suppressWarnings(as.integer(data$LotFrontage)) #convert to integer, which also converts the 'NA' strings to NAs
  # require(mice)
  # set.seed(129)
  # mice_imp = mice(data[, c('MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle')],
  #     printFlag = F)
  # mice_output = complete(mice_imp)
  # data$LotFrontage = mice_output$LotFrontage

  return(data)
}

oneHotEncode = function(data) {
  dmy = caret::dummyVars('~.', data, fullRank=T)
  data = data.frame(predict(dmy, data))
  return(data)
}

featureEngineer = function(data) {
  return(data)
}

getData = function(yName) {
  #read data from file as character
  train = read.csv('data/train.csv', stringsAsFactors=F, na.strings=c(''), colClasses='character')
  test = read.csv('data/test.csv', stringsAsFactors=F, na.strings=c(''), colClasses='character')
  full = bind_rows(train, test)

  #manually convert cols to numeric
  for (col in NUMERIC_COLS) {
    full[[col]] = suppressWarnings(as.integer(full[[col]]))
  }

  #impute missing values
  full = imputeMissingValues(full)

  #do feature engineering
  full = featureEngineer(full)

  #split the data back into train and test
  train = full[1:nrow(train),]
  test = full[(nrow(train)+1):nrow(full), names(full) != yName]

  return(list(train=train, test=test, full=full))
}
