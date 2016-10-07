#todo:
#D-impute missing values
#-try making quality cols factors instead of ints (OverallQual, OverallCond)
#-convert cols that have qualities as strings (eg. "Ex"/"Gd"/"TA"/"Fa"/"Po") to ints
#-convert GarageYrBlt from factor to int (which means converting the NAs to ints somehow)

setwd('/Users/dan/Desktop/Kaggle/Housing')

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

TARGET_COL = 'SalePrice'
FACTOR_COLS = c('MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
                'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'RoofStyle',
                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                'GarageYrBlt', #make GarageYrBlt a factor because there are NAs
                'MiscFeature', 'SaleType', 'SaleCondition' )
questionableFactorCols = c('YearBuilt', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces')

imputeMissingValues = function(data) {
  cat('Imputing missing values...\n')
  #to investigate:
    #D-LotFrontage: should be numbers, but has 259 NAs - use mice
    #D-MasVnrArea: should be numbers, but has 8 NAs - remove the NA rows
    #D-MasVnrType: should be factor without 'NA', but has 8 'NA's (the number of NAs is the same as the number of NAs in MasVnrArea) - remove the NA rows
    #D-Electrical: should be factor without 'NA', but has 1 'NA' - leave it be since it probably indicates weird electrical issue with the house
    #D-GarageYrBlt: should be numbers, but has 81 NAs - the NAs mean there is no garage, so leave it be as is

  #Fill LotFrontage using mice
  data$LotFrontage = suppressWarnings(as.integer(data$LotFrontage)) #convert to integer, which also converts the 'NA' strings to NAs
  require(mice)
  set.seed(129)
  mice_imp = mice(data[, c('MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle')],
      printFlag = F)
  mice_output = complete(mice_imp)
  data$LotFrontage = mice_output$LotFrontage

  #Remove rows that have NA for MasVnrArea and MasVnrType (they're the same 8 rows)
  data$MasVnrArea = suppressWarnings(as.integer(data$MasVnrArea)) #convert to integer, which also converts the 'NA' strings to NAs
  data = data[-data[is.na(data$MasVnrArea), 'Id'],]

  return(data)
}

featureEngineer = function(data) {
  return(data)
}

getData = function() {
  #read data from file
  train = read.csv('data/train.csv', stringsAsFactors=F, na.strings=c(''))
  test = read.csv('data/test.csv', stringsAsFactors=F, na.strings=c(''))

  #manually create factors
  for (col in FACTOR_COLS) {
    train[, col] = factor(train[, col])
  }

  #do feature engineering
  train = imputeMissingValues(train)

  return(list(train=train, test=test))
}
