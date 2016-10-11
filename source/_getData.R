#todo:
#D-fill NAs using roughfix
#D-combine rare values in all factor cols
#D-change NAs to -1s for GarageYrBlt before roughfix b/c the NAs represent no garage
#-try making quality cols factors instead of ints (OverallQual, OverallCond)


setwd('/Users/dan/Desktop/Kaggle/Housing')

library(dplyr) #bind_rows
library(randomForest) #na.roughfix

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
    #-TotalBsmtSF: should be numbers, but has 1 NA in test
    #-BsmtFullBath: should be numbers, but has 2 NAs in test
    #-BsmtHalfBath: should be numbers, but has 2 NAs in test
    #-GarageYrBlt: should be numbers, but has 81 NAs in train, 78 NAs in test - the NAs mean there is no garage
    #-GarageCars: should be numbers, but has 1 NA in test
    #-GarageArea: should be numbers, but has 1 NA in test
    #-KitchenQual: should be factor without 'NA', but has 1 'NA' in test - set it to TA

  #Replace NAs with -1s in GarageYrBlt
  data[is.na(data$GarageYrBlt), 'GarageYrBlt'] = -1

  #use roughfix to impute missing values in numeric cols
  for (col in NUMERIC_COLS) {
    data[[col]] = na.roughfix(suppressWarnings(as.integer(data[[col]])))
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


replaceValues = function(col, newValue, oldValue) {
  #Note: oldValue can be a string or an array of strings
  for (str in oldValue) {
    col[col == str] = newValue
  }
  return(col)
}

featureEngineer = function(data) {
  #Combine rare values

  #BsmtCond #TA/Gd/Fa&Po/NA
  data$BsmtCond = factor(replaceValues(data$BsmtCond, 'Fa', 'Po'))
  #BsmtExposure #as-is
  #BsmtFinType1 #as-is
  #BsmtFinType2 #Unf/ALQ&GLQ/BLQ/Rec/LwQ/NA
  data$BsmtFinType2 = factor(replaceValues(data$BsmtFinType2, 'ALQ', 'GLQ'))
  #BsmtQual #as-is
  #Condition1 #Norm/Other(Artery&Feedr&PosA&PosN&RRAe&RRAn&RRNe&RRNn)
  data$Condition1 = factor(replaceValues(data$Condition1, 'Other', c('Artery', 'Feedr', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNe', 'RRNn')))
  #Condition2 #Norm/Other(Artery&Feedr&PosA&PosN&RRAe&RRAn&RRNe&RRNn)
  data$Condition2 = factor(replaceValues(data$Condition2, 'Other', c('Artery', 'Feedr', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNe', 'RRNn')))
  #ExterQual #as-is
  #Fence #NA/Minimal(MnPrv&MnWw)/Good(GdPrv&GdWo)
  data$Fence = replaceValues(data$Fence, 'Minimal', c('MnPrv', 'MnWw'))
  data$Fence = replaceValues(data$Fence, 'Good', c('GdPrv', 'GdWo'))
  data$Fence = factor(data$Fence)
  #Foundation #PConc&Slab/CBlock/Other(BrkTil&Stone&Wood)
  data$Foundation = replaceValues(data$Foundation, 'PConc', 'Slab')
  data$Foundation = replaceValues(data$Foundation, 'Other', c('BrkTil', 'Stone', 'Wood'))
  data$Foundation = factor(data$Foundation)
  #Functional #Typ&NA/Min(Min1&Min2)/Maj(Mod&Maj1&Maj2&Sev)
  data$Functional = replaceValues(data$Functional, 'Typ', 'NA')
  data$Functional = replaceValues(data$Functional, 'Min', c('Min1', 'Min2'))
  data$Functional = replaceValues(data$Functional, 'Maj', c('Maj1', 'Maj2', 'Mod', 'Sev'))
  data$Functional = factor(data$Functional)
  #GarageCond #TA&Ex&Gd/NA/Bad(Fa&Po)
  data$GarageCond = replaceValues(data$GarageCond, 'TA', c('Gd', 'Ex'))
  data$GarageCond = replaceValues(data$GarageCond, 'Bad', c('Fa', 'Po'))
  data$GarageCond = factor(data$GarageCond)
  #GarageQual #TA&Ex&Gd/NA/Bad(Fa&Po)
  data$GarageQual = replaceValues(data$GarageQual, 'TA', c('Gd', 'Ex'))
  data$GarageQual = replaceValues(data$GarageQual, 'Bad', c('Fa', 'Po'))
  data$GarageQual = factor(data$GarageQual)
  #GarageType #Attchd/Detchd/BuiltIn/NA/Other(2Types&Basment&CarPort)
  data$GarageType = factor(replaceValues(data$GarageType, 'Other', c('2Types', 'Basment', 'CarPort')))
  #Heating #GasA/Other(Floor&GasW&Grav&OthW&Wall)
  data$Heating = factor(replaceValues(data$Heating, 'Other', c('Floor', 'GasW', 'Grav', 'OthW', 'Wall')))
  #HouseStyle #1Story/2Story/1.5Fin/SLvl/SFoyer/Other(1.5Unf&2.5Fin&2.5Unf)
  data$HouseStyle = factor(replaceValues(data$HouseStyle, 'Other', c('1.5Unf', '2.5Fin', '2.5Unf')))
  #KitchenQual #as-is, but set the 1 test NA to TA
  data$KitchenQual = factor(replaceValues(data$KitchenQual, 'TA', 'NA'))
  #LandSlope #Gtl/Mod&Sev
  data$LandSlope = factor(replaceValues(data$LandSlope, 'Mod', 'Sev'))
  #LotConfig #Inside/Corner/CulDSac/Other(FR2&FR3)
  data$LotConfig = factor(replaceValues(data$LotConfig, 'Other', c('FR2', 'FR3')))
  #MSZoning #RL/FV/RM&RH/C
  data$MSZoning = replaceValues(data$MSZoning, 'RL', 'NA')
  data$MSZoning = replaceValues(data$MSZoning, 'RM', 'RH')
  data$MSZoning = factor(data$MSZoning)
  #Neighborhood #as-is
  #PoolQC #NA/Other(Ex&Gd&Fa)
  data$PoolQC = factor(replaceValues(data$PoolQC, 'Other', c('Ex', 'Gd', 'Fa')))
  #RoofMatl #CompShg/Other('Tar&Grv'&WdShake&WdShngl&ClyTile&Membran&Metal&Roll)
  data$RoofMatl = factor(replaceValues(data$RoofMatl, 'Other', c('Tar&Grv', 'WdShake', 'WdShngl', 'ClyTile', 'Membran', 'Metal', 'Roll')))
  #RoofStyle #Gable/Hip/Other(Gambrel&Flat&Mansard&Shed)
  data$RoofStyle = factor(replaceValues(data$RoofStyle, 'Other', c('Gambrel', 'Flat', 'Mansard', 'Shed')))
  #SaleCondition #Normal/Other(Abnorml&AdjLand&Alloca&Family&Partial)
  data$SaleCondition = factor(replaceValues(data$SaleCondition, 'Other', c('Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial')))
  #Street #as-is

  #Convert PoolQA to numeric b/c it only has 2 levels, and one is very rare
  data$PoolQC = as.numeric(data$PoolQC)

  return(data)
}

getData = function(yName) {
  #read data from file as character
  train = read.csv('data/train.csv', stringsAsFactors=F, na.strings=c(''), colClasses='character')
  test = read.csv('data/test.csv', stringsAsFactors=F, na.strings=c(''), colClasses='character')
  full = bind_rows(train, test)

  #impute missing values
  full = imputeMissingValues(full)

  #do feature engineering
  full = featureEngineer(full)

  #split the data back into train and test
  train = full[1:nrow(train),]
  test = full[(nrow(train)+1):nrow(full), names(full) != yName]

  return(list(train=train, test=test))
}
