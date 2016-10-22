library(dplyr) #bind_rows
library(randomForest) #na.roughfix
library(caret) #dummyVars

#cols=Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,
#     LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,
#     Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,
#     YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,
#     MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,
#     BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,
#     Heating,HeatingQC,CentralAir,Electrical,X1stFlrSF,X2ndFlrSF,LowQualFinSF,
#     GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,
#     KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,
#     GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,
#     GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,X3SsnPorch,
#     ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,
#     SaleType,SaleCondition,SalePrice

NUMERIC_COLS = c("Id", "LotFrontage", "LotArea", "YearBuilt",
                 "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "X1stFlrSF", "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
                 "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
                 "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                 "X3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold", "SalePrice")
FACTOR_COLS_WITH_REAL_NAS = c('MasVnrType', 'KitchenQual')

loadData = function() {
  train = read.csv('data/train.csv', stringsAsFactors=F, na.strings=c(''), colClasses='character')
  test = read.csv('data/test.csv', stringsAsFactors=F, na.strings=c(''), colClasses='character')
  full = bind_rows(train, test)

  #manually convert cols to numeric
  for (col in NUMERIC_COLS) {
    full[[col]] = suppressWarnings(as.integer(full[[col]]))
  }

  #Convert 'NA' to NA for factors that have real NAs
  for (col in FACTOR_COLS_WITH_REAL_NAS) {
    full[full[[col]] == 'NA', col] = NA
  }

  #manually convert cols to factors
  factorCols = setdiff(names(full), NUMERIC_COLS)
  for (col in factorCols) {
    full[[col]] = factor(full[[col]])
  }

  return(list(train=train, test=test, full=full))
}

getMiceImputedValues = function(colName, data, colNamesToUseInMice) {
  require(mice)

  naRows = data[is.na(data[[colName]]), 'Id']

  set.seed(129)
  mice_imp = mice(data[, colNamesToUseInMice], printFlag = T)
  mice_output = complete(mice_imp)

  return(mice_output[[colName]][naRows])
}

setValues = function(dataCol, rowIndices, values) {
  for (i in 1:length(rowIndices)) {
    dataCol[rowIndices[i]] = values[i]
  }
  return(dataCol)
}

imputeMissingValues = function(data) {
  cat('    Imputing missing values...\n')

  #to investigate:
    #-LotFrontage: should be numbers, but has 259 NAs in train, 227 NAs in test - use mice
    #-MasVnrArea: should be numbers, but has 8 NAs in train, 15 NAs in test - use mice
    #-MasVnrType: should be factor without 'NA', but has 8 'NA's in train, 16 'NA's in test (the number of NAs is the same as the number of NAs in MasVnrArea) - use mice
    #-Electrical: should be factor without 'NA', but has 1 'NA' in train - leave it be since it probably indicates weird electrical issue with the house
    #-BsmtFinSF1: should be numbers, but has 1 NA in test
    #-BsmtFinSF2: should be numbers, but has 1 NA in test
    #-BsmtUnfSF: should be numbers, but has 1 NA in test
    #-TotalBsmtSF: should be numbers, but has 1 NA in test
    #-BsmtFullBath: should be numbers, but has 2 NAs in test
    #-BsmtHalfBath: should be numbers, but has 2 NAs in test
    #-GarageYrBlt: should be numbers, but has 81 NAs in train, 78 NAs in test - the NAs mean there is no garage, so set them to -1
    #-GarageCars: should be numbers, but has 1 NA in test
    #-GarageArea: should be numbers, but has 1 NA in test
    #-KitchenQual: should be factor without 'NA', but has 1 'NA' in test

  #GarageYrBlt: replace NAs with -1s
  data[is.na(data$GarageYrBlt), 'GarageYrBlt'] = -1

  #LotFrontage: use mice
  #colsToUseInMice = c('MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle')
  naRows = c(8, 13, 15, 17, 25, 32, 43, 44, 51, 65, 67, 77, 85, 96, 101, 105, 112, 114, 117, 121, 127, 132, 134, 137, 148, 150, 153, 154, 161, 167, 170, 171, 178, 181, 187, 192, 204, 208, 209, 215, 219, 222, 235, 238, 245, 250, 270, 288, 289, 294, 308, 309, 311, 320, 329, 331, 336, 343, 347, 348, 352, 357, 361, 362, 365, 367, 370, 371, 376, 385, 393, 394, 405, 406, 413, 422, 427, 448, 453, 458, 459, 460, 466, 471, 485, 491, 497, 517, 519, 530, 538, 539, 540, 542, 546, 560, 561, 565, 570, 581, 594, 611, 612, 613, 617, 624, 627, 642, 646, 661, 667, 669, 673, 680, 683, 686, 688, 691, 707, 710, 715, 721, 722, 727, 735, 746, 747, 752, 758, 771, 784, 786, 790, 792, 795, 812, 817, 818, 823, 829, 841, 846, 852, 854, 856, 857, 860, 866, 869, 880, 883, 894, 901, 905, 909, 912, 918, 926, 928, 929, 930, 940, 942, 945, 954, 962, 968, 976, 981, 984, 989, 997, 998, 1004, 1007, 1018, 1019, 1025, 1031, 1033, 1034, 1036, 1038, 1042, 1046, 1058, 1060, 1065, 1078, 1085, 1087, 1098, 1109, 1111, 1117, 1123, 1125, 1139, 1142, 1144, 1147, 1149, 1154, 1155, 1162, 1165, 1178, 1181, 1191, 1194, 1207, 1214, 1231, 1234, 1245, 1248, 1252, 1254, 1261, 1263, 1269, 1271, 1272, 1273, 1277, 1278, 1287, 1288, 1291, 1301, 1302, 1310, 1313, 1319, 1322, 1343, 1347, 1349, 1355, 1357, 1358, 1359, 1363, 1366, 1369, 1374, 1382, 1384, 1397, 1408, 1418, 1420, 1424, 1425, 1430, 1432, 1442, 1444, 1447, 1467, 1501, 1502, 1506, 1508, 1513, 1520, 1536, 1543, 1559, 1564, 1566, 1568, 1574, 1580, 1585, 1593, 1607, 1613, 1628, 1635, 1638, 1640, 1643, 1644, 1645, 1648, 1649, 1660, 1690, 1691, 1692, 1696, 1699, 1701, 1729, 1732, 1733, 1734, 1735, 1737, 1738, 1740, 1741, 1744, 1747, 1751, 1755, 1758, 1759, 1762, 1769, 1820, 1824, 1834, 1841, 1844, 1847, 1848, 1849, 1862, 1863, 1864, 1873, 1879, 1882, 1884, 1886, 1903, 1911, 1912, 1923, 1937, 1942, 1946, 1948, 1950, 1956, 1958, 1985, 1986, 1989, 1990, 1993, 1997, 2000, 2024, 2030, 2031, 2040, 2042, 2043, 2045, 2050, 2053, 2065, 2075, 2111, 2112, 2123, 2129, 2132, 2138, 2141, 2142, 2143, 2147, 2149, 2156, 2158, 2164, 2165, 2167, 2168, 2171, 2172, 2174, 2175, 2176, 2179, 2187, 2203, 2204, 2205, 2223, 2224, 2235, 2242, 2246, 2251, 2254, 2255, 2258, 2259, 2269, 2280, 2301, 2321, 2326, 2328, 2358, 2362, 2373, 2380, 2388, 2390, 2396, 2397, 2404, 2422, 2424, 2432, 2447, 2460, 2467, 2481, 2484, 2485, 2491, 2492, 2493, 2494, 2499, 2513, 2515, 2522, 2531, 2535, 2536, 2548, 2568, 2569, 2571, 2573, 2584, 2587, 2588, 2595, 2597, 2598, 2603, 2607, 2612, 2615, 2617, 2618, 2621, 2626, 2635, 2637, 2663, 2673, 2674, 2677, 2678, 2681, 2684, 2685, 2701, 2704, 2705, 2707, 2708, 2709, 2710, 2715, 2716, 2725, 2728, 2738, 2739, 2742, 2765, 2808, 2811, 2812, 2813, 2815, 2816, 2819, 2840, 2846, 2848, 2851, 2901, 2902, 2909)
  valuesFromMice = c(81, 74, 75, 50, 79, 44, 50, 47, 110, 99, 81, 74, 85, 96, 72, 60, 105, 91, 75, 35, 50, 89, 70, 90, 60, 60, 57, 66, 47, 83, 160, 85, 80, 34, 63, 42, 30, 85, 105, 96, 125, 50, 65, 60, 80, 50, 65, 70, 75, 102, 66, 75, 68, 77, 63, 72, 68, 80, 50, 85, 59, 73, 41, 50, 78, 133, 89, 75, 44, 68, 80, 80, 65, 85, 55, 80, 88, 85, 90, 90, 43, 51, 44, 60, 70, 34, 80, 107, 53, 75, 80, 105, 83, 90, 95, 48, 80, 61, 98, 71, 30, 85, 60, 60, 75, 44, 105, 60, 130, 57, 84, 93, 85, 92, 64, 24, 60, 30, 313, 76, 42, 48, 30, 94, 80, 85, 62, 60, 86, 48, 68, 80, 90, 100, 140, 30, 90, 66, 80, 60, 68, 56, 43, 101, 88, 70, 94, 68, 105, 52, 168, 76, 96, 60, 60, 24, 60, 101, 80, 70, 98, 87, 75, 50, 80, 84, 50, 34, 99, 110, 82, 60, 87, 85, 80, 24, 80, 91, 43, 120, 73, 60, 80, 80, 63, 60, 84, 57, 101, 91, 44, 53, 85, 75, 80, 80, 82, 86, 60, 80, 79, 50, 60, 76, 52, 90, 44, 80, 100, 30, 92, 57, 138, 61, 105, 89, 53, 90, 80, 52, 85, 70, 77, 44, 60, 75, 62, 61, 101, 50, 60, 65, 69, 50, 58, 82, 81, 99, 108, 90, 60, 34, 75, 96, 30, 83, 76, 94, 47, 73, 85, 60, 98, 60, 100, 55, 30, 82, 66, 100, 68, 60, 96, 78, 63, 67, 60, 80, 67, 77, 100, 60, 60, 61, 30, 75, 110, 72, 75, 129, 81, 98, 96, 81, 41, 82, 87, 78, 60, 65, 100, 35, 43, 80, 60, 75, 85, 63, 73, 71, 122, 24, 26, 130, 43, 54, 114, 79, 122, 68, 75, 60, 60, 60, 73, 68, 73, 60, 33, 74, 65, 60, 40, 60, 98, 66, 79, 65, 58, 78, 56, 125, 94, 131, 50, 80, 90, 110, 53, 34, 70, 75, 60, 61, 61, 80, 42, 55, 70, 72, 88, 56, 60, 84, 60, 47, 75, 50, 51, 60, 44, 76, 46, 85, 30, 85, 60, 131, 70, 110, 50, 60, 90, 85, 80, 75, 44, 78, 30, 60, 80, 80, 60, 160, 95, 80, 34, 140, 313, 43, 68, 160, 44, 67, 90, 85, 51, 81, 64, 98, 90, 50, 85, 45, 62, 65, 129, 51, 110, 50, 60, 63, 80, 85, 73, 62, 75, 80, 75, 113, 108, 60, 70, 33, 58, 80, 70, 94, 47, 64, 60, 74, 69, 41, 68, 38, 72, 73, 65, 24, 129, 92, 91, 140, 75, 80, 91, 79, 80, 102, 43, 34, 91, 118, 72, 60, 80, 78, 50, 60, 80, 64, 77, 62, 48, 55, 88, 35, 68, 100, 66, 94, 70, 80, 47, 80, 74, 80, 30, 44, 93, 85, 108, 200, 78, 80)
  data$LotFrontage = setValues(data$LotFrontage, naRows, valuesFromMice)

  #MasVnrArea: use mice
  #colsToUseInMice = c('BldgType', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'MasVnrType', 'MasVnrArea', 'X1stFlrSF', 'X2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', 'PoolArea')
  naRows = c(235, 530, 651, 937, 974, 978, 1244, 1279, 1692, 1707, 1883, 1993, 2005, 2042, 2312, 2326, 2341, 2350, 2369, 2593, 2658, 2687, 2863)
  valuesFromMice = c(500, 0, 0, 216, 0, 306, 256, 0, 0, 0, 0, 11, 0, 0, 170, 0, 323, 0, 246, 0, 142, 522, 340)
  data$MasVnrArea = setValues(data$MasVnrArea, naRows, valuesFromMice)

  #MasVnrType: use mice
  #colsToUseInMice = c('BldgType', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'MasVnrType', 'MasVnrArea', 'X1stFlrSF', 'X2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', 'PoolArea')
  naRows = c(235, 530, 651, 937, 974, 978, 1244, 1279, 1692, 1707, 1883, 1993, 2005, 2042, 2312, 2326, 2341, 2350, 2369, 2593, 2611, 2658, 2687, 2863)
  valuesFromMice = c('Stone', 'BrkCmn', 'None', 'BrkFace', 'None', 'Stone', 'Stone', 'None', 'None', 'None', 'None', 'BrkFace', 'None', 'None', 'BrkFace', 'None', 'BrkFace', 'None', 'BrkFace', 'None', 'BrkFace', 'BrkFace', 'BrkFace', 'BrkFace')
  data$MasVnrType = setValues(data$MasVnrType, naRows, valuesFromMice)

  #roughfix the rest since they only have 1 or 2 missing values
  data = na.roughfix(data)

  return(data)
}

oneHotEncode = function(data) {
  cat('    One hot encoding variables...\n')
  dmy = caret::dummyVars('~.', data, fullRank=T)
  data = data.frame(predict(dmy, data))
  return(data)
}

featureEngineer = function(data) {
  return(data)
}

getData = function(yName, oneHotEncode=F) {
  cat('Getting data...\n')

  #read data from file as character
  data = loadData()
  train = data$train
  test = data$test
  full = data$full

  #impute missing values
  full = imputeMissingValues(full)

  #do feature engineering
  full = featureEngineer(full)

  #one hot encode factors
  #todo: perhaps one-hot-encode train and test separately.  I believe the reason
  #to do it separate is so that I don't 'leak' information to the train set when
  #there is a factor value in test that is not in train (but this would result in
  #an all-0s column for train, which wouldn't be used, so I'm not sure how big of
  #a problem the leaking information is).  For now, keep one-hot-encoding them together
  if (oneHotEncode) {
    full = oneHotEncode(full)
  }

  #split the data back into train and test
  train = full[1:nrow(train),]
  test = full[(nrow(train)+1):nrow(full), names(full) != yName]

  return(list(train=train, test=test, full=full))
}
