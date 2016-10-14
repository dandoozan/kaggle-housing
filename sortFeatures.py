
NA_VALUE = 10
PREPEND_COMMENT = True

f = open('newSigs.txt')
sigLines = f.readlines()
f.close()

def parseLine(line):
    sp = filter(None, line.replace('< 2e-16', '2e-16').split(' '))
    featureName = sp[0].strip()
    pValueStr = sp[4].strip()
    sigMark = sp[5].strip() if len(sp) == 6 else ''
    pValue = NA_VALUE if pValueStr == 'NA' else float(pValueStr)
    return featureName, pValue, sigMark

features = []
for line in sigLines:
    featureName, pValue, sigMark = parseLine(line)

    #add pValue first so that i can sort by that
    features.append((pValue, featureName, sigMark))

features.sort()

for feature in features:
    featureName = feature[1]
    pValue = feature[0]
    sigMark = feature[2]
    if pValue < 0.1:
        print '%s%s + #%g %s' % ('#' if PREPEND_COMMENT else '', featureName, pValue, sigMark)
    elif pValue == NA_VALUE:
        print '#%s + #NA' % featureName




