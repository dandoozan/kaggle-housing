f = open('oldSigs.txt')
oldSigLines = f.readlines()
f.close()

f = open('newSigs.txt')
newSigLines = f.readlines()
f.close()

sigMap = {
    '.': 1,
    '*': 2,
    '**': 3,
    '***': 4
}

def parseLine(line):
    sp = filter(None, line.replace('< 2e-16', '<2e-16').split(' '))
    featureName = sp[0]
    if len(sp) < 6:
        return (featureName, 0)
    sigMark = sp[5].strip()
    return (featureName, sigMap[sigMark])


#put the old sigs in a dict
oldSigs = {}
for line in oldSigLines:
    featureName, sigLevel = parseLine(line)
    oldSigs[featureName] = sigLevel

#compare the new sigs and old sigs
for line in newSigLines:
    featureName, sigLevel = parseLine(line)
    if featureName not in oldSigs:
        if sigLevel > 0:
            print '***NEW FEATURE: %s = %d' % (featureName, sigLevel)
    elif oldSigs[featureName] != sigLevel:
        print '%s, %d -> %d' % (featureName, oldSigs[featureName], sigLevel)




