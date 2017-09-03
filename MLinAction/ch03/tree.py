from math import log

def calShannonEnt(dataset):
	numData = len(dataset)
	labelCounts = {}
	for featVec in dataset:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numData
		shannonEnt -= prob*log(prob, 2)
	return shannonEnt

def creatDataset():
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

def splitDataSet(dataset, axis, value):
	retDataSet = []
	for featVec in dataset:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeaSplit(dataset):
	numFea = len(dataset[0]) - 1
	baseEnt = calShannonEnt(dataset)
	bestInfoGain, bestFea = 0.0, -1 # information gain,
	for i in range(numFea):
		featList = [example[i] for example in dataset]
		uniqueVals = set(featList)
		newEnt = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataset, i, value)
			prob = len(subDataSet)/float(len(dataset))
			newEnt += prob*calShannonEnt(subDataSet)
		infoGain = baseEnt - newEnt
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFea = i
	return bestFea