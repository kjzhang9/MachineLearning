from numpy import *
import operator
import os

def createDataSet():
	group = array(mat('1.0 1.1; 1.0 1.0; 0 0; 0 0.1'))
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX,dataSet,lables,k):     
    dataSetSize = dataSet.shape[0]    
    diffMat = tile(inX,(dataSetSize,1))- dataSet      
    sqDiffMat = diffMat**2  
    sqDistances = sqDiffMat.sum(axis=1)  
    distance = sqDistances**0.5      
    sortedDistance = distance.argsort()  
     
    classCount= {}  
    for i in range(k):  
        voteLable = lables[sortedDistance[i]]  
        classCount[voteLable] = classCount.get(voteLable,0)+1  
    #print(classCount)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  
    
    return sortedClassCount[0][0] 

def file2matrix(filename):
	dataset, labels = [], []
	with open(filename, 'r') as f:
		for line in f:
			splited_line = [float(i) for i in line.strip().split('\t')]
			dataset.append(splited_line[:-1])
			labels.append(splited_line[-1])
	dataset = array(dataset)
	labels = array(labels)
	return dataset, labels

def autoNorm(dataset):
	minVals = dataset.min(0)
	maxVals = dataset.max(0)
	ranges = maxVals - minVals
	m = dataset.shape[0]
	norm_dataset = dataset - minVals
	norm_dataset = norm_dataset/ranges
	return norm_dataset, ranges, minVals

def datingClassTest():
	hoRtio = 0.10
	datingMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRtio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classResult = classify0(normMat[i, :], normMat[numTestVecs:, :], \
								datingLabels[numTestVecs:], 3)
		print("the classifier came back with: %d, the real answer is: %d" \
			  %(classResult, datingLabels[i]))
		if (classResult != datingLabels[i]): errorCount += 1.0
	print("the total error rate is: %f" %(errorCount/float(numTestVecs)))

def classifyPerson():
	resList = ['not at all', 'in small doses', 'in large doses']
	percentT = float(input("percentage of time spent playing video games:"))
	ffMiles = float(input("frequent flier miles earned per year:"))
	iceCream = float(input("liters of ice cream consumed per year:"))
	dateMat, dateLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(dateMat)
	inArr = array([ffMiles, percentT, iceCream])
	classResult = classify0((inArr-minVals)/ranges, normMat, dateLabels, 3)
	#print(classResult)
	print("You will probably like this person: ", resList[int(classResult)-1])

def img2vector(filename):
	returnVect = []
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip() # abandon \n for each line
			returnVect += [int(i) for i in line]
	returnVect = array(returnVect)
	return returnVect

def hwClassTest():
	hwLabels = []
	trainFile = os.listdir('trainingDigits')
	m = len(trainFile)
	trainMat = zeros((m, 1024))
	# read training data
	for i in range(m):
		classNumStr = int(trainFile[i][0])
		hwLabels.append(classNumStr)
		trainMat[i, :] = img2vector('trainingDigits/%s' % trainFile[i])

	testFile = os.listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFile)
	for i in range(mTest):
		classNumStr = int(testFile[i][0])
		testVec = img2vector('testDigits/%s' % testFile[i])
		classResult = classify0(testVec, trainMat, hwLabels, 3)
		print("The classifier came back with: %d, the real answer is: %d" \
			  % (classResult, classNumStr))
		if (classResult != classNumStr): errorCount += 1.0

	print("\nThe total number of errors is : %d" % errorCount)
	print("\nThe total error rate is : %f" %(errorCount/float(mTest)))
