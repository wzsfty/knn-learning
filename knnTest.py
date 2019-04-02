
import numpy as np 

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

def file2matrix(filename):
	fr = open(filename,'r',encoding = 'utf-8')
	arrayOfLines = fr.readlines()
	numberOfLines = len(arrayOfLines)
	returnMat = np.zeros((numberOfLines,3));
	classLabelVector = []
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		# print(line)
		listFromLine = line.split('\t')
		# print(listFromLine)
		returnMat[index,:] = listFromLine[0:3]
		if listFromLine[-1] == 'didntLike':
			classLabelVector.append(1)
		elif listFromLine[-1] == 'smallDoses':
			classLabelVector.append(2)
		elif listFromLine[-1] == 'largeDoses':
			classLabelVector.append(3)
		index += 1
	return returnMat,classLabelVector

def showDatas(datingDataMat,datingLabels):
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	fig,axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))
	numberOfLabels = len(datingLabels)
	LabelColors = []
	for i in datingLabels:
		if i == 1:
			LabelColors.append('black')
		if i == 2:
			LabelColors.append('orange')
		if i == 3:
			LabelColors.append('red')
	axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelColors,s=15,alpha=0.5)
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	plt.setp(axs0_title_text, size=9, weight='bold', color='red')  
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')  
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black') 
	didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',
	                  markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',
	                  markersize=6, label='largeDoses')
	#添加图例
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	plt.show()


def autoNorm(dataSet):
	minVals = dataSet.min(axis = 0)
	# print(minVals)
	maxVals = dataSet.max(axis = 0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(np.shape(dataSet))
	# print(normDataSet)
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals,(m,1))
	normDataSet = normDataSet / np.tile(ranges,(m,1))
	return normDataSet,ranges,minVals

def classify0(input,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(input,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDist = sqDiffMat.sum(axis=1)
	dist = sqDist**0.5
	sortedDistIndex = dist.argsort()
	classCount = {}
	for i in range(k):
		voteLabel = labels[sortedDistIndex[i]]
		classCount[voteLabel] = classCount.get(voteLabel,0) + 1
	maxCount = 0
	for key,value in classCount.items():
		if value > maxCount:
			maxCount = value
			classes = key

	return classes



def datingClassTest():
	filename = 'datingTestSet.txt'
	datingDataMat,datingLabels = file2matrix(filename)
	normDataSet,ranges,minVals =  autoNorm(datingDataMat)
	m = normDataSet.shape[0]
	hoRatio = 0.10
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normDataSet[i,:], normDataSet[numTestVecs:m,:], datingLabels[numTestVecs:m], 4)
		print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))



def classifyPerson():
	#输出结果
	resultList = ['讨厌','有些喜欢','非常喜欢']
	#三维特征用户输入
	precentTats = float(input("玩视频游戏所耗时间百分比:"))
	ffMiles = float(input("每年获得的飞行常客里程数:"))
	iceCream = float(input("每周消费的冰激淋公升数:"))
	#打开的文件名
	filename = "datingTestSet.txt"
	#打开并处理数据
	datingDataMat, datingLabels = file2matrix(filename)
	#训练集归一化
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#生成NumPy数组,测试集
	inArr = np.array([ffMiles, precentTats, iceCream])
	#测试集归一化
	norminArr = (inArr - minVals) / ranges
	#返回分类结果
	classifierResult = classify0(norminArr, normMat, datingLabels, 3)
	#打印结果
	print("你可能%s这个人" % (resultList[classifierResult-1]))


if __name__ == '__main__':
	# filename = 'datingTestSet.txt'
	# datingDataMat,datingLabels = file2matrix(filename)
	# # print(datingDataMat)
	# # print(datingLabels)
	# # showDatas(datingDataMat,datingLabels)
	# normDataSet,ranges,minVals =  autoNorm(datingDataMat)
	# print(normDataSet)
	# print(ranges)
	# print(minVals)
	classifyPerson()

