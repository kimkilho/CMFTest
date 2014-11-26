import numpy
from numpy import *
import scipy
from scipy.sparse import *

def loadMeta(filename):
	fData = open(filename, 'r')
	numAttrGroups = 1
	numAttributes = 0

	for line in fData:
		numAttributes += 1
		currFeatureId = int(line)
		numAttrGroups = max(currFeatureId, numAttrGroups)
	fData.close()
	numAttrPerGroup = numpy.zeros(shape=numAttrGroups)
	fData = open(filename, 'r')
	for line in fData:
		currFeatureId = int(line)
		numAttrPerGroup[currFeatureId-1] += 1
	fData.close()

	return [numAttrPerGroup, numAttributes]


def loadData(metafile, datafile):
	fData = open(datafile, 'r')
	[numAttrPerGroup, numAttributes] = loadMeta(metafile)

	target = numpy.array([])
	numUsers = numAttrPerGroup[0]
	print "numUsers=", numUsers
	numItems = numAttrPerGroup[1]
	print "numItems=", numItems
	numContexts = numAttributes - numUsers - numItems
	print "numContexts=", numContexts
	user_item = numpy.zeros(shape=(numUsers, numItems))
	item_context = numpy.zeros(shape=(numItems, numContexts))
	user_context = numpy.zeros(shape=(numUsers, numContexts))
	# user_item = scipy.sparse.csc_matrix((numUsers, numItems), dtype=float)
	# item_context = scipy.sparse.csc_matrix((numItems, numContexts), dtype=float)
	# user_context = scipy.sparse.csc_matrix((numUsers, numContexts), dtype=float)

	for line in fData:
		# print line
		lineList = line.split()
		currUserId = 0
		currItemId = 0
		currContextIds = numpy.array([])
		for curr in lineList:
			if isDouble(curr):
				currTarget = float(curr)
				target = numpy.append(target, currTarget)
			else:
				pair = curr.split(':')
				currFeatureId = int(pair[0])
				# currFeatureValue = int(pair[1])
				if currFeatureId > 0 and currFeatureId <= numUsers:
					# currFeatureId is from "user"
					currUserId = currFeatureId

				if currFeatureId > numUsers and currFeatureId <= numUsers+numItems:
					# currFeatureId is from "item"
					currItemId = currFeatureId - numUsers

				if currFeatureId > numUsers+numItems:
					# currFeatureId is from "context"
					currContextIds = numpy.append(currContextIds, currFeatureId-numUsers-numItems)
		print currUserId, currItemId, currContextIds
		user_item[currUserId-1, currItemId-1] += 1
		for id in currContextIds:
			item_context[currItemId-1, id-1] += 1
			user_context[currUserId-1, id-1] += 1

	return [user_item, item_context, user_context]


def isDouble(str):
	try:
		float(str)
		return True
	except ValueError:
		return False

if __name__ == "__main__":
	[user_item, item_context, user_context] = loadData('meta.txt', 'em_66_f2_m11_tr.libsvm')
	for i in xrange(0, 67):
		print "user_item[%d] = "%i, user_item[i]
	print item_context
	for j in xrange(0, 10057-10009):
		print "user_context[%d] = "%j, user_context[j]
