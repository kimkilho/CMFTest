# various evaluation measures for collaboration filtering
import numpy
from numpy import *
from scipy.sparse import *

'''
X is prediction, Y is ground truth
Both X and Y should be scipy.sparse.csc_matrix
'''

##########################
# signed RMSE
##########################
#def srmse(X, Y):
#	signs = zeros(X.data.size)
#	errs = X.data-Y.data
#	signs[errs>0] = 1.0  # overestimation
#	signs[errs<0] = -1.0 # underestimation
#	errs = errs**2
#	return sqrt(sum(multiply(signs,errs))/X.size)
	
def rmse(X, Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	if X.size > 0:
		return numpy.sqrt(sum((X.data - Y.data) ** 2) / X.size)
	else:
		return 0

#############################
## signed MAE
#############################
#def smae(X, Y):
#	signs = zeros(X.data.size)
#	errs = X.data-Y.data
#	signs[errs>0] = 1.0  # overestimation
#	signs[errs<0] = -1.0 # underestimation
#	errs = abs(errs)
#	return sum(multiply(signs,errs))/X.size
	
def mae(X, Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	if X.size > 0:
		return sum(abs(X.data - Y.data)) / X.size
	else:
		return 0


def map(X, Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	n = Y.shape[1]
	res = 0
	nvalid = 0
	Xdata = X.data
	Ydata = Y.data
	indices = Y.indices
	indptr = Y.indptr
	for i in xrange(n):
		[j0, j1] = [indptr[i], indptr[i + 1]]
		if j0 == j1: # skip empty column
			continue
		Xi = Xdata[j0:j1]
		Yi = Ydata[j0:j1]
		if len(unique(Yi)) == 1:
			continue
		I = argsort(-Xi) 
		[inds1] = where(Yi[I] >= 1)
		nvalid += 1
		pres = numpy.divide(arange(1, inds1.size + 1), 1.0 + inds1) # to avoid integer arithmetic
		res += mean(pres)
	if nvalid > 0:
		res = res / nvalid
	else:
		print "map warning! nvalid==0"
	return res

	
	
###############################################################################
# a version of map based on ratings data that treat ratings 
# greater than r0 as relevant and all others as irreleant
###############################################################################
def map_rating(X,Y,r0):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	assert(r0>0)
	n = Y.shape[1]
	res = 0
	nvalid = 0
	Xdata = X.data
	Ydata = Y.data
	indices = Y.indices
	indptr = Y.indptr
	for i in xrange(n):
		[j0, j1] = [indptr[i], indptr[i + 1]]
		if j0 == j1: # skip empty column
			continue
		Xi = Xdata[j0:j1]
		Yi = Ydata[j0:j1]
		if all(Yi<r0):
			continue
		I = argsort(-Xi) 
		[inds1] = where(Yi[I] >= r0)
		nvalid += 1
		pres = numpy.divide(arange(1, inds1.size + 1), 1.0 + inds1) # to avoid integer arithmetic
		res += mean(pres)
	if nvalid > 0:
		res = res / nvalid
	else:
		print "map warning! nvalid==0"
	return res


#################################
# hit ratio of top-k list
#################################
def f1_topk(Xpred, Xtst, r0, k):
	M,N = Xtst.shape
	ys = zeros(M)
	nvalid = 0
	res = 0
	for i in xrange(N):
		ni = Xpred.indptr[i+1]-Xpred.indptr[i]
		assert(ni>=k)
		xs_i = Xpred.data[Xpred.indptr[i]:Xpred.indptr[i]+k]
		ids_i = Xpred.indices[Xpred.indptr[i]:Xpred.indptr[i]+k]
		assert(all(xs_i[1:k]-xs_i[0:k-1]<=0))
		ys[:] = 0
		ys[Xtst.indices[Xtst.indptr[i]:Xtst.indptr[i+1]]] = Xtst.data[Xtst.indptr[i]:Xtst.indptr[i+1]]
		n1_total = sum(ys>=r0)
		if n1_total==0:
			continue
		nvalid += 1
		ys_i = ys[ids_i]
		n1_topk = sum(ys_i>=r0)
		pre = float(n1_topk)/float(k)
		rec = float(n1_topk)/float(n1_total)
		if pre + rec > 0:
			res += 2*pre*rec/(pre+rec)
	if nvalid > 0:
		res /= nvalid
	return res
		
#################################
# hit ratio of top-k list
#################################
def hit_topk(Xpred, Xtst, r0, k):
	M,N = Xtst.shape
	ys = zeros(M)
	nvalid = 0
	res = 0
	for i in xrange(N):
		ni = Xpred.indptr[i+1]-Xpred.indptr[i]
		assert(ni>=k)
		xs_i = Xpred.data[Xpred.indptr[i]:Xpred.indptr[i]+k]
		ids_i = Xpred.indices[Xpred.indptr[i]:Xpred.indptr[i]+k]
		assert(all(xs_i[1:k]-xs_i[0:k-1]<=0))
		ys[:] = 0
		ys[Xtst.indices[Xtst.indptr[i]:Xtst.indptr[i+1]]] = Xtst.data[Xtst.indptr[i]:Xtst.indptr[i+1]]
		n1_total = sum(ys>=r0)
		if n1_total==0:
			continue
		nvalid += 1
		ys_i = ys[ids_i]
		n1_topk = sum(ys_i>=r0)
		
		res += float(n1_topk)/float(k)
	if nvalid > 0:
		res /= nvalid
	return res
		
	
	
# do not compute average precision for each 
def mpr(X, Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	n = Y.shape[1]
	res = 0
	nvalid = 0
	Xdata = X.data
	Ydata = Y.data
	indices = Y.indices
	indptr = Y.indptr
	for i in xrange(n):
		[j0, j1] = [indptr[i], indptr[i + 1]]
		if j0 == j1: # skip empty column
			continue
		Xi = Xdata[j0:j1]
		Yi = Ydata[j0:j1]
		if len(unique(Yi)) == 1:
			continue
		I = argsort(-Xi) 
		[inds1] = where(Yi[I] >= 1)
		nvalid += inds1.size
		pres = numpy.divide(arange(1, inds1.size + 1), 1.0 + inds1) # to avoid integer arithmetic
		res += sum(pres)	
	assert(nvalid > 0)
	res = res / nvalid
	return res

# mean rank predicion, for each relevant item, we compute the proportion of 
# irrelevant items ranked below it
def mrpr(X, Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	n = Y.shape[1]
	res = 0
	nvalid = 0
	Xdata = X.data
	Ydata = Y.data
	indices = Y.indices
	indptr = Y.indptr
	for i in xrange(n):
		[j0, j1] = [indptr[i], indptr[i + 1]]
		if j0 == j1: # skip empty column
			continue
		Xi = Xdata[j0:j1]
		Yi = Ydata[j0:j1]
		if len(unique(Yi)) == 1: # must have multiple rating classes
			continue
		I = argsort(-Xi) 
		[inds1] = where(Yi[I] >= 1)
		n0 = j1-j0-inds1.size # total number of irrelevant items
		nvalid += inds1.size
		rpr = 1-numpy.divide(inds1-arange(inds1.size), float(n0)) # to avoid integer arithmetic
		res += sum(rpr)	
	assert(nvalid > 0)
	
	res = res / nvalid
	return res

####################################
# mean reciprocal rank for implicit
####################################
def mrr(X,Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	n = Y.shape[1]
	res = 0
	nvalid = 0
	Xdata = X.data
	Ydata = Y.data
	indices = Y.indices
	indptr = Y.indptr
	for i in xrange(n):
		[j0, j1] = [indptr[i], indptr[i + 1]]
		if j0 == j1: # skip empty column
			continue
		Xi = Xdata[j0:j1]
		Yi = Ydata[j0:j1]
		if len(unique(Yi)) == 1: # must have multiple rating classes
			continue
		I = argsort(-Xi) 
		[inds1] = where(Yi[I] >= 1)
		nvalid += inds1.size
		res += sum(1.0/(1+inds1))
			
	assert(nvalid > 0)
	res = res / nvalid
	return res
	
	
def ap_global(X,Y):
	Xdata = X.data
	Ydata = Y.data
	if all(Ydata < 1):
		assert False, "no relevant items found in the test data"
	I = argsort(-Xdata) 
	[inds1] = where(Ydata[I] >= 1)
	pres = numpy.divide(arange(1, inds1.size + 1), 1.0 + inds1) # to avoid integer arithmetic
	return mean(pres)
	

def ndcgk_global(X,Y,K):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	(M,N) = Y.shape
	Xdata = X.data
	Ydata = Y.data
	nnz = Ydata.size
	I = argsort(-Xdata)
	Y_pred = numpy.exp(Ydata[I])-1.0
	Y_best = numpy.exp(-(sort(-Ydata)))-1.0
	Wi = numpy.log(numpy.exp(1) + Ydata.size-1)
	Yi_pred = numpy.divide(Y_pred, Wi)
	Yi_best = numpy.divide(Y_best, Wi)
	K = min([K, nnz])
	res = sum(Yi_pred[0:K]) / sum(Yi_best[0:K])
	return res

'''
Normalized Discounted Cummulative Gain at K
'''
def ndcg_k(X, Y, K):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	n = Y.shape[1]
	res = 0
	nvalid = 0
	Xdata = X.data
	Ydata = Y.data
	indices = Y.indices
	indptr = Y.indptr
	for i in xrange(n):
		[j0, j1] = [indptr[i], indptr[i+1]]
		if j0 == j1: # skip empty column
			continue
		
		Xi = Xdata[j0:j1]
		Yi = Ydata[j0:j1]
		if all(Yi==0):
			continue
		nvalid += 1
		I = argsort(-Xi)
		Yi_pred = numpy.exp(Yi[I])-1.0
		Yi_best = numpy.exp(-(sort(-Yi)))-1.0
		Wi = numpy.log(numpy.exp(1) + arange(j1 - j0))
		Yi_pred = numpy.divide(Yi_pred, Wi)
		Yi_best = numpy.divide(Yi_best, Wi)
		Ki = min([K, j1 - j0])
		res += sum(Yi_pred[0:Ki]) / sum(Yi_best[0:Ki])
	assert(nvalid > 0)
	res /= nvalid
	return res



def ndcg_multi(X, Y, Ks):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	n = Y.shape[1]
	res = zeros(len(Ks))
	nvalid = 0
	Xdata = X.data
	Ydata = Y.data
	indices = Y.indices
	indptr = Y.indptr
	for i in xrange(n):
		[j0, j1] = [indptr[i], indptr[i + 1]]
		if j0 == j1: # skip empty column
			continue
		nvalid += 1
		Xi = Xdata[j0:j1]
		Yi = Ydata[j0:j1]
		I = argsort(-Xi)
		Yi_pred = numpy.exp(Yi[I])-1.0
		Yi_best = numpy.exp(-(sort(-Yi)))-1.0
		Wi = numpy.log(numpy.exp(1) + arange(j1 - j0))
		Yi_pred = numpy.divide(Yi_pred, Wi)
		Yi_best = numpy.divide(Yi_best, Wi)
		for k in xrange(len(Ks)):
			K = Ks[k]
			Ki = min([K, j1 - j0])
			res[k] += sum(Yi_pred[0:Ki]) / sum(Yi_best[0:Ki])
	assert(nvalid > 0)
	res /= nvalid
	return res




#####################################
# Test MAP
#####################################

def test_map():
	import scipy.io
	import time
	import scipy
	import scipy.sparse
	matdat = scipy.io.loadmat('D:\\Users\\nliu\\PyCF\\map_test.mat')
	Y = matdat['Ytest']
	X = matdat['Xtest']
	
	#X.data = scipy.random.rand(X.data.size)
#	tic = time.time()
#	res1 = map_old(X, Y)
#	toc = time.time()
#	print "res1 : %f , time : %f" % (res1, toc - tic)
	
	tic = time.time()
	res2 = map(X, Y)
	toc = time.time()
	print "res2 : %f , time : %f" % (res2, toc - tic)





if __name__ == "__main__":
	#test_metrics()
	pass
	
	
	
	
	
