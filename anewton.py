'''
	implements relational als that is the core algorithm for all
	models in this package
'''
import numpy

def update(Us, Xs, Xts, rc_schema, r0s, r1s, alphas, modes, Ns, t, C, K, step):
	assert(t <= len(Ns) and t >= 0)
	eyeK = C * numpy.eye(K, K)
	N = Ns[t] # number of instances for type t
	V = Us[t]
	A = numpy.zeros((K, K)) # place holders for hessian
	b = numpy.zeros(K) # place holders for gradient
	UtUs = numpy.empty(len(Xs),object)
	change = 0
	for j in xrange(len(Xs)):
		if modes[j] == 'densemf':
			if rc_schema[0, j] == t:		
				U = Us[rc_schema[1, j]]
			else:
				U = Us[rc_schema[0, j]] 
			UtUs[j] = numpy.dot(U.T,U)
	for i in xrange(N):
		A[:] = 0
		b[:] = 0
		for j in xrange(len(Xs)):
			if alphas[j] == 0:
				continue
			if rc_schema[0, j] == t or rc_schema[1, j] == t:
				if rc_schema[0, j] == t:
					X = Xts[j]
					U = Us[rc_schema[1, j]]
				else:
					X = Xs[j]
					U = Us[rc_schema[0, j]]
				data = X.data
				indptr = X.indptr
				indices = X.indices
				
				ind_i0, ind_i1 = (indptr[i], indptr[i+1])
				if ind_i0 == ind_i1:
					continue
				
				inds_i = indices[ind_i0:ind_i1] 
				data_i = data[ind_i0:ind_i1]
				
				if modes[j] == "densemf": # square loss, dense binary representation
					UtU = UtUs[j]
					Utemp = U[inds_i, :]
					A += alphas[j] * UtU
					b += alphas[j] * (numpy.dot(UtU,V[i,:])-numpy.dot(data_i, Utemp))
				elif modes[j] == "denselogmf": # logistic loss
					Xi = numpy.dot(U, V[i, :])
					Yi = - 1 * numpy.ones(U.shape[0])
					Yi[inds_i] = 1
					# (sigma(yx)-1)
					Wi = 1.0 / (1 + numpy.exp(-1 * numpy.multiply(Yi, Xi))) - 1 
					Wi = numpy.multiply(Wi, Yi)
					gv = numpy.dot(Wi, U)
					# compute sigmoid(x)
					Ai = 1 / (1 + numpy.exp(-Xi))
					Ai = numpy.multiply(Ai, 1 - Ai)
					Ai = Ai.reshape(Ai.size, 1)
					AiU = numpy.multiply(Ai, U)
					Hv = numpy.dot(AiU.T, U)
					A += alphas[j] * Hv
					b += alphas[j] * gv
					
				elif modes[j] == "sparsemf": # square loss
					Utemp = U[inds_i, :]
					UtU = numpy.dot(Utemp.T, Utemp)
					A += alphas[j] * UtU
					b += alphas[j] * (numpy.dot(UtU, V[i,:])-numpy.dot(data_i, Utemp))
					
				elif modes[j] == "sparselogmf": # sparse logistic loss, sparse representation
					if r1s[j] != r0s[j]:
						Xi = (data_i-r0s[j])/(r1s[j]-r0s[j])
					else:
						Xi = data_i
					Ui = U[inds_i,:]
					Yi = numpy.dot(Ui, V[i,:])
					# sigmoid(Y)
					Wi = 1.0/(1+numpy.exp(-Yi)) 
					b += alphas[j] * numpy.dot(Wi-Xi, Ui)
					Wi = numpy.multiply(Wi,1-Wi)
					UiW = numpy.multiply(Wi.reshape(Wi.size,1),Ui)
					A += alphas[j] * numpy.dot(UiW.T, Ui)
				else:
					assert False, "Unrecognized mode: %s"%modes[j]
		A += eyeK
		b += C*V[i, :]
		d = numpy.dot(numpy.linalg.inv(A), b)
		vi = V[i,:].copy()
		V[i, :] -= step*d
		change += numpy.linalg.norm(vi-V[i,:])/(numpy.linalg.norm(vi)+1e-10)
	return change



	