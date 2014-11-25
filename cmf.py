'''
Collective Matrix Factorization
Input: data1.mat (item X user), data2.mat (item X user)
Eval: RMSE, MAE
--------------------------------------
Author: Zhongqi Lu (zluab@cse.ust.hk)
http://ihome.ust.hk/~zluab
Sep. 2011
--------------------------------------
'''

import numpy
import time
import anewton
import scipy.sparse

def learn(Xs, Xstst, rc_schema, r0s, r1s, alphas, modes, K, C, T=40, tol=0.005):
    assert(rc_schema.shape[1] == len(Xs) and rc_schema.shape[0] == 2) # schema match data
    assert(numpy.all(rc_schema[0, :] != rc_schema[1, :])) # should not have symmetric relations
    assert(r0s != None and r1s != None)
    assert(alphas != None)
    res = 0
    Xts = numpy.empty(len(Xs),object)
    
    
    for i in xrange(len(Xs)):
        Xts[i] = Xs[i].T.tocsc()
        if modes[i] == 'sparselogmf' or modes[i] == 'denselogmf':
            assert r0s[i]!=r1s[i]
        
    # S: numober of types, Ns: sizes of each type 
    [S, Ns] = rel_config(Xs, rc_schema)
    print S,Ns

    # random initialize factor matrices
    Us = numpy.empty(S,object)
    print Us.shape
    for i in xrange(S):
        Us[i] = numpy.random.rand(Ns[i], K)/K

    i = 0
    step = 0.6
    tgood = 0
#    loss_prev = loss(Us, Xs, rc_schema, r0s, r1s, modes, alphas, C)
    while i < T:
        i += 1
        tic = time.time()
        change = 0
        for t in xrange(S):
            change += anewton.update(Us, Xs, Xts, rc_schema, r0s, r1s, alphas, modes, Ns, t, C, K, step)
        change /= numpy.sum(Ns)
        toc = time.time()

        if Xstst == None:
            print "iter %d,  change %f, time %f" % (i, change, toc - tic)
        else:
            loss_tst = loss(Us, Xstst, rc_schema, r0s, r1s, modes, alphas)
            print "iter %d,  tst loss %.2f, change %f, time %.2f" % (i, loss_tst, change, toc - tic)
        if change < tol:
            print "Early terminate due to insufficient change!"
            break
    return [Us, r0s, r1s]


def loss(Us, Xs, rc_schema, r0s, r1s, modes, alphas, C=0):
    assert(rc_schema.shape[1] == len(Xs) and rc_schema.shape[0] == 2)
    res = 0
    for U in Us:
        res += C*numpy.dot(U.flat,U.flat)
    for t in xrange(len(Xs)):
        alpha_t = alphas[t]
        X = Xs[t]
        if X==None or X.size == 0 or alpha_t == 0:
            continue
        data = X.data
        indices = X.indices
        indptr = X.indptr
        ri = rc_schema[0, t]
        ci = rc_schema[1, t]
        U = Us[ri]
        V = Us[ci]
        if modes[t] == "densemf" or modes[t] == "denselogmf":
            X_i = numpy.zeros(X.shape[0])
        r0 = r0s[t]
        r1 = r1s[t]
        # computing loss for each matrix
        if modes[t] == 'densemf':
            for i in xrange(X.shape[1]):
                # compute loss for each column
                inds_i = indices[indptr[i]:indptr[i+1]]
                if inds_i.size == 0:
                    continue
                inds_i = indices[indptr[i]:indptr[i+1]]
                Y_i = numpy.dot(U,V[i,:])
                X_i[inds_i] = data[indptr[i]:indptr[i+1]]
                res += alpha_t * numpy.sum((Y_i-X_i)**2)
                X_i[inds_i] = 0
                
        elif modes[t] == 'denselogmf':            
            for i in xrange(X.shape[1]):
                # compute loss for each column
                inds_i = indices[indptr[i]:indptr[i + 1]]
                if inds_i.size == 0:
                    continue
                # HACK: currently assume the matrix is binary
                X_i[:] = - 1
                Y_i = numpy.dot(U, V[i, :])
                X_i[inds_i] = 1
                res += alpha_t * numpy.sum(numpy.log(1 + numpy.exp(-1 * numpy.multiply(Y_i, X_i))))
        
        elif modes[t] == 'sparsemf':
            for i in xrange(X.shape[1]):
                # compute loss for each column
                inds_i = indices[indptr[i]:indptr[i + 1]]
                if inds_i.size == 0:
                    continue
                Y_i = numpy.dot(U[inds_i, :], V[i, :])
                X_i = data[indptr[i]:indptr[i + 1]]
                res += alpha_t * numpy.sum((Y_i - X_i) ** 2)
        
        elif modes[t] == 'sparselogmf':
            for i in xrange(X.shape[1]):
                # compute loss for each column
                inds_i = indices[indptr[i]:indptr[i + 1]]
                if inds_i.size == 0:
                    continue
                Yi = 1/(1+numpy.exp(-numpy.dot(U[inds_i,:],V[i,:])))
                Xi = (data[indptr[i]:indptr[i+1]]-r0)/(r1-r0) #normalize the data to [0,1]
                res -= alpha_t*numpy.sum(numpy.multiply(Xi,numpy.log(Yi)))
                res -= alpha_t*numpy.sum(numpy.multiply(1-Xi,numpy.log(1-Yi)))
        
        else:
            assert False,'Unrecognized mode %s'%modes[t]
    return res

'''
    get neccessary configurations of the given relation
    S : number of entity types
    Ns :  number of instances for each entity type
'''
def rel_config(Xs, rc_schema):
    
    S = rc_schema.max() + 1
    Ns = -1 * numpy.ones(S, int)
    for i in xrange(len(Xs)):
        
        ri = rc_schema[0, i]
        ci = rc_schema[1, i]
        
        [m, n] = Xs[i].shape
        
        if Ns[ri] < 0:
            Ns[ri] = m
        else:
            assert(Ns[ri] == m)
                            
        if Ns[ci] < 0:
            Ns[ci] = n
        else:
            assert(Ns[ci] == n)
    return [S, Ns]


def predict(Us, Xs, rc_schema, r0s, r1s, modes):
    Ys = []
    for i in xrange(len(Xs)):
        X = Xs[i]
        if X == None:
            Ys.append(None)
            continue
        ri = rc_schema[0, i]
        ci = rc_schema[1, i]
        U = Us[ri]
        V = Us[ci]
        data = X.data.copy()
        indices = X.indices.copy()
        indptr = X.indptr.copy()
        r0 = r0s[i]
        r1 = r1s[i]
        if modes[i] == "sparselogmf" or modes[i] == "denselogmf":
            for j in xrange(X.shape[1]):
                inds_j = indices[indptr[j]:indptr[j + 1]]
                if inds_j.size == 0:
                    continue
                data[indptr[j]:indptr[j + 1]] = r0+(r1-r0)*(1.0/(1.0+numpy.exp(-numpy.dot(U[inds_j,:],V[j,:]))))
        else:
            for j in xrange(X.shape[1]):
                inds_j = indices[indptr[j]:indptr[j + 1]]
                if inds_j.size==0:
                    continue
                data[indptr[j]:indptr[j + 1]] = numpy.dot(U[inds_j, :], V[j, :])
        Y = scipy.sparse.csc_matrix((data, indices, indptr), X.shape)
        Ys.append(Y)
    return Ys


def test_cmf():
    import scipy.io
    import cfeval
  
    matdata1 = scipy.io.loadmat('data1.mat')
    matdata2 = scipy.io.loadmat('data2.mat')
    
    Xtrn = matdata1['Xtrn']
    print Xtrn.shape
    Xaux = matdata2['Xtrn'] + matdata2['Xtst'] 
    Xaux = Xaux.T.tocsc()
    print Xaux.shape
    Xtst = matdata1['Xtst']
    print Xtst.shape
   
    Xs_trn = [Xtrn, Xaux]
    Xs_tst = [Xtst, None]
    
    rc_schema = numpy.array([[0, 2], [1, 0]])
    C = 0.9
    K = 30
    alphas = [0.9, 0.1]
    T = 20
    modes = numpy.zeros(len(Xs_trn), object)
    modes[0] = 'sparsemf'
    modes[1] = 'densemf'
    r0s = [1.0, 0.0]
    r1s = [5.0, 1.0]

    [Us, r0s, r1s] = learn(Xs_trn, Xs_tst, rc_schema, r0s, r1s, alphas, modes, K, C, T)
    print '******'
    print Us[0].shape
    print Us[1].shape
    print Us[2].shape
    print '********'
    Vt = scipy.sparse.csc_matrix(Us[0],dtype=float)
    Ut= scipy.sparse.csc_matrix(Us[1],dtype=float)
    
    Ys_tst = predict(Us, Xs_tst, rc_schema, r0s, r1s, modes)
    #tst
    X = Xs_tst[0]
    Y = Ys_tst[0]
    print Y.shape
   
    
    print "K: %d, C: %f"%(K, C)
    print "alphas: ",alphas
    print "rmse: %.4f , mae: %.4f\n" % (cfeval.rmse(X, Y), cfeval.mae(X, Y))
if __name__ == "__main__":
    test_cmf()
