import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from itertools import cycle
import scipy.spatial
from sklearn.pipeline import Pipeline

def median_trick_inv_diag(X, scale = 1.):
    N = X.shape[0]
    perm = np.random.permutation(N)
    X_perm = np.copy(X[perm])

    dd = (X - X_perm)**2
    medians = np.median(dd, axis=0) + 1e-7
    medians = scale*medians
    inv_medians = 1./medians
    assert len(medians) == X.shape[1]
    return np.diag(inv_medians) #covariance matrix, inversed already

def RBF_kernel_cov(X, diag_inv_cov):
    XX = X.reshape(X.shape[0], 1, X.shape[1])
    dd = XX - X
    mat = np.sum(dd.dot(diag_inv_cov)*dd,axis=2) 
    return np.exp(-mat)

def RBF_kernel_cov(X, Y, diag_inv_cov):
    YY = Y.reshape(Y.shape[0], 1, Y.shape[1])
    dd = YY - X
    mat = np.sum(dd.dot(diag_inv_cov)*dd,axis=2) 
    return np.exp(-mat)


def median_trick(X, scale = 1.):
    '''
    median trick for computing the sigma for RBF kernel, exp(-\|x-x'\|^2/sigma^2)
    '''
    N = X.shape[0]
    pdists = scipy.spatial.distance.pdist(X) #compute all pairwise euclidean distances. Warning: could be slow when N is large.
    return scale*np.median(pdists) #return the median as sigma.

def compute_kernel_mat(x,atoms,kernel_type='gaussian',**kwargs):
    '''
    computes len(x) x len(atoms) kernel matrix
    
    kernel types:
        gaussian
        linear
    
    args
        bias  - T/(F) adds an extra unity feature (dim 1 is now len(atoms)+1)
        
    '''
        #Feature map vs kernel mat
    dim = np.array(x).shape[1]
    if kernel_type=='gaussian':
        sigma = kwargs.get('sigma',1)
        K = np.ones((len(x),len(atoms)+kwargs.get('bias',False)))
        for i in range(len(x)):
            for j in range(len(atoms)):
                K[i,j] = np.exp(-.5*np.linalg.norm(x[i]-atoms[j])**2/sigma**2)/(sigma*np.sqrt(2*np.pi))**dim
    if kernel_type=='linear':
        K = np.ones((len(x),dim+kwargs.get('bias',False)))
        for i in range(len(x)):
            K[i,:dim] = x[i]
    #if kernel_type=='quadratic':
    return K

def compute_MMD(X,c,wen=True,**kwargs):
    '''
        Computes Max Mean Discrepancy between two dists P,Q using rbf kernel to approx.
        
        data of dim d, n samples from P, m samples from Q, c is (potentially weighted) label vector
        X = [xP1,...,xPn,xQ1,...,xQm] /in R^(n+m x d)
        c = [1/n,...,1/n,-1/m,...,-1/m]
        
        When we use these samples as basis for RKHS, f(x,w) = sum_i w_i * k(X[i,:],x)
        
        MMD = max_{w: |w|_2<=1}  E_P[f(x,w)]-E_Q[f(x,w)]
            = max_{w: |w|_2<=1}  w^T sum_j  k(.,X[j,:]) c_j
            = max_{w: |w|_2<=1}  sum_i,j  w_i k(X[i,:],X[j,:]) c_j
            = max_{w: |w|_2<=1}  w^T (K c)  (in the lit, K c = mu_p-mu_q, maximizing inner product of x,y just let x=y)
            = sqrt( c^T K K c )     where w = K c /sqrt( c^T K K c )
        
        Evaluating f(x,w) on some set of points Y 
            f(Y) = w^T k(X,Y) = c^T K k(X,Y)/sqrt( c^T K K c )
    '''
    K = compute_kernel_mat(X,X,**kwargs)
    mu_diff = np.dot(c,K)
    if wen:
        w = c/np.sqrt(np.dot(c,K).dot(c))
    else:
        w = mu_diff/np.sqrt(np.inner(mu_diff,mu_diff))
    f = lambda Y: np.dot(w,compute_kernel_mat(X,Y,**kwargs))
    return np.dot(w,K),f


class KernelDensityEstimator:
    def __init__(self,**kwargs):
        print(f'Kernel Density Estimator')
        pass
    def fit(self,X,y,**kwargs):
        assert X.shape[0]==len(y), 'Need data of shape (N_samp,M_feat) and labels of shape (N_samp), even when N_samp or M_feat=1, X shape: {}, y shape: {}'.format(X.shape,y.shape)
        inds1 = np.nonzero(y)[0]
        inds0 = np.delete(np.arange(len(y)),inds1)
        self.X0 = X[inds0,:]
        self.X1 = X[inds1,:]
        self.fit_kwargs = kwargs
        return self
    def predict_proba(self,x):
        assert np.array(x).shape[1] == self.X0.shape[1], 'Need 2-d input of shape (N_samp,M_feat), even when N_samp or M_feat=1'
        K0,K1 = compute_kernel_mat(x,self.X0),compute_kernel_mat(x,self.X1)
        return np.mean(K0,axis=1),np.mean(K1,axis=1)
    def predict_ratio(self,x):
        K0,K1 = compute_kernel_mat(x,self.X0),compute_kernel_mat(x,self.X1)
        r = np.mean(K1,axis=1)/np.mean(K0,axis=1)
        return r

class VanillaLogisticRegression:
    def __init__(self,**kwargs):
        self.vlr = LogisticRegression(**kwargs)
        print(f'Vanilla Logistic Regression')
    def fit(self,X,y,**kwargs):
        self.N1 = len(np.nonzero(y)[0])
        self.N0 = len(y)-self.N1
        self.vlr.fit(X,y)
        return self
    def predict_proba(self,x,**kwargs):
        return self.vlr.predict_proba(x,**kwargs)
    def predict_ratio(self,x,**kwargs):
        p_pred_LR = self.predict_proba(x,**kwargs)
        return p_pred_LR[:,1]*self.N0/(p_pred_LR[:,0]*self.N1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

class PolynomialLogisticRegression:
    def __init__(self,degree=2,**kwargs):
        self.clf = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('lr', LogisticRegression(**kwargs))])
        print(f'Degree {degree} Polynomial Logistic Regression')
    def fit(self,X,y,**kwargs):
        self.N1 = len(np.nonzero(y)[0])
        self.N0 = len(y)-self.N1
        self.clf.fit(X,y)
        return self
    def predict_proba(self,x,**kwargs):
        return self.clf.predict_proba(x,**kwargs)
    def predict_ratio(self,x,**kwargs):
        p_pred_LR = self.predict_proba(x,**kwargs)
        return p_pred_LR[:,1]*self.N0/(p_pred_LR[:,0]*self.N1)
        
class KernelLogisticRegression:
    def __init__(self,**kwargs):
        print(f'Kernel Logistic Regression')
        self.rand_state = kwargs.get('random_state',None)
    def fit(self,X,y,learning_rate = 0.01, init_params=None, steps_max = 1e5, grad_thresh = 1e-3,batch_size=32,**kwargs):
        '''ratio is P1(x)/P0(x) where 1 and 0 are labels'''
        self.X = X
        self.N_samp = X.shape[0]
        self.M_feats = X.shape[1]
        self.fit_kwargs = kwargs
        assert X.shape[0]==len(y), 'Need data of shape (N_samp,M_feat) and labels of shape (N_samp), even when N_samp or M_feat=1, X shape: {}, y shape: {}'.format(X.shape,y.shape)
        self.y = y
        self.inds1 = np.nonzero(y)[0]
        self.inds0 = np.delete(np.arange(len(y)),self.inds1)
        np.random.seed(self.rand_state)
        ind0_cycle = cycle(np.random.permutation(self.inds0)) #random shuffle inds
        ind1_cycle = cycle(np.random.permutation(self.inds1))
        lam = .1
        K = compute_kernel_mat(X,X,**self.fit_kwargs)
        N_feat = K.shape[1]
        self.w = init_params
        if type(init_params) is str or init_params is None:
            self.w = {'ones':np.ones((N_feat)),
                      'zeros':np.zeros((N_feat))}.get(init_params,np.random.randn(N_feat))

        step_num = 0
        grad_norm = grad_thresh*2
        while step_num<steps_max/batch_size and grad_norm>grad_thresh:
            rand_ind0,rand_ind1 = [next(ind0_cycle) for i in range(batch_size)],[next(ind1_cycle) for i in range(batch_size)]
            grad = np.mean(K[rand_ind0,:]*np.tile(np.reshape((1-1/(1+np.exp( np.dot(K[rand_ind0,:],self.w)))),(len(rand_ind0),1)),(1,N_feat)),axis=0) \
                 - np.mean(K[rand_ind1,:]*np.tile(np.reshape((1-1/(1+np.exp(-np.dot(K[rand_ind1,:],self.w)))),(len(rand_ind1),1)),(1,N_feat)),axis=0) \
                 + 2*lam*self.w
            grad_norm = np.linalg.norm(grad)
            self.w = self.w - learning_rate*grad
            if not step_num % 100: print(step_num,grad_norm)
            if not step_num%((max(len(self.inds0),len(self.inds1))//batch_size+1)*10):
                #reshuffle inds every so often
                ind0_cycle = cycle(np.random.permutation(self.inds0))
                ind1_cycle = cycle(np.random.permutation(self.inds1))
            step_num += 1
        print('Finished after {step_num} steps with grad norm {grad_norm:.3g}'.format(step_num=step_num,grad_norm=grad_norm))
        return self
    def predict_ratio(self,x):
        assert x.shape[1]==self.X.shape[1], 'Need data of shape (N_samp,M_feat)'
        K = compute_kernel_mat(x,self.X,**self.fit_kwargs)
        r = np.exp(np.dot(K,self.w))
        return r

if __name__=='__main__':

    N_samp0,N_samp1 = 500,500
    mu0,sig0 = 0,1
    mu1,sig1 = 0,1
    x0 = np.random.normal(mu0,sig0,(N_samp0,))
    x1 = np.random.laplace(mu1,sig1,(N_samp1,))
    X = np.hstack([x0,x1])[:,np.newaxis]
    c = np.hstack([np.ones((N_samp0,))/N_samp0,-np.ones((N_samp1,))/N_samp1])
    f_X,f_wen = compute_MMD(X,c,wen=True,sigma=.5)
    f_X,f = compute_MMD(X,c,wen=False,sigma=.5)
    xaxis = np.linspace(-5,5,201)

    mmd_wen = f_wen(xaxis[:,np.newaxis])
    mmd = f(xaxis[:,np.newaxis])
    pdf_true_0 = np.exp(-.5*(xaxis-mu0)**2/sig0**2)/(sig0*np.sqrt(2*np.pi))
    pdf_true_1 = np.exp(-np.abs(xaxis-mu1)/sig1)/(sig1*2)
    import matplotlib.pyplot as plt
    if 1:
        plt.figure('Densities')
        plt.plot(xaxis,pdf_true_0,label='p0(x)',lw=2)
        plt.plot(xaxis,pdf_true_1,label='p1(x)',lw=2)
        plt.plot(xaxis,mmd,label='MMD(x)',lw=2)
        plt.plot(xaxis,mmd_wen,label='MMD_wen(x)',lw=2)
        plt.legend()
        plt.show()

if __name__=='__main_':

    N_samp0,N_samp1 = 200,200
    mu0,sig0 = 0,2
    mu1,sig1 = 1,2
    x0 = np.random.normal(mu0,sig0,(N_samp0,))
    x1 = np.random.normal(mu1,sig1,(N_samp1,))
    X = np.hstack([x0,x1])[:,np.newaxis]
    y = np.hstack([np.zeros((N_samp0,)),np.ones((N_samp1,))])
    #X = x0[:,np.newaxis] #np.hstack([x0,x1])[:,np.newaxis]
    #y = np.zeros((N_samp0,))
    xaxis = np.linspace(-5,5,100)
    pdf_true_0 = np.exp(-.5*(xaxis-mu0)**2/sig0**2)/(sig0*np.sqrt(2*np.pi))
    pdf_true_1 = np.exp(-.5*(xaxis-mu1)**2/sig1**2)/(sig1*np.sqrt(2*np.pi))
    r_true = pdf_true_1/pdf_true_0

    #Estimation techniques:
    vlr = VanillaLogisticRegression(random_state=0).fit(X, y)
    rpred_VLR = vlr.predict_ratio(xaxis[:,np.newaxis])

    kde = KernelDensityEstimator().fit(X,y,sigma=.5)
    pdf_KDE_0,pdf_KDE_1 = kde.predict_proba(xaxis[:,np.newaxis])
    rpred_KDE = kde.predict_ratio(xaxis[:,np.newaxis])

    klr = KernelLogisticRegression().fit(X,y,batch_size=32,bias=True,learning_rate = 0.01, steps_max = 1e5)
    rpred_KLR = klr.predict_ratio(xaxis[:,np.newaxis])

    import matplotlib.pyplot as plt
    if 1:
        plt.figure('Densities')
        plt.plot(xaxis,pdf_true_0,label='p0(x)',lw=2)
        plt.plot(xaxis,pdf_true_1,label='p1(x)',lw=2)
        plt.plot(xaxis,pdf_KDE_0,label='p0_KDE(x)',lw=2,ls='--')
        plt.plot(xaxis,pdf_KDE_1,label='p1_KDE(x)',lw=2,ls='--')
        plt.hist(x0,density=True,bins=20,histtype='step',label='phat0',lw=2)
        plt.hist(x1,density=True,bins=20,histtype='step',label='phat1',lw=2)
        plt.legend()
        #plt.show()
    if 1:
        plt.figure('Ratio')
        plt.plot(xaxis,r_true,label='r(x)')
        plt.plot(xaxis,rpred_KDE,label='rpred_KDE',ls='--')
        plt.plot(xaxis,rpred_VLR,label='rpred_LR',ls='--')
        plt.plot(xaxis,rpred_KLR,label='rpred_KLR',ls='--')
        plt.legend()
        plt.show()

    #Logistic regression with gaussian kernels
    #Kernel density estimation
    #Dimensionality reduction, train PCA on all expert data then use local reduction (LDA)

#Arun is going to chat about career plans 
#   A) interest in industry, 
#   B) Drew cares about getting ALICE on the car
#   C) 5 wks left, what's the most productive use of my time


