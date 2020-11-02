import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
import itertools

EPS = 1e-12

def estimate_ratio_on_samps(x_0,x_1,feature_map_pipeline='',warm_start=False):
    '''Given distributions x_0,x_1 (lists of items), we flatten them, pass them
     through the feature_map_pipeline specified, and perform logistic regression,
     then return an array of ratio estimates r([x_0,x_1])

     feature_map_pipeline applies the following mappings space-delineated in order
        pca-[n]         Principal Component analysis with n components
        poly-[d]        Polynomial feature map of degree d
        std_scaler      StandardScaler, shifts dataset to zero mean, unit variance
        qtl_transform   QuantileTransform, squishes dataset to 0-1 range by quantile
        rff-[n]         RandomFourierFeatures with n components

     ie. feature_map_pipeline = 'std_scaler pca-100 poly-2'   '''
    
    X = np.stack([x.flatten() for x in itertools.chain(x_0,x_1)])
    N_0,N_1 = len(x_0),len(x_1)
    y = np.hstack([np.zeros((N_0,)),np.ones((N_1,))])
    pipeline_list = []
    for fm in feature_map_pipeline.split(' '):
        name,num = fm.split('-')[0],int(fm.split('-')[1]) if '-' in fm else None
        if name == 'linear':
            pass
        if name =='poly':
            pipeline_list.append((name, PolynomialFeatures(degree=num or 2)))
        if name =='standardscalar':
            pipeline_list.append((name, StandardScaler()))
        if name =='quantilescalar':
            pipeline_list.append((name, QuantileTransformer()))
        if name =='pca':
            pipeline_list.append((name, PCA(n_components=num or 100)))
        if name =='rff':
            pipeline_list.append((name, RBFSampler(n_components=num or 100)))
    pipeline_list.append(('lr',LogisticRegression(warm_start=warm_start)))
    clf = Pipeline(pipeline_list)
    clf.fit(X,y)
    proba = clf.predict_proba(X) #class 1 is learner, class 0 is expert
    ratio = proba[:,1]*N_0/(proba[:,0]*N_1+EPS)
    return ratio

if 1:
    
    N_samp0,N_samp1 = 200,800
    dim = 20

    mu0 = np.zeros((dim,))
    mu1 = np.ones((dim,))*(dim**2)
    sig0 = np.ones((dim,))*2*dim**.25
    sig1 = np.ones((dim,))*dim**.25
    eps = 1e-9
    p_0 = lambda x : np.exp(np.sum(-.5*((mu0-x)/sig0)**2-np.log(sig0)-np.log(2*np.pi)/2,axis=1))
    p_1 = lambda x : np.exp(np.sum(-.5*((mu1-x)/sig1)**2-np.log(sig1)-np.log(2*np.pi)/2,axis=1))
    r_true = lambda x : p_1(x)/(p_0(x)+EPS)+EPS

    x0 = np.random.randn(N_samp0,dim)*sig0 + mu0
    x1 = np.random.randn(N_samp0,dim)*sig1 + mu1
    y = np.hstack([np.zeros((N_samp0,)),np.ones((N_samp1,))])
    X = np.vstack([x0,x1])
    

    ############################################################################
    #               Try out different things here!!!
    ############################################################################
    estimation_pipelines = ['linear',
                            'rff-100',
                            'rff-100 poly-2']

    r_hats = [estimate_ratio_on_samps(x0,x1,pipe) for pipe in estimation_pipelines]
    r = r_true(X)
    print(max(r),min(r),np.mean(r),'r max min mean')
    plt.figure()
    if dim==1:
        plt.hist(x0[:,0],50,density=True,histtype='step',label='phat0',)
        plt.hist(x1[:,0],50,density=True,histtype='step',label='phat1',)
        plt.scatter(X[:,0],p_0(X),label='p0')
        plt.scatter(X[:,0],p_1(X),label='p1')
    else:
        plt.yscale('log')
    plt.scatter(X[:,0],r,marker='.',label='r')
    for i in range(len(r_hats)):
        plt.scatter(X[:,0],r_hats[i],marker='+',label=estimation_pipelines[i],alpha=.2)
    
    plt.legend()
    plt.title('1-d projection of ratios')

    plt.figure()
    for i in range(len(r_hats)):
        err = np.abs(r-r_hats[i])
        print(estimation_pipelines[i],'Mean Error',np.mean(err))
        if np.mean(err)<100:
            plt.hist(err,1000,density=True,histtype='step',range=(.1,100),cumulative=True,label=estimation_pipelines[i])
    plt.xscale('log')
    plt.title('Absolute Ratio Estimation Error CDF')
    plt.legend()
    plt.show()

