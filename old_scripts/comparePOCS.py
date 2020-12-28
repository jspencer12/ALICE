# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:19:24 2017

@author: Jonathan
"""

import numpy as np
import time

TOLERANCE = 1e-8

def projSimplex(X,axis,c,mode='on'):
    #Only works for 2 dim
    
    if axis==0:
        Proj = X[:,:].T
    elif axis==1:
        Proj = X[:,:]
    else:
        #except(ValueError,'axis must be 1 or 0')
        print('error')
        return
    if mode=='on':
        minvec = np.min(Proj,axis=1)
        Proj = Proj-(minvec*(minvec<0))[:,np.newaxis]
    elif mode=='under':
        #zero out negative values
        Proj = Proj-Proj*(Proj<0)
    elif mode=='max':
        MaxProj = np.zeros(Proj.shape)
        for row in range(MaxProj.shape[0]):
            MaxProj[row,np.argmax(Proj[row,:])]=c
        if axis==1:
            return MaxProj
        elif axis==0:
            return MaxProj.T
    else:
        print('error')
        return
    ordered_inds = np.int_(np.argsort(Proj,axis=1))
    marg = np.sum(Proj,axis=1)
    N = len(marg)
    Niter = np.zeros(marg.shape,dtype=int)
    ones = np.ones(marg.shape,dtype=int)
    support_elems = np.ones(Proj.shape,dtype=int)
    if mode=='under':
        #We only want to shrink them to simplex if they are above simplex
        vecs_2_shrink = (marg>c)*1
    elif mode=='on':
        #If we are going onto the simplex, then we shrink all vectors
        vecs_2_shrink = ones[:]
    while max(marg)>(c+TOLERANCE):
        #print 'proj',Proj
        #print 'support',support_elems
        zero_elems = (Proj==0)
        #Subtract smallest vector elements from all vectors with marg too big
        #ordered_inds[range(N),Niter] returns index of next largest element each time
        Proj = Proj-~zero_elems*((Proj[range(N),ordered_inds[range(N),Niter]])*(marg>c))[:,np.newaxis]
        #Increase iteration count for those you subtracted
        #Niter stops incrementing at the first iteration for which marg>c
        Niter = Niter+ones*(marg>c)
        #Recompute support elements
        support_elems = support_elems-(support_elems*zero_elems)*(marg>c)[:,np.newaxis]
        #Recompute marginal sum
        marg = np.sum(Proj,axis=1)
    
    #how much left do we have to get the sum to equal c?
    residual = c-marg
    #how many vectors are we dividing it among?]
    support = np.sum(support_elems,axis=1)
    #how much do we bump each vector? (only bump the ones that mode dicatates)
    bump = ((1.0*residual/support)*vecs_2_shrink)[:,np.newaxis]
    #Bump the support entries
    Proj = Proj+support_elems*bump
         
    if axis==1:
        return Proj
    elif axis==0:
        return Proj.T
    
def proj_simplex(y,c=1.0):
    #c is simplex scale factor
    n = len(y)
    val = -np.sort(-y)
    suppt_v = np.cumsum(val) - np.arange(1,n+1,1) * val
        #This allows for multiple support elements at the boundary
    k_act = np.sum(suppt_v < c)
    lam = (np.sum(val[0:k_act]) - c) / k_act
    x = np.maximum(y-lam,0.0)
    return x

def POCS(Gin,g_max,N_run,verbose):
    #Efficiently copy
    G = np.array(Gin)
    n_iter = 0
    while n_iter<N_run:
        # and (n_iter<POCS_maxiter)
        n_iter += 1
        #project rows on
        for i in range(G.shape[0]):
            G[i,:]=proj_simplex(G[i,:])
        #project columns under
        for j in range(G.shape[1]):
            gj = np.maximum(G[:,j],0.0)
            if np.sum(gj)>g_max:
                gj = proj_simplex(gj,g_max)
            G[:,j] = gj
        
        #check for convergence
        dist_onto = np.sum(abs(np.sum(G,1)-1))
        dist_under = np.sum(np.maximum(np.sum(G,0)/g_max-1,0.0))
        distance_from_simplexes = np.maximum(dist_onto,dist_under)
        
    if verbose:
        print('POCS Converged to {:.2e} in {:.2f} iterations.'.format(distance_from_simplexes,n_iter))
    return G

def POCS2(Gin,g_max,N_run,verbose):
    G = np.array(Gin)
    POCS_iters = 0
    
    while POCS_iters<N_run:
        POCS_iters +=1
        G = projSimplex(G,axis=1,c=1,mode='on')
        G = projSimplex(G,axis=0,c=g_max,mode='under')
        dist_onto = np.sum(abs(np.sum(G,1)-1))
        dist_under = np.sum(np.maximum(np.sum(G,0)/g_max-1,0.0))
        dist_away = np.maximum(dist_onto,dist_under)
    if verbose:
        print('POCS2 Converged to {:.2e} in {:.2f} iterations.'.format(dist_away,POCS_iters))
    return G

np.random.seed(1)
N_run = 1000
(N,M) = (600,60)
print('Matrix Dim: ({},{}), {} iterations'.format(N,M,N_run))
g_max = N/M #this is the minimum value possible
G1 = np.zeros((N,M))
for i in range(N):
    G1[i,int(np.floor(np.random.rand()*M))]=1
G2 = np.array(G1)
start1 = time.time()
G1_out = POCS(G1,g_max,N_run,True)
time1 = time.time()-start1
start2 = time.time()
G2_out = POCS2(G2,g_max,N_run,True)
time2 = time.time()-start2
print('Andrew POCS: {:.2f} seconds'.format(time1))
print('Jonathan POCS: {:.2f} seconds'.format(time2))
print('Matrix max difference: {:.4e}'.format(np.max(abs(G1_out-G2_out))))

#Observations, for matrix dimensions larger than 500, Andrew wins (by a lot).
#Otherwise, Jonathan wins (by a lot).
#Jonathan's scales by roughly M*N, whereas Andrew's scales just by largest dim