import numpy as np
import pandas as pd
from scipy.stats import genpareto, probplot, multivariate_normal, norm, poisson, uniform
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from sympy import *
from sympy.stats import MultivariateNormal, density, Poisson, Normal
import cupy as cp
import os

# Disable memory pool for device memory (GPU)
cp.cuda.set_allocator(None)

# Disable memory pool for pinned memory (CPU).
cp.cuda.set_pinned_memory_allocator(None)

#makes directory if it doesn't exist
try:
    os.mkdir('CDFs')
except Exception:
    pass


v1s = [395.84] #extrapolated volumes

# files = ['CMU01-08-2', 'CMU04-03', 'CMU02-05-1', 'CMU04-26', 'CMU03-30-2', 'CMU04-04']

dx = 0.1 #resolution of CDF

df = pd.read_csv('data/parameters.csv') #read in csv of thresholds

poredf = pd.read_csv('data/poredf.csv')

#Probability of poisson distribution with gaussian uncertainty in estimator 
def getprobpoiss(x, dl, fdl):
    y = Pow(dl,x) * exp(-dl) / factorial(x) * fdl
    return y

X = Symbol('N')

#Gets pmf of a poisson distribution with gaussian estimator
def getfn(stats):
    mu = stats[0]
    std = stats[1]
    l = Normal('l', mu, std)
    dl = Symbol('dl', real = True, nonnegative = True)
    fn = re(N(integrate(getprobpoiss(X, dl, density(l)(dl)), (dl, 0, oo)))/N(integrate(density(l)(dl), (dl, 0, oo))))
    return fn


with cp.cuda.Device(1):
    for v1 in v1s:
        #loops over all samples
        for i in range(2, len(df)):
            row = df.iloc[i]

            # if row.Name in files:
            #     print(row.Name)
            # else:
            #     continue

            pores = poredf[poredf.Name == row.Name]
            evd = pores[pores['Equivalent Spherical Diameter (um)']>row['Threshold (um)']]['Equivalent Spherical Diameter (um)']
            lefttail = pores[pores['Equivalent Spherical Diameter (um)']<= row['Threshold (um)']]['Equivalent Spherical Diameter (um)']

            dmax = np.max(evd)
            
            #https://bpb-us-w2.wpmucdn.com/sites.uwm.edu/dist/2/109/files/2016/04/2009_IME_final-1mmannh.pdf
            threshold = row['Threshold (um)']

            #Check for missing threshold
            if threshold == 0:
                continue

            #MOM Fit
            scale = 0.5 * (evd.mean() - threshold) * ((evd.mean() - threshold)**2/evd.var()+1)
            shape = -0.5 * ((evd.mean()-threshold)**2/(evd.var())-1)

            #Validity of MOM
            if shape < 1/4:
                cov = (1-shape)**2/(1-3*shape)/(1-4*shape)/len(evd) * np.array([[2*scale**2*(1-6*shape+12*shape**2)/(1-2*shape), scale*(1-4*shape+12*shape**2)], [scale*(1-4*shape+12*shape**2), (1-2*shape)*(1-shape+6*shape**2)]])
                fmle = multivariate_normal([scale, shape], cov.tolist()) 
            
            #Validity of MLE
            else:
                (shape, threshold, scale) = genpareto.fit(evd, floc = row['Threshold (um)'])
                cov = (1+shape)/len(evd)*np.array([[2*scale**2, scale], [scale, (1+shape)]])
                fmle = multivariate_normal([scale, shape], cov.tolist()) 

                #If neither work (could be due to inconsistent fit)
                if shape > -0.5:
                    print('Error Shape Out Of Bounds')
                    print(shape)

            #Mean of MLE and Variance of poisson dist divided by vol
            #https://stats.stackexchange.com/questions/549204/variance-of-mle-poisson-distribution
            #https://www.statlect.com/fundamentals-of-statistics/Poisson-distribution-maximum-likelihood
            stats = np.array([len(evd)/row['Volume (mm^3)'], np.sqrt(len(evd)/row['Volume (mm^3)']**2)])
            
            stats1 = np.array([len(lefttail)/row['Volume (mm^3)'], np.sqrt(len(lefttail)/row['Volume (mm^3)']**2)])

            #get poisson distribution with gaussian estimator        
            fn = getfn(stats*v1)
            fn1 = getfn(stats1*v1)
            xs = np.arange(np.max([stats[0]*v1 - 3*stats[1]*v1, 0]), stats[0]*v1 + 3*stats[1]*v1)
            xs1 = np.arange(np.max([stats1[0]*v1 - 3*stats1[1]*v1, 0]), stats1[0]*v1 + 3*stats1[1]*v1)

            probs = []

            #get probabilities for all likely values of x number of pores
            for j in range(len(xs)):
                try:
                    probs.append(N(fn.subs(X, xs[j])))
                except Exception:
                    probs.append(0)
            probs = np.array(probs, dtype = float)

            dxs = 1
            xs = cp.array(xs)
            cdf = cp.array(np.array(np.cumsum(probs*dxs), dtype = float))

            probs1 = []

            #get probabilities for all likely values of x number of pores
            for j in range(len(xs1)):
                try:
                    probs1.append(N(fn1.subs(X, xs1[j])))
                except Exception:
                    probs1.append(0)
            probs1 = np.array(probs1, dtype = float)

            dxs1 = 1
            xs1 = cp.array(xs1)
            cdf1 = cp.array(np.array(np.cumsum(probs1*dxs1), dtype = float))

            #inverse CDF method
            sampler = uniform()

            #generate random variables using inverse cdf method
            samples = cp.array(sampler.rvs(1000))


            pgs = cp.zeros_like(samples)

            pgs1 = cp.zeros_like(samples)

            #generate 1000 pore numbers
            for i in range(len(pgs)):
                try:
                    pgs[i] = xs[cp.max(np.argwhere(cdf < samples[i]))]
                except Exception:
                    if samples[i]  > 0.5:
                        pgs[i] = xs[-1]
                    else:
                        pgs[i] = xs[0]

            if stats[0]*v1 < 10:
            #generate 1000 pore numbers for under threshold
                for i in range(len(pgs1)):
                    try:
                        pgs1[i] = xs1[cp.max(np.argwhere(cdf1 < samples[i]))]
                    except Exception:
                        if samples[i]  > 0.5:
                            pgs1[i] = xs1[-1]
                        else:
                            pgs1[i] = xs1[0]

            else:
                pgs1 = cp.ones_like(pgs1) * stats1[0]*v1
            #generate 1000 shape scale parameters
            ss = cp.array(fmle.rvs(1000))


            #generate inverse CDF p values
            samples = cp.array(sampler.rvs(1000))
        

            xs = cp.zeros(len(samples)*len(pgs)*len(ss))
            ls=  cp.arange(0, len(samples), dtype =  int)

            t1 = ss[:, 0]/ss[:, 1]
            s0  = 1/pgs

            s0[pgs == 0] = -1 #case where 0 pores generated

            ones = cp.ones_like(samples)
            th = ones*threshold

            lefttail = cp.array(lefttail)

            qs = []

            #compute maximum pore size from all of these rvs
            for i in tqdm(range(len(pgs))):
                for j in range(len(ss)):
                    if pgs[i] > 0:
                        xs[ls] = t1[j]*((ones - samples**s0[i])**(-ss[j,1])-ones) + th
                    else:
                        qs = samples **(1/pgs1)
                        qs[qs > 1] = 1
                        qs[qs < 0] = 0
                        qs[cp.isnan(qs)] = 0
                        xs[ls] = cp.quantile(lefttail, qs)

                    ls+=len(samples)*ones.astype(int)

            
            #generation of CDFs
            xs2 = cp.arange(cp.min(xs), cp.quantile(xs, 0.999), dx)
            pdf, bins = cp.histogram(xs, xs2, density = True)
            cdf = cp.cumsum(pdf*dx)

            bins = (bins[1:] + bins[:-1])/2

            #saving cdf
            np.save('CDFs/' + row.Name + '_CDF.npy', np.array([bins.get(), cdf.get()]))

            del xs, xs2, pdf, bins, cdf, samples, ss, pgs, ones, pgs1, ls, qs, lefttail, cdf1, xs1, probs, probs1
            cp._default_memory_pool.free_all_blocks()