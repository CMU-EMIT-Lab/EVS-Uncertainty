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

#makes directory if it doesn't exist
try:
    os.mkdir('CDFs')
except Exception:
    pass


v1s = [395.84] #extrapolated volumes

dx = 0.001 #resolution of CDF

df = pd.read_csv('parameters.csv') #read in csv of thresholds

poredf = pd.read_csv('poredf.csv')


#moving average for the PDF
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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



#obtains CDF
def F_x(xs, l, conv = 0.0005):
    xs = cp.array(xs)
    l = cp.floor(l)
    i = 1
    rls = cp.zeros_like(xs)
    p = 1
    th = threshold*cp.ones_like(xs)
    zs = xs - th
    c1 = p

    #while loop to add up pmf areas until an area of conv is achieved (starts from middle and goes outward)
    while p > conv:
        c1 = p
        if (l - i)+ 1 >= 0:
            T = (l - i)+ 1
            try:
                pp = N(fn.subs(X, T))
            except Exception:
                pp = 0
            if T == 0:
                rls += float(pp)*cp.ones_like(xs)
            else:
                rls += (cp.ones_like(zs)-(cp.ones_like(zs)+shape/scale*(zs))**(-1/shape))**(T)*float(pp)
            p -= pp
        
        T = l+i
        try:
            pp = N(fn.subs(X, T))
        except Exception:
            pp = 0
        rls+= float(pp)*(cp.ones_like(zs)-(cp.ones_like(zs)+shape/scale*(zs))**(-1/shape))**(T)
        p -= pp
        i+= 1

        if c1 == p:
            break

    return rls.get()

#loops over all volumes
for v1 in v1s:
    #loops over all samples
    for i in range(len(df)):
        row = df.iloc[i]

        pores = poredf[poredf.Name == row.Name]
        evd = pores[pores['ESD']>row['Threshold (um)']]['ESD']

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
            (shape, threshold, scale) = genpareto.fit(evd, floc = row['t'])
            cov = (1+shape)/len(evd)*np.array([[2*scale**2, scale], [scale, (1+shape)]])
            fmle = multivariate_normal([scale, shape], cov.tolist()) 

            #If neither work (could be due to inconsistent fit)
            if shape > -0.5:
                print('Error Shape Out Of Bounds')
                print(shape)

        #Mean of MLE and Variance of poisson dist divided by vol
        #https://stats.stackexchange.com/questions/549204/variance-of-mle-poisson-distribution
        #https://www.statlect.com/fundamentals-of-statistics/Poisson-distribution-maximum-likelihood
        stats = (len(evd)/row['Volume (mm^3)']*v1, np.sqrt(len(evd)/row['Volume (mm^3)']**2) *v1)

        #get poisson distribution with gaussian estimator        
        fn = getfn(stats)
        xs = np.arange(0, 3*stats[0])

        probs = []

        #get probabilities for all likely values of x number of pores
        for j in range(len(xs)):
            try:
                probs.append(N(fn.subs(X, xs[j])))
            except Exception:
                probs.append(0)
        probs = np.array(probs, dtype = float)

        
        #get values of pore size
        xs = np.arange(threshold, np.max(evd)+threshold, dx)
        fx = F_x(xs, stats[0])

        #obtain standard cdf values of pore size cdf
        ys = (np.ones_like(xs)-(np.ones_like(xs)+shape/scale*(xs - threshold*np.ones_like(threshold)))**(-1/shape))**stats[0]

        xs = cp.array(np.arange(0, 3*stats[0]))
        cdf = cp.array(np.array(np.cumsum(probs*dx), dtype = float))

        #inverse CDF method
        sampler = uniform()

        #generate random variables using inverse cdf method
        samples = cp.array(sampler.rvs(1000))


        pgs = cp.zeros_like(samples)

        #generate 1000 pore numbers
        for i in range(len(pgs)):
            try:
                pgs[i] = xs[cp.max(np.argwhere(cdf < samples[i]))]
            except Exception:
                if samples[i]  > 0.5:
                    pgs[i] = xs[-1]
                else:
                    pgs[i] = xs[0]


        #generate 1000 shape scale parameters
        ss = cp.array(fmle.rvs(1000))


        #generate inverse CDF p values
        samples = cp.array(sampler.rvs(1000))

        xs = cp.zeros(len(samples)*len(pgs)*len(ss))
        l = 0

        t1 = ss[:, 0]/ss[:, 1]
        s0  = 1/pgs
        ones = cp.ones_like(samples)
        th = ones*threshold

        #compute maximum pore size from all of these rvs
        for i in tqdm(range(len(pgs))):
            for j in range(len(ss)):
                xs[l:l+len(samples)] = t1[j]*((ones - samples**s0)**(-ss[j,1])-ones) + th
                l+=len(samples)


        #generation of CDFs
        samples = xs.get()
        xs = np.arange(threshold, np.max(evd)+threshold, dx)
        xs2 = np.arange(threshold, np.quantile(samples, 0.99), dx)
        pdf, bins = np.histogram(samples, xs2, density = True)
        cdf = np.cumsum(pdf*dx)

        bins = (bins[1:] + bins[:-1])/2

        #saving cdf
        np.save('CDFs/' + row.Name + '_CDF_{}.npy'.format(v1), np.array([bins, cdf]))