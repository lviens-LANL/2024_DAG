#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:57:47 2024
Code to reproduce Figure 3 of the manuscript. 
This code is based on the scatter matrix module from Okubo et al. (https://github.com/kura-okubo/SeisMonitoring_Paper/blob/master/Post/ModelFit/code/scattermatrix.py)
Their code is based on the corner.corner module: https://corner.readthedocs.io/en/latest/

!!!!!!!!!!!!!!!
This code requires the low level callback function to perform the integration that was developped by Kurama Okubo. The original function can be found here:  https://github.com/kura-okubo/SeisMonitoring_Paper/tree/master/Post/ModelFit/code/LowLevel_callback_healing_distributed
Note that a copy of the C library is also included in this package and needs to be compiled (mac OS) with:
    gcc -shared -o healing_int.so healing_int.c   
!!!!!!!!!!!!!!!
@author: lviens
"""

import random
import numpy as np
import pickle, h5py 
import matplotlib as mpl
from scipy.stats import norm
import matplotlib.pyplot as plt
import os, ctypes
from scipy import integrate, LowLevelCallable
# Integration with low level callback function
# read shared library
lib_int = ctypes.CDLL(os.path.abspath('/Users/lviens/Python/DAG/DAG-2/Final_codes/LowLevel_callback_healing_distributed/healing_int.so'))
lib_int.f.restype = ctypes.c_double
lib_int.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    

def model_heal(theta, ts):
    # using Low-level caling function
    S = theta[0] # S
    taumax = theta[1]# taumax
    taumin = theta[2] # taumin    
    c = ctypes.c_double(ts) # This is the argument of time t as void * userdata
    user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)
    int1_llc = LowLevelCallable(lib_int.f, user_data) # in this way, only void* is available as argument
    
    return S*integrate.quad(int1_llc, taumin, taumax)[0]

    
def compute_y_healing(theta, ts):
    """
    return the time history of healing associated with the San Simeon and Parkfield EQ.
    """
    return [model_heal(theta, t)  for t in ts]


def calc_log_likelihood( theta, t, y, yerr):
    # We add eps to avoid divide by zero error for errorless synthetics
    sigma = yerr + 10 * 10**-16
    return norm.logpdf(y,  loc=compute_y_healing(theta, t), scale=sigma).sum()
 
    
def lnprior(theta):
    a1  = theta[0]
    a2  = theta[1]
    a3  = theta[2]
    if -2.0 < a1 < .1 and 1 < a2 < 9  and 0 < a3 < .066  :
        return 0.0
    return -np.inf

def lnprob(theta, t, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + calc_log_likelihood(theta, t, y, yerr)

title_fmt=".2f"
fmt = "{{0:{0}}}".format(title_fmt).format
titletot = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
#%%



CClim = .3
winfft = 4.0

Eps = 4.0
limval = .3
sta_gap = 1
chans =  [1965 ]#np.arange(100,2200,sta_gap)
freqs =  [10, 25]
dist_chans = np.arange(20, len(chans) + 20 )[::-1]


FFT_ref_start = 8 #*60 # start computing fft after 7 hours.
FFT_ref_end = 12 #*60 # start computing fft after 7 hours.
# Taumin = 0.034

#%% Load df/f results
dir_out = '../Figures/'
name_save = '../Data/Fig_5.h5'

# Load data
with h5py.File(name_save, 'r') as f:
    print(f.keys())
    chan = f.get('chans')[0]
    dv1 = f.get('dv1')[:]
    cc1= f.get('cc1')[:]
    tdas = f.get('tDAS')[:]
    test = f.get('curveDAS')[:]

#%%
chan = 1965  
#%%

curve = []
res0 = []

medsdtS = []
medtau =[]
out = []

finame = '../Data/Fig_3_MCMC.pickle'
with open(finame, "rb") as f:
    sampler = pickle.load(f)

xHom = np.arange(0.01, 12 , .05)
samples = sampler.flatchain
samples[np.argmax(sampler.flatlnprobability)]

dataini =  sampler.get_chain(flat=True)
median = np.zeros(3)
perc1 = np.zeros(3)
perc2 = np.zeros(3)
for i in [0,1,2] :
    data = dataini[:, i]
    (perc1[i], median[i], perc2[i]) = np.percentile(data, [16, 50, 84])

tmpcruve = compute_y_healing( median, xHom )
allfitt = []

selnb = random.sample(range(len(dataini)), 10000)
for i in selnb:
    tmp = compute_y_healing(dataini[i,  :], xHom )
    allfitt.append(tmp)

allfitt = np.array(allfitt) 
res = [median[0] ,median[1], median[2] ]  

#%%



#%%
allfitt2 = []
for i in np.arange(len(allfitt)):
    if str(np.nanmean(allfitt[i])) !=  'inf'  and str(np.nanmean(allfitt[i])) !=  '-inf' : 
        allfitt2.append(allfitt[i])
        print(np.nanmean(allfitt[i] ) )

#%%
labels = ['$s$','$\\tau_{max}$','$\\tau_{min}$' ]
dataini =  sampler.get_chain(flat=True)
Ndim = 3
fnt = 11
Ncontour = 5
Ncontour_clip = 4
Ncontourf = 51
cmap="Oranges"
xrange_sigma_factor=4
xranges = np.zeros((Ndim, 2))
# create figure and subplots
fig, axs = plt.subplots(Ndim, Ndim, figsize=((9,10)))


#% Plot df/f and models (Figure 3a)
ax = axs[0,1]
sc = ax.scatter(tdas , dv1  , c = cc1 , vmin = 0 ,vmax = 1 ,cmap = 'hot_r' , edgecolors = 'k', linewidth = .5)

for curve in allfitt2   : # in samples[np.random.randint(len(samples), size= 100 )]: # len(samples))]: #
    ax.plot(xHom, curve, color="grey", alpha=.25, linewidth = .5)    
    
ax.plot(xHom, allfitt[0], color="grey", alpha=1, label = '10,000 models')
ax.plot(xHom, tmpcruve , 'dodgerblue', label = 'Median model')
ax.set_xlabel('Time after DAG-2 (Hour)' , fontsize = fnt)
ax.set_ylabel('df/f (%)' , fontsize = fnt)
#@%%
for i in [0,1,2] :
    data = dataini[:, i]
    data = data[~np.isnan(data)]
    print(np.std(data), np.percentile(data, [16, 50, 84]) )
    (perc1[i], median[i], perc2[i]) = np.percentile(data, [16, 50, 84])


std0 = abs(  median[0] - perc1[0] )
std1 =  abs( median[1] - perc1[1] )
std2 =  abs( median[2] - perc1[2] )

std0up =   ( median[0] - perc2[0]  )
std1up =    (median[1] - perc2[1]  )
std2up =    (median[2] - perc2[2] ) 

##%% Error propagation for Dt0
errordt0 =  ( median[0] ) * np.log(median[1]/median[2]) * np.sqrt( (  std0 / median[0])**2 + (np.sqrt( ( std1/ median[1] )**2 + (std2 / median[2])**2 )/ np.log(median[1]/median[2])  )**2 )
errordt02 = ( median[0] ) * np.log(median[1]/median[2]) * np.sqrt(  (std0up/ median[0])**2 + (np.sqrt( ( std1up/ median[1] )**2 + (std2up/ median[2])**2 )/ np.log(median[1]/median[2])  )**2 )

valfindt0  =  ( median[0] ) * np.log(median[1]/median[2])


title1 = titletot.format(fmt(valfindt0), fmt( abs(errordt0) ), fmt( abs(errordt02) ) )
titlemax = titletot.format(fmt(median[1]), fmt(median[1]- perc1[1]), fmt( perc2[1] - median[1]  ))
titlemin = titletot.format(fmt(median[2]), fmt(median[2]- perc1[2]), fmt( perc2[2] - median[2]  ))


ax.set_title('DAS channel ' + str(chan )  , fontsize = fnt) 
ax.text(7, -9, '  $D_{t_0}$ = ' + title1 +' %\n' +  r' $\tau_{max}$ = ' + titlemax + ' hour '  , fontsize = fnt+ 2,bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', alpha = .5),linespacing =1.5  )


ax.legend()
ax.set_xlim(0, 12)
ax.set_ylim(-4,1.99)
ax.grid(linewidth = .4)
ax.text(-.85, 2, '(a)', fontsize = fnt)

ax.tick_params(bottom=True, top=True, left=True, right=True)
rect = plt.Rectangle( (9.2, -3.6 ) ,  2.5, 2.2, alpha = .8, facecolor = 'w', edgecolor = 'k',  zorder = 100 )
ax.add_patch(rect)
cbaxes = fig.add_axes([0.795, 0.818, 0.12, 0.02])
cb = plt.colorbar(sc,  cax = cbaxes, orientation = 'horizontal'  ) 
cb.ax.set_title('CC', fontsize = fnt)

 
#% Plot 1D histograms
for i in range(Ndim):
    ax = axs[i, i]
    if i<Ndim-1:
        ax.set(xticklabels=[])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.minorticks_off()
    data = dataini[:, i]
    if i ==2 :
        data[data<0] =np.nan
        data[data>.1] = np.nan
        
    data = data[~np.isnan(data)]
    

    if i ==0 :
        xrange = [-1 , -.5]
    elif i ==1 :
        xrange = [4, 7]
    elif i ==2:
        xrange = [0,.067]
    plt.grid()
    weights = np.ones_like(data) / len(data)
    ncount, bins, patches = ax.hist(data, bins=50 , color= 'grey',
            edgecolor='black',  weights=weights, linewidth=0.4, density=False)
    ax.set_xlim(xrange )
    if i ==0 or i ==1:
        ax.set_yticks(np.arange(0, 1.0, 0.05))
    else:
        ax.set_yticks(np.arange(0, 1.0, 0.01))
    if i ==0 :
        ax.set_ylim(0, .25)
    elif i ==1:
        ax.set_ylim(0, .15)
    else:
        ax.set_ylim(0, .05)
    ax.grid(linewidth = .4)
    ax.axvline(median[i], c="dodgerblue", lw=4.0, ls="-")
    if i==0:
        ax.set_ylabel("Probability", fontsize=fnt)
        ax.text(-1.125, .27 , '(b)', fontsize =fnt )
    if i==1 or i ==2:
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Probability", fontsize=fnt)
        ax.yaxis.tick_right()
    ax.set_title( labels[i] + ' = ' + str(round(median[i],2) ) )
    if i==Ndim-1:
        ax.set_xlabel(labels[i], fontsize=fnt)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), fontsize=fnt)
    plt.setp(ax.get_yticklabels(), fontsize=fnt)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    
    datamode = bins[ncount.argmax()]

    ax.axvline(median[i], c="dodgerblue", lw=2, ls="-")
    ax.axvline(perc1[i], c="dodgerblue", lw=2, ls="--")
    ax.axvline(perc2[i], c="dodgerblue", lw=2, ls="--")
    
    title = titletot.format(fmt(median[i]), fmt(median[i]- perc1[i]), fmt( perc2[i]-median[i]))
    ax.set_title( labels[i] + ' = ' + title )
    
#% Plot 2D histogram 
for i in range(1, Ndim):
    for j in range(0, i):
        ax = axs[i, j]
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        data_y =  dataini[:, i]
        data_x =  dataini[:, j]
        dataff = []
        for ii in np.arange(len(data_x)):
            if ~np.isnan(data_x[ii]) and ~np.isnan(data_y[ii]) :
                
                dataff.append([data_x[ii],data_y[ii] ])
        dataff = np.array(dataff)
        Z, X, Y = np.histogram2d(dataff[:, 0], dataff [:, 1], bins=18 ,
                                 weights=np.ones_like(dataff[:, 0]) / len(dataff[:, 0]))
        # Compute the bin centers.
        mX = (X[:-1] + X[1:])/2
        mY = (Y[:-1] + Y[1:])/2
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        
        mX2 = np.concatenate([[mX[0] - 2*dx, mX[0] - dx], mX, [mX[-1]+dx, mX[-1]+2*dx]])
        mY2 = np.concatenate([[mY[0] - 2*dy, mY[0] - dy], mY, [mY[-1]+dy, mY[-1]+2*dy]])
        Z2 = Z.min() + np.zeros(Z.shape+np.array([4,4]))
        Z2[2:-2, 2:-2] = Z
        for l, m in zip([1,-2], [0, -1]):
            Z2[2:-2, l] = Z[:, m]
            Z2[l, 2:-2] = Z[m, :]
            Z2[1, l] = Z[0, m]
            Z2[-2, l] = Z[m, 0]
        
        # Normalize Z2
        norm_Z = mpl.colors.Normalize(vmin=0, vmax=.04)
        mnloc = mpl.ticker.MaxNLocator(nbins=Ncontour)
        levels = mnloc.tick_values(0, Z2.max()) 
        levels_clipped = levels[Ncontour-Ncontour_clip:]# remove the lower levels
        h1 = ax.contourf(mX2, mY2, Z2.T, Ncontourf, norm=norm_Z, zorder=-10, cmap=cmap)
        ax.contour(mX2, mY2, Z2.T, levels_clipped, norm=norm_Z, colors='k', zorder=-9, linewidths=0.8)
        ax.plot(median[j], median[i],  "o", c= 'dodgerblue' ,  ms=10, markeredgecolor="w", lw=1.0)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-5, 5), useOffset=True, useLocale=True, useMathText=True)
        
        if j ==0 :
            xrange = [-1 , -.5]
        elif j ==1 :
            xrange = [4,7]
        elif j ==2:
            xrange = [0,.06]
            
        if i ==0 :
            yrange = [-1 , -.5]
        elif i ==1 :
            yrange = [4,7]
        elif i ==2:
            yrange = [0,.067]
        ax.set_xlim(xrange  )
        ax.set_ylim(yrange )
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.grid(linewidth = .5)
        if j==0:
            ax.set_ylabel(labels[i], fontsize=fnt)        
        else:
            ax.set(yticklabels=[])
        if i==Ndim-1:
            ax.set_xlabel(labels[j], fontsize=fnt)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        else:
            ax.set(xticklabels=[])
        
        plt.setp(ax.get_xticklabels(), fontsize=fnt)
        plt.setp(ax.get_yticklabels(), fontsize=fnt)

# Reposition the panels
xdimlen = .25
ydimlen = .2
pos2 = [.092 , .775  ,  .875,  .2 ] 
axs[0,1].set_position(pos2)

pos2 = [.09 , .5225 ,  xdimlen, ydimlen ] 
axs[0,0].set_position(pos2)

pos2 = [.09 , .298 ,  xdimlen, ydimlen ] 
axs[1,0].set_position(pos2)

pos2 = [.38 , .298 ,  xdimlen, ydimlen] 
axs[1,1].set_position(pos2)

pos2 = [.092 , .06225 , xdimlen,  ydimlen ] 
axs[2,0].set_position(pos2)

pos2 = [.38 , .06225 ,  xdimlen,  ydimlen ] 
axs[2,1].set_position(pos2)

pos2 = [.67 , .06225 , xdimlen,  ydimlen ] 
axs[2,2].set_position(pos2)

axs[1,2].set_visible(False)
axs[0,2].set_visible(False)
plt.show()

fig.savefig('../Figures/Fig_3.jpg', dpi = 300)
#%%
# fig.savefig('/Users/lviens/Documents/DAG/Figures/Fig_3.jpg', dpi = 300)

