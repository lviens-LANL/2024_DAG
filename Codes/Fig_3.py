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

import numpy as np
import pickle, h5py 
import matplotlib as mpl
from scipy.stats import norm
import matplotlib.pyplot as plt
import os, ctypes
from scipy import integrate, LowLevelCallable
# Integration with low level callback function

lib_int = ctypes.CDLL(os.path.abspath('/Users/lviens/Python/DAG/DAG-2/Final_codes/LowLevel_callback_healing_distributed/healing_int.so'))
lib_int.f.restype = ctypes.c_double
lib_int.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    

def model_heal(theta, ts):
    # using Low-level caling function
    S = theta[0]
    taumax = theta[1]
    taumin = 0.034  # fixed taumin so that healing starts just after incident    
    c = ctypes.c_double(ts) # This is the argument of time t as void * userdata
    user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)
    int1_llc = LowLevelCallable(lib_int.f, user_data) # in this way, only void* is available as argument
    
    return -S*integrate.quad(int1_llc, taumin, taumax)[0]

    
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
    if -10.0 < a1 < 10 and .1 < a2 < 10.0  :
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

limval = .5 # Compute \tau_{rec} for df/f = -0.5% 
Taumin = 0.034 # 0.034 hour = 2 min 
xHom = np.arange(0.01, 12 , .01) # time vector to show models

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
 
#%% Load MCMC results

finame = '../Data/Fig_3_MCMC.pickle'
with open(finame, "rb") as f:
    sampler = pickle.load(f)
    
dataini =  sampler.get_chain(flat=True)
median = np.zeros(2)
perc1 = np.zeros(2)
perc2 = np.zeros(2)
for i in [0,1] :
    data = dataini[:, i]
    (perc1[i], median[i], perc2[i]) = np.percentile(data, [16, 50, 84]) # Get median and 1 std values for D_t0 and Tau_{rec}

tmpcruve = compute_y_healing( [median[0] ,median[1] ], xHom )
res0all = []
for i in np.arange(0 , len(dataini) , 100):
    tmp = compute_y_healing( [dataini[i, 0] ,dataini[i,1] ], xHom )
    tmp = np.array(tmp)
    if  len( np.where(tmp >-limval )[0]) >0 : 
        zer =  xHom [np.where(tmp> -limval )[0][0] ] 
    else:
        zer= np.nan
    res0all.append([ dataini[i, 0] * np.log(dataini[i,1] / Taumin ), zer ]  )

res0all = np.array(res0all)

res = [median[0] ,median[1] ]  

tmpcruve = np.array(tmpcruve)
if  len( np.where(tmpcruve >-limval )[0]) >0 : 
    zer =  xHom [np.where(tmpcruve> -limval )[0][0] ] 
else:
    zer= np.nan
res0 =  [ res[0] * np.log(res[1] / Taumin ), zer ] 

#%%
print(res[0] * np.log(res[1] / Taumin ))
#%% Some parameters to plot figure 3
labels = ['$s$','$\\tau_{max}$' ]
dataini =  sampler.get_chain( thin=50, flat=True)
Ndim = 2
fnt = 11
Ncontour = 5
Ncontour_clip = 4
Ncontourf = 51
cmap="Oranges"
xrange_sigma_factor=4
xranges = np.zeros((Ndim, 2))

#%% Plot df/f and models (Figure 3a)
fig, axs = plt.subplots(Ndim, Ndim, figsize=((7,10)))
ax = axs[0,1]

sc = ax.scatter(tdas , dv1  , c = cc1  , vmin = 0 ,vmax = 1 ,cmap = 'hot_r' , edgecolors = 'k', linewidth = .5)
xHom2 = np.arange(0.01, 12 , .1)

samples = sampler.get_chain( thin=5, flat=True) # remove some samples to not have to plot 200,000 models
for theta in samples[np.random.randint(len(samples), size= 100)]: # Only plots 100 models for speed (we plot all the model in Figure 3 of the main manuscript)
    ax.plot(xHom2, compute_y_healing( theta, xHom2 ), color="grey", alpha=1, linewidth = 2)
ax.plot(xHom2, compute_y_healing( [0.7631362 , 6.44413906], xHom2 ), color="grey", alpha=1, linewidth = 2)
ax.plot(xHom2, compute_y_healing(theta, xHom2), color="grey", alpha=1, label = 'All models')
ax.plot(xHom, tmpcruve , 'dodgerblue', label = 'Median model')
ax.set_xlabel('Time after DAG-2 (Hour)' , fontsize = fnt)
ax.set_ylabel('df/f (%)' , fontsize = fnt)

(perc1dt0 , medianDt0,perc2dt0 ) = np.percentile(res0all[:,0], [16, 50, 84]) 
(perc1taurec , mediantaurec,perc2taurec ) = np.percentile(res0all[:,1], [16, 50, 84]) 
print(fmt(medianDt0), fmt(medianDt0- perc1dt0), fmt( perc2dt0-medianDt0) ) 
title1 = titletot.format(fmt(res0[0]), fmt(res0[0]- perc1dt0), fmt( perc2dt0-res0[0]) )

 
title2 = titletot.format(fmt(res0[1]), fmt(res0[1]- perc1taurec), fmt( perc2taurec - res0[1]  ))
ax.set_title('DAS channel ' + str(chan )  , fontsize = fnt) 
ax.text(4.5, -3.3, '  $D_{t_0}$ = ' + title1 +' %\n' +  r' $\tau_{rec}$ = ' + title2 + ' hour' , fontsize = fnt+ 2,bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', alpha = .5),linespacing =1.5  )


ax.set_title('DAS channel ' + str(chan )  , fontsize = fnt) 
ax.legend()
ax.set_xlim(0, 12)
ax.set_ylim(-4,1.99)
ax.grid(linewidth = .4)
ax.text(-1, 2, '(a)', fontsize = fnt)

ax.tick_params(bottom=True, top=True, left=True, right=True)
rect = plt.Rectangle( (9.2, -3.5 ) ,  2.5, 1.4, alpha = .8, facecolor = 'w', edgecolor = 'k',  zorder = 100 )
ax.add_patch(rect)
cbaxes = fig.add_axes([0.795, 0.719, 0.12, 0.02])
cb = plt.colorbar(sc,  cax = cbaxes, orientation = 'horizontal'  ) 
cb.ax.set_title('CC', fontsize = fnt)

#% Plot 1D histograms
mul_minus1 = -1 # multiply the s parameter by -1 as the model_heal(theta, ts) function has a - sign in it
for i in range(Ndim):
    ax = axs[i, i]
    if i<Ndim-1:
        ax.set(xticklabels=[])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.minorticks_off()
    if i==0:
        data = dataini[:, i]*mul_minus1
    else:
        data = dataini[:, i]
    m1, s1 = norm.fit(data)
    xranges[i, 0] = max(data.min(), m1-xrange_sigma_factor*s1) # xmin
    xranges[i, 1] = min(data.max(), m1+xrange_sigma_factor*s1) # xmax
    plt.grid()
    weights = np.ones_like(data) / len(data)
    ncount, bins, patches = ax.hist(data, bins=20 , color= 'grey',
            edgecolor='black',  weights=weights, linewidth=0.4, density=False)
    ax.set_xlim(xranges[i])
    ax.set_yticks(np.arange(0, 1.0, 0.05))
    ax.set_ylim(0, .165)
    ax.grid(linewidth = .4)
    ax.axvline(median[i], c="dodgerblue", lw=4.0, ls="-")
    if i==0:
        ax.set_ylabel("Probability", fontsize=fnt)
        ax.text(-.925, .17 , '(b)', fontsize =fnt )
        ax.set_xticks(np.arange( -.9, -.78, .02) )
    if i==1:
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Probability", fontsize=fnt)
        ax.yaxis.tick_right()
        ax.set_xticks(np.arange(5.4, 6.8, .2) )
    ax.set_title( labels[i] + ' = ' + str(round(median[i],2) ) )
    if i==Ndim-1:
        ax.set_xlabel(labels[i], fontsize=fnt)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), fontsize=fnt)
    plt.setp(ax.get_yticklabels(), fontsize=fnt)
    
    if i==0:
        title = titletot.format(fmt(median[i]*mul_minus1), fmt(perc1[i]*mul_minus1 - median[i]*mul_minus1), fmt( median[i]*mul_minus1 - perc2[i]*mul_minus1))
        ax.axvline(median[i]*mul_minus1, c="dodgerblue", lw=2, ls="-")
        ax.axvline(perc1[i]*mul_minus1, c="dodgerblue", lw=2, ls="--")
        ax.axvline(perc2[i]*mul_minus1, c="dodgerblue", lw=2, ls="--")
        
    else:
        title = titletot.format(fmt(median[i]), fmt(median[i]-perc1[i] ), fmt( perc2[i]- median[i] ))
        ax.axvline(median[i], c="dodgerblue", lw=2, ls="-")
        ax.axvline(perc1[i], c="dodgerblue", lw=2, ls="--")
        ax.axvline(perc2[i], c="dodgerblue", lw=2, ls="--")
            
    ax.set_title( labels[i] + ' = ' + title )
    
#% Plot 2D histogram 
for i in range(1, Ndim):
    for j in range(0, i):
        ax = axs[i, j]
        data_y =  dataini[:, i]
        data_x =  dataini[:, j]*mul_minus1
        Z, X, Y = np.histogram2d(data_x, data_y, bins=18 ,
                                 weights=np.ones_like(data_x) / len(data_x))
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
        ax.plot(median[j]*-1, median[i],  "o", c= 'dodgerblue' ,  ms=10, markeredgecolor="w", lw=1.0)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-5, 5), useOffset=True, useLocale=True, useMathText=True)
        ax.set_xlim(xranges[j])
        ax.set_ylim(xranges[i])
        ax.set_xticks(np.arange(-.9,-.78,  .02) )
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.grid(linewidth = .5)
        if j==0:
            ax.set_ylabel(labels[i], fontsize=fnt)        
        else:
            ax.set(yticklabels=[])
        if i==Ndim-1:
            ax.set_xlabel(labels[j], fontsize=fnt)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        else:
            ax.set(xticklabels=[])
        plt.setp(ax.get_xticklabels(), fontsize=fnt)
        plt.setp(ax.get_yticklabels(), fontsize=fnt)

# Reposition the panels
pos2 = [.092 , .669  ,  .875,  .3 ] 
axs[0,1].set_position(pos2)

pos2 = [.092 , .345 ,  .4,  .25 ] 
axs[0,0].set_position(pos2)

pos2 = [.092 , .065 ,  .4,  .25 ] 
axs[1,0].set_position(pos2)

pos2 = [.505 , .065 ,  .4,  .25 ] 
axs[1,1].set_position(pos2)
plt.show()
##%%
fig.savefig('../Figures/Fig_3.jpg', dpi = 300)
