#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:19:55 2024

@author: lviens
"""

import h5py
import os, ctypes
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, LowLevelCallable


def int1(tau, t):
    return (1/tau) * np.exp(-t/tau)

# Integration with low level callback function (from Okubo et al., 2024), read shared library
lib_int = ctypes.CDLL(os.path.abspath('./LowLevel_callback_healing/healing_int.so'))
lib_int.f.restype = ctypes.c_double
lib_int.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    

def model_heal(theta, ts):
    # using Low-level caling function
    S = theta[0]
    taumax= theta[1]
    taumin = 0.034 # fix taumin so that healing starts just after incident     (0.034 hours = 2 minutes)
    c = ctypes.c_double(ts) # This is the argument of time t as void * userdata
    user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)
    int1_llc = LowLevelCallable(lib_int.f, user_data) # in this way, only void* is available as argument
    return -S*integrate.quad(int1_llc, taumin, taumax)[0]

def compute_y_healing(theta, ts):
    """
    return the time history of healing associated with the explosion
    """
    return [model_heal(theta, t)  for t in ts]

title_fmt=".2f"
fmt = "{{0:{0}}}".format(title_fmt).format
titletot = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
#%%
fnt = 12 # Font size
dir_out = '../Figures/'
name_save = '../Data/Fig_5.h5'

# Load data
with h5py.File(name_save, 'r') as f:
    print(f.keys())
    chanplt = f.get('chans')[:]
    dv1 = f.get('dv1')[:]
    dv2 = f.get('dv2')[:]
    cc1= f.get('cc1')[:]
    cc2 = f.get('cc2')[:]
    
    dvgeo1= f.get('dvgeo1')[:]
    dvgeo2 = f.get('dvgeo2')[:]
    ccgeo1= f.get('ccgeo1')[:]
    ccgeo2 = f.get('ccgeo2')[:]
    
    dvHV1= f.get('dvHV1')[:]
    dvHV2 = f.get('dvHV2')[:]
    ccHV1= f.get('ccHV1')[:]
    ccHV2 = f.get('ccHV2')[:]
    
    res0= f.get('res0')[:]
    res0geo = f.get('res0geo')[:]
    res0HV = f.get('res0HV')[:]
    
    curveDAS= f.get('curveDAS')[:]
    curvegeo = f.get('curvegeo')[:]
    curveHV = f.get('curveHV')[:]
    
    thv = f.get('THV')[:]
    tgeo = f.get('Tgeo')[:]
    tdas = f.get('tDAS')[:]
    Tfit = f.get('Tfit')[:]
    
    med_param_DAS = f.get('med_param_DAS')[:]
    med_param_geo = f.get('med_param_geo')[:]
    med_param_HV = f.get('med_param_HV')[:]
    
    stdDt0geo = f.get('stdDt0geo')[:]
    stdtaugeo = f.get('stdtaugeo')[:]
    
    stdDt0HV = f.get('stdDt0HV')[:]
    stdtauHV = f.get('stdtauHV')[:]
    
    stdDt0 = f.get('stdDt0')[:]
    stdtau = f.get('stdtau')[:]

#%%
dv = [dv1, dv2]  
cc = [cc1, cc2] 
dvgeo = [dvgeo1, dvgeo2]  
ccgeo = [ccgeo1, ccgeo2] 
dvHV = [dvHV1, dvHV2]  
ccHV = [ccHV1, ccHV2]  


#%% Plot DAS data
fig = plt.figure(figsize = (9,12))
for i in np.arange(2):
    if i ==0 :
        ax = plt.subplot(321)
        plt.ylabel('$df/f$ (%)', fontsize = fnt)
        plt.text(-1.5 , 4 , '(a)', fontsize = fnt)
    elif i==1:
        ax = plt.subplot(322)
        plt.text(-1.5 , 4 , '(b)', fontsize = fnt)
        
    sc = plt.scatter(tdas , dv[i], c = cc[i], vmin = 0 ,vmax = 1 ,cmap = 'hot_r' , edgecolors = 'k', linewidth = .5)
    plt.plot(Tfit , curveDAS[i] , 'dodgerblue'  , linewidth = 4)
    
    theta = med_param_DAS[i]
    print(theta)
    plt.plot(Tfit, compute_y_healing(theta, Tfit), color="r", alpha=0.01)
        
    plt.title('DAS channel ' + str(chanplt[i]) , fontsize = fnt)
    plt.ylim(-4,4)
    plt.xlim(0,12)
    plt.grid()
    
    title1 = titletot.format(fmt(res0[i,0]), fmt(res0[i,0]- stdDt0[0,i]), fmt( stdDt0[1,i]-res0[i,0]) )
    title2 = titletot.format(fmt(res0[i,1]), fmt(res0[i,1]- stdtau[0,i]), fmt( stdtau[1,i] - res0[i,1]  ))
    ax.text(2.5, -3.3, '  $D_{t_0}$ = ' + title1 +' %\n' +  r' $\tau_{rec}$ = ' + title2 + ' hour' , fontsize = fnt+ 2,bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', alpha = .8),linespacing =1.5  )
    if i ==0 :   
        rect = plt.Rectangle( (3.5, 2.05 ) ,  5, 1.85, alpha = .8, facecolor = 'w', edgecolor = 'k',  zorder = 1 )
        ax.add_patch(rect)
        cbaxes = fig.add_axes([0.22, 0.924, 0.13, 0.019])
        cb = plt.colorbar(sc,  cax = cbaxes, orientation = 'horizontal'  ) 
        cb.ax.set_title('CC', fontsize=fnt)
           
 
#%% Plot Geophones
for i in np.arange(2):
    if i ==0 :
        ax = plt.subplot(323)
        plt.text(-1.5 , 4 , '(c)', fontsize = fnt)
        plt.ylabel('$df/f$ (%)', fontsize = fnt)
        plt.title('Geophone - Distance to SGZ: 224.4 m ' , fontsize = fnt)
    elif i==1:
        ax = plt.subplot(324) 
        plt.text(-1.5 , 4 , '(d)', fontsize = fnt)
        plt.title('Geophone - Distance to SGZ: 774.8 m ' , fontsize = fnt)
        
    sc = plt.scatter(tgeo, dvgeo[i] , c = ccgeo[i], vmin = 0 ,vmax = 1 ,cmap = 'hot_r' , edgecolors = 'k', linewidth = .5)
    theta = med_param_geo[i]
    print(theta)
    plt.plot(Tfit, compute_y_healing(theta, Tfit), color='dodgerblue', alpha=1,  linewidth =4)
        
    
    plt.grid()
    title1 = titletot.format(fmt(res0geo[i,0]), fmt(res0geo[i,0]- stdDt0geo[0,i]), fmt( stdDt0geo[1,i]-res0geo[i,0]) )
    title2 = titletot.format(fmt(res0geo[i,1]), fmt(res0geo[i,1]- stdtaugeo[0,i]), fmt( stdtaugeo[1,i] - res0geo[i,1]  ))
    ax.text(2.5, -3.3, '  $D_{t_0}$ = ' + title1 +' %\n' +  r' $\tau_{rec}$ = ' + title2 + ' hour' , fontsize = fnt+ 2,bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', alpha = .8),linespacing =1.5  )
    plt.ylim(-4,4)
    plt.xlim(0,12)
      

#%% Plot HVSR
for i in np.arange(2):
    if i ==0 :
        ax = plt.subplot(325)
        plt.text(-1.5 , 4 , '(e)', fontsize = fnt)
        plt.ylabel('$df/f$ (%)', fontsize = fnt)
    elif i==1:
        ax = plt.subplot(326)
        plt.text(-1.5 , 4 , '(f)', fontsize = fnt)
        
    sc = plt.scatter(thv, dvHV[i] , c = ccHV[i], vmin = 0 ,vmax = 1 ,cmap = 'hot_r' , edgecolors = 'k', linewidth = .5)
    theta = med_param_HV[i]
    print(theta)
    plt.plot(Tfit, compute_y_healing(theta, Tfit), color='dodgerblue', alpha=1 , linewidth =4)
        
    plt.title('Geophone - HVSR' , fontsize = fnt)
    plt.grid()
    title1 = titletot.format(fmt(res0HV[i,0]), fmt(res0HV[i,0]- stdDt0HV[0,i]), fmt( stdDt0HV[1,i]-res0HV[i,0]) )
    title2 = titletot.format(fmt(res0HV[i,1]), fmt(res0HV[i,1]- stdtauHV[0,i]), fmt( stdtauHV[1,i] - res0HV[i,1]  ))
    ax.text(2.5, -3.3, '  $D_{t_0}$ = ' + title1 +' %\n' +  r' $\tau_{rec}$ = ' + title2 + ' hour' , fontsize = fnt+ 2,bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', alpha = .8),linespacing =1.5  )
    plt.ylim(-4,4)
    plt.xlim(0,12)
    plt.xlabel('Time after DAG-2 (Hour)' )
               
plt.tight_layout()
fig.savefig(dir_out + '/Fig_5.jpg', dpi=300)

