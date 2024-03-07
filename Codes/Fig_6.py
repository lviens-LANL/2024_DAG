#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:07:33 2023

@author: lviens
"""

import h5py, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt
from matplotlib.ticker import (MultipleLocator)

dir_out = '../Figures/'
name_save = '../Data/Fig_6.h5'

# Load data
with h5py.File(name_save, 'r') as f:
    D_DAS = f.get('D_DAS')[:]
    Tau_DAS = f.get('Tau_DAS')[:]
    dist_chans = f.get('dist_chans')[:]
    dist_line= f.get('dist_line')[:]
    D_HV = f.get('D_HV')[:]
    Tau_HV = f.get('Tau_HV')[:]
    D_geo = f.get('D_geo')[:]
    Tau_geo = f.get('Tau_geo')[:]


# Tau_geo +=.5
#%% Compute MAE
das_geo = []
for i in np.arange(len(dist_line)):  
    if dist_line[i]<2050:
        argdas = np.argmin(abs(dist_line[i]- dist_chans))
        print(Tau_DAS[argdas] - Tau_geo[i])
        das_geo.append([dist_line[i],D_DAS[argdas] - D_geo[i] ,Tau_DAS[argdas] - Tau_geo[i]  ]) 
 
das_geo = np.array(das_geo)



das_HV = []
for i in np.arange(len(dist_line)):  
    if dist_line[i]<2050:
        argdas = np.argmin(abs(dist_line[i]- dist_chans))
        das_HV.append([dist_line[i],D_DAS[argdas] - D_HV[i] ,Tau_DAS[argdas] - Tau_HV[i]  ]) #out0[argdas], out0geo_line[i]
        
das_HV = np.array(das_HV)
 

#%% Plot data
fnt = 12
sort_line_only = np.sort(dist_line)
sort_line_onlyarg = np.argsort(dist_line)
fig = plt.figure(figsize = (8,8))
smoothval = 100
ax1 = plt.subplot(221)
plt.plot(dist_chans, D_DAS, 'k', linewidth = 2, label = 'DAS')

plt.scatter(dist_line, D_geo,s =50 ,color =  'r' , edgecolor = 'k', zorder = 100, label = 'Geo. radial comp.')
plt.ylabel('Frequency change at $t_0$ (%)', fontsize = fnt)
plt.ylabel('$D_{t_0}$ (%)', fontsize = fnt)
plt.title('MAE$=$' + str(round( np.nanmean(abs(das_geo[:,1])),2 )) +'% ($\sigma=$'  + str(round( np.nanstd(abs(das_geo[:,1])),2 ))+ '%)' )

plt.grid()
plt.text(-170, 10 , '(a)', fontsize = fnt)
plt.legend()
plt.ylim(0, 9.99)
plt.xlim(20,2000)
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)
ax1.xaxis.set_minor_locator(MultipleLocator(100))
ax1.tick_params(axis='x', which='minor', top=True )
ax1.tick_params(bottom=True, top=True, left=True, right=True)
 


ax2 = plt.subplot(222)
plt.plot(dist_chans, Tau_DAS , 'k', linewidth = 2)
plt.scatter(dist_line,  Tau_geo,s = 50 ,color =  'r' , edgecolor = 'k' , zorder = 100, label = 'Geo. Radial C.')
# plt.title('MAE$=$' + str(round( np.nanmean(abs(das_geo[:,2])),2) ) +' min ($\sigma=$' + str(round( np.nanstd(abs(das_geo[:,2])),2 ) )+ ' min)' )
plt.title('MAE$=$' + str(round( np.nanmean(abs(das_geo[:,2]))*60,2) ) +' min ($\sigma=$' + str(round( np.nanstd(abs(das_geo[:,2]))*60,2 ) )+ ' min)' )

# plt.ylabel('Recovery time (hour)', fontsize = fnt)
plt.ylabel(r'$\tau_{rec}$ (hour)', fontsize = fnt)

plt.grid()
plt.text(-170 , 5 , '(b)', fontsize = fnt)
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.xaxis.set_minor_locator(MultipleLocator(100))
ax2.tick_params(axis='y', which='minor', right=True )
ax2.tick_params(axis='x', which='minor', top=True )
plt.ylim(0, 4.99)
plt.xlim(20,2000)
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)
ax2.tick_params(bottom=True, top=True, left=True, right=True)




ax3 = plt.subplot(223)
plt.plot(dist_chans, D_DAS, 'k', linewidth = 2, label = 'DAS')
plt.scatter(dist_line, D_HV,s = 50 ,color =  'orange' , edgecolor = 'k', zorder = 100, label = 'Geo. HVSR')
# plt.ylabel('Frequency change at $t_0$ (%)', fontsize = fnt)
plt.ylabel('$D_{t_0}$ (%)', fontsize = fnt)

plt.xlabel('Distance to SGZ (m)', fontsize = fnt)
# plt.title('MAE$=$' + str(round( np.nanmean(abs(das_HV[:,1])),2 )) +'% ($\sigma=$'  + str(round( np.nanstd(abs(das_HV[:,1])),2 ))+ '%)' )
plt.title('MAE$=$' + str(round( np.nanmean(abs(das_HV[:,1])),2 )) +'% ($\sigma=$'  + str(round( np.nanstd(abs(das_HV[:,1])),2 ))+ '%)' )

plt.grid()
plt.text(-170 , 10 , '(c)', fontsize = fnt)
plt.legend()
plt.ylim(0, 9.99)
plt.xlim(20,2000)
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)
ax3.tick_params(bottom=True, top=True, left=True, right=True)
ax3.xaxis.set_minor_locator(MultipleLocator(100))
ax3.tick_params(axis='x', which='minor', top=True )
 


ax4 = plt.subplot(224)
plt.plot(dist_chans, Tau_DAS, 'k', linewidth = 2)
plt.scatter(dist_line, Tau_HV,s = 50 ,color =  'orange' , edgecolor = 'k' , zorder = 100, label = 'Geo. HVSR')
# plt.title('MAE$=$' + str(round( np.nanmean(abs(das_HV[:,2])),2 )) +' min ($\sigma=$' + str(round( np.nanstd(abs(das_HV[:,2])),2 ) )+ ' min)' )
plt.title('MAE$=$' + str(round( np.nanmean(abs(das_HV[:,2])*60),2 )) +' min ($\sigma=$' + str(round( np.nanstd(abs(das_HV[:,2]))*60,2 ) )+ ' min)' )
plt.xlabel('Distance to SGZ (m)', fontsize = fnt)
# plt.ylabel('Recovery time (hour)', fontsize = fnt)
plt.ylabel(r'$\tau_{rec}$ (hour)', fontsize = fnt)
plt.grid()
plt.text(-170 , 5 , '(d)', fontsize = fnt)
ax4.yaxis.set_major_locator(MultipleLocator(1))
ax4.xaxis.set_minor_locator(MultipleLocator(100))
ax4.tick_params(axis='y', which='minor', right=True )
ax4.tick_params(axis='x', which='minor', top=True )
plt.ylim(0, 4.99)
plt.xlim(20,2000)
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)
ax4.tick_params(bottom=True, top=True, left=True, right=True)


plt.tight_layout()
# dir_out = '/Users/lviens/Documents/DAG/Figures/'
fig.savefig(dir_out + '/Fig_6.jpg', dpi=300)


#%%

