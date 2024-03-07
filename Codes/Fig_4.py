#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:03:15 2024

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
name_save = '../Data/Fig_4.h5'

# Load data
with h5py.File(name_save, 'r') as f:
    print(f.keys())
    dvDAS = f.get('dvDAS')[:]
    ccDAS = f.get('ccDAS')[:]
    dist_chans = f.get('dist_chans')[:]
    freEPS= f.get('freEPS')[:]
    t0DAS = f.get('tfftDAS')[:]
    Geophonedv = f.get('Geophonedv')[:]
    Geophonecc = f.get('Geophonecc')[:]
    dist_plot = f.get('distplot')[:]
    t0 = f.get('tfftgeo')[:]
    dat_finHV = f.get('dat_finHV')[:]
    dat_finHVcc = f.get('dat_finHVcc')[:]
 
#%% Plot Figure 4

Eps = freEPS[2]
cldv = Eps/2  
fnt = 11
fig = plt.figure(figsize = (10,12))


ylim = [12,0]
cm = plt.cm.get_cmap('rainbow')

ax3 = plt.subplot(321)
plt.imshow(dvDAS.T, aspect = 'auto', extent=(dist_chans[0], dist_chans[-1],t0DAS[0], t0DAS[-1]), clim = (-cldv,cldv), cmap = 'seismic_r')
plt.ylabel('Time after DAG-2 (hour)', fontsize = fnt)
plt.title('Relative frequency change between ' + str(round(freEPS[0]) ) + '-' + str(round(freEPS[1]) )+ ' Hz\n DAS', fontsize = fnt )
plt.ylim(ylim)
plt.xlim(dist_chans[-1], dist_chans[0])
plt.text(-100, -.3, '(a)', fontsize = fnt)
ax3.tick_params(bottom=True, top=True, left=True, right=True)
rect = plt.Rectangle( (1760, 8.5 ) ,  2200, 3.5, alpha = .8, facecolor = 'w', edgecolor = 'k',  zorder = 100 )
ax3.add_patch(rect)
cbaxes = fig.add_axes([0.4425, 0.7, 0.02, 0.045])
cb = plt.colorbar( cax = cbaxes, orientation = 'vertical'  ) 
cb.ax.set_title('  df/f (%)', fontsize = fnt)
cm = plt.cm.get_cmap('rainbow')


ax4 = plt.subplot(322)
plt.imshow(ccDAS.T, aspect = 'auto', extent=(dist_chans[0], dist_chans[-1],t0DAS[0], t0DAS[-1]), clim = (0,1), cmap = cm)
plt.xlim(dist_chans[-1], dist_chans[0])
plt.title('Correlation Coefficient\n DAS' , fontsize = fnt)
plt.ylim(ylim)
plt.text(-100, -.3, '(b)', fontsize = fnt)
ax4.tick_params(bottom=True, top=True, left=True, right=True)
rect = plt.Rectangle( (1790, 8.5 ) ,  2200, 3.5, alpha = .8, facecolor = 'w', edgecolor = 'k',  zorder = 100 )
ax4.add_patch(rect)
cbaxes = fig.add_axes([0.925, 0.7, 0.02, 0.045])
cb = plt.colorbar( cax = cbaxes, orientation = 'vertical'  ) 
cb.ax.set_title('CC', fontsize = fnt)

 
ax1 = plt.subplot(323)
plt.imshow(Geophonedv.T, aspect = 'auto', extent=(dist_plot[0], dist_plot[-1],t0[0], t0[-1]) , clim = (-cldv, cldv), cmap = 'seismic_r')
plt.ylabel('Time after DAG-2 (hour)', fontsize = fnt)
plt.title('Radial geophones', fontsize = fnt )
plt.xlim(dist_chans[-1], dist_chans[0])
plt.ylim(ylim)
plt.text(-100, -.3, '(c)', fontsize = fnt)
ax1.tick_params(bottom=True, top=True, left=True, right=True)



ax2 = plt.subplot(324)
plt.imshow(Geophonecc.T, aspect = 'auto', extent=(dist_plot[0], dist_plot[-1],t0[0], t0[-1]) , clim = (0,1), cmap = cm)
plt.xlim(dist_chans[-1], dist_chans[0])
plt.title('Radial geophones', fontsize = fnt )
plt.ylim(ylim)
plt.text(-100, -.3, '(d)', fontsize = fnt)
ax2.tick_params(bottom=True, top=True, left=True, right=True)


ax5 = plt.subplot(325)
plt.imshow(dat_finHV.T, aspect = 'auto', extent=(dist_plot[0], dist_plot[-1],t0[0], t0[-1]) , clim = (-cldv , cldv), cmap = 'seismic_r')
plt.xlabel('Distance to SGZ (m)', fontsize = fnt)
plt.ylabel('Time after DAG-2 (hour)', fontsize = fnt)
plt.title('HVSR geophones', fontsize = fnt )
plt.xlim(dist_chans[-1], dist_chans[0])
plt.ylim(ylim)
plt.text(-100, -.3, '(e)', fontsize = fnt)
ax5.tick_params(bottom=True, top=True, left=True, right=True)
 

ax6 = plt.subplot(326)
plt.imshow(dat_finHVcc.T, aspect = 'auto', extent=(dist_plot[0], dist_plot[-1],t0[0], t0[-1]) , clim = (0,1), cmap = cm)
plt.xlim(dist_chans[-1], dist_chans[0])
plt.xlabel('Distance to SGZ (m)', fontsize = fnt)
plt.title('HVSR geophones', fontsize = fnt )
plt.ylim(ylim)
plt.text(-100, -.3, '(f)', fontsize = fnt)
ax2.tick_params(bottom=True, top=True, left=True, right=True)

 
plt.tight_layout()
fig.savefig(dir_out + '/Fig_4.jpg', dpi=300)