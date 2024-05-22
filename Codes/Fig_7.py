#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:49:09 2023

@author: lviens
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.ticker import (MultipleLocator)

#%%
dir_out = '../Figures/'
name_save = '../Data/Fig_7.h5'

fact = 2400 # Apparent wave velocity in m/s to convert PSR to PGA
#%%
# Load data
with h5py.File(name_save, 'r') as f:
    T_DAS = f.get('T_DAS')[:]
    D_DAS = f.get('D_DAS')[:]
    peak_SRDAS= f.get('peak_SRDAS')[:]
    PGA = f.get('PGA')[:]
    dist_chans= f.get('dist_chans')[:]
    D_geo= f.get('D_geo')[:]
    T_geo = f.get('T_geo')[:]
    dist_only= f.get('dist_only')[:]

#%% Plot data
color_clipped = 'grey'
fnt = 11
fig = plt.figure(figsize = (7,7))
ax1 = plt.subplot(221)
pl  = plt.scatter(peak_SRDAS , D_DAS, s = 15 ,c = dist_chans , cmap = 'magma', zorder = 1, edgecolors = 'k', linewidth = .4)
ax1.set_xscale('log')

plt.grid(linewidth = .5)
plt.xlabel('PSR ($s^{-1}$)', fontsize = fnt)
# plt.ylabel('Frequency change at $t_0$ (%)', fontsize = fnt)
plt.ylabel('$D_{t_0}$ (%)', fontsize = fnt)
plt.xticks(fontsize = fnt)
plt.yticks(fontsize = fnt)
plt.ylim( -9, 0)
plt.text(3.*10**-5, 0 , '(a)' )
plt.text(.0007, -8, 'Clipped\n  data',bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', alpha = .5))
plt.plot([0.00175, .003] , [ -7.5, -7.8], 'k')
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.tick_params(axis='y', which='minor', right=True )
ax1.tick_params(axis='x', which='minor', top=True )
ax1.tick_params(bottom=True, top=True, left=True, right=True)




plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)
ax1.axvspan(7/fact, 10/fact, ymin=0, ymax=10, alpha=0.3, color=color_clipped, zorder = -10)
ax1.set_xlim(.15/fact, 10/fact)



ax2 = plt.subplot(222)
pl = plt.scatter(peak_SRDAS , np.array(T_DAS) , s = 15 ,c = dist_chans , cmap = 'magma', zorder = 1, edgecolors = 'k', linewidth = .4)
plt.xticks(fontsize = fnt)
plt.yticks(fontsize = fnt)
plt.text(3*10**-5, 10 , '(b)' )
plt.grid(linewidth = .5)
plt.xlabel('PSR ($s^{-1}$)', fontsize = fnt)
ax2.axvspan(7/fact, 10/fact, ymin=0, ymax=10, alpha=0.3, color=color_clipped, zorder = -10)
plt.ylabel(r'$\tau_{max}$ (hour)', fontsize = fnt)
plt.ylim( 1, 9.99)
ax2.set_xlim(.15/fact, 10/fact)
ax2.set_xscale('log')
ax2.yaxis.set_minor_locator(MultipleLocator(5))
ax2.tick_params(axis='y', which='minor', right=True )
ax2.tick_params(axis='x', which='minor', top=True )
ax2.tick_params(bottom=True, top=True, left=True, right=True)
cbaxes = fig.add_axes([0.66, 0.83, 0.04, 0.09])
cb = fig.colorbar(pl, cax = cbaxes, orientation = 'vertical', ticks = [500,1000, 1500, 2000]  ) 
cb.ax.set_title('   Dist. to SGZ (m)', fontsize=fnt)


##%%
ax3 = plt.subplot(223)
plt.scatter(peak_SRDAS*fact , D_DAS, s = 15 ,c = dist_chans , cmap = 'magma', zorder = 1, edgecolors = 'k', linewidth = .4, alpha = .35)
plt.scatter(PGA, D_geo, s = 15 ,c = dist_only , cmap = 'magma',marker='s', zorder = 1, edgecolors = 'blue', linewidth = .4 )
ax3.set_xscale('log')
plt.text(.045, .0 , '(c)' )
plt.grid(linewidth = .5)
plt.xlabel('PGA (m/s$^2$)', fontsize = fnt)
plt.ylabel('$D_{t_0}$ (%)', fontsize = fnt)
plt.xticks(fontsize = fnt)
plt.yticks(fontsize = fnt)
plt.ylim(-9 , 0)
plt.xlim(.1,10)
ax3.axvspan(7, 10, ymin=0, ymax=10, alpha=0.3, color=color_clipped, zorder = -10)
ax3.yaxis.set_major_locator(MultipleLocator(1))
ax3.tick_params(axis='y', which='minor', right=True )
ax3.tick_params(axis='x', which='minor', top=True )
ax3.tick_params(bottom=True, top=True, left=True, right=True)




ax4 = plt.subplot(224)
pl = plt.scatter(peak_SRDAS*fact , np.array(T_DAS)  , s = 15 ,c = dist_chans , cmap = 'magma', zorder = 1, edgecolors = 'k', linewidth = .4, alpha = .35)
plt.scatter(PGA, np.array(T_geo) , s = 15 ,c = dist_only , cmap = 'magma',marker='s', zorder = 1, edgecolors = 'blue', linewidth = .4 )
plt.xticks(fontsize = fnt)
plt.yticks(fontsize = fnt)
plt.xlim(.1,10)
plt.grid(linewidth = .5)
ax4.axvspan(7, 10, ymin=0, ymax=10, alpha=0.3, color=color_clipped, zorder = -10)
plt.xlabel('PGA (m/s$^2$)', fontsize = fnt)
plt.ylabel(r'$\tau_{max}$ (hour)', fontsize = fnt)
plt.ylim( 1, 9.99)
ax4.set_xscale('log')
ax4.yaxis.set_minor_locator(MultipleLocator(5))
ax4.tick_params(axis='y', which='minor', right=True )
ax4.tick_params(axis='x', which='minor', top=True )
ax4.tick_params(bottom=True, top=True, left=True, right=True) 
plt.text(.045 , 10 , '(d)' )


plt.tight_layout()
fig.savefig(dir_out + '/Fig_7.jpg', dpi=300)

#%%

print( 7/2400)
