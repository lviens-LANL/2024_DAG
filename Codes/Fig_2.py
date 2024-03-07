#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:11:52 2023

@author: lviens
Reproduce Figure 2 of the paper. The spectrograms have a lower frequency resolution to keep the file size low.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import (MultipleLocator)

dir_out = '../Figures/'
name_save = '../Data/Fig_2.h5'

# Load data
with h5py.File(name_save, 'r') as f:
    DAS = f.get('DAS')[:]
    refdDAS = f.get('refdDAS')[:]
    fs = f.get('fs')[:]
    tinifinDAS = f.get('tinifinDAS')[:]
    chan = f.get('chan_dist')[:]
    Geophone = f.get('Geophone')[:]
    Georef = f.get('Georef')[:]
    fsgeo = f.get('fsgeo')[:]
    tinifinGEO = f.get('tinifinGEO')[:]
    HVSR = f.get('HVSR')[:]
    HVref = f.get('HVref')[:]
    spect_time_step = f.get('spect_time_step')[:]
    

#%% Plot Figure 2
fnt = 12
arg1h = np.where(spect_time_step ==1)[0][0]

fig = plt.figure(figsize = (12,12))  
    
ax2 = plt.subplot(3, 2, 1 )
cl1 = [-115, -85]
sc = plt.imshow(DAS, aspect = 'auto', extent=(fs[0], fs[-1],tinifinDAS[0], tinifinDAS[-1]  ) , clim = cl1, cmap = 'viridis')
plt.plot(fs ,   refdDAS*-.12 -2, 'w' , linewidth = 3)
plt.text(2, -.3 , '(a)')
plt.xlim(5 ,30)
plt.grid(linewidth = .5)
ax2.yaxis.set_minor_locator(MultipleLocator(.5))
ax2.tick_params(axis='y', which='minor', right=True)
ax2.xaxis.set_minor_locator(MultipleLocator(1))
plt.ylabel('Time after DAG-2 (hour)', fontsize = fnt)
plt.title('DAS channel ' + str(int(chan[0]) )  )  
plt.ylim(12,-.12)
ax2.tick_params(bottom=True, top=True, left=True, right=True)
ax2.add_patch(Rectangle((5, 9),4,9, facecolor = 'w', edgecolor = 'k', alpha = .7 ))
cbaxes = fig.add_axes([0.06, 0.705, 0.02, 0.04])
cb = plt.colorbar(sc, cax = cbaxes, orientation = 'vertical'  )   
plt.title('dB',  fontsize = fnt-2)
plt.yticks( fontsize = fnt-2)



ax22 = plt.subplot(3, 2, 2)
preex = np.mean(DAS[:3, : ], axis =0 )
postex = DAS[arg1h, : ]  
plt.plot(fs ,preex, label =  'Pre DAG-2' , linewidth =2 )
plt.plot(fs ,postex , label =  '1 h after DAG-2', linewidth =2  )
plt.title('DAS channel ' + str( int( chan[0]) )  )  
plt.plot(fs ,  refdDAS , label = 'Reference', linewidth =2 )
plt.xlim(10, 25)
plt.legend()
plt.grid()
plt.ylabel('Fourier amplitude (dB)')
plt.ylim(-110, -80.01)
plt.text(9, -80, '(b)')
ax22.xaxis.set_minor_locator(MultipleLocator(1))
ax22.tick_params(bottom=True, top=True, left=True, right=True)


ax1 = plt.subplot(3, 2, 3)
cl = [-110, -80]
sc2 = plt.imshow(Geophone, aspect = 'auto', extent=(fsgeo[0], fsgeo[-1], tinifinGEO[0], tinifinGEO[-1]  ) , clim = cl, cmap = 'viridis') 
plt.plot(fsgeo ,  Georef*-.12 + -2, 'w', linewidth =3 )
plt.xlim(5 ,30)
plt.grid(linewidth = .5)
plt.ylabel('Time after DAG-2 (hour)', fontsize = fnt)
plt.title('Geophone - Distance to SGZ: ' + str(round(chan[1] , 1)) + ' m' ) #, fontsize = fnt)
plt.text(2, -.3 , '(c)')
plt.ylim(12,-.12)
ax1.tick_params(bottom=True, top=True, left=True, right=True)
ax1.yaxis.set_minor_locator(MultipleLocator(.5))
ax1.tick_params(axis='y', which='minor', right=True)
ax1.xaxis.set_minor_locator(MultipleLocator(1))
cbaxes = fig.add_axes([0.06, 0.382, 0.02, 0.04])
ax1.add_patch(Rectangle((5,9), 3.75, 9, facecolor = 'w', edgecolor = 'k', alpha = .7 ))
cb = plt.colorbar(sc2, cax = cbaxes, orientation = 'vertical'  )  #, ticks=[-2.1,-1.4]
plt.title('dB',  fontsize = fnt-2)
plt.yticks( fontsize = fnt-2)


ax11 = plt.subplot(3, 2, 4 )
preex = np.mean(Geophone[:3, : ], axis =0 )
postex =  Geophone[arg1h, : ]
plt.plot(fsgeo ,preex, label =  'Pre DAG-2', linewidth =2  )
plt.plot(fsgeo ,postex , label =  '1 h after DAG-2' , linewidth =2 )
plt.plot(fsgeo ,  Georef , label = 'Reference', linewidth = 2)
plt.title('Geophone - Distance to SGZ: 224.4 m' ) 
plt.xlim(10, 25)
plt.legend()
plt.grid()
plt.ylabel('Fourier amplitude (dB)')
plt.text(9, -75, '(d)')
plt.ylim(-105,-75.01)
ax11.xaxis.set_minor_locator(MultipleLocator(1))
ax11.tick_params(bottom=True, top=True, left=True, right=True)


ax3 = plt.subplot(325)
cl = [0, 5]
sc3 = plt.imshow(HVSR,aspect = 'auto', extent=(fsgeo[0], fsgeo[-1], tinifinGEO[0], tinifinGEO[-1] ) , clim = cl, cmap = 'viridis') 
plt.plot(fsgeo ,  HVref*-.3 +11  , 'w' , linewidth = 3 )

plt.xlabel('Frequency (Hz)', fontsize = fnt)
plt.title('Geophone - HVSR')
plt.text(2, -.3, '(e)')
plt.grid()
plt.ylim(12,-.12)
plt.xlim(5,30)
ax3.yaxis.set_minor_locator(MultipleLocator(.5))
ax3.xaxis.set_minor_locator(MultipleLocator(1))
ax3.tick_params(axis='y', which='minor', right=True)
plt.ylabel('Time after DAG-2 (hour)')
ax3.tick_params(bottom=True, top=True, left=True, right=True)
ax3.add_patch(Rectangle((5,9), 3.3, 9, facecolor = 'w', edgecolor = 'k', alpha = .7 ))
cbaxes = fig.add_axes([0.063, 0.059, 0.02, 0.04])
cb = plt.colorbar(sc3, cax = cbaxes, orientation = 'vertical'  )  #, ticks=[-2.1,-1.4]
plt.title('  HVSR',  fontsize = fnt-2)


ax33 = plt.subplot(326)
preex = np.mean(HVSR[:1, : ], axis =0 )
postex = HVSR[arg1h, : ]
plt.plot(fsgeo, preex, label =  'Pre DAG-2' , linewidth =2 )
plt.plot(fsgeo, postex , label =  '1 h after DAG-2' , linewidth =2 )
plt.plot(fsgeo, HVref , label = 'Reference', linewidth =2 )
plt.xlim(10, 25)
plt.legend()
plt.grid()
plt.title('Geophone - HVSR')
plt.xlabel('Frequency (Hz)')
plt.ylabel('HVSR amplitude')
plt.ylim(0, 11.9)
plt.text(9, 12, '(f)')
ax33.xaxis.set_minor_locator(MultipleLocator(1))
ax33.tick_params(bottom=True, top=True, left=True, right=True)


plt.tight_layout()
plt.show()
fig.savefig(dir_out + '/Fig_2.jpg', dpi=300)
plt.close()



