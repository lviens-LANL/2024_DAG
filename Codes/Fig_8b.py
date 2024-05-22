#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:59:39 2023

@author: lviens
"""

   

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy import optimize
from matplotlib.ticker import (MultipleLocator)

#%% Bowl shap model
def piecewise_linear2(x, x0, y0, k2):
    return np.piecewise(x, [x < x0], [lambda x:0*x + y0-0*x0, lambda x:k2*x + y0-k2*x0])

#%%
dir_out = '../Figures/'
name_save = '../Data/Fig_8b.h5'


# Load data
with h5py.File(name_save, 'r') as f:
    spalldur2 = f.get('spalldur2')[:]
    disthzfin = f.get('disthzfin')[:]

#%% Fit data

# initial point
x = np.array(disthzfin )  
y =  np.array(spalldur2 )  
p_init = [ 2.11997704e+02 , 1.18929724e-01 , 1.20365824e-06, -5.08256792e-06]
p_init = [ 200, .15 ,   -10e-05]  
# set initial parameter estimates
p, pconv = optimize.curve_fit(piecewise_linear2, x, y , p0=p_init , method = 'lm')
xnew = np.arange(20,470,.01) 

#%% plot data
fnt =12
fig = plt.figure(figsize = (6,4) )

ax2 = plt.subplot(111)
plt.scatter(x , y , s = 80 , edgecolors='k', label = 'Data')
plt.plot(xnew  ,  piecewise_linear2(xnew, *p) ,'orange' , linewidth = 4, label = 'Model' )
plt.legend()

yl = [0 , .1199]
plt.xlabel('Distance to SGZ (m)',fontsize = fnt)
plt.ylabel('Slapdown phase duration',fontsize = fnt)
plt.grid(linewidth = .5 )
plt.xlim(20, 470 )
plt.ylim(yl)
plt.xticks(fontsize = fnt)
plt.yticks(fontsize = fnt)
plt.text(-39, .12 , '(b)', fontsize = fnt)
ax2.tick_params(bottom=True, top=True, left=True, right=True)

ax3 = ax2.twinx()

VPToney = (480 * 3 + 500* 4 + 530 * 8 + 570* 5 )/ 20 *1.5
VPSchramm = (410 * 3 + 4* 560 + 8*690 + 5*830) /20

VP = VPToney
ax3.set_ylim( 0,  yl[1]/2 *VP )


spallde = piecewise_linear2(xnew, *p)/2*VP
plt.plot(xnew  ,  spallde  ,'orange' , linewidth = 1, label = 'Model' )
print(spallde)

arrval = np.arange(0, yl[1] , .02) # [0 , .05  , .1 , .15 ]
vb =[]
for i in arrval :
    vb.append( round(i /2*VP  ,1) )
print(vb)  
plt.yticks(  vb )
plt.ylabel('Estimated spall depth (m)')


plt.tight_layout()
fig.savefig(dir_out + '/Fig_8b.jpg', dpi=300)
