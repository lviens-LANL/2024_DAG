#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:06:02 2023

@author: lviens
This code requires the basemap package and requires Matplotlib 3.1 (or below)
"""

import h5py, os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
os.environ['PROJ_LIB'] = '/opt/anaconda3/envs/py36/share/proj/'
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata

maps = ['ESRI_Imagery_World_2D',    # 0
        'ESRI_StreetMap_World_2D',  # 1
        'NatGeo_World_Map',         # 2
        'NGS_Topo_US_2D',           # 3
        'Ocean_Basemap',            # 4
        'USA_Topo_Maps',            # 5
        'World_Imagery',            # 6
        'World_Physical_Map',       # 7
        'World_Shaded_Relief',      # 8
        'World_Street_Map',         # 9
        'World_Terrain_Base',       # 10
        'World_Topo_Map'  ,          # 11
        ]
#%%
GZ_loc = [ -116.06926431, 37.114644234]

 
#%% Read geophones

dir_out = '../Figures/'
name_save = '../Data/Fig_9.h5'

with h5py.File(name_save, 'r') as hdf:
    peak_SR = hdf['peak_SR'][:] #))
    chans = hdf['chans'][:]
    dist_only = hdf['dist_only'][:]
    lon_only = hdf['lon_only'][:]
    lat_only = hdf['lat_only'][:]
    az = hdf['az'][:]
    res0geo = hdf['res0geo'][:]
    
a_geo = res0geo[:,0]
b_geo = res0geo[:,1]
    

#%%
chanW= ['03344360', '03344086', '03341974', '03345324', '03355788', '03360516', '03342886', '03363404', '03341690', '03342514', '03348032', '03342092', '03362614', '03345158', '03346918', '03344916', '03349772', '03341408', '03365444', '03374358', '03344294', '03356064', '03354840', '03342312', '03363096', '03348380', '03345748', '03356632', '03364802', '03345342', '03363850', '03349326', '03358154', '03351686']  
 
chanS= ['03358090', '03341986', '03347964', '03355086', '03363316', '03363188', '03359148', '03333260', '03361180', '03350818', '03348778', '03370634', '03380036', '03360666', '03353996', '03342824', '03342190', '03358664', '03344666', '03344974', '03359476', '03342172', '03363770', '03350112', '03356040', '03351278', '03375024' , '03343260' , '03343588']
#%% drop nans
a_geo2 = []
lat2 = []
lon2 =[]
chanWb2 = []
chanWa2 =[]
peak_SRa2 = []
peak_SRb2 = []
dist_onlya = []
for i in np.arange(len(a_geo)):
    if np.isnan(a_geo[i]):
        print('drop')
    else:
        a_geo2.append(a_geo[i])
        lat2.append(lat_only[i])
        lon2.append(lon_only[i])
        chanWa2.append(chans[i])
        dist_onlya.append(dist_only[i])
        peak_SRa2.append(peak_SR[i])
        
b_geo2 = []
latb2 = []
lonb2 =[]
dist_onlyb = []
for i in np.arange(len(b_geo)):
    if np.isnan(b_geo[i]):
        print('drop')
    else:
        b_geo2.append(b_geo[i] )
        latb2.append(lat_only[i])
        lonb2.append(lon_only[i])
        chanWb2.append(chans[i])
        dist_onlyb.append(dist_only[i])
        peak_SRb2.append(peak_SR[i])

#%% Get data along the two lines
line1a = []
line1b = []
for i in np.arange(len(chanW)):
    for ii in np.arange(len(chanWa2)):
        if chanW[i]==chanWa2[ii].decode("utf-8")and  (a_geo2[ii])<0:
             line1a.append([a_geo2[ii], peak_SRb2[ii] ,dist_onlya[ii] , lat2[ii], lon2[ii]])  
        if chanW[i]==chanWa2[ii].decode("utf-8")  and  (a_geo2[ii])<0:
             line1b.append([b_geo2[ii], peak_SRb2[ii] ,dist_onlyb[ii], lat2[ii], lon2[ii] ])
             # if dist_onlyb[ii] > 1200 : #and dist_onlyb[ii] < 600:
             #      print(chanWa2[ii].decode("utf-8"), dist_onlyb[ii], a_geo2[ii])

line2a = []
line2b = []
for i in np.arange(len(chanS)):
    for ii in np.arange(len(chanWa2)):
        if chanS[i]==chanWa2[ii].decode("utf-8")  and  (a_geo2[ii])<0:
             line2a.append([a_geo2[ii], peak_SRb2[ii] ,dist_onlya[ii] , latb2[ii], lonb2[ii]])
        if chanS[i]==chanWa2[ii].decode("utf-8")   and  (a_geo2[ii])<0:
             line2b.append([b_geo2[ii], peak_SRb2[ii], dist_onlyb[ii], latb2[ii], lonb2[ii] ])
             if dist_onlyb[ii] <400  :
                  print(chanWa2[ii].decode("utf-8"), dist_onlyb[ii] , a_geo2[ii])
             
             

line1a = np.array(line1a)
line2a = np.array(line2a)     
line1b = np.array(line1b)
line2b = np.array(line2b)         
# print(line2a) 
#%%
for ii in np.arange(len(chanWa2)):
    if dist_onlyb[ii] >300 and dist_onlyb[ii] <400:
                  print(chanWa2[ii].decode("utf-8"), dist_onlyb[ii] , b_geo2[ii])
#%%
GZ_loc = [ -116.06926431, 37.114644234]
fnt =12

x = np.linspace(np.min(lon2), np.max(lon2), 1000)
y = np.linspace( np.max(lat2), np.min(lat2),1000)
xv, yv = np.meshgrid(x, y)
grid_z0 = griddata(((lon2,lat2)), a_geo2, (xv, yv), method='linear')

#%%
fig1 = plt.figure(figsize = (10,9) )
ax2 = fig1.add_subplot(221)
north = 37.123
south = 37.095
west = -116.09
east = -116.055  


m3 = Basemap(projection='merc', lat_0 = south, lon_0 = west, llcrnrlat = south, urcrnrlat = north, llcrnrlon =  west, urcrnrlon = east, resolution = 'f', epsg = 3832)      

m3.arcgisimage(service='USA_Topo_Maps', xpixels = 2000) #, xpixels = 1000
 
lon_ticks_proj, _=m3(np.arange(round(west,2),round(east,2 ), .01), np.zeros(len(np.arange(round(west,2),round(east,2 ), .01))))
_, lat_ticks_proj=m3(np.zeros(len(np.arange( south, north,.01))), np.arange( south, north,.01))

ax2.set_xticks(lon_ticks_proj)
ax2.set_yticks(lat_ticks_proj)
ax2.tick_params(axis='both',which='major' )
# add ticks to the opposite side as well
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')
# remove the tick labels
ax2.xaxis.set_ticklabels([])
ax2.yaxis.set_ticklabels([])

parallels = np.arange(south,north,.01)
m3.drawparallels(parallels,labels=[1,0,0,0],linewidth=0,
        xoffset=0.03*abs(m3.xmax-m3.xmin), zorder=-2) # need to enlarge the offset a bit
meridians = np.arange(round(west,2),round(east,2 ),.01)
m3.drawmeridians(meridians ,labels=[0,0,0,1],linewidth=0,
        yoffset=0.03*abs(m3.ymax-m3.ymin), zorder=-2) # need to enlarge the offset a bit

xp, yp = m3(-116.09275 , 37.123 )
plt.text(xp, yp , '(a)', fontsize=fnt)

xp, yp = m3(xv, yv )
plt.contourf(xp, yp ,grid_z0 , cmap = 'hot', vmin = -5 ,vmax = 0, alpha = .7)

xp, yp = m3(lon2,lat2 )
sc = plt.scatter(xp, yp, c = a_geo2  , vmin = -5,vmax = 0 ,cmap = 'hot', edgecolors = 'k', linewidth = .5, alpha = .7)

xp, yp = m3(-116.089 , 37.119 )
plt.text(xp, yp , '                      \n                      \n                      ',   fontsize = fnt,bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', alpha = .75),zorder = 1)


arrowlonS = -116.0617
arrowlatS = 37.0957
x,y = m3(arrowlonS, arrowlatS)
x2, y2 = m3(arrowlonS-0.00083,arrowlatS+0.002)
plt.arrow(x,y,x2-x,y2-y,fc="r", ec="r", linewidth = 3, head_width=100, head_length=150  )


arrowlonS = -116.0875
arrowlatS = 37.0973
x,y = m3(arrowlonS, arrowlatS)
x2, y2 = m3(arrowlonS+0.002,arrowlatS+0.002)
plt.arrow(x,y,x2-x,y2-y,fc="g", ec="g", linewidth = 3, head_width=100, head_length=150  )

x,y = m3(-116.0615 , 37.097)
plt.text(x , y, 'SE line', weight='bold')

x,y = m3(-116.0894 , 37.0958)
plt.text(x , y, 'SW line', weight='bold')


m3.drawmapscale(-116.084, 37.12, -116.084, 37.12, 1, barstyle='fancy', fontsize =fnt, format ='%s',zorder = 10) # Plot scale

xp, yp = m3(GZ_loc[0],GZ_loc[1] )
plt.scatter(xp, yp, s = 150, edgecolors = 'k' ,color = 'r', linewidth = 1, marker='*', label = 'GZ' )

plt.grid(linewidth =.5)
cbaxes = fig1.add_axes([0.21, 0.94, 0.175, 0.019])
cb = plt.colorbar(sc,  cax = cbaxes, orientation = 'horizontal', ticks=np.arange(-5, 1, 1)  ) 
cb.ax.set_title('$D_{t_0}$ (%)', fontsize=fnt)


#%%

ax4 = plt.subplot(222)
m3 = Basemap(projection='merc', lat_0 = south, lon_0 = west, llcrnrlat = south, urcrnrlat = north, llcrnrlon =  west, urcrnrlon = east, resolution = 'f', epsg = 3832)      

m3.arcgisimage(service='USA_Topo_Maps', xpixels = 2000) #, xpixels = 1000
 
lon_ticks_proj, _=m3(np.arange(round(west,2),round(east,2 ), .01), np.zeros(len(np.arange(round(west,2),round(east,2 ), .01))))
_, lat_ticks_proj=m3(np.zeros(len(np.arange( south, north,.01))), np.arange( south, north,.01))

ax4.set_xticks(lon_ticks_proj)
ax4.set_yticks(lat_ticks_proj)
ax4.tick_params(axis='both',which='major' )
# add ticks to the opposite side as well
ax4.xaxis.set_ticks_position('both')
ax4.yaxis.set_ticks_position('both')
# remove the tick labels
ax4.xaxis.set_ticklabels([])
ax4.yaxis.set_ticklabels([])

parallels = np.arange(south,north,.01)
meridians = np.arange(round(west,2),round(east,2 ),.01)
m3.drawmeridians(meridians ,labels=[0,0,0,1],linewidth=0,
        yoffset=0.03*abs(m3.ymax-m3.ymin), zorder=-2) # need to enlarge the offset a bit


grid_z1 = griddata(((lonb2,latb2)), b_geo2, (xv, yv), method='linear')
xp, yp = m3(xv, yv )

plt.contourf(xp, yp ,grid_z1 , cmap = 'hot_r', vmin = 1 ,vmax = 7, alpha = .7)


xp, yp = m3(lonb2,latb2 )
sc = plt.scatter(xp, yp , c = b_geo2  , vmin = 1 ,vmax =7 ,cmap = 'hot_r', edgecolors = 'k', linewidth = .5, alpha = .7)

xp, yp = m3(GZ_loc[0],GZ_loc[1] )
plt.scatter(xp, yp, s = 150, edgecolors = 'k' ,color = 'r', linewidth = 1, marker='*', label = 'GZ' )
 
plt.grid(linewidth =.5)

xp, yp = m3(-116.09275 , 37.123 )
plt.text(xp, yp,  '(b)', fontsize=fnt)

arrowlonS = -116.0617
arrowlatS = 37.0957
x,y = m3(arrowlonS, arrowlatS)
x2, y2 = m3(arrowlonS-0.00083,arrowlatS+0.002)
plt.arrow(x,y,x2-x,y2-y,fc="r", ec="r", linewidth = 3, head_width=100, head_length=150  )


arrowlonS = -116.0875
arrowlatS = 37.0973
x,y = m3(arrowlonS, arrowlatS)
x2, y2 = m3(arrowlonS+0.002,arrowlatS+0.002)
plt.arrow(x,y,x2-x,y2-y,fc="g", ec="g", linewidth = 3, head_width=100, head_length=150  )

x,y = m3(-116.0615 , 37.097)
plt.text(x , y, 'SE line', weight='bold')

x,y = m3(-116.0894 , 37.0958)
plt.text(x , y, 'SW line', weight='bold')

cbaxes = fig1.add_axes([0.71, 0.94, 0.175, 0.019])
cb = plt.colorbar(sc,  cax = cbaxes, orientation = 'horizontal' , ticks= np.arange(1, 8 ,1) ) 
cb.ax.set_title('$\\tau_{max}$ (hour)', fontsize=fnt)





#%%
from scipy import stats
slope1a, intercept1a, r_value1a, p_value1a, std_err1a = stats.linregress( np.log10(line1a[:,2]),line1a[:,0])
slope2a, intercept2a, r_value2a, p_value2a, std_err2a = stats.linregress(np.log10(line2a[:,2]) , line2a[:,0] )

slope1b, intercept1b, r_value1b, p_value1b, std_err1b = stats.linregress(np.log10(line1b[:,2]),line1b[:,0])
slope2b, intercept2b, r_value2b, p_value2b, std_err2b = stats.linregress(np.log10(line2b[:,2]) , line2b[:,0] )

# print(r_valuea, r_valueb, slopea ,intercepta, slopeb, interceptb)
#%%
print(line1b)
           #%%
from matplotlib.ticker import (MultipleLocator)

xnew = np.arange(0,2200)
ax11 = plt.subplot(223)
plt.scatter(line1a[:,2] ,  line1a[:,0] ,s = 80, c = 'g' , edgecolor = 'k' )#, label = 'SW Line' )   
plt.scatter(line2a[:,2] ,  line2a[:,0] ,s = 80, c= 'r', edgecolor = 'k' )#,label = 'SSE Line' ) 

plt.plot(xnew,intercept1a + slope1a*np.log10(xnew) , 'g' , label = 'SW line: ' + str(round(intercept1a,2) ) + '+'  + str(round(slope1a,2) ) +' x log$_{10}$(d)' ) #' \n$R^2$: ' + str(round(r_value1a**2,2)) + f', p-value: {p_value1a:.1e}' ) 
plt.text(165, 0.5,  '(c)', fontsize=fnt)

plt.plot(xnew, intercept2a +slope2a*np.log10(xnew) , 'r' , label = 'SE line: ' + str(round(intercept2a,2) )  + '+' +  str(round(slope2a,2) ) +' x log$_{10}$(d) ') #' \n$R^2$: ' + str(round(r_value2a**2,2)) + f', p-val.: {p_value2a:.1e}' )
ax11.semilogx()
plt.xlim(200,2100)
plt.title(  'SW line - $R^2$: ' + str(round(r_value1a**2,2)) + f', p-value: {p_value1a:.1e}'  +'\n SE line - $R^2$: ' + str(round(r_value2a**2,2)) + f', p-value: {p_value2a:.1e}'   )

import matplotlib.ticker
plt.grid(linewidth = .5)  
plt.legend(loc = 8, fontsize = fnt)
plt.ylim(-4.9, 0) 
ax11.set_xlabel('Distance to SGZ (m)', fontsize=fnt)
ax11.set_ylabel('$D_{t_0}$ (%)', fontsize = fnt)
# ax11.xaxis.set_minor_locator(np.arange(200, 2000, 100) )
# ax11.xaxis.get_ticklocs(minor=True) 
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(.1, 2, .1),numticks=10)
ax11.xaxis.set_minor_locator(locmin)
ax11.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# ax11.set_xticks(np.arange(200, 2000, 100), minor=True)

ax11.tick_params(axis='y', which='minor', right=True )
ax11.tick_params(axis='x', which='minor', top=True )
ax11.tick_params(bottom=True, top=True, left=True, right=True)

ax11.set_xticks((200, 500, 1000, 2000))
ax11.set_xticklabels([200,500, 1000, 2000])



ax12 = plt.subplot(224)
plt.scatter(line1b[:,2] ,  line1b[:,0] , s = 80, c= 'g', edgecolor = 'k'  )    
plt.scatter(line2b[:,2] ,  line2b[:,0] ,s = 80, c= 'r', edgecolor = 'k'  )
 
plt.plot(xnew,intercept1b + slope1b*np.log10(xnew) , 'g' , label = 'SW line: ' +str(round(intercept1b,2) )  + ''+ str(round(slope1b,2) ) +' x log$_{10}$(d)')
plt.plot(xnew, intercept2b +slope2b*np.log10(xnew) , 'r' , label ='SE line: ' + str(round(intercept2b,2) )  + ''+ str(round(slope2b,2) ) +' x log$_{10}$(d)')
 
plt.text(165, 8.8,  '(d)', fontsize=fnt)

plt.legend(loc = 9, fontsize = fnt)

plt.grid(linewidth = .5) 
plt.xlim(200,2200)  
ax12.set_xlabel('Distance to SGZ (m)', fontsize=fnt)
ax12.set_ylabel('$\\tau_{max}$ (hour)', fontsize = fnt)
plt.title('SW line - $R^2$: ' + str(round(r_value1b**2,2)) + f', p-value: {p_value1b:.1e}'  +'\n SE line - $R^2$: ' + str(round(r_value2b**2,2)) + f', p-value: {p_value2b:.1e}'   )

ax12.semilogx()
plt.ylim(0, 8) 
 

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(.1, 2, .1),numticks=10)
ax12.xaxis.set_minor_locator(locmin)
ax12.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax12.set_xticks((200, 500, 1000, 2000))
ax12.set_xticklabels([200,500, 1000, 2000])


ax12.tick_params(axis='y', which='minor', right=True )
ax12.tick_params(axis='x', which='minor', top=True )
ax12.tick_params(bottom=True, top=True, left=True, right=True)

ax2.set_position([.08, .43 , .4, .5])
ax4.set_position([.58, .43 , .4, .5])
ax11.set_position([.08, .065 , .4, .3])
ax12.set_position([.58, .065 , .4, .3])

fig1.savefig(dir_out + '/Fig_9.jpg', dpi=300)

#%%
print(p_value1b)

