# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:30:38 2020

@author: gutoa
"""
import femder as fd
import numpy as np

path_to_geo = "Mshs\\FEM_3D\\cplx_room.geo"



AP = fd.AirProperties(c0 = 343)
AC = fd.AlgControls(AP,200,200,1)

S = fd.Source("spherical")
S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.q = np.array([[0.0001],[0.0001]])

R = fd.Receiver([0,1,1.2])

BC = fd.BC(AC,AP)
BC.normalized_admittance(2,0.02)
BC.rigid(3)
#%%
grid = fd.GridImport3D(AP,path_to_geo,S,R,fmax = 200,num_freq=6,scale=1)
# grid = fd.GridImport3D(AP,path_to_geo,fmax = 200,num_freq=6,scale=1)


#%%
# grid.plot_mesh(False)
obj = fd.FEM3D(grid,S,AP,AC,BC)
obj.npg = 4
#%%
obj.compute()
#%%
pN = obj.evaluate(R,plot=True)


#%%
obj.plot_problem(renderer='browser')
obj.surf_evaluate(freq = 200,renderer = 'browser',d_range = 45)
#%%
import matplotlib.pyplot as plt
pp = np.genfromtxt('../3d_valid_Z_refine.txt')
plt.semilogx(AC.freq,pp[:,1],label='Validation',linewidth = 5,alpha=0.5)
plt.legend()