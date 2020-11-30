# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:30:38 2020

@author: gutoa
"""
import femder as fd
import numpy as np

path_to_geo = "..\\Mshs\\FEM_3D\\cplx_room.geo"



AP = fd.AirProperties(c0 = 343)
AC = fd.AlgControls(AP,20,200,1)

S = fd.Source("spherical")
S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.q = np.array([[0.0001],[0.0001]])

R = fd.Receiver([0,1,1.2])

BC = fd.BC(AC,AP)
BC.normalized_admittance(2,0.02)
BC.rigid(3)

grid = fd.GridImport3D(AP,S,R,path_to_geo, fmax=200, num_freq=6, plot=False,scale=1)

obj = fd.FEM3D(grid,S,AP,AC,BC)
#%%
obj.compute()
pN = obj.evaluate(R,plot=True)

#%%
import matplotlib.pyplot as plt
pp = np.genfromtxt('../../3d_valid_Z.txt')
plt.semilogx(AC.freq,pp[:,1])