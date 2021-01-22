# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:43:11 2021

@author: gutoa
"""
import femder as fd
import numpy as np
import matplotlib.pyplot as plt
msh_path = '../Mshs/BEM_3D/hemi_30_cm_132cm.geo'

AP = fd.AirProperties()
AC = fd.AlgControls(AP,500,500,2)
# AC.third_octave_fvec(500,500,7)
# AC.freq = np.array([500])
# w = 2*np.pi*AC.freq
# k0 = w/AC.c0
S = fd.Source(coord = [100,0,0.72],q=[1])

R = fd.Receiver()
R.arc_receivers(50,2,angle_span=(-90,90),d=0.72,axis='z')

grid = fd.GridImport3D(AP,msh_path,fmax=500,num_freq=6,scale=1,meshDim=2,plot=False)

obj = fd.BEM3D(grid,S,R,AP,AC,BC=None)

obj.compute()
# obj.surf_evaluate(500,renderer='browser')
pT,pS = obj.evaluate(R,plot=True)

# pS = np.array(pss)
# pfs = np.mean(np.abs(pS),axis=0)
# pS = pfs
#%%
data = np.genfromtxt('hemi_valid_500hz.txt')
data = data[1:]
# pS = np.array(pss)
# pfs = np.mean(np.abs(pS),axis=0)
# pS = pfs
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# ax.set_thetamin(np.amin(np.rad2deg(R.theta)))
# ax.set_thetamax(np.amax(np.rad2deg(R.theta)))
# ax.set_ylim([-60,2])

ax.plot(R.theta, fd.p2SPL(pS.ravel()))
ax.plot(R.theta, data)
