# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:30:38 2020

@author: gutoa
"""
import femder as fd
import numpy as np

path_to_geo = "..\\Mshs\\FEM_3D\\cplx_room.geo"



AP = fd.AirProperties(c0 = 343)
AC = fd.AlgControls(AP,20,150,1)

S = fd.Source("spherical")

S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.q = np.array([[0.0001],[0.0001]])

R = fd.Receiver([0,1,1.2])

BC = fd.BC(AC,AP)
# BC.normalized_admittance(3,0.02)
tmm = fd.TMM(20,200,1)
tmm.porous_layer(sigma=10.9, t=150, model='db')
tmm.compute()
# BC.normalized_admittance(3,tmm.y_norm)
BC.mu[3] = (tmm.y)
# BC.delany(3,10900,0.15)
# BC.normalized_admittance(3,0.02+1j*0.5)

BC.rigid(2)
#%%
grid = fd.GridImport3D(AP,path_to_geo,S,R,fmax = 200,num_freq=6,scale=1,order=1)
# grid = fd.GridImport3D(AP,path_to_geo,fmax = 200,num_freq=3,scale=1)



#%%
# grid.plot_mesh(False)
obj = fd.FEM3D(grid,S,R,AP,AC,BC)
obj.npg = 4
#%%
obj.compute()
#%%
obj.fem_save('test')
#%%
pN = obj.evaluate(R,plot=True)
#%%

objl = fd.fem_load('test')

#%%

objl.evaluate(R,True)
#%%
obj.plot_problem(renderer='browser')
obj.surf_evaluate(freq = 200,renderer = 'browser',d_range = 45)
#%%
import matplotlib.pyplot as plt
pp = np.genfromtxt('../../3d_valid_delany.txt')

plt.semilogx(pp[:,0],pp[:,1],label='Validation',linewidth = 4,alpha=0.9,linestyle=':')
plt.legend()
plt.xlim([20,150])
#%%

fn = obj.eigenfrequency(20)

obj.modal_evaluate(49,'browser',d_range=None)

#%%

obj.modal_superposition(R)

#%%

def p2SPL(p):
    SPL = 10*np.log10(0.5*p*np.conj(p)/(2e-5)**2)
    return SPL
pp = np.genfromtxt('../../3d_valid.txt')
plt.semilogx(pp[:,0],pp[:,1],label='Validation',linewidth = 5,alpha=0.5)
plt.legend()
plt.semilogx(AC.freq,p2SPL(obj.pm)-50,label='Modal')
plt.legend()
#%%

fn = obj.amort_eigenfrequency(10)
#%%
tr = 1.1/np.imag(fn)
Q = np.abs(fn)/np.imag(fn)/2

#%%

mp = np.genfromtxt('../../modos_Z.txt')

plt.scatter(fn[1:-1],mp[1:len(fn)-1],label='validation',linewidth=6)
plt.scatter(fn[1:-1],tr[1:-1],marker='*',label='femder',color='black',linewidth=4)
plt.xticks(np.round(np.real(fn)),np.round(np.real(fn)))

plt.xlabel('Eigenfrequencies [Hz]')
plt.ylabel('MT60 [s]')
plt.legend()
plt.ylim([0.1,7])
plt.xlim([45,113])
plt.grid()