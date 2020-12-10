# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:30:38 2020

@author: gutoa
"""
import femder as fd
import numpy as np
import bemder as bd
path_to_geo = "..\\Mshs\\FEM_3D\\cplx_room.geo"

AP = fd.AirProperties(c0 = 343)
AC = fd.AlgControls(AP,20,150,1)

S = fd.Source("spherical")

S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.q = np.array([[0.0001],[0.0001]])

R = fd.Receiver([0,1,1.2])

BC = fd.BC(AC,AP)
BCb = fd.BC(AC,AP)

# BC.normalized_admittance(3,0.02)
tmm = fd.TMM(20,200,1)
tmm.porous_layer(sigma=10.9, t=150, model='db')
tmm.compute()
# BC.normalized_admittance(3,tmm.y_norm)
BC.mu[3] = (tmm.y).ravel()

BC.rigid(2)

BCb.mu[2] = (tmm.y).ravel()

BCb.rigid(1)
#%%
gridf = fd.GridImport3D(AP,path_to_geo,S,R,fmax = 200,num_freq=6,scale=1,order=1)
gridb = bd.import_geo(path_to_geo, 200, 3)


#%%
# grid.plot_mesh(False)
objf = fd.FEM3D(gridf,S,R,AP,AC,BC)
objb = bd.InteriorBEM(gridb,AC,AP,S,R,BCb,'opencl')
#%%
objf.compute()
#%%
objb.impedance_bemsolve()
#%%
pN = objf.evaluate(R,plot=True)
pd = objb.point_evaluate(objb.boundData,R)
#%%

objl = fd.fem_load('test')

#%%

obj.evaluate(R,True)
#%%
obj.plot_problem(renderer='browser')
obj.surf_evaluate(freq = 200,renderer = 'browser',d_range = 45)
#%%
def p2SPL(p):
    SPL = 10*np.log10(0.5*p*np.conj(p)/(2e-5)**2)
    return SPL
import matplotlib.pyplot as plt
pp = np.genfromtxt('../../3d_valid_delany.txt')

# plt.semilogx(pp[:,0],pp[:,1],label='Validation',linewidth = 4,alpha=0.9,linestyle=':')
plt.semilogx(AC.freq,p2SPL((pd[0][:,0]*AC.w*1j)))
plt.semilogx(AC.freq,p2SPL(pN))
# plt.legend()
# plt.xlim([20,1])
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