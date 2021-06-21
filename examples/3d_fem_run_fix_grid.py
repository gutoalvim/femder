# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:30:38 2020

@author: gutoa
"""
import femder as fd
import numpy as np
import matplotlib.pyplot as plt

def p2SPL(p):
    SPL = 10*np.log10(0.5*p*np.conj(p)/(2e-5)**2)
    return SPL


path_to_geo = "..\\Mshs\\FEM_3D\\room_vol.geo"



AP = fd.AirProperties(c0 = 343)
AC = fd.AlgControls(AP,20,200,1)

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
# BC.mu[3] = (tmm.y)
# BC.delany(3,10900,0.15)
BC.normalized_admittance(3,0.02)

BC.rigid(2)
#%%
grid = fd.GridImport3D(AP,path_to_geo,S,R,fmax = 200,num_freq=6,scale=1,order=1)
# grid = fd.GridImport3D(AP,path_to_geo,fmax = 200,num_freq=3,scale=1)
#%%


#%%
# grid.plot_mesh(False)
obj = fd.FEM3D(grid,S,R,AP,AC,BC)
#%%
obj.plot_problem('browser')
#%%
obj.optimize_source_receiver_pos([4,4],plot_geom=True,renderer='browser')


#%%
obj.compute()
#%%
obj.fem_save('test')
#%%
pN = obj.evaluate(R,plot=True)

#%%

pn_win = np.convolve(pN.ravel(),freq_win/20)
freq = np.linspace(20,200,len(pn_win))
plt.semilogx(freq,p2SPL(pn_win))
# plt.semilogx(AC.freq,p2SPL(pN))

#%%


IR = fd.IR(44100,1,20,200)
ir = IR.compute_room_impulse_response(pN.ravel())
t_ir = np.linspace(0,1,len(ir))
sbir = fd.SBIR(ir,t_ir,20,200,method='peak')
fft = np.fft.fft(ir)
# plt.plot(ir)
plt.semilogx(sbir[0],p2SPL(sbir[1]))
# plt.semilogx(np.abs(pN))



#%%
from scipy.signal import find_peaks



pR = np.abs(obj.pR).flatten()
peaks, _ = find_peaks(pR)#,prominence=1,width=3)
# pR = p2SPL(obj.pR).ravel()
plt.plot(AC.freq,(pR))
plt.vlines(x=peaks,ymin=np.amin(pR),ymax=np.amax(pR),linestyle=':')

#%%

objl = fd.fem_load('test')

#%%

obj.evaluate(R,True)
#%%

obj.plot_problem(renderer='browser')
obj.surf_evaluate(freq = 49,renderer = 'browser',d_range = 45)


#%%
import matplotlib.pyplot as plt
pp = np.genfromtxt('../../3d_valid_delany.txt')

plt.semilogx(pp[:,0],pp[:,1],label='Validation',linewidth = 4,alpha=0.9,linestyle=':')
plt.legend()
# plt.xlim([20,150])
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

#%%
obj.pressure_field(frequencies = 49, renderer='browser')
    