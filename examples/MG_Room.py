# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:56:03 2020

@author: gutoa
"""
import sys
sys.path.append('C:\\Users\\gutoa\\Documents\\UFSM\\MNAV\\MENAV')
import femder as fd
import numpy as np
import matplotlib.pyplot as plt

project_folder = 'C:\\Users\\gutoa\\Documents\\Room Acoustics\\Cipriano_Rina_Gargel\\Room Study\\Rinaldi_Study\\Impedance_Data'

tmmM = fd.TMM(fmin=20,fmax=200,df=1,incidence='diffuse')

X_ = [1.000e+00, 5.718e+03, 2.230e+02, 8.000e+00, 3.400e+01, 2.100e+01, 1.400e+01,
 1.000e+01, 1.600e+01, 2.310e+02, 5.000e+00, 1.980e+02, 5.000e+00]

tmmM.perforated_panel_layer(t=40, d=4, s=15)
tmmM.porous_layer(model='mac', t=X_[11], sigma=X_[12])
tmmM.membrane_layer(t=X_[0], rho=X_[1])
tmmM.porous_layer(model='mac', t=X_[2], sigma=X_[3])
tmmM.air_layer(t=X_[4])
tmmM.perforated_panel_layer(t=X_[5],d = X_[6], s=X_[7])
tmmM.porous_layer(model='mac', t=X_[9], sigma=X_[10])
tmmM.air_layer(t=X_[8])
tmmM.compute(rigid_backing=True, show_layers=False)

tmm66 = fd.TMM(fmin=20,fmax=200,df=1, incidence='normal', project_folder=project_folder)
tmm66.slotted_panel_layer(t=250, w=2, s=30, method='barrier')
tmm66.porous_layer(model='mac', t=150, sigma=15)
tmm66.compute(rigid_backing=True, show_layers=True, conj=False)
# tmm66.plot(figsize=(7, 5), plots=['alpha'], saveFig=False, timestamp=False, filename='miki_test')

tmm116 = fd.TMM(fmin=20,fmax=200,df=1, incidence='normal', project_folder=project_folder)
tmm116.slotted_panel_layer(t=200, w=3, s=30, method='barrier')
tmm116.porous_layer(model='mac', t=150, sigma=15)
tmm116.compute(rigid_backing=True, show_layers=True, conj=False)
# tmm116.plot(figsize=(7, 5), plots=['alpha'], saveFig=False, timestamp=False, filename='miki_test')

tmm163 = fd.TMM(fmin=20,fmax=200,df=1, incidence='normal', project_folder=project_folder)
tmm163.slotted_panel_layer(t=100, w=3, s=30, method='barrier')
tmm163.porous_layer(model='mac', t=150, sigma=15)
tmm163.compute(rigid_backing=True, show_layers=True, conj=False)
# tmm163.plot(figsize=(7, 5), plots=['alpha'], saveFig=False, timestamp=False, filename='miki_test')

tmm230 = fd.TMM(fmin=20,fmax=200,df=1, incidence='normal', project_folder=project_folder)
tmm230.slotted_panel_layer(t=75, w=2, s=15, method='barrier')
tmm230.porous_layer(model='mac', t=150, sigma=15)
tmm230.compute(rigid_backing=True, show_layers=True, conj=False)
# tmm230.plot(figsize=(7, 5), plots=['alpha'], saveFig=False, timestamp=False, filename='miki_test')

tmmP = fd.TMM(fmin=20,fmax=200,df=1, incidence='diffuse', project_folder=project_folder)
tmmP.porous_layer(model='mac', t=150, sigma=15)
tmmP.compute(rigid_backing=True, show_layers=True, conj=False)
#%% Ola Amigo
import femder as fd

path_to_geo = '../Mshs/FEM_3D/room_mesh_MG_treatments - Copy.geo'
AP = fd.AirProperties(c0 = 343)
AC = fd.AlgControls(AP,400,400,1)

S = fd.Source("spherical")
S.coord = np.array([[1.53/2,2.7+1.32,1.14],[-1.53/2,2.7+1.32,1.14]])
S.q = np.array([[0.0001],[0.0001]])
        


R = fd.Receiver()
R.star([0,2.7,1.14],0.15)
grid = fd.GridImport3D(AP,path_to_geo,S,R,400,6,1,plot=False,order=1)
# grid.plot_mesh(True)

bc = fd.BC(AC,AP)
bc.normalized_admittance(2,0.02)
bc.mu[3] = tmmP.y
bc.mu[4] = tmmM.y
bc.mu[5] = tmm66.y
bc.mu[6] = tmm116.y
bc.mu[7] = tmm163.y
bc.mu[8] = tmm230.y
#%%
obj = fd.FEM3D(grid,S,R,AP,AC,bc)
obj.compute()

#%%
pN = obj.evaluate(R,plot=True)

#%%

obj.pressure_field(frequencies = 400    , renderer='browser')

#%%

import matplotlib.pyplot as plt

data = np.genfromtxt('../../201207_frequency_response_with_treatments.txt')
dataC = np.genfromtxt('../../config_rina_Q_mains.txt')

plt.semilogx(data[:,0],data[:,1],label = 'NIRO')
plt.semilogx(data[:,0],np.mean(dataC[:,1:-1],axis=1)-20,label='Evil Software')

Lp = (pN.T/(1j*AC.w))
Lp = fd.p2SPL(np.mean(Lp,axis=0))
plt.semilogx(AC.freq,Lp+80,label='Femder')
plt.legend()

plt.xticks([20,40,60,80,100,120,150,200],[20,40,60,80,100,120,150,200])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.grid()
plt.show()