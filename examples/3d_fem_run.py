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


path_to_geo = "..\\Mshs\\FEM_3D\\cplx_room.geo"



AP = fd.AirProperties(c0 = 343)
AC = fd.AlgControls(AP,10,250,1)

S = fd.Source("spherical")

S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.q = np.array([[0.0001],[0.0001]])

R = fd.Receiver(coord=[0,1,1.2])
# R.star([0,1,1.2],0.15)
BC = fd.BC(AC,AP)
# BC.normalized_admittance(3,0.02)
tmm = fd.TMM(20,2000,1)
tmm.porous_layer(sigma=10.9, t=150, model='mac')
tmm.compute()
# BC.normalized_admittance(3,tmm.y_norm)
# BC.mu[2] = (tmm.y)
# BC.delany(3,10900,0.15)
BC.normalized_admittance(2,0.02)
# BC.normalized_admittance(,0.02)

# BC.rigid(2)
#%%
grid = fd.GridImport3D(AP,path_to_geo,S=None,R=None,fmax = 250,num_freq=6,scale=1,order=1)
# grid = fd.GridImport3D(AP,path_to_geo,fmax = 200,num_freq=3,scale=1)



#%%
    # grid.plot_mesh(False)
obj = fd.FEM3D(grid,S,R,AP,AC,BC)
#%%
obj.optimize_source_receiver_pos([3,3],minimum_distance_between_speakers=0.9,
                                      max_distance_from_wall=0.7,speaker_receiver_height=1.24,
                                      min_distance_from_backwall=0.6,max_distance_from_backwall=1.5,
                                      method='modal',neigs=200,plot_geom=False,renderer='browser',
                                      print_info=False,saveFig=False,camera_angles=['diagonal_front'],
                                      plot_evaluate=False,plotBest=False)

#%%

obj.plot_problem(renderer='browser')
#%%
obj.compute()
#%%
obj.fem_save('test')
#%%
pN = obj.evaluate(R,plot=True)

#%%

r = obj.fitness_metric(w1=0.5, w2=0.5,fmin=20,fmax=200, dip_penalty=True, center_penalty=True, mode_penalty=True,
                       ref_curve='mean', dB_oct=2, nOct=2, lf_boost=10, infoLoc=(0.12, -0.03),
                       returnValues=True, plot=True, figsize=(17, 9), std_dev='assymmetric')
#%%

sf,ss = fd.FEM_3D.SBIR_SPL(obj.pR,obj.R.coord, AC,10,250)

#%%
import scipy.signal.windows as win
t_IR = np.linspace(0,1,len(obj.freq))
N = len(obj.freq)*2 + 1
peak = 0  # Window from the start of the IR
dt = (max(t_IR) / len(t_IR))  # Time axis resolution
tt_ms = round((ms / 1000) / dt)  # Number of samples equivalent to 64 ms

# Windows
post_peak = np.zeros((len(t_IR[:])))
pre_peak = np.zeros((len(t_IR[:])))


win_cos = win.tukey(int(2 * tt_ms), 1)  # Cosine window

window = np.zeros((len(t_IR[:])))  # Final window
##
win_cos[0:int(tt_ms)] = 1
window[0:int(2 * tt_ms)] = win_cos

win_freq = np.fft.fft(window,N)
pR = np.pad(obj.pR[:,0],N-len(obj.pR[:,0]),mode='edge')

sbir = np.convolve(win_freq,pR)

plt.semilogx(fd.p2SPL(sbir))

#%%
plt.semilogx(sf,ss,label='Direct')
plt.semilogx(sf2,ss2,label='Modal')
plt.legend()
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

fn = obj.eigenfrequency(200)

# obj.modal_evaluate(49,'browser',d_range=None)

#%%

pM = obj.modal_superposition(R)

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
    