# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:29:31 2020

@author: gutoa
"""
import numpy as np
from scipy.sparse.linalg import spsolve
def find_no(nos,coord=[0,0,0]):
    gpts = nos
    coord = np.array(coord)
    # no_ind = np.zeros_like(gpts)
    no_ind = []
    for i in range(len(gpts)):
        no_ind.append(np.linalg.norm(gpts[i,:]-coord))
        # print(gpts[i,:])
    # print(no_ind)    
    indx = no_ind.index(min(no_ind))

    return indx
class FEM1D:
    def __init__(self,Grid,AP,AC,A_dict,BC_dict):
        self.A_dict = A_dict
        self.BC_dict = BC_dict
        self.grid = Grid
        self.freq = AC.freq
        self.c0 = AP.c0
        self.rho0 = AP.rho0
        self.nos = Grid.nos
        self.nnos = len(Grid.nos)
        self.elem = Grid.elem
        self.nelem = self.nnos-1
        self.w = AC.w
        self.domain_index = Grid.domain_index
        
        # print(self.freq)
        # self.ComputeFEM1D = self.ComputeFEM1D()
        # self.AirProperties = self.AirProperties()
        # self.AlgControls = self.AlgControls()
        # self.GridImport = self.GridImport()
    
        



        
    def compute(self,coordF,coordZ,qi=10e-6):

        self.K = np.zeros([self.nnos,self.nnos],dtype=complex)
        self.M = np.zeros([self.nnos,self.nnos],dtype=complex)
        self.C = np.zeros([self.nnos,self.nnos],dtype=complex)

            
        for n in range(self.nelem):
            for key, value in self.A_dict.items():
                if self.domain_index[n] == key:
                    A = value

            # print(A)
            con = self.elem[n,:]
            he = np.sqrt((self.nos[con[0],0]-self.nos[con[1],0])**2 + (self.nos[con[0],1]-self.nos[con[1],1])**2)
            Ke = (A/self.rho0)*np.array([[1,-1],[-1,1]])/he
            Me = (A/(self.rho0*self.c0**2))*np.array([[2,1],[1,2]])*(he/6)
            # print(con[1])
            # print(Ke.shape)
            # print(K[con][con])
            self.K[con[0],con[0]] = self.K[con[0],con[0]] + Ke[0,0]
            self.K[con[0],con[1]] = self.K[con[0],con[1]] + Ke[0,1]
            self.K[con[1],con[0]] = self.K[con[1],con[0]] + Ke[1,0]
            self.K[con[1],con[1]] = self.K[con[1],con[1]] + Ke[1,1]
            
            self.M[con[0],con[0]] = self.M[con[0],con[0]] + Me[0,0]
            self.M[con[0],con[1]] = self.M[con[0],con[1]] + Me[0,1]
            self.M[con[1],con[0]] = self.M[con[1],con[0]] + Me[1,0]
            self.M[con[1],con[1]] = self.M[con[1],con[1]] + Me[1,1]

        # print(self.K)
        f = np.zeros([self.nnos,1],dtype=complex)
        # p = np.zeros([len(self.freq),self.nnos],dtype=complex)
        p = []
        self.NoF = find_no(self.nos,coordF)
        self.NoZ = find_no(self.nos,coordZ)
        fi = -1j*self.w*qi
        for j in range(len(self.freq)):
            f[self.NoF] = fi[j]
            Zrad = self.rho0*self.c0 * (.25*(self.w[j]/self.c0*0.0063/2)**2 -1j*0.6133*(self.w[j]/self.c0*0.0063/2))
            self.C[self.NoZ,self.NoZ] = self.A_dict[1]/Zrad
            # print(Zrad)
            # C = csc_matrix(C)
            G = self.K+1j*self.w[j]*self.C - (self.w[j]**2)*self.M
            # p_f = np.linalg.inv(G)*f
            p_f = spsolve(G,f)
            p.append(p_f)

        self.pN = np.array(p)
        
        return self.pN