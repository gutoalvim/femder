# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:33:54 2020

@author: gutoa
"""
import numpy as np
from scipy.sparse.linalg import spsolve
import time 
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

def int_tetra_simpl(coord_el,c0,rho0,npg):

    He = np.zeros([4,4])
    Qe = np.zeros([4,4])
    
# if npg == 1:
    #Pontos de Gauss para um tetraedro
    ptx = 1/4 
    pty = 1/4
    ptz = 1/4
    wtz= 1#/6 * 6 # Pesos de Gauss
    qsi1 = ptx
    qsi2 = pty
    qsi3 = ptz
    
    Ni = np.array([[1-qsi1-qsi2-qsi3],[qsi1],[qsi2],[qsi3]])
    GNi = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]])

    Ja = (GNi@coord_el)
    detJa = (1/6) * np.linalg.det(Ja)
    # print(detJa)
    B = (np.linalg.inv(Ja)@GNi)
    # B = spsolve(Ja,GNi)
    # print(B.shape)              
    argHe1 = (1/rho0)*(np.transpose(B)@B)*detJa
    # print(np.matmul(Ni,np.transpose(Ni)).shape)
    argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
    
    He = He + wtz*wtz*wtz*argHe1   
    Qe = Qe + wtz*wtz*wtz*argQe1 
    
    return He,Qe

def int_tri_impedance_simpl(coord_el,npg):


    Ae = np.zeros([3,3])
    xe = np.array(coord_el[:,0])[0]
    ye = np.array(coord_el[:,1])[0]
    ze = np.array(coord_el[:,2])[0]
    #Formula de Heron - Area do Triangulo
    
    a = np.sqrt((xe[0]-xe[1])**2+(ye[0]-ye[1])**2+(ze[0]-ze[1])**2)
    b = np.sqrt((xe[1]-xe[2])**2+(ye[1]-ye[2])**2+(ze[1]-ze[2])**2)
    c = np.sqrt((xe[2]-xe[0])**2+(ye[2]-ye[0])**2+(ze[2]-ze[0])**2)
    p = (a+b+c)/2
    area_elm = np.abs(np.sqrt(p*(p-a)*(p-b)*(p-c)))
    
    if npg == 1:
    #Pontos de Gauss para um tetraedro
        ptx = 1/3
        pty = 1/3
        wtz= 1#/6 * 6 # Pesos de Gauss
        qsi1 = ptx
        qsi2 = pty
        wtx = wtz
        wty = wtz
        
        
    if npg == 3:
    #Pontos de Gauss para um tetraedro
        aa = 1/6
        bb = 2/3
        ptx = np.array([aa,aa,bb])
        pty = np.array([aa,bb,aa])
        wtz= np.array([1/6,1/6,1/6])*2 # Pesos de Gauss
    
        for indx in range(npg):
            qsi1 = ptx[indx]
            wtx =  wtz[indx]
        for indx in range(npg):
            qsi2 = pty[indx]
            wty =  wtz[indx]
        
    Ni = np.array([[qsi1],[qsi2],[1-qsi1-qsi2]])
    
        
    detJa= area_elm
    argAe1 = Ni@np.transpose(Ni)*detJa
    
    Ae = Ae + wtx*wty*argAe1
    
    return Ae
class FEM3D:
    def __init__(self,Grid,S,AP,AC,BC=None):
        """
        

        Parameters
        ----------
        Grid : GridImport()
            GridImport object created with MENAV.GridImport('YOUGEO.geo',fmax,maxSize,scale).
        AP : AirProperties
            AirPropeties object containing, well, air properties.
        AC : AlgControls
            Defines frequency configuration for calculation.
        BC : BoundaryConditions()
            BoundaryConditions object containg impedances for each assigned Physical Gropu in gmsh.

        Returns
        -------
        None.

        """
        
        self.BC= BC
        self.mu = BC.mu
        #AirProperties
        self.freq = AC.freq
        self.w = AC.w
        
        ##AlgControls
        self.c0 = AP.c0
        self.rho0 = AP.rho0
        
        self.S = S
        #%Mesh
        self.grid = Grid
        self.nos = Grid.nos
        self.elem_surf = Grid.elem_surf
        self.elem_vol =  Grid.elem_vol
        self.domain_index_surf =  Grid.domain_index_surf
        self.domain_index_vol =Grid.domain_index_vol
        self.number_ID_faces =Grid.number_ID_faces
        self.number_ID_vol = Grid.number_ID_vol
        self.NumNosC = Grid.NumNosC
        self.NumElemC = Grid.NumElemC
        self.npg = 1
        
    def compute(self,timeit=True):
        then = time.time()
        self.H = np.zeros([self.NumNosC,self.NumNosC],dtype = np.complex128)
        self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype = np.complex128)
        self.A = np.zeros([self.NumNosC,self.NumNosC,len(self.number_ID_faces)],dtype = np.complex128)
        self.q = np.zeros([self.NumNosC,1],dtype = np.complex128)
        
        #Assemble H(Massa) and Q(Rigidez) matrix
        for e in range(self.NumElemC):
            con = self.elem_vol[e,:]
            coord_el = self.nos[con,:]
            
            He, Qe = int_tetra_simpl(coord_el,self.c0,self.rho0,self.npg)   
            
            self.H[con[:,np.newaxis],con] = self.H[con[:,np.newaxis],con] + He
            self.Q[con[:,np.newaxis],con] = self.Q[con[:,np.newaxis],con] + Qe
   
        
        #Assemble A(Amortecimento)
        npg = 3
        if self.BC != None:
            i = 0
            for bl in self.number_ID_faces:
                indx = np.argwhere(self.domain_index_surf==bl)
                for es in range(len(self.elem_surf[indx])):
                    con = self.elem_surf[indx[es],:]
                    coord_el = self.nos[con,:]
                    con = con[0]
                    Ae = int_tri_impedance_simpl(coord_el, npg)
                    self.A[con[:,np.newaxis],con,i] = self.A[con[:,np.newaxis],con] + Ae
                i += 1
            
        pN = []
        
        
        for ii in range(len(self.S.coord)):
            self.q[find_no(self.S.coord[:,ii])] = self.S.q
        i = 0
        for N in range(len(self.freq)):
            Ag = np.zeros_like(self.Q)
            for bl in self.number_ID_faces:
                Ag = self.A[:,:,i]*self.mu[bl][N]
            G = self.H + 1j*self.w[N]*Ag - (self.w[N]**2)*self.Q
            b = -1j*self.w[N]*self.q
            ps = spsolve(G,b)
            pN.append(ps)
            
        self.pN = np.array(pN)
        self.t = time.time()-then
        
        if timeit:
            if self.t <= 60:
                print(f'Time taken: {self.t} s')
            elif 60 < self.t < 3600:
                print(f'Time taken: {self.t/60} min')
            elif self.t >= 3600:
                print(f'Time taken: {self.t/60} min')
                
    def evaluate(self,R):
        self.R = R
        
        
        pp = 
        for i in range(len(R.Coord))
        return 