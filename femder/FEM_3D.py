# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:33:54 2020

@author: gutoa
"""
import numpy as np
from scipy.sparse.linalg import spsolve
# from pypardiso import spsolve
# from scipy.sparse.linalg import gmres
import time 
from tqdm import tqdm
import warnings
from numba import jit
import cloudpickle
# from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def fem_load(filename,ext='.pickle'):
    
        import pickle
        
        infile = open(filename + ext, 'rb')
        simulation_data = pickle.load(infile)
        # simulation_data = ['simulation_data']
        infile.close()
        # Loading simulation data

        AP = simulation_data['AP']
        AC = simulation_data['AC']
        S = simulation_data["S"]
        R = simulation_data["R"]
        Grid = simulation_data['grid']
        # self.set_status = True
        BC = simulation_data["BC"]


        obj = FEM3D(Grid=None,AC=AC,AP=AP,S=S,R=R,BC=BC)
        obj.freq = AC.freq
        obj.w = AC.w
        obj.AC = AC
        obj.AP = AP
        ##AlgControls
        obj.c0 = AP.c0
        obj.rho0 = AP.rho0
        
        obj.S = S
        #%Mesh
        obj.grid = Grid
        obj.nos = Grid['nos']
        obj.elem_surf = Grid['elem_surf']
        obj.elem_vol =  Grid['elem_vol']
        obj.domain_index_surf =  Grid['domain_index_surf']
        obj.domain_index_vol =Grid['domain_index_vol']
        obj.number_ID_faces =Grid['number_ID_faces']
        obj.number_ID_vol = Grid['number_ID_vol']
        obj.NumNosC = Grid['NumNosC']
        obj.NumElemC = Grid['NumElemC']
        obj.order = Grid["order"]
        
        obj.pR = simulation_data['pR']
        obj.pN = simulation_data['pN']
        obj.F_n = simulation_data['F_n']
        obj.Vc = simulation_data['Vc']
        obj.H = simulation_data['H']
        obj.Q = simulation_data['Q']
        obj.A = simulation_data['A']
        obj.q = simulation_data['q']
        print('FEM loaded successfully.')
        return obj


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def p2SPL(p):
    SPL = 10*np.log10(0.5*p*np.conj(p)/(2e-5)**2)
    return SPL

@jit
def Tetrahedron10N(qsi):

    t1 = qsi[0]
    t2 = qsi[1]
    t3 = qsi[2]
    t4 = 1 - qsi[0] - qsi[1] - qsi[2];
    # print(t1)
    N = np.array([t4*(2*t4 - 1),t1*(2*t1 - 1),t2*(2*t2 - 1),t3*(2*t3 - 1),
                  4*t1*t4,4*t1*t2,4*t2*t4,4*t3*t4,4*t2*t3,4*t3*t1]);
    return N[:,np.newaxis]

@jit
def Triangle10N(qsi):
    
    N = np.array([(-qsi[0] - qsi[1] + 1) * (2*(-qsi[0] - qsi[1] + 1) - 1),
    qsi[0]*(2*qsi[0] - 1),
    qsi[1]*(2*qsi[1] - 1),
    4*qsi[0]*qsi[1],
    4*qsi[1]*(-qsi[0] - qsi[1] + 1),
    4*qsi[0]*(-qsi[0] - qsi[1] + 1)])
    
    # deltaN = np.array([[(4*qsi[0] + 4*qsi[1] - 3),
    # (4*qsi[0] - 1),
    # 0,
    # 4*qsi[1],
    # -4*qsi[1],
    # (4 - 4*qsi[1] - 8*qsi[0])],
    # [(4*qsi[0] + 4*qsi[1] - 3),
    # 0,
    # (4*qsi[1] - 1),
    # 4*qsi[0],
    # 4 - 8*qsi[1] - 4*qsi[0],
    # -4*qsi[0]]
    # ]);
    
    return N[:,np.newaxis]#,deltaN
@jit
def Tetrahedron10deltaN(qsi):
    t1 = 4*qsi[0]
    t2 = 4*qsi[1]
    t3 = 4*qsi[2]
    # print(t1)
    deltaN = np.array([[t1 + t2 + t3 - 3,t1 + t2 + t3 - 3,t1 + t2 + t3 - 3],[t1 - 1,0,0],[0,t2 - 1,0],[0,0,t3 - 1],
                        [4 - t2 - t3 - 2*t1,-t1,-t1],[t2,t1,0],[-t2,4 - 2*t2 - t3 - t1,-t2],
                        [-t3,-t3,4 - t2 - 2*t3 - t1],[0,t3,t2],[t3,0,t1]])
    
    return deltaN.T
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
    # print(min(no_ind))
    return indx

def assemble_Q_H_4(H_zero,Q_zero,NumElemC,elem_vol,nos,c0,rho0):
    H = H_zero
    Q = Q_zero
    for e in tqdm(range(NumElemC)):
        con = elem_vol[e,:]
        coord_el = nos[con,:]
    
        He, Qe = int_tetra_4gauss(coord_el,c0,rho0)   
        
        H[con[:,np.newaxis],con] = H[con[:,np.newaxis],con] + He
        Q[con[:,np.newaxis],con] = Q[con[:,np.newaxis],con] + Qe
    return H,Q

def assemble_Q_H_4_FAST(NumElemC,NumNosC,elem_vol,nos,c0,rho0):

    Hez = np.zeros([4,4,NumElemC])
    Qez = np.zeros([4,4,NumElemC])
    for e in tqdm(range(NumElemC)):
        con = elem_vol[e,:]
        coord_el = nos[con,:]
        
        He, Qe = int_tetra_4gauss(coord_el,c0,rho0)    
        Hez[:,:,e] = He
        Qez[:,:,e] = Qe
    
    NLB=np.size(Hez,1)
    Y=np.matlib.repmat(elem_vol[0:NumElemC,:],1,NLB).T.reshape(NLB,NLB,NumElemC)
    X = np.transpose(Y, (1, 0, 2))
    H= coo_matrix((Hez.ravel(),(X.ravel(),Y.ravel())), shape=[NumNosC, NumNosC]);
    Q= coo_matrix((Qez.ravel(),(X.ravel(),Y.ravel())), shape=[NumNosC, NumNosC]);
    H = H.tocsc()
    Q = Q.tocsc()
    return H,Q

def assemble_Q_H_4_FAST_2order(NumElemC,NumNosC,elem_vol,nos,c0,rho0):

    Hez = np.zeros([10,10,NumElemC])
    Qez = np.zeros([10,10,NumElemC])
    for e in tqdm(range(NumElemC)):
        con = elem_vol[e,:]
        coord_el = nos[con,:]
        
        He, Qe = int_tetra10_4gauss(coord_el,c0,rho0)    
        Hez[:,:,e] = He
        Qez[:,:,e] = Qe
    
    NLB=np.size(Hez,1)
    Y=np.matlib.repmat(elem_vol[0:NumElemC,:],1,NLB).T.reshape(NLB,NLB,NumElemC)
    X = np.transpose(Y, (1, 0, 2))
    H= coo_matrix((Hez.ravel(),(X.ravel(),Y.ravel())), shape=[NumNosC, NumNosC]);
    Q= coo_matrix((Qez.ravel(),(X.ravel(),Y.ravel())), shape=[NumNosC, NumNosC]);
    H = H.tocsc()
    Q = Q.tocsc()
    return H,Q

def assemble_A_3_FAST(domain_index_surf,number_ID_faces,NumElemC,NumNosC,elem_surf,nos,c0,rho0):
    
    Aa = []
    for bl in number_ID_faces:
        indx = np.argwhere(domain_index_surf==bl)
        A = np.zeros([NumNosC,NumNosC])
        for es in range(len(elem_surf[indx])):
            con = elem_surf[indx[es],:][0]
            coord_el = nos[con,:]
            Ae = int_tri_impedance_simpl(coord_el,3)
            A[con[:,np.newaxis],con] = A[con[:,np.newaxis],con] + Ae
        Aa.append(csc_matrix(A))
  
       
    return Aa

def assemble_A10_3_FAST(domain_index_surf,number_ID_faces,NumElemC,NumNosC,elem_surf,nos,c0,rho0):
    
    Aa = []
    for bl in number_ID_faces:
        indx = np.argwhere(domain_index_surf==bl)
        A = np.zeros([NumNosC,NumNosC])
        for es in range(len(elem_surf[indx])):
            con = elem_surf[indx[es],:][0]
            coord_el = nos[con,:]
            Ae = int_tri10_3gauss(coord_el)
            A[con[:,np.newaxis],con] = A[con[:,np.newaxis],con] + Ae
        Aa.append(csc_matrix(A))
  
       
    return Aa
@jit
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

# @jit
# def int_tetra_4gauss(coord_el,c0,rho0):

#     He = np.zeros([4,4])
#     Qe = np.zeros([4,4])
    
# # if npg == 1:
#     #Pontos de Gauss para um tetraedro
#     a = 0.5854101966249685#(5-np.sqrt(5))/20 
#     b = 0.1381966011250105 #(5-3*np.sqrt(5))/20 #
#     ptx = np.array([a,b,b,a])
#     pty = np.array([b,a,b,b])
#     ptz = np.array([b,b,a,b])
    
#     weigths = np.array([1/24,1/24,1/24,1/24])*6
    
#     ## argHe1 is independent of qsi's, therefore it can be pre computed
#     GNi = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]])
#     Ja = (GNi@coord_el)
#     detJa = (1/6) * np.linalg.det(Ja)
#     B = (np.linalg.inv(Ja)@GNi)
#     argHe1 = (1/rho0)*(np.transpose(B)@B)*detJa
#     for indx in range(4):
#         qsi1 = ptx[indx]
#         wtx =  weigths[indx]
#         for indy in range(4):
#             qsi2 = pty[indy]
#             wty =  weigths[indx]
#             for indz in range(4):
#                 qsi3 = ptz[indz]
#                 wtz =  weigths[indx]
                
#                 Ni = np.array([[1-qsi1-qsi2-qsi3],[qsi1],[qsi2],[qsi3]])

#                 argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
                
#                 He = He + wtx*wty*wtz*argHe1   
#                 Qe = Qe + wtx*wty*wtz*argQe1 
    
#     return He,Qe
@jit
def int_tetra_4gauss(coord_el,c0,rho0):

    He = np.zeros([4,4])
    Qe = np.zeros([4,4])
    
# if npg == 1:
    #Pontos de Gauss para um tetraedro
    a = 0.5854101966249685#(5-np.sqrt(5))/20 
    b = 0.1381966011250105 #(5-3*np.sqrt(5))/20 #
    ptx = np.array([a,b,b,b])
    pty = np.array([b,a,b,b])
    ptz = np.array([b,b,a,b])
    
    ## argHe1 is independent of qsi's, therefore it can be pre computed
    GNi = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]])
    Ja = (GNi@coord_el)
    detJa =  np.linalg.det(Ja)
    B = (np.linalg.inv(Ja)@GNi)
    argHe1 = (1/rho0)*(np.transpose(B)@B)*detJa
    weigths = 1/24#**(1/3)

    qsi = np.zeros([3,1]).ravel()
    for indx in range(4):
        qsi[0] = ptx[indx]
        qsi[1]= pty[indx]
        qsi[2] = ptz[indx]
                
        Ni = np.array([[1-qsi[0]-qsi[1]-qsi[2]],[qsi[0]],[qsi[1]],[qsi[2]]])

        argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
        
        He = He + weigths*argHe1   
        Qe = Qe + weigths*argQe1 
    
    return He,Qe

@jit
def int_tetra10_4gauss(coord_el,c0,rho0):
    
    He = np.zeros([10,10])
    Qe = np.zeros([10,10])
    
# if npg == 1:
    #Pontos de Gauss para um tetraedro
    a = 0.5854101966249685
    b = 0.1381966011250105
    ptx = [a,b,b,b]
    pty = [b,a,b,b]
    ptz = [b,b,a,b]
    
    weigths = 1/24#**(1/3)

    qsi = np.zeros([3,1]).ravel()
    for indx in range(4):
        qsi[0] = ptx[indx]
        qsi[1]= pty[indx]
        qsi[2] = ptz[indx]
        Ni = Tetrahedron10N(qsi)
        GNi = Tetrahedron10deltaN(qsi)
        Ja = (GNi@coord_el)
        detJa = ((np.linalg.det(Ja)))
        B = (np.linalg.inv(Ja)@GNi)
        argHe1 = (1/rho0)*(np.transpose(B)@B)*detJa
        argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
        He = He + weigths*argHe1   
        Qe = Qe + weigths*argQe1
    
    return He,Qe
@jit
def int_tri_impedance_1gauss(coord_el):


    Ae = np.zeros([3,3])
    xe = np.array(coord_el[:,0])
    ye = np.array(coord_el[:,1])
    ze = np.array(coord_el[:,2])
    #Formula de Heron - Area do Triangulo
    
    a = np.sqrt((xe[0]-xe[1])**2+(ye[0]-ye[1])**2+(ze[0]-ze[1])**2)
    b = np.sqrt((xe[1]-xe[2])**2+(ye[1]-ye[2])**2+(ze[1]-ze[2])**2)
    c = np.sqrt((xe[2]-xe[0])**2+(ye[2]-ye[0])**2+(ze[2]-ze[0])**2)
    p = (a+b+c)/2
    area_elm = np.abs(np.sqrt(p*(p-a)*(p-b)*(p-c)))
    
    # if npg == 1:
    # #Pontos de Gauss para um tetraedro
    qsi1 = 1/3
    qsi2 = 1/3
    wtz= 1#/6 * 6 # Pesos de Gauss

      
    Ni = np.array([[qsi1],[qsi2],[1-qsi1-qsi2]])
    
        
    detJa= area_elm
    argAe1 = Ni@np.transpose(Ni)*detJa
    
    Ae = Ae + wtz*wtz*argAe1
    
    return Ae
@jit
def int_tri_impedance_simpl(coord_el,npg):


    Ae = np.zeros([3,3])
    xe = np.array(coord_el[:,0])
    ye = np.array(coord_el[:,1])
    ze = np.array(coord_el[:,2])
    #Formula de Heron - Area do Triangulo
    
    a = np.sqrt((xe[0]-xe[1])**2+(ye[0]-ye[1])**2+(ze[0]-ze[1])**2)
    b = np.sqrt((xe[1]-xe[2])**2+(ye[1]-ye[2])**2+(ze[1]-ze[2])**2)
    c = np.sqrt((xe[2]-xe[0])**2+(ye[2]-ye[0])**2+(ze[2]-ze[0])**2)
    p = (a+b+c)/2
    area_elm = np.abs(np.sqrt(p*(p-a)*(p-b)*(p-c)))
    # if npg == 3:
    #Pontos de Gauss para um tetraedro
    aa = 1/6
    bb = 2/3
    ptx = np.array([aa,aa,bb])
    pty = np.array([aa,bb,aa])
    wtz= np.array([1/6,1/6,1/6])*2 # Pesos de Gauss

    for indx in range(3):
        qsi1 = ptx[indx]
        wtx =  wtz[indx]
        for indx in range(3):
            qsi2 = pty[indx]
            wty =  wtz[indx]
            
            Ni = np.array([[qsi1],[qsi2],[1-qsi1-qsi2]])
            
                
            detJa= area_elm
            argAe1 = Ni@np.transpose(Ni)*detJa
            
            Ae = Ae + wtx*wty*argAe1
    
    return Ae

@jit
def int_tri10_3gauss(coord_el):


    Ae = np.zeros([6,6])
    xe = np.array(coord_el[:,0])
    ye = np.array(coord_el[:,1])
    ze = np.array(coord_el[:,2])
    #Formula de Heron - Area do Triangulo
    
    a = np.sqrt((xe[0]-xe[1])**2+(ye[0]-ye[1])**2+(ze[0]-ze[1])**2)
    b = np.sqrt((xe[1]-xe[2])**2+(ye[1]-ye[2])**2+(ze[1]-ze[2])**2)
    c = np.sqrt((xe[2]-xe[0])**2+(ye[2]-ye[0])**2+(ze[2]-ze[0])**2)
    p = (a+b+c)/2
    area_elm = np.abs(np.sqrt(p*(p-a)*(p-b)*(p-c)))
    # if npg == 3:
    #Pontos de Gauss para um triangulo
    aa = 1/6
    bb = 2/3
    ptx = np.array([aa,aa,bb])
    pty = np.array([aa,bb,aa])
    # wtz= np.array([1/6,1/6,1/6])#*2 # Pesos de Gauss
    weight = 1/6*2
    qsi = np.zeros([2,1]).ravel()
    for indx in range(3):
        qsi[0] = ptx[indx]
        # wtx =  wtz[indx]
    # for indx in range(3):
        qsi[1] = pty[indx]
        # wty =  wtz[indx]
        
        Ni = Triangle10N(qsi)
        
            
        detJa= area_elm
        argAe1 = Ni@np.transpose(Ni)*detJa
        
        Ae = Ae + weight*argAe1
    
    return Ae
    # def damped_eigen(self,Q,H,A,mu)
def solve_damped_system(Q,H,A,number_ID_faces,mu,w,q,N):
    Ag = np.zeros_like(Q,dtype=np.complex128)
    i = 0
    for bl in number_ID_faces:
        Ag += A[:,:,i]*mu[bl][N]#/(self.rho0*self.c0)
        i+=1
    G = H + 1j*w[N]*Ag - (w[N]**2)*Q
    b = -1j*w[N]*q
    ps = spsolve(G,b)
    return ps

class FEM3D:
    def __init__(self,Grid,S,R,AP,AC,BC=None):
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
        if BC != None:
            self.mu = BC.mu
        #AirProperties
        self.freq = AC.freq
        self.w = AC.w
        self.AC = AC
        self.AP = AP
        ##AlgControls
        self.c0 = AP.c0
        self.rho0 = AP.rho0
        
        self.S = S
        self.R = R
        #%Mesh
        if Grid != None:
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
            self.order = Grid.order
            
        self.npg = 4
        self.pR = None
        self.pN = None
        self.F_n = None
        self.Vc = None
        
    def compute(self,timeit=True):
        then = time.time()
        if isinstance(self.c0, complex) or isinstance(self.rho0, complex):
            self.H = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)
            self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)

        else:
            self.H = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)
            self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)
        # self.A = np.zeros([self.NumNosC,self.NumNosC,len(self.number_ID_faces)],dtype =  np.cfloat)
        self.q = np.zeros([self.NumNosC,1],dtype = np.cfloat)
        
        if self.order == 1:
            self.H,self.Q = assemble_Q_H_4_FAST(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
        elif self.order == 2:
            self.H,self.Q = assemble_Q_H_4_FAST_2order(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
        #Assemble A(Amortecimento)
        if self.BC != None:
        #     i = 0
        #     for bl in self.number_ID_faces:
        #         indx = np.argwhere(self.domain_index_surf==bl)
        #         for es in range(len(self.elem_surf[indx])):
        #             con = self.elem_surf[indx[es],:][0]
        #             coord_el = self.nos[con,:]
        #             Ae = int_tri_impedance_simpl(coord_el,npg)
        #             self.A[con[:,np.newaxis],con,i] = self.A[con[:,np.newaxis],con,i] + Ae
        #         i += 1
            if self.order == 1:
                self.A = assemble_A_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
            elif self.order == 2:
                self.A = assemble_A10_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
            pN = []
            
            # print('Solving System')
            for ii in range(len(self.S.coord)):
                self.q[find_no(self.nos,self.S.coord[ii,:])] = self.S.q[ii].ravel()
                
            self.q = csc_matrix(self.q)
            for N in tqdm(range(len(self.freq))):
                # ps = solve_damped_system(self.Q, self.H, self.A, self.number_ID_faces, self.mu, self.w, q, N)
                # Ag = np.zeros_like(self.Q,dtype=np.cfloat)
                i = 0
                Ag = 0
                for bl in self.number_ID_faces:
                    Ag += self.A[i]*self.mu[bl].ravel()[N]#/(self.rho0*self.c0)
                    i+=1
                G = self.H + 1j*self.w[N]*Ag - (self.w[N]**2)*self.Q
                b = -1j*self.w[N]*self.q
                ps = spsolve(G,b)
                pN.append(ps)
                  
        else:
            pN = []
            
            
            for ii in range(len(self.S.coord)):
                self.q[find_no(self.nos,self.S.coord[ii,:])] = self.S.q[ii].ravel()
            i = 0
            for N in tqdm(range(len(self.freq))):
                G = self.H - (self.w[N]**2)*self.Q
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
    
    def eigenfrequency(self,neigs=12,near_freq=None,timeit=True):
        self.neigs = neigs
        self.near = near_freq
        
        # from numpy.linalg import inv
        # from scipy.sparse.linalg import eigsh
        from scipy.sparse.linalg import eigs,inv
        # from numpy.linalg import inv
        
        then = time.time()
        self.H = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)
        self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)

        self.H,self.Q = assemble_Q_H_4_FAST(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
            
        G = inv(self.Q)*(self.H)
        if self.near != None:
            [wc,Vc] = eigs(G,self.neigs,sigma = 2*np.pi*(self.near**2),which='SM')
        else:
            [wc,Vc] = eigs(G,self.neigs,which='SM')
        
        k = np.sort(np.sqrt(wc))
        indk = np.argsort(wc)
        # Vcn = Vc/np.amax(Vc)
        self.Vc = Vc
        
        self.F_n = k/(2*np.pi)
        
        
        self.t = time.time()-then       
        if timeit:
            if self.t <= 60:
                print(f'Time taken: {self.t} s')
            elif 60 < self.t < 3600:
                print(f'Time taken: {self.t/60} min')
            elif self.t >= 3600:
                print(f'Time taken: {self.t/60} min')
                
        return self.F_n
    
    def amort_eigenfrequency(self,neigs=12,near_freq=None,timeit=True):
        self.neigs = neigs
        self.near = near_freq
        
        from numpy.linalg import inv
        # from scipy.sparse.linalg import eigsh
        from scipy.sparse.linalg import eigs
        # from numpy.linalg import inv
        
        then = time.time()
        self.H = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)
        self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)

        
        self.H,self.Q = assemble_Q_H_4(self.H,self.Q,self.NumElemC,self.elem_vol,self.nos,self.c0,self.rho0)
            
        G = inv(self.Q)@(self.H)
        if self.near != None:
            [wc,Vc] = eigs(G,self.neigs,sigma = 2*np.pi*(self.near**2),which='SM')
        else:
            [wc,Vc] = eigs(G,self.neigs,which='SM')
        
        # k = np.sort(np.sqrt(wc))
        # indk = np.argsort(wc)
        # Vcn = Vc/np.amax(Vc)
        self.Vc = Vc
        
        fn = np.sqrt(wc)/(2*np.pi)
        
        self.A = np.zeros([self.NumNosC,self.NumNosC,len(self.number_ID_faces)],dtype = np.complex128)
        npg = 3
        if self.BC != None:
            i = 0
            for bl in self.number_ID_faces:
                indx = np.argwhere(self.domain_index_surf==bl)
                for es in range(len(self.elem_surf[indx])):
                    con = self.elem_surf[indx[es],:][0]
                    coord_el = self.nos[con,:]
                    Ae = int_tri_impedance_simpl(coord_el,npg)
                    self.A[con[:,np.newaxis],con,i] = self.A[con[:,np.newaxis],con,i] + Ae
                i += 1
        
        fcn = np.zeros_like(fn,dtype=np.complex128)
        for icc in tqdm(range(len(fn))):
            Ag = np.zeros_like(self.Q,dtype=np.complex128)
            i = 0
            idxF = find_nearest(self.freq,fn[icc])
            # print(idxF)
            for bl in self.number_ID_faces:
                Ag += self.A[:,:,i]*self.mu[bl][idxF]#/(self.rho0*self.c0)
                # print((self.rho0*self.c0)*self.mu[bl][idxF])
                i+=1
            wn = 2*np.pi*fn[icc]
            HA = self.H + 1j*wn*Ag
            Ga = inv(self.Q)@(HA)
            [wcc,Vcc] = eigs(Ga,neigs,which='SM')
            fnc = np.sqrt(wcc)/(2*np.pi)
            print(fnc)
            indfn = find_nearest(np.real(fnc), fn[icc])
            fcn[icc] = fnc[indfn]
        self.F_n = fcn
        self.t = time.time()-then       
        if timeit:
            if self.t <= 60:
                print(f'Time taken: {self.t} s')
            elif 60 < self.t < 3600:
                print(f'Time taken: {self.t/60} min')
            elif self.t >= 3600:
                print(f'Time taken: {self.t/60} min')
                
        return self.F_n
        
    def modal_superposition(self,R):
        self.R = R
        Mn = np.diag(self.Vc.T@self.Q@self.Vc)
        # print(Mn.shape)
        self.A = np.zeros([self.NumNosC,self.NumNosC,len(self.number_ID_faces)],dtype = np.complex128)
        npg = 3
        if self.BC != None:
            i = 0
            for bl in self.number_ID_faces:
                indx = np.argwhere(self.domain_index_surf==bl)
                for es in range(len(self.elem_surf[indx])):
                    con = self.elem_surf[indx[es],:][0]
                    coord_el = self.nos[con,:]
                    Ae = int_tri_impedance_simpl(coord_el, npg)
                    self.A[con[:,np.newaxis],con,i] = self.A[con[:,np.newaxis],con,i] + Ae
                i += 1
                
            indS = [] 
            indR = []
            qindS = []
            for ii in range(len(self.S.coord)): 
                indS.append(find_no(self.nos,self.S.coord[ii,:]))
                qindS.append(self.S.q[ii].ravel())  
            for ii in range(len(self.R.coord)): 
                indR.append([find_no(self.nos,self.R.coord[ii,:])])
                
            # print(qindS[1])
            pmN = [] # np.zeros_like(self.freq,dtype=np.complex128)
            for N in tqdm(range(len(self.freq))):
                D = np.zeros_like(self.Q,dtype = np.complex128)
                i = 0
                
                for bl in self.number_ID_faces:
                    D += self.A[:,:,i]*self.mu[bl][N]
                    i+=1
                    
                hn = np.diag(self.Vc.T@D@self.Vc)
                # print(hn[0].shape)
                An = 0 + 1j*0
                for ir in range(len(indR)):
                    for e in range(len(self.F_n)):
                        for ii in range(len(indS)):
                            wn = self.F_n[e]*2*np.pi
                            # print(self.Vc[indS[ii],e].T*(1j*self.w[N]*qindS[ii])*self.Vc[indR,e])
                            # print(((wn-self.w[N])*Mn[e]))
                            An += self.Vc[indS[ii],e].T*(1j*self.w[N]*qindS[ii])*self.Vc[indR,e]/((wn-self.w[N])*Mn[e])#+1j*2*hn[e]*wn*self.w[N])
                            # print(An)
                    pmN.append(An[0][0])
                
            self.pm = np.array(pmN)
            
        return self.pm
            
    def modal_evaluate(self,freq,renderer='notebook',d_range = None):
        import plotly.graph_objs as go
        
        fi = find_nearest((np.real(self.F_n)),freq)
        # print(fi)
        unq = np.unique(self.elem_surf)
        uind = np.arange(np.amin(unq),np.amax(unq)+1,1)
        
        vertices = self.nos[uind].T
        # vertices = self.nos[np.unique(self.elem_surf)].T
        elements = self.elem_surf.T
        
        values = np.abs((self.Vc.T[fi,uind]))
        if d_range != None:
            d_range = np.amax(values)-d_range
            
            values[values<d_range] = np.amax(values)-d_range
        
        
        fig =  go.Figure(go.Mesh3d(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            i=elements[0,:],
            j=elements[1,:],
            k=elements[2,:],
            intensity = values,
            colorscale= 'Jet',
            intensitymode='vertex'
            
 
        ))  
        fig.update_layout(title=dict(text = f'Frequency: {(np.real(self.F_n[fi])):.2f} Hz | Mode: {fi}'))
        import plotly.io as pio
        pio.renderers.default = renderer
        fig.show()       
    def evaluate(self,R,plot=False):
        
        self.R = R

        self.pR = np.ones([len(self.freq),len(R.coord)],dtype = np.complex128)
        if plot:
            plt.style.use('seaborn-notebook')
            # plt.figure(figsize=(5*1.62,5))
            for i in range(len(self.R.coord)):
                self.pR[:,i] = self.pN[:,find_no(self.nos,R.coord[i,:])]
                plt.semilogx(self.freq,p2SPL(self.pR[:,i]),label=f'R{i} | {self.R.coord[0]}m')
                
            if len(self.R.coord) > 1:
                plt.semilogx(self.freq,np.mean(p2SPL(self.pR)),label='Average')
            
            plt.grid()
            plt.legend()
            plt.xlabel('Frequency[Hz]')
            plt.ylabel('SPL [dB]')
            # plt.show()
        else:
            for i in range(len(self.R.coord)):
                self.pR[:,i] = self.pN[:,find_no(self.nos,R.coord[i,:])]
        return self.pR
    
        
    def surf_evaluate(self,freq,renderer='notebook',d_range = 45):
        import plotly.graph_objs as go
        
        fi = np.argwhere(self.freq==freq)[0][0]
        unq = np.unique(self.elem_surf)
        uind = np.arange(np.amin(unq),np.amax(unq)+1,1)
        
        vertices = self.nos[uind].T
        # vertices = self.nos[np.unique(self.elem_surf)].T
        elements = self.elem_surf.T
        
        values = np.real(p2SPL(self.pN[fi,uind]))
        if d_range != None:
            d_range = np.amax(values)-d_range
            
            values[values<d_range] = np.amax(values)-d_range
        
        
        print(np.amin(values),np.amax(values))
        print(vertices.shape)
        print(elements.shape)
        print(values.shape)
        fig =  go.Figure(go.Mesh3d(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            i=elements[0,:],
            j=elements[1,:],
            k=elements[2,:],
            intensity = values,
            colorscale= 'Jet',
            intensitymode='vertex',
 
        ))  
        import plotly.io as pio
        pio.renderers.default = renderer
        fig.show()
        
    def plot_problem(self,renderer='notebook'):
        
        import plotly.figure_factory as ff
        import plotly.graph_objs as go
        
        vertices = self.nos.T#[np.unique(self.elem_surf)].T
        elements = self.elem_surf.T
        
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
        )
        fig['data'][0].update(opacity=0.3)
        
        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        try:
            if self.R != None:
                fig.add_trace(go.Scatter3d(x = self.R.coord[:,0], y = self.R.coord[:,1], z = self.R.coord[:,2],name="Receivers",mode='markers'))
        except:
            pass
        
        if self.S != None:    
            if self.S.wavetype == "spherical":
                fig.add_trace(go.Scatter3d(x = self.S.coord[:,0], y = self.S.coord[:,1], z = self.S.coord[:,2],name="Sources",mode='markers'))
        import plotly.io as pio
        pio.renderers.default = renderer
        fig.show()
        
    def fem_save(self, filename=time.strftime("%Y%m%d-%H%M%S"), ext = ".pickle"):
        # Simulation data
        gridpack = {'nos': self.nos,
                'elem_vol': self.elem_vol,
                'elem_surf': self.elem_surf,
                'NumNosC': self.NumNosC,
                'NumElemC': self.NumElemC,
                'domain_index_surf': self.domain_index_surf,
                'domain_index_vol': self.domain_index_vol,
                'number_ID_faces': self.number_ID_faces,
                'number_ID_vol': self.number_ID_vol,
                'order': self.order}

    
        
            
        simulation_data = {'AC': self.AC,
                           "AP": self.AP,
                           'R': self.R,
                           'S': self.S,
                           'BC': self.BC,
                           'A': self.A,
                           'H': self.H,
                           'Q': self.Q,
                           'q':self.q,
                           'grid': gridpack,
                           'pN': self.pN,
                           'pR':self.pR,
                           'F_n': self.F_n,
                           'Vc':self.Vc}
                           # 'incident_traces': incident_traces}

                
        outfile = open(filename + ext, 'wb')
                
        cloudpickle.dump(simulation_data, outfile)
        outfile.close()
        print('FEM saved successfully.')