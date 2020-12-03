# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:33:54 2020

@author: gutoa
"""
import numpy as np
from scipy.sparse.linalg import spsolve
import time 
from tqdm import tqdm
import warnings
from numba import jit
from numba import njit

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def p2SPL(p):
    SPL = 10*np.log10(0.5*p*np.conj(p)/(2e-5)**2)
    return SPL
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
@jit
def Tetrahedron10N(qsi):

    t1 = qsi[0][0]
    t2 = qsi[1][0]
    t3 = qsi[2][0]
    t4 = 1 - qsi[0] - qsi[1] - qsi[2];
    # print(t1)
    N = np.array([t4*(2*t4 - 1),t1*(2*t1 - 1),t2*(2*t2 - 1),t3*(2*t3 - 1),
                  4*t1*t4,4*t1*t2,4*t2*t4,4*t3*t4,4*t2*t3,4*t3*t1]);
    return N
@jit
def Tetrahedron10deltaN(qsi):
    t1 = 4*qsi[0][0]
    t2 = 4*qsi[1][0]
    t3 = 4*qsi[2][0]
    # print(t1)
    deltaN = np.array([[t1 + t2 + t3 - 3,t1 + t2 + t3 - 3,t1 + t2 + t3 - 3],[t1 - 1,0,0],[0,t2 - 1,0],[0,0,t3 - 1],
                       [4 - t2 - t3 - 2*t1,-t1,-t1],[t2,t1,0],[-t2,4 - 2*t2 - t3 - t1,-t2],
                       [-t3,-t3,4 - t2 - 2*t3 - t1],[0,t3,t2],[t3,0,t1]])
    
    return deltaN.T

@jit
def Triangle10N(qsi):
    
    N = np.array([(-qsi[0] - qsi[1] + 1) * (2*(-qsi[0] - qsi[1] + 1) - 1),
    qsi[0]*(2*qsi[0] - 1),
    qsi[1]*(2*qsi[1] - 1),
    4*qsi[0]*qsi[1],
    4*qsi[1]*(-qsi[0] - qsi[1] + 1),
    4*qsi[0]*(-qsi[0] - qsi[1] + 1)])
    
    deltaN = [[(4*qsi[0] + 4*qsi[1] - 3),
    (4*qsi[0] - 1),
    0,
    4*qsi[1],
    -4*qsi[1],
    (4 - 4*qsi[1] - 8*qsi[0])],
    [(4*qsi[0] + 4*qsi[1] - 3),
    0,
    (4*qsi[1] - 1),
    4*qsi[0],
    4 - 8*qsi[1] - 4*qsi[0],
    -4*qsi[0]]
    ];
    
    return N,deltaN
@jit
def int_tetra10_gauss(coord_el,c0,rho0):
    
    He = np.zeros([10,10])
    Qe = np.zeros([10,10])
    
# if npg == 1:
    #Pontos de Gauss para um tetraedro
    a = 0.5854101966249685
    b = 0.1381966011250105
    ptx = np.array([a,b,b,a])
    pty = np.array([b,a,b,b])
    ptz = np.array([b,b,a,b])
    
    weigths = np.array([1/24,1/24,1/24,1/24])*6

    qsi = np.zeros([3,1])
    for indx in range(4):
        qsi[0] = ptx[indx]
        wtx =  weigths[indx]
        for indy in range(4):
            qsi[1]= pty[indy]
            wty =  weigths[indx]
            for indz in range(4):
                qsi[2] = ptz[indz]
                wtz =  weigths[indx]
                
                Ni = Tetrahedron10N(qsi)
                
                GNi = Tetrahedron10deltaN(qsi)
                # print(GNi)
                # print(coord_el)
                Ja = (GNi@coord_el)
                # print(Ja)
                detJa = (1/6) * np.linalg.det(Ja)
                
                # if detJa < 0:
                    # print(detJa)
                    # print(coord_el)
                B = (np.linalg.inv(Ja)@GNi)
                argHe1 = (1/rho0)*(np.transpose(B)@B)*detJa
                argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
                # print(Ni.shape)
                He = He + wtx*wty*wtz*argHe1   
                Qe = Qe + wtx*wty*wtz*argQe1
    
    return He,Qe
def int_tetra_4gauss(coord_el,c0,rho0):

    He = np.zeros([4,4])
    Qe = np.zeros([4,4])
    
# if npg == 1:
    #Pontos de Gauss para um tetraedro
    a = 0.5854101966249685
    b = 0.1381966011250105
    ptx = np.array([a,b,b,a])
    pty = np.array([b,a,b,b])
    ptz = np.array([b,b,a,b])
    
    weigths = np.array([1/24,1/24,1/24,1/24])*6
    
    ## argHe1 is independent of qsi's, therefore it can be pre computed
    GNi = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]])
    Ja = (GNi@coord_el)
    detJa = (1/6) * np.linalg.det(Ja)
    B = (np.linalg.inv(Ja)@GNi)
    argHe1 = (1/rho0)*(np.transpose(B)@B)*detJa
    for indx in range(4):
        qsi1 = ptx[indx]
        wtx =  weigths[indx]
        for indy in range(4):
            qsi2 = pty[indy]
            wty =  weigths[indx]
            for indz in range(4):
                qsi3 = ptz[indz]
                wtz =  weigths[indx]
                
                Ni = np.array([[1-qsi1-qsi2-qsi3],[qsi1],[qsi2],[qsi3]])

                # B = spsolve(Ja,GNi)
                # print(B.shape)              
                
                # print(np.matmul(Ni,np.transpose(Ni)).shape)
                argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
                
                He = He + wtx*wty*wtz*argHe1   
                Qe = Qe + wtx*wty*wtz*argQe1 
    
    return He,Qe

@jit
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
    
    # if npg == 1:
    # #Pontos de Gauss para um tetraedro
    #     ptx = 1/3
    #     pty = 1/3
    #     wtz= 1#/6 * 6 # Pesos de Gauss
    #     qsi1 = ptx
    #     qsi2 = pty
    #     wtx = wtz
    #     wty = wtz
        
        
    # if npg == 3:
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

@jit
def int_tri_impedance_10(coord_el,npg):


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
    
    #Pontos de Gauss para um tetraedro
    aa = 1/6
    bb = 2/3
    ptx = np.array([aa,aa,bb])
    pty = np.array([aa,bb,aa])
    wtz= np.array([1/6,1/6,1/6])*2 # Pesos de Gauss

    qsi = np.zeros([2,1])
    for indx in range(npg):
        qsi[0] = ptx[indx]
        wtx =  wtz[indx]
        for indx in range(npg):
            qsi[1] = pty[indx]
            wty =  wtz[indx]
            
            Ni,GNi = Triangle10N(qsi)
            
                
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
        if BC != None:
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
        self.npg = 4
        
    def compute(self,timeit=True):
        then = time.time()
        self.H = np.zeros([self.NumNosC,self.NumNosC],dtype = np.complex128)
        self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype = np.complex128)
        self.A = np.zeros([self.NumNosC,self.NumNosC,len(self.number_ID_faces)],dtype = np.complex128)
        self.q = np.zeros([self.NumNosC,1],dtype = np.complex128)
        
        #Assemble H(Massa) and Q(Rigidez) matrix
        # print('Assembling Matrix')
        for e in tqdm(range(self.NumElemC)):
            con = self.elem_vol[e,:][0]
            # print(con)
            coord_el = self.nos[con,:]

            He, Qe = int_tetra10_gauss(coord_el,self.c0,self.rho0)   
            
            # print(Qe.shape)
            # print(self.Q[con[:,np.newaxis],con].shape)
            self.H[con[:,np.newaxis],con] = self.H[con[:,np.newaxis],con] + He
            self.Q[con[:,np.newaxis],con] = self.Q[con[:,np.newaxis],con] + Qe
   
        
        #Assemble A(Amortecimento)
        npg = 3
        if self.BC != None:
            i = 0
            for bl in self.number_ID_faces:
                indx = np.argwhere(self.domain_index_surf==bl)
                for es in range(len(self.elem_surf[indx])):
                    con = self.elem_surf[indx[es],:][0]
                    coord_el = self.nos[con,:]
                    
                    Ae = int_tri_impedance_10(coord_el, npg)
                    self.A[con[:,np.newaxis],con,i] = self.A[con[:,np.newaxis],con,i] + Ae
                i += 1
            
            
            pN = []
            
            # print('Solving System')
            for ii in range(len(self.S.coord)):
                self.q[find_no(self.nos,self.S.coord[ii,:])] = self.S.q[ii].ravel()
            for N in tqdm(range(len(self.freq))):
                Ag = np.zeros_like(self.Q)
                i = 0
                for bl in self.number_ID_faces:
                    Ag += self.A[:,:,i]*self.mu[bl][N]/(self.rho0*self.c0)
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
                
    def evaluate(self,R,plot=False):
        self.R = R

        self.pR = np.ones([len(self.freq),len(R.coord)],dtype = np.complex128)
        if plot:
            plt.style.use('seaborn-notebook')
            plt.figure(figsize=(5*1.62,5))
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