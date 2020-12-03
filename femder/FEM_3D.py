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
def int_tetra_4gauss(coord_el,c0,rho0):

    He = np.zeros([4,4])
    Qe = np.zeros([4,4])
    
# if npg == 1:
    #Pontos de Gauss para um tetraedro
    a = 0.5854101966249685#(5-np.sqrt(5))/20 
    b = 0.1381966011250105 #(5-3*np.sqrt(5))/20 #
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
            con = self.elem_vol[e,:]
            coord_el = self.nos[con,:]
            if self.npg == 1:
                He, Qe = int_tetra_simpl(coord_el,self.c0,self.rho0,self.npg)   
            elif self.npg == 4:
                He, Qe = int_tetra_4gauss(coord_el,self.c0,self.rho0)   
            
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
    
    def eigenfrequency(self,neigs=12,near_freq=None,timeit=True):
        self.neigs = neigs
        self.near = near_freq
        
        from numpy.linalg import inv
        # from scipy.sparse.linalg import eigsh
        from scipy.sparse.linalg import eigs
        # from numpy.linalg import inv
        
        then = time.time()
        self.H = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)
        self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)

        #Assemble H(Massa) and Q(Rigidez) matrix
        # print('Assembling Matrix')
        # for e in tqdm(range(self.NumElemC)):
        #     con = self.elem_vol[e,:]
        #     coord_el = self.nos[con,:]
        #     if self.npg == 1:
        #         He, Qe = int_tetra_simpl(coord_el,self.c0,self.rho0,self.npg)   
        #     elif self.npg == 4:
        #         He, Qe = int_tetra_4gauss(coord_el,self.c0,self.rho0)   
            
        #     self.H[con[:,np.newaxis],con] = self.H[con[:,np.newaxis],con] + He
        #     self.Q[con[:,np.newaxis],con] = self.Q[con[:,np.newaxis],con] + Qe
        
        self.H,self.Q = assemble_Q_H_4(self.H,self.Q,self.NumElemC,self.elem_vol,self.nos,self.c0,self.rho0)
            
        G = inv(self.Q)@(self.H)
        if self.near != None:
            [wc,Vc] = eigs(G,self.neigs,sigma = 2*np.pi*(self.near**2),which='SM')
        else:
            [wc,Vc] = eigs(G,self.neigs,which='SM')
        
        k = np.sort(np.sqrt(wc))
        # indk = np.argsort(wc)
        
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