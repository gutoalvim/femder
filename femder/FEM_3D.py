# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:33:54 2020

@author: Luiz Augusto Alvim & Paulo Mareze
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from pyMKL import pardisoSolver

from matplotlib import ticker, gridspec, style, rcParams
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
import seaborn
# from pypardiso import spsolve
# from scipy.sparse.linalg import gmres
import time 
from tqdm import tqdm
import warnings
from numba import jit
import cloudpickle
from numba import njit
import numba
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from femder.utils import detect_peaks
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import plotly.io as pio

from contextlib import contextmanager
import sys, os
os.environ['KMP_WARNINGS'] = '0'
import femder as fd

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
 
def asymetric_std_dev(curve):
    total_sum = 0
    for i in range(len(curve)):
        if curve[i] < np.mean(curve):
            total_sum += abs(np.sum((curve[i] - np.mean(curve)) ** 2))
        else:
            total_sum += np.sum((curve[i] - np.mean(curve)) ** 1)
    std_dev = np.sqrt(1 / (len(curve) - 1) * total_sum)

    return std_dev

def first_cuboid_mode(Lx,Ly,Lz,c0):
    idx_max = np.argmax([Lx,Ly,Lz])
    odr = np.zeros((3,))
    odr[idx_max] = 1
    fn = (c0/2)*np.sqrt((odr[0]/Lx)**2+(odr[1]/Ly)**2+(odr[2]/Lz)**2)
    return fn
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
            
def fem_load(filename,ext='.pickle'):
    """
    Load FEM3D simulation

    Parameters
    ----------
    filename : str
        File name saved with fem_save.
    ext : TYPE, optional
        File extension. The default is '.pickle'.

    Returns
    -------
    obj : TYPE
        DESCRIPTION.

    """
    
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

def SBIR_SPL(complex_pressure,rC, AC,fmin,fmax):
    sbirspl = []
    fs = 44100
    
    # fmin_indx = np.argwhere(AC.freq==fmin)[0][0]
    # fmax_indx = np.argwhere(AC.freq==fmax)[0][0]

    
    df = (AC.freq[1]-AC.freq[0])
    ir_duration = 1/df
    for i in range(len(rC)):

        
        ir = fd.IR(fs,ir_duration,fmin,fmax).compute_room_impulse_response(complex_pressure[:,i].ravel())
        t_ir = np.linspace(0,ir_duration,len(ir))
        sbir = fd.SBIR(ir,t_ir,fmin,fmax,winCheck=False,method='constant')
        sbir_spl= sbir[1][:,1]
        sbir_SPL = np.real(p2SPL(sbir_spl))
        sbirspl.append(sbir_SPL[find_nearest(sbir[0], fmin):find_nearest(sbir[0], fmax)])
        sbir_freq = sbir[0][find_nearest(sbir[0], fmin):find_nearest(sbir[0], fmax)]
    # print(sbirspl[0].shape)
    plt.semilogx(sbir_freq,sbirspl[0])
    return sbir_freq,sbirspl
    
def IR(complex_pressure,AC,fmin,fmax):
    fs = 44100

    df = (AC.freq[-1]-AC.freq[0])/len(AC.freq)
    
    ir_duration = 1/df
    
    ir = fd.IR(fs,ir_duration,fmin,fmax).compute_room_impulse_response(complex_pressure.ravel())
    t_ir = np.linspace(0,ir_duration,len(ir))
    
    import pytta
    
    r = pytta.SignalObj(ir)
    return r
@jit
def coord_interpolation(nos,elem_vol,coord,pN):
    coord = np.array(coord)
    pelem,pind = prob_elem(nos, elem_vol, coord)
    indx = which_tetra(nos,pelem,coord)
    indx = pind[indx]
    con = elem_vol[indx,:][0]
    coord_el = nos[con,:]
    GNi = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]])
    Ja = (GNi@coord_el).T

    icoord = coord - coord_el[0,:]
    qsi = (np.linalg.inv(Ja)@icoord)
    Ni = np.array([[1-qsi[0]-qsi[1]-qsi[2]],[qsi[0]],[qsi[1]],[qsi[2]]])

    Nip = Ni.T@pN[:,con].T
    return Nip.T

@jit
def prob_elem(nos,elem,coord):
    cl1 = closest_node(nos, coord)
    eln = np.where(elem==cl1)
    pelem = elem[eln[0]]
    return pelem,eln[0]
@jit
def which_tetra(node_coordinates, node_ids, p):
    ori=node_coordinates[node_ids[:,0],:]
    v1=node_coordinates[node_ids[:,1],:]-ori
    v2=node_coordinates[node_ids[:,2],:]-ori
    v3=node_coordinates[node_ids[:,3],:]-ori
    n_tet=len(node_ids)
    v1r=v1.T.reshape((3,1,n_tet))
    v2r=v2.T.reshape((3,1,n_tet))
    v3r=v3.T.reshape((3,1,n_tet))
    mat = np.concatenate((v1r,v2r,v3r), axis=1)
    inv_mat = np.linalg.inv(mat.T).T    # https://stackoverflow.com/a/41851137/12056867        
    if p.size==3:
        p=p.reshape((1,3))
    n_p=p.shape[0]
    orir=np.repeat(ori[:,:,np.newaxis], n_p, axis=2)
    newp=np.einsum('imk,kmj->kij',inv_mat,p.T-orir)
    val=np.all(newp>=0, axis=1) & np.all(newp <=1, axis=1) & (np.sum(newp, axis=1)<=1)
    id_tet, id_p = np.nonzero(val)
    res = -np.ones(n_p, dtype=id_tet.dtype) # Sentinel value
    res[id_p]=id_tet
    return res

def closest_node(nodes, node):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest2(array, value):
    """
    Function to find closest frequency in frequency array. Returns closest value and position index.
    """
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
def p2SPL(p):
    SPL = 10*np.log10(0.5*p*np.conj(p)/(2e-5)**2)
    return np.real(SPL)

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
@jit
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

def mu2alpha(mu,c0,rho0):
    mu[mu==0] = 0.0001
    z0 = rho0*c0
    z = 1/(mu)
    R = (z-z0)/(z+z0)
    alpha = 1-np.abs(R)**2
    alpha[alpha<0] = 0
    return alpha.flatten()
    
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


@jit
def assemble_Q_H_4_FAST_equifluid(NumElemC,NumNosC,elem_vol,nos,c,rho,domain_index_vol,fi):

    Hez = np.zeros([4,4,NumElemC],dtype='cfloat')
    Qez = np.zeros([4,4,NumElemC],dtype='cfloat')
    for e in range(NumElemC):
        con = elem_vol[e,:]
        coord_el = nos[con,:]
        
        He, Qe = int_tetra_4gauss(coord_el,c[domain_index_vol[e]][fi],rho[domain_index_vol[e]][fi])    
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


def assemble_Q_H_5_FAST(NumElemC,NumNosC,elem_vol,nos,c0,rho0):

    Hez = np.zeros([4,4,NumElemC])
    Qez = np.zeros([4,4,NumElemC])
    for e in tqdm(range(NumElemC)):
        con = elem_vol[e,:]
        coord_el = nos[con,:]
        
        He, Qe = int_tetra_5gauss(coord_el,c0,rho0)    
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

def assemble_Q_H_4_FAST_2order_equifluid(NumElemC,NumNosC,elem_vol,nos,c,rho,domain_index_vol,fi):

    Hez = np.zeros([10,10,NumElemC])
    Qez = np.zeros([10,10,NumElemC])
    for e in range(NumElemC):
        con = elem_vol[e,:]
        coord_el = nos[con,:]
        
        He, Qe = int_tetra10_4gauss(coord_el,c[domain_index_vol[e]][fi],rho[domain_index_vol[e]][fi])    
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

def integration_tri3_3gauss(shape_function):
    """
    Gauss integration for 3 points and tri3
    Parameters
    ----------
    area_elem: float
        Area of current element
    Returns
    -------

    """
    damp_elem = np.zeros((3, 3), dtype=np.float64)

    gauss_weight = 1 / 9
    # damp_elem = (np.einsum("i,j,k, j,i,k -> i,j", shape_function, shape_function)*area_elem)*gauss_weight
    for index_x in range(3):
        damp_elem += (np.dot(shape_function[index_x, :, :],
                             shape_function[index_x, :, :].transpose(1, 0)) * gauss_weight)

    return damp_elem

def gauss_3_points():
    """
    Initializes the 3 points for gauss integration in tri3

    Returns
    -------
    ptx: array
        X coordinate for gauss integration
    pty: array
        Y coordinate for gauss integration

    """
    return np.array([1 / 6, 1 / 6, 2 / 3]), np.array([1 / 6, 2 / 3, 1 / 6])

def assemble_surface_matrices(elem_surf, vertices, areas, domain_indices_surface, unique_domain_indices, order):
    """
    Assemble dampening FEM matrix for surface elements.

    Parameters
    ----------
    unique_domain_indices: array
    elem_surf: array

    vertices: array
    areas: array
    domain_indices_surface: array
    order: int

    Returns
    -------

    """
    gauss_integration = None
    ptx, pty = gauss_3_points()
    if order == 1:
        gauss_integration = integration_tri3_3gauss
    if order == 2:
        return None
    damp_global = []
    shape_function = np.array([np.broadcast_to(ptx[:, None], (3, 3)).T,
                               np.broadcast_to(pty[:, None], (3, 3)),
                               1 - ptx - np.broadcast_to(pty[:, None], (3, 3))]).transpose(2, 0, 1)
    _damp_elem = gauss_integration(shape_function)
    len_vertices = len(vertices)
    for bl in tqdm(unique_domain_indices, desc="FEM | Assembling surface matrix", bar_format='{l_bar}{bar:25}{r_bar}'):
        surface_index = np.argwhere(domain_indices_surface == bl).ravel()
        damp = np.zeros((3, 3, len(elem_surf[surface_index])), dtype="float64")
        for es in range(len(elem_surf[surface_index])):
            con = elem_surf[surface_index[es], :]
            area_elem = areas[surface_index[es]]
            damp_elem = _damp_elem * area_elem
            damp[:, :, es] = damp_elem

        _nlb = np.size(damp, 1)
        len_elem_vol = len(elem_surf[surface_index])
        assemble_y = np.matlib.repmat(elem_surf[surface_index], 1, _nlb).T.reshape(_nlb, _nlb, len_elem_vol)
        assemble_x = np.transpose(assemble_y, (1, 0, 2))
        csc_damp = coo_matrix((damp.ravel(),
                               (assemble_x.ravel(), assemble_y.ravel())),
                              shape=[len_vertices, len_vertices])
        damp_global.append(csc_damp.tocsc())

    return damp_global

def assemble_A_3_FAST(domain_index_surf,number_ID_faces,NumElemC,NumNosC,elem_surf,nos,c0,rho0):
    
    Aa = []
    for bl in number_ID_faces:
        indx = np.argwhere(domain_index_surf==bl)
        A = np.zeros([NumNosC,NumNosC])
        for es in range(len(elem_surf[indx])):
            con = elem_surf[indx[es],:][0]
            coord_el = nos[con,:]
            Ae = int_tri_impedance_3gauss(coord_el)
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
def assemble_Q_H_4_ULTRAFAST(NumElemC,NumNosC,elem_vol,nos,c0,rho0):

    Hez = np.zeros([4,4,NumElemC])
    Qez = np.zeros([4,4,NumElemC])
    coord_el_e = coord_el_pre(NumElemC,elem_vol,nos)
    Ja_,GNi = Ja_pre(coord_el_e)
    argHe_,detJa_= detJa_pre(Ja_, GNi, rho0)
    ptx,pty,ptz = gauss_4_points()
    for e in tqdm(range(NumElemC)):
        argHe = argHe_[:,:,e]
        detJa = detJa_[e]
        He, Qe = nint_tetra_4gauss(np.complex64(c0),numba.complex64(rho0),
                                   numba.complex64(argHe),detJa,ptx,pty,ptz)    
        Hez[:,:,e] = He
        Qez[:,:,e] = Qe
    
    NLB=np.size(Hez,1)
    Y=np.matlib.repmat(elem_vol[0:NumElemC,:],1,NLB).T.reshape(NLB,NLB,NumElemC)
    X = np.transpose(Y, (1, 0, 2))

    H= coo_matrix((Hez.ravel(),(X.ravel(), Y.ravel())), shape=[NumNosC, NumNosC])
    Q= coo_matrix((Qez.ravel(),(X.ravel(), Y.ravel())), shape=[NumNosC, NumNosC])
    
    H = H.tocsc()
    Q = Q.tocsc()
    
    return H,Q

def assemble_Q_H_4_ULTRAFAST_2order(NumElemC,NumNosC,elem_vol,nos,c0,rho0):

    Hez = np.zeros([10,10,NumElemC])
    Qez = np.zeros([10,10,NumElemC])
    coord_el_e = coord_el_pre(NumElemC,elem_vol,nos)
    Ja_,GNi = Ja_pre(coord_el_e)
    argHe_,detJa_= detJa_pre(Ja_, GNi, rho0)
    ptx,pty,ptz = gauss_4_points()
    for e in tqdm(range(NumElemC)):
        argHe = argHe_[:,:,e]
        detJa = detJa_[e]
        He, Qe = nint_tetra10_4gauss(np.complex64(c0),numba.complex64(rho0),
                                   numba.complex64(argHe),detJa,ptx,pty,ptz)    
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


def compute_volume(NumElemC,NumNosC,elem_vol,nos,c0,rho0):


    coord_el_e = coord_el_pre(NumElemC,elem_vol,nos)
    Ja_,GNi = Ja_pre(coord_el_e)
    argHe_,detJa_= detJa_pre(Ja_, GNi, rho0)
    V = []
    for e in (range(NumElemC)):
        detJa = detJa_[e]
        V.append(detJa/6) 
    return sum(V)

def coord_el_pre(NumElemC,elem_vol,nos):
    coord_el_= []
    for e in range(NumElemC):
        con = elem_vol[e,:]
        coord_el_.append(np.complex64(nos[con,:]))
    return coord_el_

def Ja_pre(coord_el):
    
    GNi = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]],dtype=np.complex64)
    Ja = np.zeros((3,3,len(coord_el)))
    for i in range(len(coord_el)): 
        Ja[:,:,i] = np.dot(GNi,coord_el[i])
    return Ja,GNi

def detJa_pre(Ja,GNi,rho0):
    detJa =  np.zeros((len(Ja[0,0,:]),))
    B = np.zeros((3,4,len(Ja[0,0,:])))
    argHe1 = np.zeros((4,4,len(Ja[0,0,:])))
    for i in range(len(Ja[0,0,:])):
        detJa[i] =  np.linalg.det(Ja[:,:,i])
        B[:,:,i] = (np.linalg.inv(Ja[:,:,i])@GNi)
        argHe1[:,:,i] = (1/rho0)*(np.transpose(B[:,:,i] )@B[:,:,i])*detJa[i]
    return argHe1,detJa

def compute_areas(vertices, faces):
    """Calculate the area of all elements in the mesh.

    Calls area_normal_ph to compute the area.

    Parameters
    ----------
    vertices : numpy ndArray
        The vertices in the mesh.
    faces : numpy ndArray
        The connectivity matrix of the mesh.

    Returns
    -------
    areas : numpy 1dArray
        the area of each triangle in the mesh.
    """
    areas = np.zeros((len(faces), 1))
    for i in range(len(faces)):
        rS = vertices[faces[i, :], :]
        areas[i] = area_normal_ph(rS)
    return numba.complex128(areas.ravel())

@njit
def area_normal_ph(re):
    """Calculate the area of one triangle.

    Parameters
    ----------
    re : numpy ndArray
        The vertices in the triangle.


    Returns
    -------
    area_elm : float
        the area of a triangle.
    """
    xe = (re[:, 0])
    ye = (re[:, 1])
    ze = (re[:, 2])
    # Formula de Heron - Area do Triangulo

    a = np.sqrt((xe[0] - xe[1]) ** 2 + (ye[0] - ye[1]) ** 2 + (ze[0] - ze[1]) ** 2)
    b = np.sqrt((xe[1] - xe[2]) ** 2 + (ye[1] - ye[2]) ** 2 + (ze[1] - ze[2]) ** 2)
    c = np.sqrt((xe[2] - xe[0]) ** 2 + (ye[2] - ye[0]) ** 2 + (ze[2] - ze[0]) ** 2)
    p = (a + b + c) / 2
    area_elm = np.abs(np.sqrt(p * (p - a) * (p - b) * (p - c)))

    return area_elm

def int_tetra_4gauss(coord_el,c0,rho0):

    He = np.zeros([4,4],dtype='cfloat')
    Qe = np.zeros([4,4],dtype='cfloat')
    
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
    weigths = 1/24

    qsi = np.zeros([3,1]).ravel()
    for indx in range(4):
        qsi[0] = ptx[indx]
        qsi[1]= pty[indx]
        qsi[2] = ptz[indx]
                
        Ni = np.array([[1-qsi[0]-qsi[1]-qsi[2]],[qsi[0]],[qsi[1]],[qsi[2]]],dtype=np.complex64)

        argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
        
        He = He + weigths*argHe1   
        Qe = Qe + weigths*argQe1 
    
    return He,Qe
def gauss_4_points():
        #Pontos de Gauss para um tetraedro
    a = 0.5854101966249685#(5-np.sqrt(5))/20 
    b = 0.1381966011250105 #(5-3*np.sqrt(5))/20 #
    ptx = np.array([a,b,b,b])
    pty = np.array([b,a,b,b])
    ptz = np.array([b,b,a,b])
    return ptx,pty,ptz
@njit
def nint_tetra_4gauss(c0,rho0,argHe,detJa,ptx,pty,ptz):
    He = np.zeros((4,4),dtype=np.complex64)
    Qe = np.zeros((4,4),dtype=np.complex64)
    
    for indx in range(4):
        Ni = np.array([[1-ptx[indx]-pty[indx]-ptz[indx]],[ptx[indx]],[pty[indx]],[ptz[indx]]],dtype=np.complex64)  
        argQe1 = (1/(rho0*c0**2))*np.dot(Ni,Ni.transpose())*detJa
        He += 1/24*argHe  
        Qe += 1/24*argQe1 

    return He,Qe

@njit
def nint_tetra10_4gauss(c0,rho0,argHe,detJa,ptx,pty,ptz):
    He = np.zeros((10,10),dtype=np.complex64)
    Qe = np.zeros((10,10),dtype=np.complex64)
    
    for indx in range(4):
        Ni = np.array([[1-ptx[indx]-pty[indx]-ptz[indx]],[ptx[indx]],[pty[indx]],[ptz[indx]]],dtype=np.complex64)  
        argQe1 = (1/(rho0*c0**2))*np.dot(Ni,Ni.transpose())*detJa
        He += 1/24*argHe  
        Qe += 1/24*argQe1 

    return He,Qe




@jit
def int_tetra_5gauss(coord_el,c0,rho0):

    He = np.zeros([4,4])
    Qe = np.zeros([4,4])
    
# if npg == 1:
    #Pontos de Gauss para um tetraedro
    a = 1/4
    b = 1/6
    c = 1/2
    ptx = np.array([a,b,b,b,c])
    pty = np.array([a,b,b,c,b])
    ptz = np.array([a,b,c,b,b])
    
    ## argHe1 is independent of qsi's, therefore it can be pre computed
    GNi = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]])
    Ja = (GNi@coord_el)
    detJa = 1/6* np.linalg.det(Ja)
    B = (np.linalg.inv(Ja)@GNi)
    argHe1 = (1/rho0)*(np.transpose(B)@B)*detJa
    weigths = np.array([-2/15,3/40,3/40,3/40,3/40])*6

    qsi = np.zeros([3,1]).ravel()
    for indx in range(5):
        qsi[0] = ptx[indx]
        wtx =  weigths[indx]
        for indy in range(5):
            qsi[1] = pty[indy]
            wty =  weigths[indx]
            for indz in range(5):
                qsi[2] = ptz[indz]
                wtz =  weigths[indx]

                
                Ni = np.array([[1-qsi[0]-qsi[1]-qsi[2]],[qsi[0]],[qsi[1]],[qsi[2]]])
        
                argQe1 = (1/(rho0*c0**2))*(Ni@np.transpose(Ni))*detJa
                
                He = He + wtx*wty*wtz*argHe1   
                Qe = Qe + wtx*wty*wtz*argQe1 
    
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
# @jit
def int_tri_impedance_3gauss(coord_el):


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

def compute_tri_area(domain_index_surf,number_ID_faces,NumElemC,NumNosC,elem_surf,nos,c0,rho0):
    Aa = {}
    for bl in number_ID_faces:
        indx = np.argwhere(domain_index_surf==bl)
        Ab = []
        for es in range(len(elem_surf[indx])):
            con = elem_surf[indx[es],:][0]
            coord_el = nos[con,:]
            xe = np.array(coord_el[:,0])
            ye = np.array(coord_el[:,1])
            ze = np.array(coord_el[:,2])
            #Formula de Heron - Area do Triangulo            
            a = np.sqrt((xe[0]-xe[1])**2+(ye[0]-ye[1])**2+(ze[0]-ze[1])**2)
            b = np.sqrt((xe[1]-xe[2])**2+(ye[1]-ye[2])**2+(ze[1]-ze[2])**2)
            c = np.sqrt((xe[2]-xe[0])**2+(ye[2]-ye[0])**2+(ze[2]-ze[0])**2)
            p = (a+b+c)/2
            area_elm = np.abs(np.sqrt(p*(p-a)*(p-b)*(p-c)))
            Ab.append(area_elm)
        Aa[bl] = sum(Ab)

    return Aa
@jit
def int_tri_impedance_4gauss(coord_el):


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
    aa = 1/3
    bb = 1/5
    cc = 3/5
    ptx = np.array([aa,bb,bb,cc])
    pty = np.array([aa,bb,cc,aa])
    wtz= np.array([-27/96,25/96,25/96,25/96])##*2 # Pesos de Gauss

    for indx in range(4):
        qsi1 = ptx[indx]
        wtx =  wtz[indx]
        for indx in range(4):
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

@njit
def solve_modal_superposition(indR,indS,F_n,Vc,Vc_T,w,qindS,N,hn,Mn,ir):
    lenS = numba.int64(len(indS))
    lenfn = numba.int64(len(F_n))
    
    An = np.zeros((1,1),dtype=np.complex64)
    # pmN = np.zeros((len(indR),),dtype=np.complex64)
    # An = 0+0*1j
    for ii in range(lenS):
        for e in range(lenfn):
        
            wn = F_n[e]*2*np.pi
            # print(self.Vc[indS[ii],e].T*(1j*self.w[N]*qindS[ii])*self.Vc[indR,e])
            # print(((wn-self.w[N])*Mn[e]))
            An[0] += Vc_T[e,indS[ii]]*(1j*w[N]*qindS[ii])*Vc[indR[ir],e]/((wn**2-w[N]**2)*Mn[e]+1j*hn[e]*w[N])
            # An[0] += Vc_T[e,2]*(1j*w[N]*qindS[ii])*Vc[2,e]/((wn**2-w[N]**2)*Mn[e]+1j*hn[e]*w[N])

    return An[0]
    


def std_obj(pR,rC):
    std_dev = []
    if len(rC) > 1:
        for i in range(len(rC)):
            std_r = np.std(pR[:,i])
            std_dev.append(std_r)
    else:
        std_dev =  np.std(pR)
    return [std_dev]

class FEM3D:
    def __init__(self,Grid,S,R,AP,AC,BC=None):
        """
        Initializes FEM3D Class

        Parameters
        ----------
        Grid : GridImport()
            GridImport object created with femder.GridImport('YOURGEO.geo',fmax,maxSize,scale).
        S: Source
            Source object containing source coordinates
        R: Receiver
            Receiver object containing receiver coordinates
        AP : AirProperties
            AirPropeties object containing, well, air properties.
        AC : AlgControls
            Defines frequency configuration for calculation.
        BC : BoundaryConditions()
            BoundaryConditions object containg impedances for each assigned Physical Group in gmsh.

        Returns
        -------
        None.

        """
        self.BC= BC
        if BC != None:
            self.mu = BC.mu
            self.v = BC.v
        
        
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
            self.path_to_geo = Grid.path_to_geo
            # if Grid.path_to_geo_unrolled != None:
            self.path_to_geo_unrolled = Grid.path_to_geo_unrolled
        self.npg = 4
        self.pR = None
        self.pN = None
        self.F_n = None
        self.Vc = None
        self.H = None
        self.Q = None
        self.rho = {}
        self.c = {}
        
        if len(self.BC.rhoc) > 0:
            rhoc_keys=np.array([*self.BC.rhoc])[0]
            rho0_keys = self.number_ID_vol
            rho_list = np.setdiff1d(rho0_keys,rhoc_keys)
            for i in rho_list:
                self.rho[i] = np.ones_like(self.freq)*self.rho0
                
            self.rho.update(self.BC.rhoc)
            
        if len(self.BC.cc) > 0:
            cc_keys=np.array([*self.BC.cc])[0]
            c0_keys = self.number_ID_vol
            cc_list = np.setdiff1d(c0_keys,cc_keys)
            for i in cc_list:
                self.c[i] = np.ones_like(self.freq)*self.c0
                
            self.c.update(self.BC.cc)

    @property
    def spl(self):
        return fd.p2SPL(self.pR)

    @property
    def avg_spl(self):
        return np.mean(self.spl, axis=1)

    @property
    def spl_S(self):
        return fd.p2SPL(self.pR.T/(1j*self.w*self.rho0)).T

    @property
    def avg_spl_S(self):
        return np.mean(self.spl_S, axis=1)

    @property
    def internal_volume(self):
        return compute_volume(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)

    def compute(self,timeit=True,printless=True):
        """
        Computes acoustic pressure for every node in the mesh.

        Parameters
        ----------
        timeit : TYPE, optional
            Prints solve time. The default is True.

        Returns
        -------
        None.

        """
        
        then = time.time()
        # if isinstance(self.c0, complex) or isinstance(self.rho0, complex):
        #     self.H = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)
        #     self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)

        # else:
        #     self.H = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)
        #     self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype =  np.cfloat)
        # self.A = np.zeros([self.NumNosC,self.NumNosC,len(self.number_ID_faces)],dtype =  np.cfloat)
        self.q = np.zeros([self.NumNosC,1],dtype = np.cfloat)
        self.areas = compute_areas(self.nos, self.elem_surf)
        if len(self.rho) == 0:
            if self.H is None:
                if self.order == 1:
                    # self.H,self.Q = assemble_Q_H_4_ULTRAFAST(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
                    fluid_c = {1: np.ones((len(self.elem_vol),)) * self.c0}
                    fluid_rho = {1: np.ones((len(self.elem_vol),)) * self.rho0}
                    det_ja, arg_stiff = fd.fem_numerical.pre_compute_volume_assemble_vars(self.elem_vol,self.nos, 1)
                    self.H,self.Q = fd.fem_numerical.assemble_volume_matrices(self.elem_vol,self.nos, fluid_c, fluid_rho, 1, self.domain_index_vol, 0, det_ja, arg_stiff)
                elif self.order == 2:
                    self.H,self.Q = assemble_Q_H_4_FAST_2order(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
                #Assemble A(Amortecimento)
            if self.BC != None:
                
                if self.order == 1:
                    # self.A = assemble_A_3_FAST(self.domain_index_surf,np.sort([*self.mu]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                    self.A = assemble_surface_matrices(self.elem_surf, self.nos, self.areas, self.domain_index_surf, np.sort([*self.mu]), 1)
                        # assemble_A_3_FAST(self.domain_index_surf,np.sort([*self.mu]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                    if len(self.v) > 0:
                        # self.V = assemble_A_3_FAST(self.domain_index_surf,np.sort([*self.v]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                        self.V = assemble_surface_matrices(self.elem_surf, self.nos, self.areas, self.domain_index_surf, np.sort([*self.v]), 1)

                elif self.order == 2:
                    self.A = assemble_A10_3_FAST(self.domain_index_surf,np.sort([*self.mu]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                    if len(self.v) > 0:
                        self.V = assemble_A10_3_FAST(self.domain_index_surf,np.sort([*self.v]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)

                pN = []

                
                # print('Solving System')
                if self.S is not None:
                    for ii in range(len(self.S.coord)):
                        self.q[closest_node(self.nos,self.S.coord[ii,:])] = self.S.q[ii].ravel()
                    
                self.q = csc_matrix(self.q)
                if len(self.v) == 0:
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
                        # ps = spsolve(G,b)
                        pSolve = pardisoSolver(G, mtype=13)
                        pSolve.run_pardiso(12)
                        ps = pSolve.run_pardiso(33, b.todense())
                        pSolve.clear()
                        pN.append(ps)
                if len(self.v) > 0:
                    
                    for N in tqdm(range(len(self.freq))):
                    # ps = solve_damped_system(self.Q, self.H, self.A, self.number_ID_faces, self.mu, self.w, q, N)
                    # Ag = np.zeros_like(self.Q,dtype=np.cfloat)
                        i = 0
                        Ag = 0
                        Vn = 0
                        V = np.zeros([self.NumNosC,1],dtype = np.cfloat)
                        
                        for bl in np.sort([*self.mu]):
                            Ag += self.A[i]*self.mu[bl].ravel()[N]#/(self.rho0*self.c0)
                            i += 1
                            
                        i=0
                        for bl in np.sort([*self.v]):
                            indx = np.argwhere(self.domain_index_surf==bl)
                            indx = np.unique(self.elem_surf[indx, :].ravel())
                            V[indx] = self.v[bl][N]
                            Vn += self.V[i]*csc_matrix(V)
                            i+=1
                        G = self.H + 1j*self.w[N]*Ag - (self.w[N]**2)*self.Q
                        b = -1j*self.w[N]*Vn
                        pSolve = pardisoSolver(G, mtype=13)
                        pSolve.run_pardiso(12)
                        ps = pSolve.run_pardiso(33, b.todense())
                        pSolve.clear()
                        pN.append(ps)
                # self.Vn = Vn
            else:
                pN = []
                
                
                for ii in range(len(self.S.coord)):
                    self.q[closest_node(self.nos,self.S.coord[ii,:])] = self.S.q[ii].ravel()
                    
                self.q = csc_matrix(self.q)
                i = 0
                
                for N in tqdm(range(len(self.freq))):
                    G = self.H - (self.w[N]**2)*self.Q
                    b = -1j*self.w[N]*self.q
                    pSolve = pardisoSolver(G, mtype=13)
                    pSolve.run_pardiso(12)
                    ps = pSolve.run_pardiso(33, b.todense())
                    pSolve.clear()
                    pN.append(ps)
        else:
            if self.BC != None:
    
                if self.order == 1:
                    self.A = assemble_A_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                elif self.order == 2:
                    self.A = assemble_A10_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                pN = []
                
                # print('Solving System')
                for ii in range(len(self.S.coord)):
                    self.q[closest_node(self.nos,self.S.coord[ii,:])] = self.S.q[ii].ravel()
                    
                self.q = csc_matrix(self.q)
                for N in tqdm(range(len(self.freq))):
                    # ps = solve_damped_system(self.Q, self.H, self.A, self.number_ID_faces, self.mu, self.w, q, N)
                    # Ag = np.zeros_like(self.Q,dtype=np.cfloat)
                    i = 0
                    Ag = 0
                    for bl in self.number_ID_faces:
                        Ag += self.A[i]*self.mu[bl].ravel()[N]#/(self.rho0*self.c0)
                        i+=1
                    if self.order == 1:
                        self.H,self.Q = assemble_Q_H_4_FAST_equifluid(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c,self.rho,self.domain_index_vol,N)
                    elif self.order == 2:
                        self.H,self.Q = assemble_Q_H_4_FAST_2order_equifluid(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c,self.rho,self.domain_index_vol,N)
                    
                    G = self.H + 1j*self.w[N]*Ag - (self.w[N]**2)*self.Q
                    b = -1j*self.w[N]*self.q
                    pSolve = pardisoSolver(G, mtype=13)
                    pSolve.run_pardiso(12)
                    ps = pSolve.run_pardiso(33, b.todense())
                    pSolve.clear()
                    pN.append(ps)
            else:
                pN = []
                
                
                for ii in range(len(self.S.coord)):
                    self.q[closest_node(self.nos,self.S.coord[ii,:])] = self.S.q[ii].ravel()
                self.q = csc_matrix(self.q)
                i = 0
                for N in tqdm(range(len(self.freq))):
                    if self.order == 1:
                        self.H,self.Q = assemble_Q_H_4_FAST_equifluid(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c,self.rho,self.domain_index_vol,N)
                    elif self.order == 2:
                        self.H,self.Q = assemble_Q_H_4_FAST_2order_equifluid(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c,self.rho,self.domain_index_vol,N)
                    
                    G = self.H - (self.w[N]**2)*self.Q
                    b = -1j*self.w[N]*self.q
                    pSolve = pardisoSolver(G, mtype=13)
                    pSolve.run_pardiso(12)
                    ps = pSolve.run_pardiso(33, b.todense())
                    pSolve.clear()
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

    # def compute(self, timeit=True, printless=True):
    #     """
    #     Computes acoustic pressure for every node in the mesh.
    #
    #     Parameters
    #     ----------
    #     timeit : TYPE, optional
    #         Prints solve time. The default is True.
    #
    #     Returns
    #     -------
    #     None.
    #
    #     """
    #
    #     node_pressure = []
    #     damp_global_admittance = []
    #     unique_velocity_index = fd.utils.bigger_than_n_unique_dict(self.v, 1)
    #     self._pre_compute_vars = fd.numerical.pre_compute_volume_assemble_vars(self.elem_vol,
    #                                                                         self.nos,
    #                                                                         self.order)
    #     # if self.eq_fluid_bool is True:
    #
    #     for frequency_index in tqdm(range(len(self.freq)), desc="FEM | " + self.active_src_str,
    #                                 bar_format='{l_bar}{bar:25}{r_bar}'):
    #         if self.eq_fluid_bool is True or frequency_index == 0:
    #             self._global_stiff, self._global_mass = fd.numerical.assemble_volume_matrices(
    #                 self.elem_vol,
    #                 self.nos,
    #                 self.c,
    #                 self.rho,
    #                 self.order,
    #                 self.domain_index_vol, frequency_index, self._pre_compute_vars[0], self._pre_compute_vars[1])
    #         i = 0
    #
    #         if self.admittance_bool is True or frequency_index == 0:
    #             _damp_global_admittance = []
    #             for_damp_obj = np.sort([*self._admittances])
    #             for bl in for_damp_obj:
    #                 _damp_global_admittance.append(self._global_damp[i] * self._admittances[bl][frequency_index] \
    #                                                / (self._eq_fluid_c[1][0] * self._eq_fluid_rho[1][0]))
    #                 i += 1
    #             damp_global_admittance = sum(_damp_global_admittance)
    #
    #         if self.boundary_velocity_bool:
    #             velocity_init = np.zeros([len(self.nos), 1], dtype=np.complex64)
    #             boundary_velocity_vector = 0
    #             i = 0
    #             for bl in unique_velocity_index:
    #                 velocity_index = np.argwhere(self.domain_index_surf == bl)
    #                 con = self.elem_vol[velocity_index].ravel()
    #                 velocity_init[con] = self.v[bl][frequency_index]
    #                 boundary_velocity_vector += self._global_vel[i] * velocity_init
    #                 i += 1
    #             rhs_source_arg = self._source_vector + csc_matrix(boundary_velocity_vector)
    #         else:
    #             rhs_source_arg = self._source_vector
    #
    #         lhs = self._global_stiff + 1j * self.room.w0[frequency_index] * damp_global_admittance - (
    #                 self.room.w0[frequency_index] ** 2) * self._global_mass
    #         rhs = -1j * self.room.w0[frequency_index] * rhs_source_arg
    #         fem_solver = pardisoSolver(lhs, mtype=13)
    #         fem_solver.run_pardiso(12)
    #         solution_data = fem_solver.run_pardiso(33, rhs.todense())
    #         fem_solver.clear()
    #         node_pressure.append(solution_data)
    #
    #     self._node_pressure = np.array(node_pressure, dtype="complex64")
    #     self.clear()
    #
    #     if timeit:
    #         if self.t <= 60:
    #             print(f'Time taken: {self.t} s')
    #         elif 60 < self.t < 3600:
    #             print(f'Time taken: {self.t / 60} min')
    #         elif self.t >= 3600:
    #             print(f'Time taken: {self.t / 60} min')

    def optimize_source_receiver_pos(self,geometry_points, geometry_height, geometry_angle, num_freq, num_grid_pts,star_average=True,fmin=20,fmax=200,max_distance_from_wall=0.5,method='direct',
                                     minimum_distance_between_speakers=1.2,speaker_receiver_height=1.2,min_distance_from_backwall=0.6,
                                     max_distance_from_backwall=1.5,neigs=50,
                                     plot_geom=False,renderer=None,plot_evaluate=False, plotBest=False,
                                     print_info=True,saveFig=False,camera_angles=['floorplan', 'section', 'diagonal'],timeit=True):
        """
        Implements a search for the best source-receiver configuration in a control room symmetric in the y axis.

        Parameters
        ----------
        num_grid_pts : TYPE
            DESCRIPTION.
        star_average : TYPE, optional
            DESCRIPTION. The default is True.
        fmin : TYPE, optional
            DESCRIPTION. The default is 20.
        fmax : TYPE, optional
            DESCRIPTION. The default is 200.
        max_distance_from_wall : TYPE, optional
            DESCRIPTION. The default is 0.5.
        method : TYPE, optional
            DESCRIPTION. The default is 'direct'.
        minimum_distance_between_speakers : TYPE, optional
            DESCRIPTION. The default is 1.2.
        speaker_receiver_height : TYPE, optional
            DESCRIPTION. The default is 1.2.
        min_distance_from_backwall : TYPE, optional
            DESCRIPTION. The default is 0.6.
        max_distance_from_backwall : TYPE, optional
            DESCRIPTION. The default is 1.5.
        neigs : TYPE, optional
            DESCRIPTION. The default is 50.
        plot_geom : TYPE, optional
            DESCRIPTION. The default is False.
        renderer : TYPE, optional
            DESCRIPTION. The default is 'notebook'.
        plot_evaluate : TYPE, optional
            DESCRIPTION. The default is False.
        plotBest : TYPE, optional
            DESCRIPTION. The default is False.
        print_info : TYPE, optional
            DESCRIPTION. The default is True.
        saveFig : TYPE, optional
            DESCRIPTION. The default is False.
        camera_angles : TYPE, optional
            DESCRIPTION. The default is ['floorplan', 'section', 'diagonal'].
        timeit : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        print('Initializing optimization')
        then = time.time()
        _Grid = fd.GeometryGenerator(self.AP, np.amax(self.freq), num_freq)
        _Grid.generate_symmetric_polygon_variheight(
            geometry_points, geometry_height, geometry_angle)
        sC,rC = fd.r_s_from_grid(_Grid,num_grid_pts,star_average=star_average,
                                 max_distance_from_wall=max_distance_from_wall,
                                 minimum_distance_between_speakers=minimum_distance_between_speakers,
                                 speaker_receiver_height = speaker_receiver_height,
                                 min_distance_from_backwall=min_distance_from_backwall,
                                 max_distance_from_backwall=max_distance_from_backwall)
        
        R_all = []
        S_all = []
        for i in range(len(rC)):
            R_all.append(rC[i].coord)
            S_all.append(sC[i].coord)
        
        S_all = np.vstack((np.array(S_all)[:,0,0,:],np.array(S_all)[:,1,0,:]))
        R_all = np.array(R_all)[:,0,:]
        
        self.H = None
        self.Q = None
        self.R = fd.Receiver()
        self.R.coord = R_all 
        self.S = fd.Source()
        self.S.coord = S_all
        self.Scand = self.S
        self.Rcand = self.R
        
        if plot_geom:
            self.plot_problem(renderer=renderer,saveFig=saveFig,camera_angles=camera_angles)  
        fom = []
        

        # Grid = fd.GeometryGenerator(self.AP,np.amax(self.freq),num_freq).generate_symmetric_polygon_variheight(geometry_points,geometry_height,geometry_angle)
        # Grid = fd.GridImport3D(self.AP, self.path_to_geo_unrolled, S=self.S, R=self.R, fmax = self.grid.fmax, num_freq = self.grid.num_freq, order=self.order, plot=True)
        Grid = fd.GeometryGenerator(self.AP, np.amax(self.freq), num_freq)
        Grid.generate_symmetric_polygon_variheight(
            geometry_points, geometry_height, geometry_angle)
        self.nos = Grid.nos
        self.elem_surf = Grid.elem_surf
        self.elem_vol = Grid.elem_vol
        self.domain_index_surf =  Grid.domain_index_surf
        self.domain_index_vol = Grid.domain_index_vol
        self.number_ID_faces = Grid.number_ID_faces
        self.number_ID_vol = Grid.number_ID_vol
        self.NumNosC = Grid.NumNosC
        self.NumElemC = Grid.NumElemC
        self.order = Grid.order

        
        self.pOptim = []
        pOptim = []
        fom = []
        fig = plt.figure(figsize=(12,8))
        Grid.plot_mesh()
        if method != 'None':
            if method == 'modal':
                self.eigenfrequency(neigs,timeit=False)
                if self.BC != None:
                    if self.order == 1:
                        self.A = assemble_A_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                    elif self.order == 2:
                        self.A = assemble_A10_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                    
            for i in range(len(rC)):
                
                self.R = rC[i]
                self.S = sC[i]
                if method == 'direct':
                    self.compute(timeit=False)
                    self.pR = self.evaluate(self.R,False)
                elif method == 'modal':
                    self.pR = self.modal_superposition(self.R)
                pR_mean = np.real(p2SPL(self.pR))
                pOptim.append(self.pR)
                fm = self.fitness_metric(w1=0.8, w2=0.2,fmin=fmin,fmax=fmax, dip_penalty=True, center_penalty=True, mode_penalty=True,
                       ref_curve='mean', dB_oct=2, nOct=2, lf_boost=10, infoLoc=(0.12, -0.03),
                       returnValues=True, plot=False, figsize=(17, 9), std_dev='symmetric')
                
                fom.append(np.real(-fm))
                
                if plot_evaluate:
                    plt.semilogx(self.freq,pR_mean,label=f'{fm:.2f}')
                    plt.legend()
                    plt.xlabel('Frequency [Hz]')
                    plt.ylabel('SPL [dB]')
                    plt.grid()
                    # plt.show()
                
            
            # plt.savefig('optim_cand.png',dpi=300)
            min_indx = np.argmin(np.array(fom))
            self.min_indx = min_indx

            if plotBest:
                fig = plt.figure(figsize=(12,8))
         
                plt.semilogx(self.freq,np.real(p2SPL(pOptim[min_indx])),label='Total')
                # sbir_freq, sbir_spl = SBIR_SPL(self.pR,rC, self.AC, fmin,  fmax) 
                # plt.semilogx(sbir_freq,sbir_spl,label='SBIR')
                
                plt.legend()
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('SPL [dB]')
                plt.title(f'Fitness: {fom[min_indx]:.3f}')
                plt.grid()
                # plt.savefig('best_modal_Sbir.png',dpi=300, transparent=True)
                plt.show()
                
            if print_info:
                print(f'Fitness Metric: {fom[min_indx]:.2f} \n Source Position: {sC[min_indx].coord:.2f} \n Receiver Position: {rC[min_indx].coord:.2f}')
            
            self.R = rC[min_indx]
            self.S.coord = sC[min_indx].coord[:,0,:]
            self.pOptim = [fom,np.array(pOptim)]
            self.bestMetric = np.amin(fom)
            self.t = time.time()-then

            if timeit:
                if self.t <= 60:
                    print(f'Time taken: {self.t} s')
                elif 60 < self.t < 3600:
                    print(f'Time taken: {self.t/60} min')
                elif self.t >= 3600:
                    print(f'Time taken: {self.t/60} min')
                    
        
        return self.pOptim
            
    
    def eigenfrequency(self,neigs=12,near_freq=None,timeit=True):
        """
        Solves eigenvalue problem 

        Parameters
        ----------
        neigs : TYPE, optional
            Number of eigenvalues to solve. The default is 12.
        near_freq : TYPE, optional
            Search for eigenvalues close to this frequency. The default is None.
        timeit : TYPE, optional
            Print solve time. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        self.neigs = neigs
        self.near = near_freq
        
        # from numpy.linalg import inv
        # from scipy.sparse.linalg import eigsh
        from scipy.sparse.linalg import eigs
        # from numpy.linalg import inv
        
        then = time.time()
        self.H = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)
        self.Q = np.zeros([self.NumNosC,self.NumNosC],dtype = np.float64)
        self.A = None
        if self.order == 1:
            self.H,self.Q = assemble_Q_H_4_ULTRAFAST(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
        elif self.order == 2:
            self.H,self.Q = assemble_Q_H_4_FAST_2order(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
         
        print('Solving System ...')
        # G = inv(self.Q)*(self.H)
        # pSolve = pardisoSolver(self.Q, mtype=13)
        # pSolve.run_pardiso(12)
        # G = pSolve.run_pardiso(33, self.H.todense())
        # pSolve.clear()
        G = spsolve(self.Q,self.H)
        # G = gmres(self.Q,self.H)

        print('Finding Eigenvalues and Eigenvectors ...')
        if self.near != None:
            [wc,Vc] = eigs(G,self.neigs,sigma = 2*np.pi*(self.near**2),which='SM')
        else:
            [wc,Vc] = eigs(G,self.neigs,which='SM')
        
        k = np.sort(np.sqrt(wc))
        # indk = np.argsort(wc)
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
        if self.BC != None:
            if self.order == 1:
                self.A = assemble_A_3_FAST(self.domain_index_surf,np.sort([*self.mu]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                if len(self.v) > 0:
                    self.V = assemble_A_3_FAST(self.domain_index_surf,np.sort([*self.v]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
            
            elif self.order == 2:
                self.A = assemble_A10_3_FAST(self.domain_index_surf,np.sort([*self.mu]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                if len(self.v) > 0:
                    self.V = assemble_A10_3_FAST(self.domain_index_surf,np.sort([*self.v]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)

        
        fcn = np.zeros_like(fn,dtype=np.complex128)
        for icc in tqdm(range(len(fn))):
            Ag = np.zeros_like(self.Q,dtype=np.complex128)
            i = 0
            idxF = find_nearest(self.freq,fn[icc])
            # print(idxF)
            i = 0
            Ag = 0
            for bl in self.number_ID_faces:
                Ag += self.A[i]*self.mu[bl].ravel()[idxF]#/(self.rho0*self.c0)
                i+=1
            wn = 2*np.pi*fn[icc]
            HA = self.H + 1j*wn*Ag
            Ga = spsolve(self.Q,HA)
            [wcc,Vcc] = eigs(Ga,neigs,which='SM')
            fnc = np.sqrt(wcc)/(2*np.pi)
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
        
    def modal_superposition(self,R,plot=False):
        """
        Implements modal superposition method to calculate FRF for source-recevier configuration.

        Parameters
        ----------
        R : TYPE
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.R = R
        Mn = np.diag(self.Vc.T@self.Q@self.Vc)
        
        
        if self.BC != None:
            
            if self.A == None:
                print(self.A)
                if self.order == 1:
                    self.A = assemble_A_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                elif self.order == 2:
                    self.A = assemble_A10_3_FAST(self.domain_index_surf,self.number_ID_faces,self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
                
                
            indS = [] 
            indR = []
            qindS = []
            for ii in range(len(self.S.coord)): 
                indS.append(closest_node(self.nos,self.S.coord[ii,:]))
                qindS.append(self.S.q[ii].ravel())  
            for ii in range(len(self.R.coord)): 
                indR.append([closest_node(self.nos,self.R.coord[ii,:])])
                
            # print(qindS[1])
            pmN = [] # np.zeros_like(self.freq,dtype=np.complex128)
            for N in range(len(self.freq)):
                i = 0
                Ag = 0
                for bl in self.number_ID_faces:
                    Ag += self.A[i]*self.mu[bl].ravel()[N]#/(self.rho0*self.c0)
                    i+=1
                
                    
                hn = np.diag(self.Vc.T@Ag@self.Vc)
                # print(hn)
                # An = 0 + 1j*0
                # for ir in range(len(indR)):
                #     for ii in range(len(indS)):
                #         for e in range(len(self.F_n)):
                        
                #             wn = self.F_n[e]*2*np.pi
                #             # print(self.Vc[indS[ii],e].T*(1j*self.w[N]*qindS[ii])*self.Vc[indR,e])
                #             # print(((wn-self.w[N])*Mn[e]))
                #             An += self.Vc[indS[ii],e].T*(1j*self.w[N]*qindS[ii])*self.Vc[indR[ir],e]/((wn**2-self.w[N]**2)*Mn[e]+1j*hn[e]*self.w[N])
                            
                #     pmN.append(An[0])
                for ir in range(len(indR)):
                    i = int(ir)
                    pf = solve_modal_superposition(numba.int16(indR),numba.int16(indS),
                                                   numba.complex64(self.F_n),numba.complex64(self.Vc),
                                                   numba.complex64(self.Vc.T),numba.complex64(self.w),
                                                   numba.complex64(qindS),(N),
                                                   numba.complex64(hn),numba.complex64(Mn),ir)
                    pmN.append(pf)
            self.pm = np.array(pmN).reshape((len(self.freq),-1))
            if plot:
                plt.style.use('seaborn-notebook')
                plt.figure(figsize=(5*1.62,5))
                if len(self.pm[0,:]) > 1:
                    linest = ':'
                else:
                    linest = '-'
                for i in range(len(self.R.coord)):
                    # self.pR[:,i] = coord_interpolation(self.nos, self.elem_vol, R.coord[i,:], self.pN)
                    plt.semilogx(self.freq,p2SPL(self.pm[:,i]),linestyle = linest,label=f'R{i} | {self.R.coord[i,:]}m')
                    
                if len(self.R.coord) > 1:
                    plt.semilogx(self.freq,np.mean(p2SPL(self.pm),axis=1),label='Average',linewidth = 5)
                
                plt.grid()
                plt.legend()
                plt.xlabel('Frequency[Hz]')
                plt.ylabel('SPL [dB]')
                # plt.show()
            
        return self.pm
            
    def modal_evaluate(self,freq,renderer=None,d_range = None,saveFig=False,filename=None,
                     camera_angles=['floorplan', 'section', 'diagonal'],transparent_bg=True,title=None,extension='png'):
        """
        Plot modal pressure on the boundaries

        Parameters
        ----------
        freq : TYPE
            DESCRIPTION.
        renderer : TYPE, optional
            DESCRIPTION. The default is 'notebook'.
        d_range : TYPE, optional
            DESCRIPTION. The default is None.
        saveFig : TYPE, optional
            DESCRIPTION. The default is False.
        filename : TYPE, optional
            DESCRIPTION. The default is None.
        camera_angles : TYPE, optional
            DESCRIPTION. The default is ['floorplan', 'section', 'diagonal'].
        transparent_bg : TYPE, optional
            DESCRIPTION. The default is True.
        title : TYPE, optional
            DESCRIPTION. The default is None.
        extension : TYPE, optional
            DESCRIPTION. The default is 'png'.

        Returns
        -------
        None.

        """
        import plotly.graph_objs as go
        
        fi = find_nearest((np.real(self.F_n)),freq)
        # print(fi)
        unq = np.unique(self.elem_surf)
        uind = np.arange(np.amin(unq),np.amax(unq)+1,1,dtype = int)
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
        if renderer is not None:
            pio.renderers.default = renderer
        if title is False:
            fig.update_layout(title="")
        if transparent_bg:
            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                               'paper_bgcolor': 'rgba(0, 0, 0, 0)', }, )
        if saveFig:
            # folderCheck = os.path.exists('/Layout')
            # if folderCheck is False:
            #     os.mkdir('/Layout')
            if filename is None:
                for camera in camera_angles:
                    if camera == 'top' or camera == 'floorplan':
                        camera_dict = dict(eye=dict(x=0., y=0., z=2.5),
                                           up=dict(x=0, y=1, z=0),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'lateral' or camera == 'side' or camera == 'section':
                        camera_dict = dict(eye=dict(x=2.5, y=0., z=0.0),
                                           up=dict(x=0, y=0, z=1),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'front':
                        camera_dict = dict(eye=dict(x=0., y=2.5, z=0.),
                                           up=dict(x=0, y=1, z=0),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'rear' or camera == 'back':
                        camera_dict = dict(eye=dict(x=0., y=-2.5, z=0.),
                                           up=dict(x=0, y=1, z=0),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'diagonal_front':
                        camera_dict = dict(eye=dict(x=1.50, y=1.50, z=1.50),
                                           up=dict(x=0, y=0, z=1),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'diagonal_rear':
                        camera_dict = dict(eye=dict(x=-1.50, y=-1.50, z=1.50),
                                           up=dict(x=0, y=0, z=1),
                                           center=dict(x=0, y=0, z=0), )
                    fig.update_layout(scene_camera=camera_dict)
    
                    fig.write_image(f'_3D_{camera}_{time.strftime("%Y%m%d-%H%M%S")}.{extension}', scale=2)
            else:
                fig.write_image(filename+'.'+extension, scale=2)
        fig.show()
        fig.show()       
    def evaluate(self,R,interpolation_tolerance = 0.001, plot=False):
        """
        Evaluates pressure at a given receiver coordinate, for best results, include receiver
        coordinates as nodes in mesh, by passing Receiver() in GridImport3D().

        Parameters
        ----------
        R : Receiver()
            Receiver object with receiver coodinates.
        plot : Bool, optional
            Plots SPL for given nodes, if len(R)>1, also plots average. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # pressure_receiver = []
        # for i in range(len(self.receiver_positions)):
        #     closest_node_to_receiver = numerical.closest_node(self.vertices, evaluation_positions[i, :])
        #     if np.linalg.norm(
        #             evaluation_positions[i, :] - self.vertices[closest_node_to_receiver]) < interpolation_tolerance:
        #         pressure_receiver.append(self.node_pressure[:, closest_node_to_receiver])
        #     else:
        #         pressure_receiver.append(numerical.node_pressure_interpolation(self.vertices, self.volume_elements,
        #                                                                        evaluation_positions[i, :],
        #                                                                        self.node_pressure))
        #
        # self.pressure_receiver = np.asarray(pressure_receiver).T
        
        self.R = R

        self.pR = [] #np.ones([len(self.freq),len(R.coord)],dtype = np.complex128)

        for i in range(len(self.R.coord)):
            closest_node_to_receiver = closest_node(self.nos, self.R.coord[i, :])
            pt_dist = np.linalg.norm(
                    self.R.coord[i, :] - self.nos[closest_node_to_receiver])
            print(pt_dist)
            if pt_dist < interpolation_tolerance:
                self.pR.append(self.pN[:, closest_node(self.nos, R.coord[i, :])])
            else:
                self.pR.append(coord_interpolation(self.nos, self.elem_vol, self.R.coord[i, :], self.pN))
        self.pR = np.asarray(self.pR).squeeze().T

        if len(self.pR.shape) == 1:
            self.pR = self.pR.reshape(self.pN.shape[0],1)
        return self.pR
    
    def evaluate_physical_group(self,domain_index,average=True,plot=False):
        """
        Evaluates pressure at a given receiver coordinate, for best results, include receiver
        coordinates as nodes in mesh, by passing Receiver() in GridImport3D().

        Parameters
        ----------
        domain_index : List / Int()
            physical groups to be evaluated
        plot : Bool, optional
            Plots SPL for given nodes, if len(R)>1, also plots average. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        
        self.pR = np.zeros([len(self.freq),len(domain_index)],dtype = np.complex128)
        if plot:
            plt.style.use('seaborn-notebook')
            plt.figure(figsize=(5*1.62,5))
                # linest = ':'
            i = 0
            for bl in domain_index:
                indx = np.array(np.argwhere(self.domain_index_surf==bl))
                # print(indx)
                self.pR[:,i] = np.mean(p2SPL(self.pN[:,indx][:,:,0]),axis=1)

                
                plt.semilogx(self.freq,self.pR[:,i],label=f'Average - Physical Group: {i}',linewidth = 5)
                i+=1
            plt.grid()
            plt.legend()
            plt.xlabel('Frequency[Hz]')
            plt.ylabel('SPL [dB]')
            # plt.show()
        return self.pR
    
        
    def surf_evaluate(self,freq,renderer=None,d_range = 45):
        """
        Evaluates pressure in the boundary of the mesh for a given frequency, and plots with plotly.
        Choose adequate rederer, if using Spyder or similar, use renderer='browser'.

        Parameters
        ----------
        freq : float
            Frequency to evaluate.
        renderer : str, optional
            Plotly render engine. The default is 'notebook'.
        d_range : float, optional
            Dynamic range of plot. The default is 45dB.

        Returns
        -------
        None.

        """
        
        import plotly.graph_objs as go
        
        fi = np.argwhere(self.freq==freq)[0][0]
        unq = np.unique(self.elem_surf)
        uind = np.arange(np.amin(unq),np.amax(unq)+1,1,dtype=int)
        
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

        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        if renderer is not None:
            pio.renderers.default = renderer
        fig.show()
        
    def plot_problem(self,renderer=None,saveFig=False,filename=None, surface_opacity=0.3, surface_labels = None,
                     camera_angles=['floorplan', 'section', 'diagonal'],transparent_bg=True,title=None,
                     extension='png',centerc=None,eyec=None,upc=None, only_mesh=False, surface_visible=None):
        """
        Plots surface mesh, source and receivers in 3D.
        
        Parameters
        ----------
        renderer : str, optional
            Plotly render engine. The default is 'notebook'.

        Returns
        -------
        None.

        """
        
        import plotly.figure_factory as ff
        import plotly.graph_objs as go


        colors = 20 * ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
                        '#FF97FF', '#FECB52']
        vertices = self.nos.T#[np.unique(self.elem_surf)].T
        elements = self.elem_surf.T
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(225, 225, 225)"],
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
        
        if self.BC != None:
            i = 0
            if surface_visible is None:
                surface_visible = self.number_ID_faces
            for bl in self.number_ID_faces:
                indx = np.argwhere(self.domain_index_surf==bl)
                con = self.elem_surf[indx,:][:,0,:]
                vertices = self.nos.T#[con,:].T
                con = con.T
                fig.add_trace(go.Mesh3d(
                x=vertices[0, :],
                y=vertices[1, :],
                z=vertices[2, :],
                i=con[0, :], j=con[1, :], k=con[2, :],opacity=surface_opacity,showlegend=True,visible= True if bl in surface_visible else False, color = colors[i],
                    name=f'PG {int(bl)}' if surface_labels is None else surface_labels[i]))
                i+=1
                # fig['data'][0].update(opacity=0.3)
            # 
                # fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
                
        if renderer is not None:
            pio.renderers.default = renderer
        
        if title is None:
            fig.update_layout(title="")
        if transparent_bg:
            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                               'paper_bgcolor': 'rgba(0, 0, 0, 0)', }, )

        if only_mesh is True:
            fig = fd.plot_tools.remove_bg_and_axis(fig, 1)
            fig.update_layout(showlegend=False)

        if saveFig:
            # folderCheck = os.path.exists('/Layout')
            # if folderCheck is False:
            #     os.mkdir('/Layout')
            if filename is None or camera_angles is not None:
                for camera in camera_angles:
                    if camera == 'top' or camera == 'floorplan':
                        camera_dict = dict(eye=dict(x=0., y=0., z=2.5),
                                           up=dict(x=0, y=1, z=0),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'lateral' or camera == 'side' or camera == 'section':
                        camera_dict = dict(eye=dict(x=2.5, y=0., z=0.0),
                                           up=dict(x=0, y=0, z=1),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'front':
                        camera_dict = dict(eye=dict(x=0., y=2.5, z=0.),
                                           up=dict(x=0, y=1, z=0),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'rear' or camera == 'back':
                        camera_dict = dict(eye=dict(x=0., y=-2.5, z=0.),
                                           up=dict(x=0, y=1, z=0),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'diagonal_front':
                        camera_dict = dict(eye=dict(x=1.50, y=1.50, z=1.50),
                                           up=dict(x=0, y=0, z=1),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'diagonal_rear':
                        camera_dict = dict(eye=dict(x=-1.50, y=-1.50, z=1.50),
                                           up=dict(x=0, y=0, z=1),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'custom':
                        camera_dict = dict(eye=dict(x=eyec[0], y=eyec[1], z=eyec[2]),
                                           up=dict(x=upc[0], y=upc[1], z=upc[2]),
                                           center=dict(x=centerc[0], y=centerc[1], z=centerc[2]), )
                    fig.update_layout(scene_camera=camera_dict)

                    fig.write_image(filename + f'_3D_{camera}_.{extension}', scale=2)
            else:
                fig.write_image(filename+'.'+extension, scale=2)
        fig.show()
        return fig
    def pressure_field(self, Pmin=None, frequencies=[60], Pmax=None, axis=['xy', 'yz', 'xz', 'boundary'],
                       axis_visibility={'xy': True, 'yz': True, 'xz': 'legendonly', 'boundary': True},
                       coord_axis={'xy': None, 'yz': None, 'xz': None, 'boundary': None}, dilate_amount=0.9,
                       view_planes=False, gridsize=0.1, gridColor="rgb(230, 230, 255)",
                       opacity=0.2, opacityP=1, hide_dots=False, figsize=(950, 800),
                       showbackground=True, showlegend=True, showedges=True, colormap='jet',
                       saveFig=False,extension='png',room_opacity=0.3, colorbar=True, showticklabels=True, info=True, title=True,
                       axis_labels=['(X) Width [m]', '(Y) Length [m]', '(Z) Height [m]'], showgrid=True,
                       camera_angles=['floorplan', 'section', 'diagonal'], device='CPU',
                       transparent_bg=True, returnFig=False, show=True, filename=None,
                       renderer=None,centerc=None,eyec=None,upc=None):
        """
        Plots pressure field in boundaries and sections.

        Parameters
        ----------
        Pmin : TYPE, optional
            DESCRIPTION. The default is None.
        frequencies : TYPE, optional
            DESCRIPTION. The default is [60].
        Pmax : TYPE, optional
            DESCRIPTION. The default is None.
        axis : TYPE, optional
            DESCRIPTION. The default is ['xy', 'yz', 'xz', 'boundary'].
        axis_visibility : TYPE, optional
            DESCRIPTION. The default is {'xy': True, 'yz': True, 'xz': 'legendonly', 'boundary': True}.
        coord_axis : TYPE, optional
            DESCRIPTION. The default is {'xy': None, 'yz': None, 'xz': None, 'boundary': None}.
        dilate_amount : TYPE, optional
            DESCRIPTION. The default is 0.9.
        view_planes : TYPE, optional
            DESCRIPTION. The default is False.
        gridsize : TYPE, optional
            DESCRIPTION. The default is 0.1.
        gridColor : TYPE, optional
            DESCRIPTION. The default is "rgb(230, 230, 255)".
        opacity : TYPE, optional
            DESCRIPTION. The default is 0.2.
        opacityP : TYPE, optional
            DESCRIPTION. The default is 1.
        hide_dots : TYPE, optional
            DESCRIPTION. The default is False.
        figsize : TYPE, optional
            DESCRIPTION. The default is (950, 800).
        showbackground : TYPE, optional
            DESCRIPTION. The default is True.
        showlegend : TYPE, optional
            DESCRIPTION. The default is True.
        showedges : TYPE, optional
            DESCRIPTION. The default is True.
        colormap : TYPE, optional
            DESCRIPTION. The default is 'jet'.
        saveFig : TYPE, optional
            DESCRIPTION. The default is False.
        extension : TYPE, optional
            DESCRIPTION. The default is 'png'.
        room_opacity : TYPE, optional
            DESCRIPTION. The default is 0.3.
        colorbar : TYPE, optional
            DESCRIPTION. The default is True.
        showticklabels : TYPE, optional
            DESCRIPTION. The default is True.
        info : TYPE, optional
            DESCRIPTION. The default is True.
        title : TYPE, optional
            DESCRIPTION. The default is True.
        axis_labels : TYPE, optional
            DESCRIPTION. The default is ['(X) Width [m]', '(Y) Length [m]', '(Z) Height [m]'].
        showgrid : TYPE, optional
            DESCRIPTION. The default is True.
        camera_angles : TYPE, optional
            DESCRIPTION. The default is ['floorplan', 'section', 'diagonal'].
        device : TYPE, optional
            DESCRIPTION. The default is 'CPU'.
        transparent_bg : TYPE, optional
            DESCRIPTION. The default is True.
        returnFig : TYPE, optional
            DESCRIPTION. The default is False.
        show : TYPE, optional
            DESCRIPTION. The default is True.
        filename : TYPE, optional
            DESCRIPTION. The default is None.
        renderer : TYPE, optional
            DESCRIPTION. The default is 'notebook'.
        centerc : TYPE, optional
            DESCRIPTION. The default is None.
        eyec : TYPE, optional
            DESCRIPTION. The default is None.
        upc : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.

        """
        import gmsh

        import sys
        # from matplotlib.colors import Normalize
        import plotly
        import plotly.figure_factory as ff
        import plotly.graph_objs as go
        import os
        # import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings("ignore")

        # from utils.helpers import set_cpu, set_gpu, progress_bar
    
        start = time.time()
        # Creating planes
        # self.mesh_room()
        try:
            gmsh.finalize()
        except:
            pass
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", gridsize * 0.95)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", gridsize)
        # model = self.model
        if self.path_to_geo_unrolled != None:
            path_to_geo = self.path_to_geo_unrolled
        else:
            path_to_geo = self.path_to_geo
            
        print(path_to_geo)
        filenamez, file_extension = os.path.splitext(path_to_geo)
        path_name = os.path.dirname(path_to_geo)
        tgv = gmsh.model.getEntities(3)
        # ab = gmsh.model.getBoundingBox(3, tgv[0][1])
    
        xmin = np.amin(self.nos[:,0])
        xmax = np.amax(self.nos[:,0])
        ymin = np.amin(self.nos[:,1])
        ymax = np.amax(self.nos[:,1])
        zmin = np.amin(self.nos[:,2])
        zmax = np.amax(self.nos[:,2])
    
        if coord_axis['xy'] is None:
            coord_axis['xy'] = self.R.coord[0, 2] - 0.01
    
        if coord_axis['yz'] is None:
            coord_axis['yz'] = self.R.coord[0, 0]
    
        if coord_axis['xz'] is None:
            coord_axis['xz'] = self.R.coord[0, 1]
    
        if coord_axis['boundary'] is None:
            coord_axis['boundary'] = (zmin + zmax) / 2
        # with suppress_stdout():
        if 'xy' in axis:
            gmsh.clear()
            gmsh.open(path_to_geo)
            tgv = gmsh.model.getEntities(3)
            gmsh.model.occ.addPoint(xmin, ymin, coord_axis['xy'], 0., 3001)
            gmsh.model.occ.addPoint(xmax, ymin, coord_axis['xy'], 0., 3002)
            gmsh.model.occ.addPoint(xmax, ymax, coord_axis['xy'], 0., 3003)
            gmsh.model.occ.addPoint(xmin, ymax, coord_axis['xy'], 0., 3004)
            gmsh.model.occ.addLine(3001, 3004, 3001)
            gmsh.model.occ.addLine(3004, 3003, 3002)
            gmsh.model.occ.addLine(3003, 3002, 3003)
            gmsh.model.occ.addLine(3002, 3001, 3004)
            gmsh.model.occ.addCurveLoop([3004, 3001, 3002, 3003], 15000)
            gmsh.model.occ.addPlaneSurface([15000], 15000)
            gmsh.model.addPhysicalGroup(2, [15000], 15000)
    
            gmsh.model.occ.intersect(tgv, [(2, 15000)], 15000, True, True)
    
            # gmsh.model.occ.dilate([(2, 15000)],
            #                       (xmin + xmax) / 2, (ymin + ymax) / 2, coord_axis['xy'],
            #                       dilate_amount, dilate_amount, dilate_amount)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            vtags, vxy, _ = gmsh.model.mesh.getNodes()
            nxy = vxy.reshape((-1, 3))
            elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
            nxysurf = np.array(nodeTagss,dtype=int).reshape(-1,3)-1
    
        if 'yz' in axis:
            gmsh.clear()
            gmsh.open(path_to_geo)
            tgv = gmsh.model.getEntities(3)
            gmsh.model.occ.addPoint(coord_axis['yz'], ymin, zmin, 0., 3001)
            gmsh.model.occ.addPoint(coord_axis['yz'], ymax, zmin, 0., 3002)
            gmsh.model.occ.addPoint(coord_axis['yz'], ymax, zmax, 0., 3003)
            gmsh.model.occ.addPoint(coord_axis['yz'], ymin, zmax, 0., 3004)
            gmsh.model.occ.addLine(3001, 3004, 3001)
            gmsh.model.occ.addLine(3004, 3003, 3002)
            gmsh.model.occ.addLine(3003, 3002, 3003)
            gmsh.model.occ.addLine(3002, 3001, 3004)
            gmsh.model.occ.addCurveLoop([3004, 3001, 3002, 3003], 15000)
            gmsh.model.occ.addPlaneSurface([15000], 15000)
            gmsh.model.addPhysicalGroup(2, [15000], 15000)
    
            gmsh.model.occ.intersect(tgv, [(2, 15000)], 15000, True, True)
    
            # gmsh.model.occ.dilate([(2, 15000)],
            #                       coord_axis['yz'], (ymin + ymax) / 2, coord_axis['boundary'],
            #                       dilate_amount, dilate_amount, dilate_amount)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            # gmsh.write(path_name + 'current_field_yz.msh')
            # gmsh.write(outputs + 'current_field_yz.brep')
            vtags, vyz, _ = gmsh.model.mesh.getNodes()
            nyz = vyz.reshape((-1, 3))
            elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
            nyzsurf = np.array(nodeTagss,dtype=int).reshape(-1,3)-1
            
    
        if 'xz' in axis:
            gmsh.clear()
            gmsh.open(path_to_geo)
            tgv = gmsh.model.getEntities(3)
            gmsh.model.occ.addPoint(xmin, coord_axis['xz'], zmin, 0., 3001)
            gmsh.model.occ.addPoint(xmax, coord_axis['xz'], zmin, 0., 3002)
            gmsh.model.occ.addPoint(xmax, coord_axis['xz'], zmax, 0., 3003)
            gmsh.model.occ.addPoint(xmin, coord_axis['xz'], zmax, 0., 3004)
            gmsh.model.occ.addLine(3001, 3004, 3001)
            gmsh.model.occ.addLine(3004, 3003, 3002)
            gmsh.model.occ.addLine(3003, 3002, 3003)
            gmsh.model.occ.addLine(3002, 3001, 3004)
            gmsh.model.occ.addCurveLoop([3004, 3001, 3002, 3003], 15000)
            gmsh.model.occ.addPlaneSurface([15000], 15000)
            gmsh.model.addPhysicalGroup(2, [15000], 15000)
    
            gmsh.model.occ.intersect(tgv, [(2, 15000)], 15000, True, True)
    
            # gmsh.model.occ.dilate([(2, 15000)],
            #                       (xmin + xmax) / 2, coord_axis['xz'], (zmin + zmax) / 2,
            #                       dilate_amount, dilate_amount, dilate_amount)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            vtags, vxz, _ = gmsh.model.mesh.getNodes()
            nxz = vxz.reshape((-1, 3))
            elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
            nxzsurf = np.array(nodeTagss,dtype=int).reshape(-1,3)-1
    
        # if view_planes:
        #     gmsh.clear()
        #     gmsh.merge(outputs + 'current_mesh.brep')
        #     gmsh.merge(outputs + 'boundary_field.brep')
        #     gmsh.merge(outputs + 'current_field_xy.brep')
        #     gmsh.merge(outputs + 'current_field_yz.brep')
        #     gmsh.merge(outputs + 'current_field_xz.brep')
        #     gmsh.model.mesh.generate(2)
        #     gmsh.model.occ.synchronize()
        #     gmsh.fltk.run()
        gmsh.finalize()

        # Field plane evaluation
        prog = 0
        # for fi in frequencies:
        # if len(frequencies) > 1:
        #     progress_bar(prog / len(frequencies))
            
        fi = np.argwhere(self.freq==frequencies)[0][0]
        # boundData = self.bem.simulation._solution_data[idx]



            
        # print(fi)
        unq = np.unique(self.elem_surf)
        uind = np.arange(np.amin(unq),np.amax(unq)+1,1,dtype=int)
        if 'xy' in axis:
            pxy = np.zeros([len(nxy),1],dtype = np.complex128).ravel()
            for i in tqdm(range(len(nxy))):
                # pxy[i] = closest_node(self.nos,nxy[i,:])
                # print(coord_interpolation(self.nos, self.elem_vol, nxy[i,:], self.pN)[fi])
                pxy[i] = coord_interpolation(self.nos, self.elem_vol, nxy[i,:], self.pN)[fi][0]
            values_xy = np.real(p2SPL(pxy))

        if 'yz' in axis:             
            pyz = np.zeros([len(nyz),1],dtype = np.complex128).ravel()
            for i in tqdm(range(len(nyz))):
                pyz[i] = coord_interpolation(self.nos, self.elem_vol, nyz[i,:], self.pN)[fi][0]
            values_yz = np.real(p2SPL(pyz))
        if 'xz' in axis:
            pxz = np.zeros([len(nxz),1],dtype = np.complex128).ravel()
            for i in tqdm(range(len(nxz))):
                pxz[i] = coord_interpolation(self.nos, self.elem_vol, nxz[i,:], self.pN)[fi][0]
                # print(coord_interpolation(self.nos, self.elem_vol, nxz[i,:], self.pN)[fi][0])
            # print(pxz)                
            values_xz = np.real(p2SPL(pxz))
        if 'boundary' in axis:     

            values_boundary = np.real(p2SPL(self.pN[fi,uind]))  
        # Plotting
        if renderer is not None:
            plotly.io.renderers.default = renderer

        if info is False:
            showgrid = False
            title = False
            showticklabels = False
            colorbar = False
            showlegend = False
            showbackground = False
            axis_labels = ['', '', '']

        # Room
        vertices = self.nos.T#[np.unique(self.elem_surf)].T
        elements = self.elem_surf.T
        
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 222, 173)"],)
        fig['data'][0].update(opacity=room_opacity)
        fig.update_layout(title=dict(text = f'Frequency: {(np.real(self.freq[fi])):.2f} Hz'))
        # Planes
        # grid = boundData[0].space.grid
        # vertices = grid.vertices
        # elements = grid.elements
        # local_coordinates = np.array([[1.0 / 3], [1.0 / 3]])
        # values = np.zeros(grid.entity_count(0), dtype="float64")
        # for element in grid.entity_iterator(0):
        #     index = element.index
        #     local_values = np.real(20 * np.log10(np.abs((boundData[0].evaluate(index, local_coordinates))) / 2e-5))
        #     values[index] = local_values.flatten()
        if Pmin is None:
            Pmin = min(values_xy)
        if Pmax is None:
            Pmax = max(values_xy)

        colorbar_dict = {'title': 'SPL [dB]',
                         'titlefont': {'color': 'black'},
                         'title_side': 'right',
                         'tickangle': -90,
                         'tickcolor': 'black',
                         'tickfont': {'color': 'black'}, }

        if 'xy' in axis:
            vertices = nxy.T
            elements = nxysurf.T
            fig.add_trace(go.Mesh3d(x=vertices[0, :], y=vertices[1, :], z=vertices[2, :],
                                    i=elements[0, :], j=elements[1, :], k=elements[2, :], intensity=values_xy,
                                    colorscale=colormap, intensitymode='vertex', name='XY', showlegend=showlegend,
                                    visible=axis_visibility['xy'], cmin=Pmin, cmax=Pmax, opacity=opacityP,
                                    showscale=colorbar, colorbar=colorbar_dict))
            fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        if 'yz' in axis:
            vertices = nyz.T
            elements = nyzsurf.T
            fig.add_trace(go.Mesh3d(x=vertices[0, :], y=vertices[1, :], z=vertices[2, :],
                                    i=elements[0, :], j=elements[1, :], k=elements[2, :], intensity=values_yz,
                                    colorscale=colormap, intensitymode='vertex', name='YZ', showlegend=showlegend,
                                    visible=axis_visibility['yz'], cmin=Pmin, cmax=Pmax, opacity=opacityP,
                                    showscale=colorbar, colorbar=colorbar_dict))
            fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        if 'xz' in axis:
            vertices = nxz.T
            elements = nxzsurf.T
            fig.add_trace(go.Mesh3d(x=vertices[0, :], y=vertices[1, :], z=vertices[2, :],
                                    i=elements[0, :], j=elements[1, :], k=elements[2, :], intensity=values_xz,
                                    colorscale=colormap, intensitymode='vertex', name='XZ', showlegend=showlegend,
                                    visible=axis_visibility['xz'], cmin=Pmin, cmax=Pmax, opacity=opacityP,
                                    showscale=colorbar, colorbar=colorbar_dict))
            fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        if 'boundary' in axis:
            vertices = self.nos[uind].T
            elements = self.elem_surf.T
            fig.add_trace(go.Mesh3d(x=vertices[0, :], y=vertices[1, :], z=vertices[2, :],
                                    i=elements[0, :], j=elements[1, :], k=elements[2, :], intensity=values_boundary,
                                    colorscale=colormap, intensitymode='vertex', name='Boundary', showlegend=showlegend,
                                    visible=axis_visibility['boundary'], cmin=Pmin, cmax=Pmax, opacity=opacityP,
                                    showscale=colorbar, colorbar=colorbar_dict))
            fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        if not hide_dots:
            try:
                if self.R != None:
                    fig.add_trace(go.Scatter3d(x = self.R.coord[:,0], y = self.R.coord[:,1], z = self.R.coord[:,2],name="Receivers",mode='markers'))
            except:
                pass
            
            if self.S != None:    
                if self.S.wavetype == "spherical":
                    fig.add_trace(go.Scatter3d(x = self.S.coord[:,0], y = self.S.coord[:,1], z = self.S.coord[:,2],name="Sources",mode='markers'))
                   
                    
                    
        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1),
                          width=figsize[0], height=figsize[1],
                          scene=dict(xaxis_title=axis_labels[0],
                                     yaxis_title=axis_labels[1],
                                     zaxis_title=axis_labels[2],
                                     xaxis=dict(showticklabels=showticklabels, showgrid=showgrid,
                                                showline=showgrid, zeroline=showgrid),
                                     yaxis=dict(showticklabels=showticklabels, showgrid=showgrid,
                                                showline=showgrid, zeroline=showgrid),
                                     zaxis=dict(showticklabels=showticklabels, showgrid=showgrid,
                                                showline=showgrid, zeroline=showgrid),
                                     ))
        if title is False:
            fig.update_layout(title="")
        if transparent_bg:
            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                               'paper_bgcolor': 'rgba(0, 0, 0, 0)', }, )
        if saveFig:
            # folderCheck = os.path.exists('/Layout')
            # if folderCheck is False:
            #     os.mkdir('/Layout')
            for camera in camera_angles:
                if camera == 'top' or camera == 'floorplan':
                    camera_dict = dict(eye=dict(x=0., y=0., z=2.5),
                                       up=dict(x=0, y=1, z=0),
                                       center=dict(x=0, y=0, z=0), )
                elif camera == 'lateral' or camera == 'side' or camera == 'section':
                    camera_dict = dict(eye=dict(x=2.5, y=0., z=0.0),
                                       up=dict(x=0, y=0, z=1),
                                       center=dict(x=0, y=0, z=0), )
                elif camera == 'front':
                    camera_dict = dict(eye=dict(x=0., y=2.5, z=0.),
                                       up=dict(x=0, y=1, z=0),
                                       center=dict(x=0, y=0, z=0), )
                elif camera == 'rear' or camera == 'back':
                    camera_dict = dict(eye=dict(x=0., y=-2.5, z=0.),
                                       up=dict(x=0, y=1, z=0),
                                       center=dict(x=0, y=0, z=0), )
                elif camera == 'diagonal_front':
                    camera_dict = dict(eye=dict(x=1.50, y=1.50, z=1.50),
                                       up=dict(x=0, y=0, z=1),
                                       center=dict(x=0, y=0, z=0), )
                elif camera == 'diagonal_rear':
                    camera_dict = dict(eye=dict(x=-1.50, y=-1.50, z=1.50),
                                       up=dict(x=0, y=0, z=1),
                                       center=dict(x=0, y=0, z=0), )
                elif camera == 'custom':
                    camera_dict = dict(eye=dict(x=eyec[0], y=eyec[1], z=eyec[2]),
                                       up=dict(x=upc[0], y=upc[1], z=upc[2]),
                                       center=dict(x=centerc[0], y=centerc[1], z=centerc[2]), )
                fig.update_layout(scene_camera=camera_dict)

            if filename is None:
                fig.write_image(f'_3D_{camera}_{time.strftime("%Y%m%d-%H%M%S")}.{extension}', scale=2)
            else:
                fig.write_image(filename+'.'+extension, scale=2)

        if show:
            plotly.offline.iplot(fig)
        prog += 1
    
        end = time.time()
        elapsed_time = (end - start) / 60
        print(f'\n\tElapsed time to evaluate acoustic field: {elapsed_time:.2f} minutes\n')
        if returnFig:
            return fig        
        
    def pytta_obj(self):
        return IR(self.pR,self.AC,self.freq[0],self.freq[-1])
    
    def sabine_tr(self):
        """
        Calculates TR with Sabine's equation.

        Returns
        -------
        T60_sabine : TYPE
            DESCRIPTION.

        """
        V = compute_volume(self.NumElemC,self.NumNosC,self.elem_vol,self.nos,self.c0,self.rho0)
        Areas = compute_tri_area(self.domain_index_surf,np.sort([*self.mu]),self.NumElemC,self.NumNosC,self.elem_surf,self.nos,self.c0,self.rho0)
        print(f'Volume is: {V:.2f} m^3')
        Al = []
        Ar = []
        for bl in self.number_ID_faces:
            Al.append(Areas[bl]*mu2alpha(self.mu[bl],self.c0,self.rho0))
            print(mu2alpha(self.mu[bl],self.c0,self.rho0))
            Ar.append(Areas[bl])
        Al = np.array(Al)
        Area_t = np.sum(Ar)
        alp = np.sum(Al,axis=0)/Area_t
        print(f'Total Area is: {Area_t}')
        print(Ar)
        T60_sabine = 0.161*V/(np.sum(Ar)*alp)
        plt.semilogx(self.freq,T60_sabine)
        return T60_sabine
        

    def fitness_metric(self, w1=0.5, w2=0.5,fmin=20,fmax=200, dip_penalty=True, center_penalty=True, mode_penalty=True,
                       ref_curve='mean', dB_oct=2, nOct=2, lf_boost=10, infoLoc=(0.12, -0.03),
                       returnValues=False, plot=False, figsize=(17, 9), std_dev='symmetric'):
        """
        Fitness Metric.
        w1 is the weighting for the modal response and w2 for the SBIR.
        The threshold is the value below the average where the dips start to get penalized.
        """
        from scipy import interpolate
        from itertools import groupby
        
        AC = self.AC
        rC = self.R.coord
        if self.pR is None and self.pm is not None:
            self.pR = self.pm
        std_dev = np.std(p2SPL(self.pR),axis=0)
        
        self.fmin = fmin
        self.fmax = fmax
        self.wfreq, sbir_spl = SBIR_SPL(self.pR, rC, AC, fmin, fmax) 
        xmin, idx_min = find_nearest2(self.freq, self.fmin)
        xmax, idx_max = find_nearest2(self.freq, self.fmax)
        wxmin, widx_min = find_nearest2(self.wfreq, self.fmin)
        wxmax, widx_max = find_nearest2(self.wfreq, self.fmax)
        freq = self.freq[idx_min:idx_max + 1]
        pRdB = p2SPL(self.pR.T)
        
        std_dev = np.std(p2SPL(self.pR),axis=0)
        wstd_dev = np.std(sbir_spl, axis=0)
        modes = [first_cuboid_mode(max(self.nos[:, 0]), max(self.nos[:, 1]), max(self.nos[:, 2]),self.c0)]
        if len(rC)>1:
            self.hjwdB_average_rLw = np.mean(pRdB,axis=0)
            self.whjwdB_average_rLw = np.mean(sbir_spl,axis=0)
            self.std_dev_average_rLw  = np.mean(std_dev)
            self.wstd_dev_average_rLw = np.mean(wstd_dev)
    
        else:
            self.hjwdB_average_rLw = pRdB.flatten()
            self.whjwdB_average_rLw = np.array(sbir_spl).flatten()
            self.std_dev_average_rLw = std_dev[0]
            self.wstd_dev_average_rLw = wstd_dev[0]
            
        # Reference curve
        # if ref_curve == 'slope':
        #     oct_freq, oct_fr = self.oct_filter(plot=False, nOct=nOct, correction='max')
        #     reference_curve = [np.mean(self.hjwdB_average_rLw[idx_min:idx_max + 1]) for i in range(len(oct_freq))]
        #     for i in range(len(reference_curve)):
        #         reference_curve[i] = reference_curve[i] + lf_boost
        #         lf_boost -= dB_oct
        #         if lf_boost < 0:
        #             lf_boost = 0
        #     f = interpolate.interp1d(oct_freq, reference_curve)
        #     reference_curve = f(self.freq[idx_min:idx_max + 1])
        if ref_curve == 'mean':
            reference_curve = [
                np.mean(self.hjwdB_average_rLw[idx_min:idx_max + 1]) for i in
                range(len(self.freq[idx_min:idx_max + 1]))
            ]

        # Average Fitness rLw
        curve = np.copy(self.hjwdB_average_rLw[idx_min:idx_max + 1])
        asym_std_dev = asymetric_std_dev(curve)
        if self.wstd_dev_average_rLw is not None:
            if std_dev == 'asymmetric':
                fm_average_rLw = np.sqrt(w1 * asym_std_dev ** 2 + w2 * self.wstd_dev_average_rLw ** 2)
            else:
                fm_average_rLw = np.sqrt(w1 * self.std_dev_average_rLw ** 2 + w2 * self.wstd_dev_average_rLw ** 2)

        else:
            print('SBIR standard deviation not available.')
            if std_dev == 'asymmetric':
                fm_average_rLw = np.sqrt(w1 * asym_std_dev ** 2)
            else:
                fm_average_rLw = np.sqrt(w1 * self.std_dev_average_rLw ** 2)

        fm_average_rLw = 100 / (fm_average_rLw)
        axial_penalty = 0

        delta_f = 10
        if mode_penalty:
            for i in range(len(modes)):
                peak_mode = detect_peaks(curve, mph=np.mean(curve), show=False, valley=False)[0]
                if modes[i] > self.fmin:
                    axial_mode = modes[i]
                    if self.freq[peak_mode + idx_min] - delta_f < axial_mode < self.freq[
                        peak_mode + idx_min] + delta_f:
                        diff = curve[peak_mode] - reference_curve[peak_mode]
                        axial_penalty = np.sqrt(diff) if diff >= 0 else -np.sqrt(abs(diff))
                        #                         axial_penalty = 0.5 * diff
                        break
                    else:
                        peak_mode = detect_peaks(curve, mph=np.mean(curve), show=False, valley=False)[0]
        dips_depth = np.zeros_like(curve)
        dips_bandwidth = np.zeros_like(curve)
        dips_freq_weight = np.zeros_like(curve)
        curve = self.hjwdB_average_rLw[peak_mode:idx_max + 1]
        for idx_f in range(len(curve)):
            if curve[idx_f] < reference_curve[idx_f]:
                dips_depth[idx_f] = reference_curve[idx_f] - curve[idx_f]
                #                 dips_depth[idx_f] = (reference_curve[idx_f] - curve[idx_f] / np.sqrt(freq[idx_f]))
                dips_bandwidth[idx_f] = 1
                dips_freq_weight[idx_f] = np.log10(freq[idx_f])

        self.penalty = 0
        if dip_penalty is True:
            depth = np.asarray(
                [sum(val) for keys, val in groupby(dips_depth.tolist(), key=lambda x: x != 0) if keys != 0])
            bandwidth = np.asarray([sum(val) for keys, val in groupby(dips_bandwidth.tolist(),
                                                                      key=lambda x: x != 0) if keys != 0])
            freq_weight = np.asarray([sum(val) for keys, val in groupby(dips_freq_weight.tolist(),
                                                                        key=lambda x: x != 0) if keys != 0])

            penalty_value = np.sqrt(depth * freq_weight * bandwidth)
            self.penalty = penalty_value.sum() / bandwidth.sum()
        else:
            self.penalty = 0

        # Penalty Distance
        dist_penalty = 0
        if center_penalty:
            threshold = max(self.nos[:, 1]) / 20
            if max(self.nos[:, 1]) / 2 - threshold <= rC[0, 1] <= max(
                    self.nos[:, 1]) / 2 + threshold:
                dist_penalty = (threshold - (abs(max(self.nos[:, 1]) / 2 - rC[0, 1]))) * max(
                    self.nos[:, 1]) / 2
            self.penalty += dist_penalty

        # First order mode penalty

            self.penalty += -axial_penalty

        fm_average_rLw = fm_average_rLw - self.penalty

        if fm_average_rLw <= 0:
            fm_average_rLw = 0.1

        self.fitness = fm_average_rLw

        if plot:
            ymax = max(self.hjwdB_average_rLw)+10
            ymin = min(self.hjwdB_average_rLw)-20
            for i in range(len(rC)):
                while max(pRdB[i, idx_min:idx_max + 1]) > ymax or \
                        min(pRdB[i, idx_min:idx_max + 1]) < ymin:
                    if max(pRdB[i, idx_min:idx_max + 1]) > ymax:
                        ymax += 5
                        ymin += 2
                    if min(pRdB[i, idx_min:idx_max + 1]) < ymin:
                        ymax -= 2
                        ymin -= 5

            fig, ax = plt.subplots(figsize=figsize)

            curve = self.hjwdB_average_rLw[idx_min:idx_max + 1]
            ax.semilogx(freq, curve, linewidth=8, label=f'Average WMR rLW | SD: {np.std(curve):.2f} [dB]')

            ax.semilogx(self.wfreq[widx_min:widx_max + 1], self.whjwdB_average_rLw[widx_min:widx_max + 1],
                        linewidth=6, label=f'Average SBIR rLW| SD: '
                                           f'{np.std(self.whjwdB_average_rLw[widx_min:widx_max + 1]):.2f} [dB]')
            ax.semilogx(freq, reference_curve, label='Reference Curve [dB]', color='r', linestyle=':', alpha=0.8,
                        linewidth=2)
            ax.fill_between(freq, curve, reference_curve, where=(reference_curve > curve), color='r', alpha=0.5)
            ax.set_xlim([xmin, xmax])
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax.tick_params(which='minor', length=5)  # Set major and minor ticks to same length
            ax.tick_params(which='both', labelsize=15)
            ax.set_xlabel('Frequency [Hz]', fontsize=18)
            ax.set_ylabel('SPL [dB ref. 20 $\mu$Pa]', fontsize=18)
            ax.set_title('Fitness Metric', fontsize=20, pad=10)
            ax.grid('on')
            ax.legend(bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=14, loc="upper center")
            ax.set_ylim([ymin, ymax])

            ax2 = ax.twiny()
            ax2.semilogx(self.freq[idx_min:idx_max + 1], [0 for freq in self.freq[idx_min:idx_max + 1]], alpha=0)
            ax2.set_xlim([xmin, xmax])
            ax2.tick_params(axis='y', which='both', right=False, labelright=False)
            ax2.tick_params(axis='x', which='major', labelsize=15, rotation=90)
            ax2.tick_params(axis='x', which='minor', labeltop=False)
            frequencies = []

            delta = self.std_dev_average_rLw / 2
            curve = np.copy(self.hjwdB_average_rLw[idx_min:idx_max + 1])
            peak_frequencies = []
            peaks = detect_peaks(curve, mph=np.mean(curve) + delta,
                                 show=False, valley=False)
            dips = detect_peaks(curve, mph=np.mean(curve) - delta,
                                show=False, valley=True)
            # for peak in peaks:
            #     peak = int((peak + ((self.fmin - self.fminOverhead) / self.df)))
            #     peak_frequencies.append(self.freq[peak])

            num_colors = len(modes)
            cm = ListedColormap(seaborn.color_palette("bright", num_colors))
            colors = [cm(1. * i / num_colors) for i in range(num_colors)]
            j = 0
            for val in modes:
                i = 0

                if val <= self.fmax:
                    frequencies.append(round(val))
                    if i == 0:
                        ax2.axvline(x=round(val), linestyle='--', color=colors[j], linewidth=3, label=round(val))
                    else:
                        ax2.axvline(x=round(val), linestyle='--', color=colors[j], linewidth=3)
                    i += 1
                j += 1

                ax2.set_xticks([freq for freq in frequencies])
                ax2.set_xticklabels([str(freq) for freq in frequencies])
                ax2.legend(fontsize=14, ncol=3, loc='lower center', title='Cuboid Room Modes', title_fontsize=15)

            if mode_penalty:
                text_y = abs(curve[peak_mode] - reference_curve[peak_mode]) / 2
                text_y = -text_y if curve[peak_mode] < reference_curve[peak_mode] else text_y

        if returnValues:
            return fm_average_rLw

    def plot_freq(self, average=False, labels=None, visible=None, hover_data=None, linewidth=5, linestyle=None,
                  colors=None, alpha=1, mode="trace", fig=None, xlim=None, ylim=None, update_layout=True,
                  fig_size=(900, 620), show_fig=True, save_fig=False, folder_path=None, ticks=None,
                  folder_name="Frequency Response", filename="freq_response", title="",
                  ylabel='SPL [dB]', jwrho=True, offset = 0, renderer=None):
        assert self.pR is not None
        if jwrho:
            y_list = list(self.spl_S.T + offset) if average == False else [self.avg_spl_S.T + offset]
        else:
            y_list = list(self.spl.T + offset) if average == False else [self.avg_spl.T + offset]

        if labels is None:
            if average:
                labels = ["Average"]
            else:
                labels = [f"r_{i}" for i in range(len(y_list))]

        fig = fd.plot_tools.freq_response_plotly(len(y_list) * [self.freq], y_list, labels=labels, visible=visible,
                                                  hover_data=hover_data, linewidth=linewidth, linestyle=linestyle,
                                                  colors=colors, alpha=alpha, mode=mode, fig=fig, xlim=xlim, ylim=ylim,
                                                  update_layout=update_layout,
                                                  fig_size=fig_size, show_fig=show_fig, save_fig=save_fig,
                                                  folder_path=folder_path, ticks=ticks,
                                                  folder_name=folder_name, filename=filename, title=title,
                                                  ylabel=ylabel)

        if self.F_n is not None:
            for i in range(len(self.F_n)):
                fig.add_vline(x = self.F_n[i], line_width=3, line_dash="dash", line_color="red")

        if renderer is not None:
            pio.renderers.default = renderer

        return fig
    def fem_save(self, filename=time.strftime("%Y%m%d-%H%M%S"), ext = ".pickle"):
        """
        Saves FEM3D simulation into a pickle file.

        Parameters
        ----------
        filename : str, optional
            File name to be saved. The default is time.strftime("%Y%m%d-%H%M%S").
        ext : str, optional
            File extension. The default is ".pickle".

        Returns
        -------
        None.

        """
        
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


if __name__ == "__main__":
    import femder as fd

    path_to_geo = r"C:\Users\gutoa\Documents\SkaterXL\UFSM\MNAV\Code Testing\CR2_modificado.geo"
    AP = fd.AirProperties(c0=343)
    fmax = 20
    AC = fd.AlgControls(AP, 20, fmax, 10)

    S = fd.Source("spherical")

    S.coord = np.array([[3, 2.25, 1.2]])
    S.q = np.array([[0.0001]])

    R = fd.Receiver()
    R.star([0, 2.026, 1.230], 0.1)
    # R.coord = np.array([2,6,1.2])

    BC = fd.BC(AC, AP)
    BC.normalized_admittance([1, 2, 3, 4, 5], 0.02)
    grid = fd.GridImport3D(AP, path_to_geo, S=None, R=None, fmax=fmax,
                           num_freq=6, scale=1, order=1, heal_shapes=False)
    obj = fd.FEM3D(grid, S, R, AP, AC, BC)