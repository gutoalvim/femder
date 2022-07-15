# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:57:06 2021

@author: gutoa
"""

import numpy as np
import numba
# from scipy.sparse.linalg import spsolve
# from pypardiso import spsolve
from scipy.sparse.linalg import gmres
# from lapsolver import solve_dense as gmres

import time 
from tqdm import tqdm
import warnings
from numba import jit
import cloudpickle
from numba import njit
# from scipy.sparse import coo_matrix
# from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from contextlib import contextmanager
import sys, os

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
            
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
            
def bem_load(filename,ext='.pickle'):
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


    obj = BEM3D(Grid=None,AC=AC,AP=AP,S=S,R=R,BC=BC)
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

def SBIR_SPL(complex_pressure,AC,fmin,fmax):
    fs = 44100
    
    fmin_indx = np.argwhere(AC.freq==fmin)[0][0]
    fmax_indx = np.argwhere(AC.freq==fmax)[0][0]

    
    df = (AC.freq[-1]-AC.freq[0])/len(AC.freq)
    
    ir_duration = 1/df
    
    ir = fd.IR(fs,ir_duration,fmin,fmax).compute_room_impulse_response(complex_pressure.ravel())
    t_ir = np.linspace(0,ir_duration,len(ir))
    sbir = fd.SBIR(ir,t_ir,AC.freq[0],AC.freq[-1],method='peak')
    sbir_freq = sbir[1]
    sbir_SPL = p2SPL(sbir_freq)[fmin_indx:fmax_indx]
    sbir_freq = np.linspace(fmin,fmax,len(sbir_SPL))
    
    return sbir_freq,sbir_SPL

def compute_normals(vertices,faces):
    # norm = np.zeros( vertices.shape, dtype=vertices.dtype )
#Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]        
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    return numba.complex128(normalize_v3(n))

def compute_areas(vertices,faces):
    areas = np.zeros((len(faces),1))
    for i in range(len(faces)):
        rS = vertices[faces[i,:],:]
        areas[i] = area_normal_ph(rS)  
    return numba.complex128(areas.ravel())

def normalize_v3(arr):
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

@njit
def area_normal_ph(re,rev=1):
    
    xe = (re[:,0])
    ye = (re[:,1])
    ze = (re[:,2])
    #Formula de Heron - Area do Triangulo
    
    a = np.sqrt((xe[0]-xe[1])**2+(ye[0]-ye[1])**2+(ze[0]-ze[1])**2)
    b = np.sqrt((xe[1]-xe[2])**2+(ye[1]-ye[2])**2+(ze[1]-ze[2])**2)
    c = np.sqrt((xe[2]-xe[0])**2+(ye[2]-ye[0])**2+(ze[2]-ze[0])**2)
    p = (a+b+c)/2
    area_elm = np.abs(np.sqrt(p*(p-a)*(p-b)*(p-c)))
    
    return area_elm
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

def p2SPL(p):
    SPL = 10*np.log10(0.5*p*np.conj(p)/(2e-5)**2)
    return SPL

# @jit(parallel=True)
def assemble_bem_3gauss(Incid,Coord,rF,w,k0,rho0,normals,areas):
    
    lenG = numba.int64((len(Coord)))
    Gs = np.zeros((lenG,lenG),dtype = np.complex64)
    I = np.zeros((lenG,lenG),dtype = np.complex64)
    Cc = np.zeros((lenG,lenG),dtype = np.complex64)
    
    Pi = np.zeros((lenG,1),dtype = np.complex64)
    GN=np.array([[1, 0, -1],[0, 1, -1]],dtype=np.float64)
    
    a=1/6
    b=2/3
    qsi1=np.array([a, a, b])
    qsi2=np.array([a, b, a]); 
    weights=np.array([1/6, 1/6, 1/6])*2
    N = np.zeros((3,3),dtype=np.float64)
    N[:,0]=np.transpose(qsi1)
    N[:,1]=np.transpose(qsi2)
    N[:,2]=np.transpose(1-qsi1-qsi2)

    for nod in tqdm(range(len(Coord))):
        rS = Coord[nod,:]
        rF_rS = np.linalg.norm(rS-rF)
        # Pi[es] = (1j*w*rho0/(4*np.pi))*np.exp(-1j*k0*rF_rS)/rF_rS
        Pi[nod] = np.exp(-1j*k0*rF_rS)/(rF_rS)
        
        for es in range(len(Incid)):
            con = Incid[es,:]
            coord_el = Coord[con,:]

            area = areas[es]
            normal = normals[es,:]
            Ge,He,Ce=bem_t3(coord_el,rS,w,k0,rho0,normal,area,N,GN,weights);
            Gs[nod,con] += Ge
            I[nod,con] += He
            Cc[nod,con] += Ce
            

    return Gs,I,Cc,Pi



def assemble_bem_3gauss_prepost(Incid,Coord,rF,w,k0,rho0,normals,areas):
    
    lenG = numba.int64((len(Coord)))
    Gs = np.zeros((lenG,lenG),dtype = np.complex64)
    I = np.zeros((lenG,lenG),dtype = np.complex64)
    Cc = np.zeros((lenG,lenG),dtype = np.complex64)
    
    Pi = np.zeros((lenG,len(rF)),dtype = np.complex64)
    GN=np.array([[1, 0, -1],[0, 1, -1]],dtype=np.float64)
    
    a=1/6
    b=2/3
    qsi1=np.array([a, a, b])
    qsi2=np.array([a, b, a]); 
    weights=np.array([1/6, 1/6, 1/6]).T*2
    N = np.zeros((3,3),dtype=np.float64)
    N[:,0]=np.transpose(qsi1)
    N[:,1]=np.transpose(qsi2)
    N[:,2]=np.transpose(1-qsi1-qsi2)
    detJa,xg,yg,zg = pre_proccess_bem_t3(Incid, Coord, N, GN)
    for nod in (range(len(Coord))):
        rS = Coord[nod,:]
        for i in range(len(rF)):
            rF_rS = np.linalg.norm(rS-rF[i,:])
            # Pi[es] = (1j*w*rho0/(4*np.pi))*np.exp(-1j*k0*rF_rS)/rF_rS
            Pi[nod,i] = np.exp(-1j*k0*rF_rS)/(rF_rS)
        
        for es in range(len(Incid)):
            con = Incid[es,:]
            coord_el = Coord[con,:]
            area = areas[es]
            normal = normals[es,:]
            Ge,He,Ce=bem_t3_post(coord_el,rS,w,k0,rho0,normal,area,N,GN,weights,detJa[es],xg[es],yg[es],zg[es]);
            Gs[nod,con] = Gs[nod,con] + Ge
            I[nod,con] = I[nod,con] + He
            Cc[nod,nod] = Cc[nod,nod] + Ce
            

    return Gs,I,Cc,Pi

def evaluate_bem_3gauss_prepost(R,Incid,Coord,rF,w,k0,rho0,normals,areas):
    
    lenG = numba.int64((len(Coord)))
    lenR = numba.int64((len(R.coord)))
    Gf = np.zeros((lenR,lenG),dtype = np.complex64)
    I2 = np.zeros((lenR,lenG),dtype = np.complex64)
    
    Pi = np.zeros((lenR,len(rF)),dtype = np.complex64)
    GN=np.array([[1, 0, -1],[0, 1, -1]],dtype=np.float64)
    
    a=1/6
    b=2/3
    qsi1=np.array([a, a, b])
    qsi2=np.array([a, b, a]); 
    weights=np.array([1/6, 1/6, 1/6]).T*2
    N = np.zeros((3,3),dtype=np.float64)
    N[:,0]=np.transpose(qsi1)
    N[:,1]=np.transpose(qsi2)
    N[:,2]=np.transpose(1-qsi1-qsi2)
    
    detJa,xg,yg,zg = pre_proccess_bem_t3(Incid, Coord, N, GN)
    for fp in (range(len(R.coord))):
        coord_FP = R.coord[fp,:]
        for i in range(len(rF)):
            
            rF_rS = np.linalg.norm(coord_FP-rF[i,:])
            # Pi[es] = (1j*w*rho0/(4*np.pi))*np.exp(-1j*k0*rF_rS)/rF_rS
            Pi[fp,i] = -np.exp(-1j*k0*rF_rS)/(rF_rS)
        
        for es in range(len(Incid)):
            con = Incid[es,:]
            coord_el = Coord[con,:]

            area = areas[es]
            normal = normals[es,:]
            Ge,He,Ce=bem_t3_post(coord_el,coord_FP,w,k0,rho0,normal,area,N,GN,weights,detJa[es],xg[es],yg[es],zg[es]);
            Gf[fp,con] += Ge
            I2[fp,con] += He
            

    return Gf,I2,Pi


def evaluate_bem_3gauss(R,Incid,Coord,rF,w,k0,rho0,normals,areas):
    
    lenG = numba.int64((len(Coord)))
    lenR = numba.int64((len(Coord)))
    Gf = np.zeros((lenR,lenG),dtype = np.complex64)
    I2 = np.zeros((lenR,lenG),dtype = np.complex64)
    
    Pi = np.zeros((lenR,1),dtype = np.complex64)

    for fp in (range(len(R))):
        coord_FP = R.coord[fp,:]
        rF_rS = np.linalg.norm(coord_FP-rF)
        # Pi[es] = (1j*w*rho0/(4*np.pi))*np.exp(-1j*k0*rF_rS)/rF_rS
        Pi[fp] = np.exp(-1j*k0*rF_rS)/(rF_rS)
        
        for es in range(len(Incid)):
            con = Incid[es,:]
            coord_el = Coord[es,:]

            area = areas[es]
            normal = normals[es,:]
            Ge,He,Ce=bem_t3(coord_el,coord_FP,w,k0,rho0,normal,area);
            Gf[fp,con] += Ge
            I2[fp,con] += He
            

    return Gf,I2,Pi


def pre_proccess_bem_t3(Incid,Coord,N,GN):
    
    xg = []
    yg = []
    zg = []
    detJa = []
    for es in range(len(Incid)):
        con = Incid[es,:]
        coord_el = Coord[con,:]
        X0=coord_el[:,0]; Y0=coord_el[:,1]; Z0=coord_el[:,2];
        dx0=X0[1]-X0[0];dx1=X0[2]-X0[1];dx2=X0[0]-X0[2];
        dy0=Y0[1]-Y0[0];dy1=Y0[2]-Y0[1];dy2=Y0[0]-Y0[2];
        dz0=Z0[1]-Z0[0];dz1=Z0[2]-Z0[1];dz2=Z0[0]-Z0[2];
        
        # comprimento das arestas
        a=np.sqrt(dx0**2+dy0**2+dz0**2)
        b=np.sqrt(dx1**2+dy1**2+dz1**2)
        c=np.sqrt(dx2**2+dy2**2+dz2**2)
        # angulo do plano do triângulo em relação ao plano x-y
        ang = np.arccos((a**2+b**2-c**2)/(2*a*b))
        coordxy = (np.array([[0, a, b*np.cos(ang)],[0,0,b*np.sin(ang)]]).T)
        xg.append(np.dot(N,coord_el[:,0]))
        yg.append(np.dot(N,coord_el[:,1]))
        zg.append(np.dot(N,coord_el[:,2]))
        Ja = np.dot(GN,coordxy); # Matriz Jacobiana (transformação de coordenadas / mapeamento para sist. global)
    
    # print(Ja.shape)
        detJa.append(np.linalg.det(Ja)/2)
        
    return detJa,xg,yg,zg
@njit
def bem_t3_post(coord_el,coord_nod,w,k,rho0,normal,area,N,GN,weights,detJa,xg,yg,zg):
    
    Ge = np.zeros((3,),dtype=np.complex64)
    He = np.zeros((3,),dtype=np.complex64)
    Ce = np.zeros((3,),dtype=np.complex64)
    
    
    x=coord_nod[0];
    y=coord_nod[1];
    z=coord_nod[2];
    #%****Vetores de Influências
    xdis=xg-x;
    ydis=yg-y;
    zdis=zg-z;
    dis=np.sqrt(xdis**2+ydis**2+zdis**2)
    # print(dis.shape)
    g_top = np.exp(-1j*k*dis)
    g= g_top/(4*np.pi*dis);  # 1i*rho*w*exp(-1j*k*dis)/(4*np.pi*dis);
    
    Ge[0]=np.sum(np.sum(g*N[:,0]*detJa*weights)*weights); 
    Ge[1]=np.sum(np.sum(g*N[:,1]*detJa*weights)*weights); 
    Ge[2]=np.sum(np.sum(g*N[:,2]*detJa*weights)*weights); 
    
    
    h1=-xdis*np.exp(-1j*k*dis)/(4*np.pi*dis**2)*(1j*k+1/dis);
    h2=-ydis*np.exp(-1j*k*dis)/(4*np.pi*dis**2)*(1j*k+1/dis);
    h3=-zdis*np.exp(-1j*k*dis)/(4*np.pi*dis**2)*(1j*k+1/dis);
    

    n = normal
    hn = np.array([[h1[0],h1[1],h1[2]],[h2[0],h2[1],h2[2]],[h3[0],h3[1],h3[2]]]).T
    # print(hn.shape)
    h= np.dot(hn,n.T); 
    
    He[0]=np.sum(np.sum(-h*N[:,0]*detJa*weights)*weights); 
    He[1]=np.sum(np.sum(-h*N[:,1]*detJa*weights)*weights); 
    He[2]=np.sum(np.sum(-h*N[:,2]*detJa*weights)*weights);
    
    c1=-xdis/(4*np.pi*dis**2)*(1/dis);
    c2=-ydis/(4*np.pi*dis**2)*(1/dis);
    c3=-zdis/(4*np.pi*dis**2)*(1/dis);
    cn = np.array([[c1[0],c1[1],c1[2]],[c2[0],c2[1],c2[2]],[c3[0],c3[1],c3[2]]],dtype=np.complex128).T
    cc=np.dot(cn,n.T); 
    # print(cc)
    Ce[0]=np.sum(np.sum(-cc*N[:,0]*detJa*weights)*weights); 
    Ce[1]=np.sum(np.sum(-cc*N[:,1]*detJa*weights)*weights); 
    Ce[2]=np.sum(np.sum(-cc*N[:,2]*detJa*weights)*weights); 
    Ce=np.sum(Ce);
    
    return Ge,He,Ce
    
def bem_t3(coord_el,coord_nod,w,k,rho0,normal,area,N,GN,weights):
    
    Ge = np.empty((3,),dtype=np.complex64)
    He = np.empty((3,),dtype=np.complex64)
    Ce = np.empty((3,),dtype=np.complex64)
    
    X0=coord_el[:,0]; Y0=coord_el[:,1]; Z0=coord_el[:,2];
    dx0=X0[1]-X0[0];dx1=X0[2]-X0[1];dx2=X0[0]-X0[2];
    dy0=Y0[1]-Y0[0];dy1=Y0[2]-Y0[1];dy2=Y0[0]-Y0[2];
    dz0=Z0[1]-Z0[0];dz1=Z0[2]-Z0[1];dz2=Z0[0]-Z0[2];
    
    # comprimento das arestas
    a=np.sqrt(dx0**2+dy0**2+dz0**2)
    b=np.sqrt(dx1**2+dy1**2+dz1**2)
    c=np.sqrt(dx2**2+dy2**2+dz2**2)
    # angulo do plano do triângulo em relação ao plano x-y
    ang = np.arccos((a**2+b**2-c**2)/(2*a*b))
    coordxy=np.array([[0, a, b*np.cos(ang)],[0,0,b*np.sin(ang)]]).T
    # y=np.array([0,b*np.sin(ang)])
    # coordxy=np.array([x,y]).T;
    # print(coordxy.shape)
    # pontos e pesos de Gauss (3 pontos de integração) - 2D TRI


    xg=np.dot(N,coord_el[:,0])
    yg=np.dot(N,coord_el[:,1])
    zg=np.dot(N,coord_el[:,2]) # coordenadas reais dos ptos de gauss, internos ao elemento
    
    # ex=coord_el[:,0];
    # ey=coord_el[:,1];
    # ez=coord_el[:,2]; 
    # dNr=np.array([GN,GN,GN]);
    
    # JTxy=dNr*[ex,ey];
    # JTyz=dNr*[ey,ez];
    # JTzx=dNr*[ez,ex];
    # detJxy=np.array([np.linalg.det(JTxy[0:1,:]),np.linalg.det(JTxy[2:3,:]),np.linalg.det(JTxy[4:5,:])]);
    # detJyz=np.array([np.linalg.det(JTyz[0:1,:]),np.linalg.det(JTyz[2:3,:]),np.linalg.det(JTyz[4:5,:])]);
    # detJzx=np.array([np.linalg.det(JTzx[0:1,:]),np.linalg.det(JTzx[2:3,:]),np.linalg.det(JTzx[4:5,:])]);
    
    Ja = np.dot(GN,coordxy); # Matriz Jacobiana (transformação de coordenadas / mapeamento para sist. global)
    
    # print(Ja.shape)
    detJa=np.linalg.det(Ja)/2;
    
    x=coord_nod[0];
    y=coord_nod[1];
    z=coord_nod[2];
    #%****Vetores de Influências
    xdis=xg-x;
    ydis=yg-y;
    zdis=zg-z;
    dis=np.sqrt(xdis**2+ydis**2+zdis**2);
    g_top = np.exp(-1j*k*dis)
    g= g_top/(4*np.pi*dis);  # 1i*rho*w*exp(-1j*k*dis)/(4*np.pi*dis);
    
    Ge[0]=np.sum(np.sum(g*N[:,0]*detJa*weights)*weights); 
    Ge[1]=np.sum(np.sum(g*N[:,1]*detJa*weights)*weights); 
    Ge[2]=np.sum(np.sum(g*N[:,2]*detJa*weights)*weights); 
    
    
    # h1=-xdis*np.exp(-1j*k*dis)/(4*np.pi*dis**2)*(1j*k+1/dis);
    # h2=-ydis*np.exp(-1j*k*dis)/(4*np.pi*dis**2)*(1j*k+1/dis);
    # h3=-zdis*np.exp(-1j*k*dis)/(4*np.pi*dis**2)*(1j*k+1/dis);
    
    n = normal
    hn = np.array([-xdis*g_top/(4*np.pi*dis**2)*(1j*k+1/dis),
                -ydis*g_top/(4*np.pi*dis**2)*(1j*k+1/dis),
                -zdis*g_top/(4*np.pi*dis**2)*(1j*k+1/dis)])
    
    h=np.dot(hn,n.T); 
    
    He[0]=np.sum(np.sum(-h*N[:,0]*detJa*weights)*weights); 
    He[1]=np.sum(np.sum(-h*N[:,1]*detJa*weights)*weights); 
    He[2]=np.sum(np.sum(-h*N[:,2]*detJa*weights)*weights);
    c1=-xdis/(4*np.pi*dis**2)*(1/dis);
    c2=-ydis/(4*np.pi*dis**2)*(1/dis);
    c3=-zdis/(4*np.pi*dis**2)*(1/dis);
    cc=np.dot(np.array([c1, c2, c3]),(n.T)); 
    Ce[0]=np.sum(np.sum(-cc*N[:,0]*detJa*weights)*weights); 
    Ce[1]=np.sum(np.sum(-cc*N[:,1]*detJa*weights)*weights); 
    Ce[2]=np.sum(np.sum(-cc*N[:,2]*detJa*weights)*weights); 
    Ce=np.sum(Ce);
    
    return He,Ge,Ce
    
    
@njit
def assemble_BEM(Incid,Coord,rF,w,k0,rho0,normals,areas):
    lenG = numba.int64((len(Incid)))
    Gs = np.zeros((lenG,lenG),dtype = np.complex64)
    I = np.zeros((lenG,lenG),dtype = np.complex64)
    
    
    centers = np.zeros((lenG,3),dtype=np.float64)
    Pi = np.zeros((lenG,1),dtype = np.complex64)

    for es in (range(len(Incid))):
        rrS = Coord[Incid[es,:],:]
        rS = (rrS[0,:] + rrS[1,:] + rrS[2,:])/3

        centers[es,:] = rS        
        rF_rS = np.linalg.norm(rS-rF)
        # Pi[es] = (1j*w*rho0/(4*np.pi))*np.exp(-1j*k0*rF_rS)/rF_rS
        Pi[es] = np.exp(-1j*k0*rF_rS)/(rF_rS)
        
        for e in range(len(Incid)):
            re = Coord[Incid[e,:],:]
            area = areas[e]
            r_rS = np.sqrt((re[:,0]-rS[0])**2 + (re[:,1]-rS[1])**2 + (re[:,2]-rS[2])**2)
            
            dG = -((re-rS)*(np.exp(-1j*k0*r_rS)/(4*np.pi*r_rS**2))*(1j*k0 +1/r_rS))

            dGdN = np.dot(dG,normals[e,:])

            Gc = np.exp(-1j*k0*r_rS)/(4*np.pi*r_rS)

            Gs[es,e] = np.mean(Gc)*area
            I[es,e] = np.mean(dGdN)*area
            

    return Gs,I,Pi

# @staticmethod
@njit
def evaluate_field_BEM(coord,Incid,Coord,rF,w,k0,rho0,normals,areas):
    
    lenG = numba.int64((len(Incid)))
    lenR = numba.int64((len(coord)))
    Gf = np.zeros((lenR,lenG),dtype = np.complex64)
    I2 = np.zeros((lenR,lenG),dtype = np.complex64)
    Pifp = np.zeros((lenR,1),dtype = np.complex64)
    # normals = compute_normals(Coord,Incid)
    for fp in range(lenR):
        
        rfp = coord[fp,:]
        rF_rfp = np.linalg.norm(rfp - rF)
        # Pifp = (1j*w*rho0/(4*np.pi))*np.exp(-1j*k0*rF_rfp)/rF_rfp
        Pifp[fp] = np.exp(-1j*k0*rF_rfp)/(rF_rfp)
        for es in range(lenG):
            rS = Coord[Incid[es,:],:]
            r_rS = np.sqrt((rS[:,0]-rfp[0])**2 + (rS[:,1]-rfp[1])**2 + (rS[:,2]-rfp[2])**2)
            area = areas[es]
            normal = normals[es,:]
            # normal = np.reshape(normal,(-1,1))
            # print(normal.shape,)
            # normals.append(normal)
            # print(r_rS)
            
            # val = (np.exp(-1j*k0*r_rS)/(4*np.pi*r_rS))
            # dG = val / (r_rS**2) * (1j * k0 * r_rS - 1)
            # dGdN = (-dG*(rS-rfp))@normal.T
            # print(dGdN.shape)
            dG = -((rS-rfp)*(np.exp(-1j*k0*r_rS)/(4*np.pi*r_rS**2))*(1j*k0 +1/r_rS))
            
            # print(normal.shape,dG.shape)

            dGdN = np.dot(dG,normal.T)
            # dGdN = dG@normal
            # print(dGdN.shape)
            Gc = np.exp(-1j*k0*r_rS)/(4*np.pi*r_rS)
            Gf[fp,es] = np.mean(Gc)*area
            I2[fp,es] = np.mean(dGdN)*area
     
    return Gf,I2,Pifp
def evaluate_p_field(I2,Gf,Pifp,pC):
    pScat = I2@pC
    pInc = Pifp
    pTotal = pScat + pInc
    return pTotal,pScat

# @numba.jitclass
class BEM3D:
    def __init__(self,Grid,S,R,AP,AC,BC=None,interp='linear'):
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

        self.AC = AC
        self.AP = AP
        ##AlgControls
        self.c0 = AP.c0
        self.rho0 = AP.rho0
        
        self.w = 2*np.pi*self.freq
        self.k = self.w/self.c0
        
        
        self.S = S
        self.R = R
        #%Mesh
        if Grid != None:
            self.grid = Grid
            self.nos = Grid.nos
            self.elem_surf = Grid.elem_surf
            self.domain_index_surf =  Grid.domain_index_surf
            self.number_ID_faces =Grid.number_ID_faces
            self.NumNosC = Grid.NumNosC
            self.NumElemC = Grid.NumElemC
            self.order = Grid.order
            self.path_to_geo = Grid.path_to_geo
        self.pR = None
        self.pN = None
        self.rho = {}
        self.c = {}
        

        self.normals = compute_normals(self.nos, self.elem_surf)
        self.areas = compute_areas(self.nos, self.elem_surf)
        self.interp = interp
        self.coloc_cte = 'integ'
        self.individual_sources = True
        

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
        # print('Computation has started')
        p = []
        pS = []

        for i in tqdm(range(len(self.freq))):
            
            
            if self.interp == 'constant':
                Gs,I,Pi = assemble_BEM(self.elem_surf,self.nos,self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
                C = 0.5*np.eye(len(self.elem_surf))
                if self.individual_sources==True:
                    for es in range(len(self.S.coord)):
                        
                        pC,info = gmres((C-I),(Pi[:,es]))
                        pS.append(pC)
                        
                elif self.individual_sources ==False:
                    pC,info = gmres((C-I),(np.sum(Pi,axis=1)))
                    pS = pC
                
                # pC,info = gmres((C-I),(Pi))
            elif self.interp == 'linear':
                Gs,I,Cc,Pi = assemble_bem_3gauss_prepost(self.elem_surf,self.nos,self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
                
                if self.coloc_cte == 'cte':
                    C = 0.5*np.eye(len(self.nos))
                elif self.coloc_cte == 'integ':
                    C = np.eye(len(self.nos)) - Cc
                # C = 0.5*np.eye(len(self.nos))
                
                if self.individual_sources==True:
                    for es in range(len(self.S.coord)):
                        
                        pC,info = gmres((C+I),(Pi[:,es]))
                        pS.append(pC)
                        
                elif self.individual_sources ==False:
                    pC,info = gmres((C+I),(np.sum(Pi,axis=1)))
                    pS = pC
                
            if info != 0:
                print('Solver failed to converge')
            
            p.append(pS)
        
        self.pC = p
        self.t = time.time()-then
        
        if timeit:
            if self.t <= 60:
                print(f'Time taken: {self.t} s')
            elif 60 < self.t < 3600:
                print(f'Time taken: {self.t/60} min')
            elif self.t >= 3600:
                print(f'Time taken: {self.t/60} min')
                            
      
    def evaluate(self,R,plot=False):
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
        
        
        self.R = R
        pT = []
        pS = []
        pTT = []
        pSS = []
        for i in tqdm(range(len(self.freq))):
                
            if self.interp == 'constant':
                if self.individual_sources==True:
                    for es in range(len(self.S.coord)):
                        
                        Gf,I2,Pifp = evaluate_field_BEM(self.R.coord, self.elem_surf,self.nos, self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
                        ptt,pss = evaluate_p_field(I2,Gf,Pifp[:,es],self.pC[i][es])                        
                        
                        pTT.append(ptt)
                        pSS.append(pss)
                
                    pT.append(np.array(pTT))
                    pS.append(np.array(pSS))
                
                elif self.individual_sources ==False:
                    Gf,I2,Pifp = evaluate_field_BEM(self.R.coord, self.elem_surf,self.nos, self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
                    ptt,pss = evaluate_p_field(I2,Gf,np.sum(Pifp,axis=1),self.pC[i])                        
                    
                    pT.append(ptt)
                    pS.append(pss)  
                    
                # Gf,I2,Pifp = evaluate_field_BEM(self.R.coord, self.elem_surf,self.nos, self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
             
            if self.interp == 'linear':
                if self.individual_sources==True:
                    for es in range(len(self.S.coord)):
                        
                        Gf,I2,Pifp = evaluate_bem_3gauss_prepost(self.R, self.elem_surf,self.nos, self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
                        ptt,pss = evaluate_p_field(I2,Gf,Pifp[:,es],self.pC[i][es])                        
                        
                        pTT.append(ptt)
                        pSS.append(pss)
                
                    pT.append(np.array(pTT))
                    pS.append(np.array(pSS))
                
                elif self.individual_sources ==False:
                    Gf,I2,Pifp = evaluate_bem_3gauss_prepost(self.R, self.elem_surf,self.nos, self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
                    ptt,pss = evaluate_p_field(I2,Gf,np.sum(Pifp,axis=1),self.pC[i])                        
                    
                    pT.append(ptt)
                    pS.append(pss)  
                # Gf,I2,Pifp = evaluate_bem_3gauss_prepost(self.R, self.elem_surf,self.nos, self.S.coord,self.w[i],self.k[i],self.rho0,self.normals,self.areas)
            
            # print(Gf.shape)
            # print(Pifp.shape)
            # ptt,pss = evaluate_p_field(I2,Gf,Pifp,self.pC[i])
                   
            # pT.append(ptt)
            # pS.append(pss)
            
        self.pT = (pT)
        self.pS = (pS)
        
        if plot:
            if len(R.theta > 0):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='polar')
                ax.set_thetamin(np.amin(np.rad2deg(R.theta)))
                ax.set_thetamax(np.amax(np.rad2deg(R.theta)))
                
                ax.plot(R.theta, fd.p2SPL(self.pS[i,:]))
        return self.pT,self.pS
    
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
    
        
    def surf_evaluate(self,freq,renderer='notebook',d_range = 45,
                      saveFig=False,title=False,transparent_bg=True,filename=None,camera_angles=['floorplan', 'section', 'diagonal'],extension='png'):
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
        import plotly.figure_factory as ff

        
        fi = find_nearest(self.freq, freq)
        vertices = self.nos.T
        # vertices = self.nos[np.unique(self.elem_surf)].T
        elements = self.elem_surf.T
        
        values = np.abs(self.pC[fi])
        if d_range != None:
            d_range = np.amax(values)-d_range
            
            values[values<d_range] = np.amax(values)-d_range
        
        
        # print(np.amin(values),np.amax(values))
        # print(vertices.shape)
        # print(elements.shape)
        # print(values.shape)
        
        if self.interp == 'linear':
            imode = 'vertex'
        elif self.interp == 'constant':
            imode = 'cell'
            
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
        )
        fig['data'][0].update(opacity=0.3)
        
        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        fig.add_trace(go.Mesh3d(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            i=elements[0,:],
            j=elements[1,:],
            k=elements[2,:],
            intensity = values,
            colorscale= 'Jet',
            intensitymode=imode,
 
        ))  

        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        import plotly.io as pio
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
                        camera_dict = dict(eye=dict(x=2, y=2, z=2),
                                           up=dict(x=0, y=0, z=1.5),
                                           center=dict(x=0, y=0, z=0), )
                    elif camera == 'diagonal_rear':
                        camera_dict = dict(eye=dict(x=-2, y=-2, z=2),
                                           up=dict(x=0, y=0, z=1.5),
                                           center=dict(x=0, y=0, z=0), )
                    fig.update_layout(scene_camera=camera_dict)
    
                    fig.write_image(f'_3D_{camera}_{time.strftime("%Y%m%d-%H%M%S")}.{extension}', scale=2)
            else:
                fig.write_image(filename+'.'+extension, scale=2)
        fig.show()
        
    def plot_problem(self,renderer='notebook',saveFig=False,filename=None,
                     camera_angles=['floorplan', 'section', 'diagonal'],transparent_bg=True,title=None,extension='png'):
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
        
        if self.BC != None:
            
            for bl in self.number_ID_faces:
                indx = np.argwhere(self.domain_index_surf==bl)
                con = self.elem_surf[indx,:][:,0,:]
                vertices = self.nos.T#[con,:].T
                con = con.T
                fig.add_trace(go.Mesh3d(
                x=vertices[0, :],
                y=vertices[1, :],
                z=vertices[2, :],
                i=con[0, :], j=con[1, :], k=con[2, :],opacity=0.3,showlegend=True,visible=True,name=f'Physical Group {int(bl)}'
                ))
                # fig['data'][0].update(opacity=0.3)
            # 
                # fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
                
        import plotly.io as pio
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
        
    def pressure_field(self, Pmin=None, frequencies=[60], Pmax=None, axis=['xy', 'yz', 'xz', 'boundary'],
                       axis_visibility={'xy': True, 'yz': True, 'xz': 'legendonly', 'boundary': True},
                       coord_axis={'xy': None, 'yz': None, 'xz': None, 'boundary': None}, dilate_amount=0.9,
                       view_planes=False, gridsize=0.1, gridColor="rgb(230, 230, 255)",
                       opacity=0.2, opacityP=1, hide_dots=False, figsize=(950, 800),
                       showbackground=True, showlegend=True, showedges=True, colormap='jet',
                       saveFig=False, colorbar=True, showticklabels=True, info=True, title=True,
                       axis_labels=['(X) Width [m]', '(Y) Length [m]', '(Z) Height [m]'], showgrid=True,
                       camera_angles=['floorplan', 'section', 'diagonal'], device='CPU',
                       transparent_bg=True, returnFig=False, show=True, filename=None,
                       renderer='notebook'):
    
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
        gmsh.initialize(sys.argv)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", gridsize * 0.95)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", gridsize)
        # model = self.model
        
        filename, file_extension = os.path.splitext(self.path_to_geo)
        path_name = os.path.dirname(self.path_to_geo)
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
        with suppress_stdout():
            if 'xy' in axis:
                gmsh.clear()
                gmsh.open(self.path_to_geo)
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
                gmsh.open(self.path_to_geo)
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
                gmsh.open(self.path_to_geo)
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
                pxy[i] = self.evaluate(nxy[i,:])
            values_xy = np.real(p2SPL(pxy))

        if 'yz' in axis:             
            pyz = np.zeros([len(nyz),1],dtype = np.complex128).ravel()
            for i in tqdm(range(len(nyz))):
                pyz[i] = self.evaluate(nyz[i,:])
            values_yz = np.real(p2SPL(pyz))
        if '' in axis:
            pxz = np.zeros([len(nxz),1],dtype = np.complex128).ravel()
            for i in tqdm(range(len(nxz))):
                pxz[i] = self.evaluate(nxz[i,:])
                # print(coord_interpolation(self.nos, self.elem_vol, nxz[i,:], self.pN)[fi][0])
            # print(pxz)                
            values_xz = np.real(p2SPL(pxz))
        if 'boundary' in axis:     

            values_boundary = np.real(p2SPL(self.pC[fi]))  
        # Plotting
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
        fig['data'][0].update(opacity=0.3)
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
        # if Pmin is None:
        #     Pmin = min(values_xy)
        # if Pmax is None:
        #     Pmax = max(values_xy)

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

                    fig.write_image(f'_3D_pressure_plot_{camera}_{int(self.freq[fi])}Hz.png', scale=2)
            else:
                fig.write_image(filename + '.png', scale=2)

        if show:
            plotly.offline.iplot(fig)
        prog += 1
    
        end = time.time()
        elapsed_time = (end - start) / 60
        print(f'\n\tElapsed time to evaluate acoustic field: {elapsed_time:.2f} minutes\n')
        if returnFig:
            return fig        
    def bem_save(self, filename=time.strftime("%Y%m%d-%H%M%S"), ext = ".pickle"):
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
        print('BEM saved successfully.')