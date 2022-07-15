# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:43:11 2021

@author: gutoa
"""
import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import numba
from tqdm import tqdm
from scipy.sparse import coo_matrix
from numba import njit
from time import time
from scipy.sparse.linalg import lgmres, spilu, LinearOperator

def computeAB(points, faces):
    """
    Compute matrices for the Laplace-Beltrami operator.

    The matrices correspond to A and B from Reuter's 2009 article.

    Note ::
        All points must be on faces. Otherwise, a singular matrix error
        is generated when inverting D.

    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex

    faces : list of lists of 3 integers
        each list contains indices to vertices that form a triangle on a mesh

    Returns
    -------
    A : csr_matrix
    B : csr_matrix

    Examples
    --------

    """
    import numpy as np
    from scipy import sparse

    points = np.array(points)
    faces = np.array(faces)
    nfaces = faces.shape[0]

    # Linear local matrices on unit triangle:
    tB = (np.ones((3, 3)) + np.eye(3)) / 24.0

    tA00 = np.array([[0.5, -0.5, 0.0],
                     [-0.5, 0.5, 0.0],
                     [0.0, 0.0, 0.0]])

    tA11 = np.array([[0.5, 0.0, -0.5],
                     [0.0, 0.0, 0.0],
                     [-0.5, 0.0, 0.5]])

    tA0110 = np.array([[1.0, -0.5, -0.5],
                       [-0.5, 0.0, 0.5],
                       [-0.5, 0.5, 0.0]])

    # Replicate into third dimension for each triangle
    # (for tB, 1st index is the 3rd index in MATLAB):
    tB = np.array([np.tile(tB, (1, 1)) for i in range(nfaces)])
    tA00 = np.array([np.tile(tA00, (1, 1)) for i in range(nfaces)])
    tA11 = np.array([np.tile(tA11, (1, 1)) for i in range(nfaces)])
    tA0110 = np.array([np.tile(tA0110, (1, 1)) for i in range(nfaces)])

    # Compute vertex coordinates and a difference vector for each triangle:
    v1 = points[faces[:, 0], :]
    v2 = points[faces[:, 1], :]
    v3 = points[faces[:, 2], :]
    v2mv1 = v2 - v1
    v3mv1 = v3 - v1

    def reshape_and_repeat(A):
        """
        For a given 1-D array A, run the MATLAB code below.

            M = reshape(M,1,1,nfaces);
            M = repmat(M,3,3);

        Please note that a0 is a 3-D matrix, but the 3rd index in NumPy
        is the 1st index in MATLAB.  Fortunately, nfaces is the size of A.

        """
        return np.array([np.ones((3, 3)) * x for x in A])

    # Compute length^2 of v3mv1 for each triangle:
    a0 = np.sum(v3mv1 * v3mv1, axis=1)
    a0 = reshape_and_repeat(a0)

    # Compute length^2 of v2mv1 for each triangle:
    a1 = np.sum(v2mv1 * v2mv1, axis=1)
    a1 = reshape_and_repeat(a1)

    # Compute dot product (v2mv1*v3mv1) for each triangle:
    a0110 = np.sum(v2mv1 * v3mv1, axis=1)
    a0110 = reshape_and_repeat(a0110)

    # Compute cross product and 2*vol for each triangle:
    cr = np.cross(v2mv1, v3mv1)
    vol = np.sqrt(np.sum(cr * cr, axis=1))
    # zero vol will cause division by zero below, so set to small value:
    vol_mean = 0.001 * np.mean(vol)
    vol = [vol_mean if x == 0 else x for x in vol]
    vol = reshape_and_repeat(vol)

    # Construct all local A and B matrices (guess: for each triangle):
    localB = vol * tB
    localA = (1.0 / vol) * (a0 * tA00 + a1 * tA11 - a0110 * tA0110)

    # Construct row and col indices.
    # (Note: J in numpy is I in MATLAB after flattening,
    #  because numpy is row-major while MATLAB is column-major.)
    J = np.array([np.tile(x, (3, 1)) for x in faces])
    I = np.array([np.transpose(np.tile(x, (3, 1))) for x in faces])

    # Flatten arrays and swap I and J:
    J_new = I.flatten()
    I_new = J.flatten()
    localA = localA.flatten()
    localB = localB.flatten()

    # Construct sparse matrix:
    A = sparse.csr_matrix((localA, (I_new, J_new)))
    B = sparse.csr_matrix((localB, (I_new, J_new)))

    return A, B

def timer_func(func):
    """
    Time function with decorator
    Parameters
    ----------
    func: function

    Returns
    -------

    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func

@timer_func
def pre_proccess_bem_t3(Incid, Coord, N, GN):
    xg = []
    yg = []
    zg = []
    detJa = []
    for es in tqdm(range(len(Incid))):
        con = Incid[es, :]
        coord_el = Coord[con, :]
        X0 = coord_el[:, 0];
        Y0 = coord_el[:, 1];
        Z0 = coord_el[:, 2];
        dx0 = X0[1] - X0[0];
        dx1 = X0[2] - X0[1];
        dx2 = X0[0] - X0[2];
        dy0 = Y0[1] - Y0[0];
        dy1 = Y0[2] - Y0[1];
        dy2 = Y0[0] - Y0[2];
        dz0 = Z0[1] - Z0[0];
        dz1 = Z0[2] - Z0[1];
        dz2 = Z0[0] - Z0[2];

        # comprimento das arestas
        a = np.sqrt(dx0 ** 2 + dy0 ** 2 + dz0 ** 2)
        b = np.sqrt(dx1 ** 2 + dy1 ** 2 + dz1 ** 2)
        c = np.sqrt(dx2 ** 2 + dy2 ** 2 + dz2 ** 2)
        # angulo do plano do triângulo em relação ao plano x-y
        ang = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        coordxy = (np.array([[0, a, b * np.cos(ang)], [0, 0, b * np.sin(ang)]]).T)
        xg.append(np.dot(N, coord_el[:, 0]))
        yg.append(np.dot(N, coord_el[:, 1]))
        zg.append(np.dot(N, coord_el[:, 2]))
        Ja = np.dot(GN, coordxy);  # Matriz Jacobiana (transformação de coordenadas / mapeamento para sist. global)

        # print(Ja.shape)
        detJa.append(np.linalg.det(Ja) / 2)

    return detJa, xg, yg, zg

@timer_func
@njit
def assemble_bem_3gauss_prepost(Incid, Coord, rF, k, normals, det_ja, xg, yg, zg, N, weights):

    lenG = numba.int64((len(Coord)))
    I = np.zeros((lenG, lenG), dtype=np.complex64)
    Cc = np.zeros((lenG, lenG), dtype=np.complex64)

    Pi = np.zeros((lenG, len(rF)), dtype=np.complex64)

    for nod in (range(len(Coord))):
        rS = Coord[nod, :]
        for i in range(len(rF)):
            rF_rS = np.linalg.norm(rS - rF[i, :])
            # Pi[es] = (1j*w*rho0/(4*np.pi))*np.exp(-1j*k0*rF_rS)/rF_rS
            Pi[nod, i] = np.exp(-1j * k * rF_rS) / (rF_rS)

        for es in range(len(Incid)):
            con = Incid[es, :]
            normal = normals[es, :]
            # He, Ce = bem_t3_post(rS, k, normal, N, weights, det_ja[es], xg[es], yg[es],
            #                      zg[es])
            He = np.zeros((3,), dtype=np.complex64)
            Ce = np.zeros((3,), dtype=np.complex64)

            x = rS[0]
            y = rS[1]
            z = rS[2]
            xdis = xg[es] - x
            ydis = yg[es] - y
            zdis = zg[es] - z
            dis = np.sqrt(xdis ** 2 + ydis ** 2 + zdis ** 2)
            # print(dis.shape)

            h1 = -xdis * np.exp(-1j * k * dis) / (4 * np.pi * dis ** 2) * (1j * k + 1 / dis)
            h2 = -ydis * np.exp(-1j * k * dis) / (4 * np.pi * dis ** 2) * (1j * k + 1 / dis)
            h3 = -zdis * np.exp(-1j * k * dis) / (4 * np.pi * dis ** 2) * (1j * k + 1 / dis)

            n = normal
            hn = np.array([[h1[0], h1[1], h1[2]], [h2[0], h2[1], h2[2]], [h3[0], h3[1], h3[2]]]).T
            # print(hn.shape)
            h = np.dot(hn, n.T);

            He[0] = np.sum(np.sum(-h * N[:, 0] * det_ja[es] * weights) * weights)
            He[1] = np.sum(np.sum(-h * N[:, 1] * det_ja[es] * weights) * weights)
            He[2] = np.sum(np.sum(-h * N[:, 2] * det_ja[es] * weights) * weights)

            c1 = -xdis / (4 * np.pi * dis ** 2) * (1 / dis)
            c2 = -ydis / (4 * np.pi * dis ** 2) * (1 / dis)
            c3 = -zdis / (4 * np.pi * dis ** 2) * (1 / dis)
            cn = np.array([[c1[0], c1[1], c1[2]], [c2[0], c2[1], c2[2]], [c3[0], c3[1], c3[2]]], dtype=np.complex128).T
            cc = np.dot(cn, n.T)
            Ce[0] = np.sum(np.sum(-cc * N[:, 0] * det_ja[es] * weights) * weights)
            Ce[1] = np.sum(np.sum(-cc * N[:, 1] * det_ja[es] * weights) * weights)
            Ce[2] = np.sum(np.sum(-cc * N[:, 2] * det_ja[es] * weights) * weights)
            Ce = np.sum(Ce)
            for j, ix in enumerate(con):
                I[nod, ix] = I[nod, ix] + He[j]
            # I[nod, con] = I[nod, con] + He
            Cc[nod, nod] = Cc[nod, nod] + Ce

    return I, Cc, Pi

# @timer_func
def vectorized(Incid, Coord, rF, k, normals, det_ja, xg, yg, zg, N, weights):

    lenG = numba.int64((len(Coord)))
    _I = np.zeros((lenG, lenG), dtype=np.complex64)
    _Cc = np.zeros((lenG, lenG), dtype=np.complex64)

    Pi = np.zeros((lenG, len(rF)), dtype=np.complex64)
    _normals = np.repeat(np.expand_dims(normals, axis=2), 3, axis=2).transpose((1, 0, 2))
    _N = np.repeat(np.expand_dims(N, axis=2), len(Incid), axis=2)#.transpose((1, 0, 2))
    _weights = np.repeat(np.expand_dims(weights, axis=1), 3, axis=1)#.transpose((1, 0, 2))
    _weights = np.repeat(np.expand_dims(_weights, axis=2), len(Incid), axis=2)#.transpose((1, 0, 2))
    _det_ja = np.asarray(det_ja)
    for _nod in tqdm(range(len(Coord))):
        rS = Coord[_nod, :]
        _dis = np.asarray([xg - rS[0], yg - rS[1], zg - rS[2]]).squeeze()
        dis = np.sqrt(_dis[0] ** 2 + _dis[1] ** 2 + _dis[2] ** 2)

        _h = -_dis * np.exp(-1j * k * dis) / (4 * np.pi * dis ** 2) * (1j * k + 1 / dis)
        _hn = np.transpose(_h, (1, 2, 0))
        the_h = np.einsum('ijk,ik->ji', _hn, normals)
        _he = np.sum(np.sum(-the_h * _N * _det_ja * _weights, axis=1) * _weights, axis=0)

        _c = -_dis / (4 * np.pi * dis ** 2) * (1 / dis)
        _cn = np.transpose(_c, (1, 2, 0))
        the_c = np.einsum('ijk,ik->ji', _cn, normals)
        _ce = np.sum(np.sum(np.sum(np.sum(-the_c * _N * _det_ja * _weights, axis=1) * _weights, axis=0),axis=0))

        # C = (normals[:, np.newaxis, :] @ _hn)
        # C = C.reshape(-1, C.shape[2])
        #
        # out_dot = np.transpose(np.dot(np.transpose(_hn, (0, 2, 1)), np.transpose(normals, (1, 0))), (0, 2, 1))
        # # for es in range(len(Incid)):
        # #     for j, ix in enumerate(Incid[es, :]):
        # #         _I[_nod, ix] = _I[_nod, ix] + _he[:,es]
        # #     # _I[_nod, con] = _I[_nod, con] + _he[:,es]
        _I = fast_assemble(Incid, _I, _nod, _he)
        _Cc[_nod,_nod] = _Cc[_nod,_nod] + _ce

        rF_rS = np.linalg.norm(rS - rF)
        Pi[_nod, :] = np.exp(-1j * k * rF_rS) / (rF_rS)


    return _I, _Cc, Pi

@njit
def fast_assemble(Incid, _I, _nod, _he):
    for es in range(len(Incid)):
        con = Incid[es, :]
        for j, ix in enumerate(con):
            _I[_nod, ix] = _I[_nod, ix] + _he[j, es]
    return _I


def preconditioner(points, faces, k):
    a, b = computeAB(points, faces)
    a = a.todense()
    min_values = np.amin(points, axis=0)
    max_values = np.amax(points, axis=0)
    total_size = max_values - min_values
    R = np.amax(total_size)
    e = 0.4*k**(1/3)*R**(-2/3)
    V = 1 / (1j * k) * np.sqrt((1 + (a / (k + 1j*e))))
    return V

# @timer_func
def inverter(A,b, precondition = True, drop_tol=1e-4, fill_factor=10):
    if precondition:
        M2 = spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
        M = LinearOperator((len(A), len(A)), M2.solve)
    else:
        M = None
    return lgmres(A, b, M=M, atol=1e-4)

@njit
def incident_field_evaluate(r, r0, k):
    Pi = np.zeros((len(r), len(r0)), dtype=np.complex64)
    for _nod in range(len(r)):
        dis = np.linalg.norm(r[_nod, :] - r0)
        Pi[_nod, :] = np.exp(-1j * k * dis) / dis
    return Pi
msh_path = r'C:\Users\gutoa\Documents\SkaterXL\UFSM\Estagio\Virtual Goniometer\Mshs\square_pyramid.IGS'

# import compiled_bem
AP = fd.AirProperties()
AC = fd.AlgControls(AP,1000,1000,1)
# AC.third_octave_fvec(1000,1000,7)
# AC.freq = np.array([500])
# w = 2*np.pi*AC.freq
# k0 = w/AC.c0
S = fd.Source(coord = [100,0,0.6],q=[1])

R = fd.Receiver()
R.arc(50, (90, -90), "x", n=1, flip=False, center=[0,0,0.6])

grid = fd.GridImport3D(AP,msh_path,fmax=1000,num_freq=6,scale=1000,meshDim=2,plot=False)

obj = fd.BEM3D(grid,S,R,AP,AC,BC=None)

GN = np.array([[1, 0, -1], [0, 1, -1]], dtype=np.float64)

a = 1 / 6
b = 2 / 3
qsi1 = np.array([a, a, b])
qsi2 = np.array([a, b, a]);
weights = np.array([1 / 6, 1 / 6, 1 / 6]).T * 2
N = np.zeros((3, 3), dtype=np.float64)
N[:, 0] = np.transpose(qsi1)
N[:, 1] = np.transpose(qsi2)
N[:, 2] = np.transpose(1 - qsi1 - qsi2)
det_ja, xg, yg, zg = pre_proccess_bem_t3(obj.elem_surf, obj.nos, N, GN)
det_ja = np.asarray(det_ja)
xg = np.asarray(xg)
yg = np.asarray(yg)
zg = np.asarray(zg)
import json
pcd = {}
pcd["elem_surf"] = obj.elem_surf
pcd["vertices"] = obj.nos
pcd["source_position"] = obj.S.coord
# pcd["receiver_position"] = obj.R.coord
pcd["k"] = 5
pcd["normals"] = obj.normals
pcd["det_ja"] = det_ja
pcd["xg"] = xg
pcd["yg"] = yg
pcd["zg"] = zg
pcd["N"] = N
pcd["weights"] = weights
d = list(pcd.values())
# np.save('file.npy', list(pcd.values()))
# with open('pre_compile_data.json', 'w') as outfile:
#     json.dump(pcd, outfile)
# I, Cc , Pi = compiled_bem.assemble(obj.elem_surf.astype(np.int64), obj.nos.astype(np.complex64),
#                                    obj.S.coord.astype(np.complex64), np.complex64(obj.k[0]),
#                                    obj.normals.astype(np.complex64), det_ja.astype(np.complex64), xg.astype(np.complex64),
#                                    yg.astype(np.complex64), zg.astype(np.complex64), N.astype(np.complex64), weights.astype(np.complex64))


# Ic, Ccc , Pic = vectorized(obj.elem_surf, obj.nos, obj.S.coord, obj.k[0], obj.normals, det_ja, xg, yg, zg, N, weights)
# I, Cc , Pi = assemble_bem_3gauss_prepost(*d)
# I, Cc , Pi = assemble_bem_3gauss_prepost(obj.elem_surf, obj.nos, obj.S.coord, obj.k[0], obj.normals, det_ja, xg, yg, zg, N, weights)
# V = preconditioner(obj.nos, obj.elem_surf, obj.k[0])

# I, Cc, Pi = vectorized(obj.elem_surf, obj.nos, obj.S.coord, obj.k[0], obj.normals, det_ja, xg, yg, zg, N, weights)
# C = np.eye(len(obj.nos)) - Cc
# # a = inverter(C+I, Pi.ravel(), False)
# b = inverter(C + I, Pi.ravel(), True, 1e-5, 9)


# pC,info = lgmres((C+I),(Pi[:,0]), M = V)

I, Cc, Pi = vectorized(obj.elem_surf, obj.nos, obj.S.coord, obj.k[0], obj.normals, det_ja, xg, yg, zg, N, weights)
P = incident_field_evaluate(obj.nos, obj.S.coord, obj.k[0])
# C = np.eye(len(obj.nos)) - Cc
# # a = inverter(C+I, Pi.ravel(), False)
# b = inverter(C + I, Pi.ravel(), True, 1e-5, 9)

# pS = np.array(pss)
# pfs = np.mean(np.abs(pS),axis=0)
# pS = pfs
#%%
data = np.genfromtxt(r'G:\My Drive\R & D\REDI_RPG_Intership\Virtual_Goniometer\Hemicylinder\1khz_normal_hemi_120cm_30cm.txt', delimiter='	')
# pS = np.array(pss)
# pfs = np.mean(np.abs(pS),axis=0)
# pS = pfs
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# ax.set_thetamin(np.amin(np.rad2deg(R.theta)))
# ax.set_thetamax(np.amax(np.rad2deg(R.theta)))
# ax.set_ylim([-60,2])

ax.plot(np.linspace(0,np.pi,180), fd.p2SPL(pS[0].ravel()))
ax.plot(np.linspace(0,np.pi,90), data)
a=1