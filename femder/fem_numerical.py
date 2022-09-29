import numpy as np
import numba
from numba import jit, njit
from scipy.sparse import coo_matrix, csc_matrix
from tqdm import tqdm



def node_pressure_interpolation(vertices, elem_vol, receiver_coord, node_pressure_value):
    shape_function_interp = []
    prob_elem, prob_ind = probable_elem(vertices, elem_vol, receiver_coord)
    which_indx = which_tetra(vertices, prob_elem, receiver_coord)
    elem_indx = prob_ind[which_indx]
    con = elem_vol[elem_indx, :][0]
    elem_coord = vertices[con, :]
    grad_interp = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    ja = (grad_interp @ elem_coord).T

    interp_coord = receiver_coord - elem_coord[0, :]
    local_coord = (np.linalg.inv(ja) @ interp_coord)
    shape_function = np.array(
        [[1 - local_coord[0] - local_coord[1] - local_coord[2]], [local_coord[0]], [local_coord[1]], [local_coord[2]]])

    for i in range(len(node_pressure_value[:, 0])):
        shape_function_interp.append((shape_function.T @ node_pressure_value[i, con].T).T)
    return np.asarray(shape_function_interp).ravel()


def probable_elem(vertices, elem_vol, receiver_coord):
    cl1 = closest_node(vertices, receiver_coord)
    eln = np.where(elem_vol == cl1)
    pelem = elem_vol[eln[0]]
    return pelem, eln[0]


def which_tetra(node_coordinates, node_ids, p):
    ori = node_coordinates[node_ids[:, 0], :]
    v1 = node_coordinates[node_ids[:, 1], :] - ori
    v2 = node_coordinates[node_ids[:, 2], :] - ori
    v3 = node_coordinates[node_ids[:, 3], :] - ori
    n_tet = len(node_ids)
    v1r = v1.T.reshape((3, 1, n_tet))
    v2r = v2.T.reshape((3, 1, n_tet))
    v3r = v3.T.reshape((3, 1, n_tet))
    mat = np.concatenate((v1r, v2r, v3r), axis=1)
    inv_mat = np.linalg.inv(mat.T).T
    if p.size == 3:
        p = p.reshape((1, 3))
    n_p = p.shape[0]
    orir = np.repeat(ori[:, :, np.newaxis], n_p, axis=2)
    newp = np.einsum('imk,kmj->kij', inv_mat, p.T - orir)
    val = np.all(newp >= 0, axis=1) & np.all(newp <= 1, axis=1) & (np.sum(newp, axis=1) <= 1)
    id_tet, id_p = np.nonzero(val)
    res = -np.ones(n_p, dtype=id_tet.dtype)  # Sentinel value
    res[id_p] = id_tet
    return res


def closest_node(nodes, node):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def tetra_grad_shape_functions(order, local_coord=None):
    """
    Generates FEM shape function gradient for 4 and 10 node tetrahedron
    Parameters
    ----------
    order: int
        Mesh order, linear (1) or quadratic (2)
    order: int
        Mesh order, linear (1) or quadratic (2)

    Returns: array
        Shape function gradient
    -------

    """

    if order == 1:
        return np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]], dtype=np.complex64)
    if order == 2:
        pass


def tet10_interp(local_coord):  #TODO not being used
    t1 = local_coord[0]
    t2 = local_coord[1]
    t3 = local_coord[2]
    t4 = 1 - local_coord[0] - local_coord[1] - local_coord[2];
    # print(t1)
    N = np.array([t4 * (2 * t4 - 1), t1 * (2 * t1 - 1), t2 * (2 * t2 - 1), t3 * (2 * t3 - 1),
                  4 * t1 * t4, 4 * t1 * t2, 4 * t2 * t4, 4 * t3 * t4, 4 * t2 * t3, 4 * t3 * t1]);
    return N[:, np.newaxis]


@jit
def tri10_interp(local_coord):  #TODO not being used
    N = np.array([(-local_coord[0] - local_coord[1] + 1) * (2 * (-local_coord[0] - local_coord[1] + 1) - 1),
                  local_coord[0] * (2 * local_coord[0] - 1),
                  local_coord[1] * (2 * local_coord[1] - 1),
                  4 * local_coord[0] * local_coord[1],
                  4 * local_coord[1] * (-local_coord[0] - local_coord[1] + 1),
                  4 * local_coord[0] * (-local_coord[0] - local_coord[1] + 1)])

    return N[:, np.newaxis]  # ,deltaN


@jit
def tetrahedron_10_delta_interp(local_coord):  #TODO not being used
    t1 = 4 * local_coord[0]
    t2 = 4 * local_coord[1]
    t3 = 4 * local_coord[2]
    delta_n = np.array(
        [[t1 + t2 + t3 - 3, t1 + t2 + t3 - 3, t1 + t2 + t3 - 3], [t1 - 1, 0, 0], [0, t2 - 1, 0], [0, 0, t3 - 1],
         [4 - t2 - t3 - 2 * t1, -t1, -t1], [t2, t1, 0], [-t2, 4 - 2 * t2 - t3 - t1, -t2],
         [-t3, -t3, 4 - t2 - 2 * t3 - t1], [0, t3, t2], [t3, 0, t1]])

    return delta_n.T


@numba.jit
def inv_nla_jit(A):
    return np.linalg.inv(A)


def pre_compute_volume_assemble_vars(elem_vol, vertices, order):
    elem_coord = vertices[elem_vol]
    grad_shape_function = tetra_grad_shape_functions(order, None)
    ja = ja_pre(elem_coord, grad_shape_function)
    det_ja = np.linalg.det(ja)
    arg_stiff = arg_stiff_pre(ja, grad_shape_function) * det_ja

    return det_ja, arg_stiff


def arg_stiff_pre(ja, grad_shape_function):

    grad_inv_ja = np.einsum("ijk, jlk -> ilk", np.linalg.inv(ja).transpose(1, 2, 0),
                            np.broadcast_to(grad_shape_function[:, :, None], (grad_shape_function.shape[0], grad_shape_function.shape[1]
                                                                      , ja.shape[0])), optimize=True)
    arg_stiff = np.einsum("ijk, ilk -> ljk", grad_inv_ja, grad_inv_ja)

    return arg_stiff


def ja_pre(elem_coord, grad_interp):
    return np.einsum("ij, kjl -> ilk", grad_interp, elem_coord).transpose(2, 0, 1)


def gauss_4_points():
    """
    Initializes the 4 points for gauss integration in tetra4

    Returns
    -------
    ptx: array
        X coordinate for gauss integration
    pty: array
        Y coordinate for gauss integration
    ptz: array
        Z coordinate for gauss integration
    """
    a = 0.5854101966249685  # (5-np.sqrt(5))/20
    b = 0.1381966011250105  # (5-3*np.sqrt(5))/20 #
    ptx = np.array([a, b, b, b])
    pty = np.array([b, a, b, b])
    ptz = np.array([b, b, a, b])
    return ptx, pty, ptz


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





# @helpers.timer_func
# def assemble_volume_matrices(elem_vol, vertices, fluid_c, fluid_rho, order, domain_indices_vol, frequency_index,
#                              det_ja, arg_stiff):
#     """
#     Assemble mass and stiffness FEM matrices.
#
#     Parameters
#     ----------
#     arg_stiff
#     det_ja
#     elem_vol: array
#         Mesh connectivity for volume elements
#     vertices: array
#         Mesh vertices
#     fluid_c: dict
#         Speed ou sound for current element
#     fluid_rho: dict
#         Fluid density for current element
#     order: int
#         Mesh order, linear (1) or quadratic (2)
#     domain_indices_vol: array
#         Domain indices for domains
#     frequency_index: int
#         Index of frequency vector
#     Returns
#     -------
#     Returns stiffness and mass matrices for FEM calculations
#
#     """
#     characteristic_matrix_dimension = None
#     gauss_integration = None
#     ptx, pty, ptz = None, None, None
#     len_elem_vol = len(elem_vol)
#     len_vertices = len(vertices)
#     if order == 1:
#         characteristic_matrix_dimension = 4
#         gauss_integration = integration_tetra4_4gauss
#         ptx, pty, ptz = gauss_4_points()
#     if order == 2:
#         characteristic_matrix_dimension = 10
#         gauss_integration = integration_tetra10_4gauss
#
#     stiff_elem = np.zeros([characteristic_matrix_dimension,
#                            characteristic_matrix_dimension,
#                            len_elem_vol], dtype=np.complex64)
#     mass_elem = np.zeros([characteristic_matrix_dimension,
#                           characteristic_matrix_dimension,
#                           len_elem_vol], dtype=np.complex64)
#
#     for e in range(len_elem_vol):
#         c_elem = fluid_c[domain_indices_vol[e]][frequency_index]
#         rho_elem = fluid_rho[domain_indices_vol[e]][frequency_index]
#         arg_stiff_elem = (1 / rho_elem) * arg_stiff[:, :, e]
#         det_ja_elem = det_ja[e]
#         stiff_temp, mass_temp = gauss_integration(np.complex64(c_elem), numba.complex64(rho_elem),
#                                                   numba.complex64(arg_stiff_elem), det_ja_elem, ptx, pty, ptz)
#         stiff_elem[:, :, e] = stiff_temp
#         mass_elem[:, :, e] = mass_temp
#
#     _nlb = np.size(stiff_elem, 1)
#     assemble_y = np.matlib.repmat(elem_vol[0:len_elem_vol, :], 1, _nlb).T.reshape(_nlb, _nlb, len_elem_vol)
#     assemble_x = np.transpose(assemble_y, (1, 0, 2))
#     stiff_global = coo_matrix((stiff_elem.ravel(),
#                                (assemble_x.ravel(), assemble_y.ravel())),
#                               shape=[len_vertices, len_vertices])
#     mass_global = coo_matrix((mass_elem.ravel(),
#                               (assemble_x.ravel(), assemble_y.ravel())),
#                              shape=[len_vertices, len_vertices])
#
#     stiff_global = stiff_global.tocsc()
#     mass_global = mass_global.tocsc()
#
#     return stiff_global, mass_global


def assemble_volume_matrices(elem_vol, vertices, fluid_c, fluid_rho, order, domain_indices_vol, frequency_index,
                             det_ja, arg_stiff):
    """
    Assemble mass and stiffness FEM matrices.

    Parameters
    ----------
    arg_stiff
    det_ja
    elem_vol: array
        Mesh connectivity for volume elements
    vertices: array
        Mesh vertices
    fluid_c: dict
        Speed ou sound for current element
    fluid_rho: dict
        Fluid density for current element
    order: int
        Mesh order, linear (1) or quadratic (2)
    domain_indices_vol: array
        Domain indices for domains
    frequency_index: int
        Index of frequency vector
    Returns
    -------
    Returns stiffness and mass matrices for FEM calculations

    """
    characteristic_matrix_dimension = None
    # gauss_integration = None
    ptx, pty, ptz = None, None, None
    len_elem_vol = len(elem_vol)
    len_vertices = len(vertices)
    if order == 1:
        characteristic_matrix_dimension = 4
        # gauss_integration = integration_tetra4_4gauss
        ptx, pty, ptz = gauss_4_points()
    if order == 2:
        raise ValueError("order must be 1")

    stiff_elem = np.zeros([characteristic_matrix_dimension,
                           characteristic_matrix_dimension,
                           len_elem_vol], dtype=np.complex64)
    mass_elem = np.zeros([characteristic_matrix_dimension,
                          characteristic_matrix_dimension,
                          len_elem_vol], dtype=np.complex64)

    shape_function = np.array(
        [[1 - ptx - pty - ptz],
         [ptx],
         [pty],
         [ptz]],
        dtype=np.float64).T

    shape_function = np.einsum("ijk, ikj ->ijk", shape_function, shape_function)
    shape_function = shape_function.sum(axis=0)

    for i, j in enumerate(np.unique(domain_indices_vol)):
        indx = np.argwhere(domain_indices_vol == j).ravel()
        stiff_elem[:, :, indx] = (1 / fluid_rho[j][frequency_index]) * arg_stiff[:, :, indx] / 6
        mass_elem[:, :, indx] = np.broadcast_to(shape_function[:, :, None],
                                                (characteristic_matrix_dimension, characteristic_matrix_dimension,
                                                 len(indx))) * det_ja[indx] * (1 / (fluid_rho[j][frequency_index] *
                                                                                    fluid_c[j][frequency_index] ** 2))/24

    assemble_y = np.broadcast_to(elem_vol[:,:,None], (elem_vol.shape[0], characteristic_matrix_dimension,
                                                      characteristic_matrix_dimension,)).T.reshape(characteristic_matrix_dimension,
                                                                                                   characteristic_matrix_dimension,
                                                                                                   elem_vol.shape[0])
    assemble_x = np.transpose(assemble_y, (1, 0, 2))
    stiff_global = csc_matrix((stiff_elem.ravel(),
                               (assemble_x.ravel(), assemble_y.ravel())),
                              shape=[len_vertices, len_vertices])

    mass_global = csc_matrix((mass_elem.ravel(),
                              (assemble_x.ravel(), assemble_y.ravel())),
                             shape=[len_vertices, len_vertices])

    return stiff_global, mass_global

def assemble_from_elem_matrices(stiff_elem, mass_elem, elem_vol, characteristic_matrix_dimension, len_vertices):

    assemble_y = np.broadcast_to(elem_vol[:,:,None], (elem_vol.shape[0], characteristic_matrix_dimension,
                                                      characteristic_matrix_dimension,)).T.reshape(characteristic_matrix_dimension,
                                                                                                   characteristic_matrix_dimension,
                                                                                                   elem_vol.shape[0])
    assemble_x = np.transpose(assemble_y, (1, 0, 2))
    stiff_global = csc_matrix((stiff_elem.ravel(),
                               (assemble_x.ravel(), assemble_y.ravel())),
                              shape=[len_vertices, len_vertices])
    mass_global = csc_matrix((mass_elem.ravel(),
                              (assemble_x.ravel(), assemble_y.ravel())),
                             shape=[len_vertices, len_vertices])

    return stiff_global, mass_global


@njit
def integration_tetra4_4gauss(c0, rho0, arg_stiff, det_ja, ptx, pty, ptz):
    """
    Gauss integration for 4 points and tetra4
    Parameters
    ----------
    c0: float
        Speed of sound
    rho0:
        Fluid density
    arg_stiff:
        Stiffness matrix argument
    det_ja:
        Determinant of the jacobian
    ptx: array
        X coordinate for gauss integration
    pty: array
        Y coordinate for gauss integration
    ptz: array
        Z coordinate for gauss integration

    Returns
    -------

    """
    stiff_elem = np.zeros((4, 4), dtype=np.complex64)
    mass_elem = np.zeros((4, 4), dtype=np.complex64)

    for n_gauss in range(4):
        shape_function = np.array(
            [[1 - ptx[n_gauss] - pty[n_gauss] - ptz[n_gauss]], [ptx[n_gauss]], [pty[n_gauss]], [ptz[n_gauss]]],
            dtype=np.complex64)
        arg_mass = (1 / (rho0 * c0 ** 2)) * np.dot(shape_function, shape_function.transpose()) * det_ja
        stiff_elem += 1 / 24 * arg_stiff
        mass_elem += 1 / 24 * arg_mass
    return stiff_elem, mass_elem


@njit
def integration_tetra10_4gauss(c0, rho0, arg_stiff, det_ja, ptx, pty, ptz):
    """
    Gauss integration for 4 points and tetra10
    Parameters
    ----------
    c0: float
        Speed of sound
    rho0:
        Fluid density
    arg_stiff:
        Stiffness matrix argument
    det_ja:
        Determinant of the jacobian
    ptx: array
        X coordinate for gauss integration
    pty: array
        Y coordinate for gauss integration
    ptz: array
        Z coordinate for gauss integration

    Returns
    -------

    """
    pass


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
        raise ValueError("order must be 1")
    damp_global = []
    shape_function = np.array([np.broadcast_to(ptx[:, None], (3, 3)).T,
                               np.broadcast_to(pty[:, None], (3, 3)),
                               1 - ptx - np.broadcast_to(pty[:, None], (3, 3))]).transpose(2, 0, 1)
    _damp_elem = gauss_integration(shape_function)
    len_vertices = len(vertices)
    for bl in tqdm(unique_domain_indices, desc="FEM | Assembling surface matrix", bar_format='{l_bar}{bar:25}{r_bar}',
                   disable=rasta.HIDE_PBAR):
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

@njit
def solve_modal_superposition(indR, indS, F_n, Vc, Vc_T, w, qindS, N, hn, Mn, ir):
    lenS = numba.int64(len(indS))
    lenfn = numba.int64(len(F_n))

    An = np.zeros((1, 1), dtype=np.complex64)
    for ii in range(lenS):
        for e in range(lenfn):
            wn = F_n[e] * 2 * np.pi
            # print(self.Vc[indS[ii],e].T*(1j*self.w[N]*qindS[ii])*self.Vc[indR,e])
            # print(((wn-self.w[N])*Mn[e]))
            An[0] += Vc_T[e, indS[ii]] * (1j * w[N] * qindS[ii]) * Vc[indR[ir], e] / (
                    (wn ** 2 - w[N] ** 2) * Mn[e] + 1j * hn[e] * w[N])
            # An[0] += Vc_T[e,2]*(1j*w[N]*qindS[ii])*Vc[2,e]/((wn**2-w[N]**2)*Mn[e]+1j*hn[e]*w[N])
    return An[0]


def pre_proccess_bem_t3(surface_elements, vertices, shape_function, grad_shape_function):
    xg = []
    yg = []
    zg = []
    det_ja = []
    for es in range(len(surface_elements)):
        con = surface_elements[es, :]
        coord_el = vertices[con, :]

        x_0 = coord_el[:, 0]
        y_0 = coord_el[:, 1]
        z_0 = coord_el[:, 2]
        dx0 = x_0[1] - x_0[0]
        dx1 = x_0[2] - x_0[1]
        dx2 = x_0[0] - x_0[2]
        dy0 = y_0[1] - y_0[0]
        dy1 = y_0[2] - y_0[1]
        dy2 = y_0[0] - y_0[2]
        dz0 = z_0[1] - z_0[0]
        dz1 = z_0[2] - z_0[1]
        dz2 = z_0[0] - z_0[2]

        a = np.sqrt(dx0 ** 2 + dy0 ** 2 + dz0 ** 2)
        b = np.sqrt(dx1 ** 2 + dy1 ** 2 + dz1 ** 2)
        c = np.sqrt(dx2 ** 2 + dy2 ** 2 + dz2 ** 2)

        ang = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

        coord_xy = np.array([[0, a, b * np.cos(ang)], [0, 0, b * np.sin(ang)]]).T
        xg.append(np.dot(shape_function, coord_el[:, 0]))
        yg.append(np.dot(shape_function, coord_el[:, 1]))
        zg.append(np.dot(shape_function, coord_el[:, 2]))
        ja = np.dot(grad_shape_function, coord_xy)

        det_ja.append(np.linalg.det(ja) / 2)

    return det_ja, xg, yg, zg


def assemble_bem_matrices(surface_elements, vertices, source_positions, w0, k0, rho0, normals, areas):  #TODO not being used
    len_vertices = numba.int64(len(vertices))
    green_s = np.zeros((len_vertices, len_vertices), dtype=np.complex64)
    grad_green = np.zeros((len_vertices, len_vertices), dtype=np.complex64)
    colocation = np.zeros((len_vertices, len_vertices), dtype=np.complex64)

    incident_pressure = np.zeros((len_vertices, len(source_positions)), dtype=np.complex64)
    grad_shape_function = np.array([[1, 0, -1], [0, 1, -1]], dtype=np.float64)

    a = 1 / 6
    b = 2 / 3

    local_coord_x = np.array([a, a, b])
    local_coord_y = np.array([a, b, a])

    weights = np.array([1 / 6, 1 / 6, 1 / 6]).T * 2

    shape_function = np.zeros((3, 3), dtype=np.float64)
    shape_function[:, 0] = np.transpose(local_coord_x)
    shape_function[:, 1] = np.transpose(local_coord_y)
    shape_function[:, 2] = np.transpose(1 - local_coord_x - local_coord_y)
    det_ja, xg, yg, zg = pre_proccess_bem_t3(surface_elements, vertices, shape_function, grad_shape_function)

    for nod in tqdm(range(len(vertices))):
        r_field = vertices[nod, :]
        for i in range(len(source_positions)):
            r_source_field = np.linalg.norm(r_field - source_positions[i, :])
            incident_pressure[nod, i] = np.exp(-1j * k0 * r_source_field) / r_source_field

        for es in range(len(surface_elements)):
            con = surface_elements[es, :]
            normal = numba.complex64(normals[es, :])
            green_elem, grad_green_elem, colocation_elem = assemble_bem_element_matrices(r_field, k0,
                                                                                         normal, shape_function,
                                                                                         weights, det_ja[es], xg[es],
                                                                                         yg[es], zg[es])
            green_s[nod, con] = green_s[nod, con] + green_elem
            grad_green[nod, con] = grad_green[nod, con] + grad_green_elem
            colocation[nod, nod] = colocation[nod, nod] + colocation_elem

    return green_s, grad_green, colocation, incident_pressure


@njit
def assemble_bem_element_matrices(vertex, k0, normal, shape_function, weights, det_ja, xg, yg, zg):
    green_elem = np.zeros((3,), dtype=np.complex64)
    grad_green_elem = np.zeros((3,), dtype=np.complex64)
    colocation_elem = np.zeros((3,), dtype=np.complex64)

    x_dist = xg - vertex[0]
    y_dist = yg - vertex[1]
    z_dist = zg - vertex[2]
    dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
    g_top = np.exp(-1j * k0 * dist)
    green_function = g_top / (4 * np.pi * dist)

    green_elem[0] = np.sum(np.sum(green_function * shape_function[:, 0] * det_ja * weights) * weights)
    green_elem[1] = np.sum(np.sum(green_function * shape_function[:, 1] * det_ja * weights) * weights)
    green_elem[2] = np.sum(np.sum(green_function * shape_function[:, 2] * det_ja * weights) * weights)

    h1 = -x_dist * np.exp(-1j * k0 * dist) / (4 * np.pi * dist ** 2) * (1j * k0 + 1 / dist)
    h2 = -y_dist * np.exp(-1j * k0 * dist) / (4 * np.pi * dist ** 2) * (1j * k0 + 1 / dist)
    h3 = -z_dist * np.exp(-1j * k0 * dist) / (4 * np.pi * dist ** 2) * (1j * k0 + 1 / dist)

    hn = np.array([[h1[0], h1[2], h1[2]], [h2[0], h2[2], h2[2]], [h3[0], h3[2], h3[2]]]).T
    h = np.dot(hn, normal.T)

    grad_green_elem[0] = np.sum(np.sum(-h * shape_function[:, 0] * det_ja * weights) * weights)
    grad_green_elem[1] = np.sum(np.sum(-h * shape_function[:, 1] * det_ja * weights) * weights)
    grad_green_elem[2] = np.sum(np.sum(-h * shape_function[:, 2] * det_ja * weights) * weights)

    c1 = -x_dist / (4 * np.pi * dist ** 2) * (1 / dist)
    c2 = -y_dist / (4 * np.pi * dist ** 2) * (1 / dist)
    c3 = -z_dist / (4 * np.pi * dist ** 2) * (1 / dist)
    cn = np.array([[c1[0], c1[1], c1[2]], [c2[0], c2[1], c2[2]], [c3[0], c3[1], c3[2]]], dtype=np.complex64).T
    cc = np.dot(cn, normal.T)

    colocation_elem[0] = np.sum(np.sum(-cc * shape_function[:, 0] * det_ja * weights) * weights)
    colocation_elem[1] = np.sum(np.sum(-cc * shape_function[:, 1] * det_ja * weights) * weights)
    colocation_elem[2] = np.sum(np.sum(-cc * shape_function[:, 2] * det_ja * weights) * weights)
    colocation_elem = np.sum(colocation_elem)

    return green_elem, grad_green_elem, colocation_elem


@njit
def evaluate_field_bem(receiver_positions, surface_elements, vertices, source_positions, k0, normals, areas):
    len_elements = numba.int64((len(surface_elements)))
    len_receivers = numba.int64((len(receiver_positions)))
    global_evaluation_matrix = np.zeros((len_receivers, len_elements), dtype=np.complex64)
    identity = np.zeros((len_receivers, len_elements), dtype=np.complex64)
    incident_pressure_evaluate = np.zeros((len_receivers, 1), dtype=np.complex64)
    for i in range(len_receivers):
        r_receiver = receiver_positions[i, :]
        r_field_source = np.linalg.norm(r_receiver - source_positions)
        incident_pressure_evaluate[i] = np.exp(-1j * k0 * r_field_source) / r_field_source

        for es in range(len_elements):
            r_source = vertices[surface_elements[es, :], :]
            r_source_receiver = np.sqrt((r_source[:, 0] - r_receiver[0]) ** 2
                                        + (r_source[:, 1] - r_receiver[1]) ** 2
                                        + (r_source[:, 2] - r_receiver[2]) ** 2)
            area = areas[es]
            normal = normals[es, :]

            dG = -((r_source - r_receiver) * (np.exp(-1j * k0 * r_source_receiver)
                                              / (4 * np.pi * r_source_receiver ** 2)) * (
                           1j * k0 + 1 / r_source_receiver))

            dGdN = np.dot(dG, normal.T)

            Gc = np.exp(-1j * k0 * r_source_receiver) / (4 * np.pi * r_source_receiver)
            global_evaluation_matrix[i, es] = np.mean(Gc) * area
            identity[i, es] = np.mean(dGdN) * area

    return global_evaluation_matrix, identity, incident_pressure_evaluate


def evaluate_pressure_field(boundary_pressure, identity, incident_pressure_evaluate):
    pScat = identity @ boundary_pressure[np.newaxis, :]
    pTotal = pScat + incident_pressure_evaluate
    return pTotal, pScat


def evaluate_wrapper(boundary_pressure, receiver_positions, surface_elements,
                     vertices, source_positions, k0, normals, areas):  # Not being used
    global_evaluation_matrix, identity, incident_pressure_evaluate = evaluate_field_bem(receiver_positions,
                                                                                        surface_elements, vertices,
                                                                                        source_positions, k0,
                                                                                        numba.complex64(normals),
                                                                                        areas)
    pressure_total, pressure_scattered = evaluate_pressure_field(boundary_pressure,
                                                                 identity, incident_pressure_evaluate)
    return pressure_total, pressure_scattered
