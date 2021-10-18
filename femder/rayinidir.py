import numpy as np
# import quaternion as qua
#from bemder.tessellation import SphereTessellator

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

#from ra.log import log

import numpy as np
from scipy import spatial as sp_spatial


class SphereTessellator():

    def __init__(self, nverts=12, depth=0):
        '''Tessellates a sphere with a number of vertices greater or equal than
        `nverts`.
        If `depth` is greater than 0 then `nverts` is ignored and the number of
        vertices is a consequence of `depth`.
        The returned object exposes the sphere vertices and indices through
        a property named `sphere`.
        '''
        self.vertices, self.indices = self.icosahedron()
        if depth > 0:
            self.depth = depth
        elif nverts >= 12:
            self.depth = self.nverts2depth(nverts)
        else:
            msg = (
                'SphereTessellator was passed nverts:{} and depth:{}, but '
                'expected nverts >= 12 or depth >=0'.format(nverts, depth)
            )
            raise SphereTessellatorBadArgsError(msg)

        self.iterate()

    def nverts2depth(self, nverts):
        '''Returns the minimum number of iterations, `depth`, to tessellate a
        sphere with `nverts` vertices'''
        nv = 12  # num of vertices of an icosahedron
        nf = 20  # num of faces of an icosahedron
        ne = 30  # num of edges of an icosahedron
        depth = 1  # iteration
        while nv < nverts:
            depth += 1
            nv += ne
            ne = 2*ne + 3*nf
            nf *= 4
        return depth

    def icosahedron(self,):
        x = .525731112119133606
        z = .850650808352039932
        vertices = np.array([
            [-x, 0.0, z], [x, 0.0, z], [-x, 0.0, -z], [x, 0.0, -z],
            [0.0, z, x], [0.0, z, -x], [0.0, -z, x], [0.0, -z, -x],
            [z, x, 0.0], [-z, x, 0.0], [z, -x, 0.0], [-z, -x, 0.0]
        ], dtype=np.float32)
        hull = sp_spatial.ConvexHull(vertices)
        indices = hull.simplices
        return vertices, indices

    def norm(self, arr):
        arr[:] = (arr.T / np.sqrt(np.sum(arr**2, axis=1))).T

    def iterate(self,):
        vs, ids = self.vertices, self.indices
        for _ in range(self.depth - 1):
            v0 = vs[ids[:, 0]]
            v1 = vs[ids[:, 1]]
            v2 = vs[ids[:, 2]]
            a = (v0 + v2) * 0.5
            b = (v0 + v1) * 0.5
            c = (v1 + v2) * 0.5
            self.norm(a)
            self.norm(b)
            self.norm(c)
            vs = np.unique(np.concatenate((vs, a, b, c)), axis=0)
            hull = sp_spatial.ConvexHull(vs)
            ids = hull.simplices
        self.vertices, self.indices = vs, ids

    @property
    def sphere(self,):
        return self.vertices, self.indices


class SphereTessellatorBadArgsError(Exception):
    '''Exception raised when the constructor is passed bad arguments.'''
    pass


if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as a3
    import scipy as sp

    tess = SphereTessellator(
        # nverts=12,
        depth=1
    )
    vertices, indices = tess.sphere
    # vertices *= 0.5

    hull = sp_spatial.ConvexHull(vertices)
    indices = hull.simplices
    faces = vertices[indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.dist = 30
    ax.azim = -140
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for f in faces:
        face = a3.art3d.Poly3DCollection([f])
        face.set_edgecolor('k')
        face.set_alpha(0.5)
        ax.add_collection3d(face)

    plt.show()

class RayInitialDirections():
    '''
    This class is used to initialize the ray directions. It returns
    a matrix of N_rays x 3 with unit vectors that are initial ray
    directions bein emmited by the sources. It contains several methods.
    '''
    def __init__(self):
        pass

    def single_ray(self, direction = [1.0, 0.0, 0.0]):
        '''
        This method is used to define a single ray direction.
        '''
        vinit = []
        vinit.append(direction)
        self.vinit = np.array(vinit, dtype=np.float32)
        self.vinit /= np.linalg.norm(self.vinit)
        self.Nrays = 1
        return self.vinit, self.Nrays

    def single_ray_elaz(self, elevation = 0.0, azimuth = 0.0):
        el = np.deg2rad(elevation)
        az = np.deg2rad(azimuth)
        vinit = []
        direction = [np.cos(az) * np.sin(el), np.sin(az) * np.sin(el), np.cos(el)]
        vinit.append(direction)
        self.vinit = np.array(vinit, dtype=np.float32)
        self.vinit /= np.linalg.norm(self.vinit)
        self.Nrays = 1
        return self.vinit, self.Nrays

    def random_rays(self, Nrays):
        '''
        This method defines ray directions that are random numbers
        on the surface of a unit sphere. The number of rays returned is
        the same as the number of rays provided by the user. This
        source is probabilistic
        '''
        rand_values = 2.0 * np.random.rand(Nrays) - 1.0
        elevation = np.arcsin(rand_values)
        azimuth = 2 * np.pi * np.random.rand(Nrays)
        self.vinit = np.zeros((Nrays, 3), dtype=np.float32)
        self.vinit[:,0] = np.cos(elevation) * np.sin(azimuth)
        self.vinit[:,1] = np.sin(elevation) * np.sin(azimuth)
        self.vinit[:,2] = np.cos(azimuth)

        self.vinit /= np.linalg.norm(self.vinit, axis = 1)[:,None]
        self.Nrays = Nrays
        return self.vinit, self.Nrays

    def isotropic_rays(self, Nrays = 12, depth=1):
        '''
        This method defines ray directions calculated according to the
        tesselation of an icosahedron. The number of rays returned is
        always bigger than the number of rays provided by the user. This
        source is deterministic
        '''
        tess = SphereTessellator(
            nverts=Nrays,
            # depth=depth
        )
        self.vinit, self.indices = tess.sphere
        self.vinit /= np.linalg.norm(self.vinit, axis = 1)[:,None]
        self.vinit = np.array(self.vinit, dtype=np.float32)
        self.Nrays = self.vinit.shape[0]
        # log.info(self.Nrays)
        return self.vinit, self.Nrays

    def conical_rays(self, Nrays, radius, direction=(1, 0, 0), tol = 1e-5):
        '''
        This method defines ray directions that are random numbers
        calculated on the circle base of a cone. The number of rays
        returned is the same as the number of rays provided by the user.
        The source is probabilistic. User should input:
        1: radius  - radius of the cone. The smaller it is, the more
            directional the source is
        2: direction - direction of the cone source.
        '''
        a = np.random.rand(Nrays) * 2.0 * np.pi
        r = radius * np.sqrt(np.random.rand(Nrays))
        self.vinit = np.ones((Nrays, 3), dtype=np.float32)
        self.vinit[:, 1] = r * np.sin(a)
        self.vinit[:, 2] = r * np.cos(a)
        self.vinit /= np.linalg.norm(self.vinit, axis = 1)[:,None]
        direction = np.array(direction, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        dot = np.dot([1, 0, 0], direction)
        if dot + 1 < tol:
            q = np.quaternion(0, 0, 1, 0)
        elif dot -1 > tol:
            q = np.quaternion(1, 0, 0, 0)
        else:
            vcross = np.cross([1, 0, 0], direction)
            q = np.quaternion(1, 0, 0, 0)
            (q.x, q.y, q.z), q.w = vcross, (dot+1.0)
            q = q.normalized()
        rotmat = qua.as_rotation_matrix(q)
        self.vinit = self.vinit.dot(rotmat.T)
        self.vinit /= np.linalg.norm(self.vinit, axis = 1)[:,None]
        self.Nrays = Nrays
        return self.vinit, self.Nrays


    def circlexy_rays(self, Nrays, direction=(1, 0, 0), tol = 1e-5):
        '''
        This method defines ray directions calculated according to the
        sub-division of a circle in xy plane. The number of rays returned is
        the same as the number of rays provided by the user. This
        source is deterministic.
        '''
        theta = np.arange(0.0, 2*np.pi, 2.0 * np.pi / Nrays)
        self.vinit = np.zeros((Nrays, 3), dtype=np.float32)
        self.vinit[:,0] = np.cos(theta)
        self.vinit[:,1] = np.sin(theta)
        direction = np.array(direction, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        dot = np.dot([1, 0, 0], direction)
        if dot + 1 < tol:
            q = np.quaternion(0, 0, 1, 0)
        elif dot -1 > tol:
            q = np.quaternion(1, 0, 0, 0)
        else:
            vcross = np.cross([1, 0, 0], direction)
            q = np.quaternion(1, 0, 0, 0)
            (q.x, q.y, q.z), q.w = vcross, (dot+1.0)
            q = q.normalized()
        rotmat = qua.as_rotation_matrix(q)
        self.vinit = self.vinit.dot(rotmat.T)
        self.vinit /= np.linalg.norm(self.vinit, axis = 1)[:,None]
        self.Nrays = Nrays
        return self.vinit, self.Nrays

    def circlexz_rays(self, Nrays, direction=(1, 0, 0), tol = 1e-5):
        '''
        This method defines ray directions calculated according to the
        sub-division of a circle in xz plane. The number of rays returned is
        the same as the number of rays provided by the user. This
        source is deterministic.
        '''
        theta = np.arange(0.0, 2*np.pi, 2.0 * np.pi / Nrays)
        self.vinit = np.zeros((Nrays, 3), dtype=np.float32)
        self.vinit[:,0] = np.cos(theta)
        self.vinit[:,2] = np.sin(theta)
        direction = np.array(direction, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        dot = np.dot([1, 0, 0], direction)
        if dot + 1 < tol:
            q = np.quaternion(0, 0, 1, 0)
        elif dot -1 > tol:
            q = np.quaternion(1, 0, 0, 0)
        else:
            vcross = np.cross([1, 0, 0], direction)
            q = np.quaternion(1, 0, 0, 0)
            (q.x, q.y, q.z), q.w = vcross, (dot+1.0)
            q = q.normalized()
        rotmat = qua.as_rotation_matrix(q)
        self.vinit = self.vinit.dot(rotmat.T)
        self.vinit /= np.linalg.norm(self.vinit, axis = 1)[:,None]
        self.Nrays = Nrays
        return self.vinit, self.Nrays

    def circleyz_rays(self, Nrays, direction=(1, 0, 0), tol = 1e-5):
        '''
        This method defines ray directions calculated according to the
        sub-division of a circle in yz plane. The number of rays returned is
        the same as the number of rays provided by the user. This
        source is deterministic.
        '''
        theta = np.arange(0.0, 2*np.pi, 2.0 * np.pi / Nrays)
        self.vinit = np.zeros((Nrays, 3), dtype=np.float32)
        self.vinit[:,1] = np.cos(theta)
        self.vinit[:,2] = np.sin(theta)
        self.vinit /= np.linalg.norm(self.vinit, axis = 1)[:,None]
        direction = np.array(direction, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        dot = np.dot([1, 0, 0], direction)
        if dot + 1 < tol:
            q = np.quaternion(0, 0, 1, 0)
        elif dot -1 > tol:
            q = np.quaternion(1, 0, 0, 0)
        else:
            vcross = np.cross([1, 0, 0], direction)
            q = np.quaternion(1, 0, 0, 0)
            (q.x, q.y, q.z), q.w = vcross, (dot+1.0)
            q = q.normalized()
        rotmat = qua.as_rotation_matrix(q)
        self.vinit = self.vinit.dot(rotmat.T)
        self.Nrays = Nrays
        return self.vinit, self.Nrays

    def plot_arrows(self):
        '''
        A simple method to plot the ray directions as arrows.
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.quiver(0, 0, 0,
                self.vinit[:,0], self.vinit[:,1], self.vinit[:,2],
                length=0.1, normalize=True)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.show()

    def plot_points(self):
        '''
        A simple method to plot the ray directions as a scatter plot.
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(self.vinit[:,0], self.vinit[:,1], self.vinit[:,2],
            color='blue')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()
