import numpy as np
#import toml
from femder.controlsair import load_cfg
from femder.controlsair import sph2cart, cart2sph
from femder.rayinidir import RayInitialDirections
# 

class Receiver():
    '''
    A receiver class to initialize the following receiver properties:
    cood - 3D coordinates of a receiver (p, u or pu)
    There are several types of receivers to be implemented.
    - single_rec: is a single receiver (this is the class __init__)
    - double_rec: is a pair of receivers (tipically used in impedance measurements - separated by a z distance)
    - line_array: an line trough z containing receivers
    - planar_array: a regular grid of microphones
    - double_planar_array: a double regular grid of microphones separated by a z distance
    - spherical_array: a sphere of receivers
    - arc: an arc of receivers
    '''
    def __init__(self, coord = [0.0, 0.0, 0.01]):
        '''
        The class constructor initializes a single receiver with a given 3D coordinates
        The default is a height of 1 [cm]. User must be sure that the receiver lies out of
        the sample being emulated. This can go wrong if we allow the sample to have a thickness
        going on z>0
        '''
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))
    
    def star(self,coord,dx):
        
        self.coord = np.array([coord,[coord[0],coord[1],coord[2]+dx],
                                      [coord[0],coord[1]+dx,coord[2]],
                                      [coord[0]+dx,coord[1],coord[2]],
                                      [coord[0],coord[1],coord[2]-dx],
                                      [coord[0],coord[1]-dx,coord[2]],
                                      [coord[0]-dx,coord[1],coord[2]]])
                                      
                                      
    def double_rec(self, z_dist = 0.01):
        '''
        This method initializes a double receiver separated by z_dist. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        '''
        self.coord = np.append(self.coord, [self.coord[0,0], self.coord[0,1], self.coord[0,2]+z_dist])
        self.coord = np.reshape(self.coord, (2,3))

    def line_array(self, line_len = 1.0, n_rec = 10):
        '''
        This method initializes a line array of receivers. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            line_len - the length of the line. The first sensor will be at coordinates given by
            the class constructor. Receivers will span in z-direction
            n_rec - the number of receivers in the line array
        '''
        pass

    def planar_array(self, x_len = 1.0, n_x = 10, y_len = 1.0, n_y = 10, zr = 0.1):
        '''
        This method initializes a planar array of receivers (z/xy plane). It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            x_len - the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x - the number of receivers in the x direction
            y_len - the length of the y direction (array goes from -x_len/2 to +x_len/2).
            n_y - the number of receivers in the y direction
            zr - distance from the closest microphone layer to the sample
        '''
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        yc = np.linspace(-y_len/2, y_len/2, n_y)
        # meshgrid
        xv, yv = np.meshgrid(xc, yc)
        # initialize receiver list in memory
        self.coord = np.zeros((n_x * n_y, 3), dtype = np.float32)
        self.coord[:, 0] = xv.flatten()
        self.coord[:, 1] = yv.flatten()
        self.coord[:, 2] = zr
    def arc(self, radius=1, angles=(-30, 30), axis="z", ref=None, n=None):
        """
        Return an arc of points in the selected axis.
        Parameters
        ----------
        radius : int or float, optional
            Radius of the arc.
        angles : tuple or list, optional
            Defines how many points the arc will contain and at which angles in degrees.
        axis : str, optional
            Defines which plane the arc will be parallel to.
        ref : int, optional
            Index of what point in self.coords will be used as the pivot of the rotation. If not defined the centroid
            of 'coords' will be used.
        n : int, optional
            Number of points that the range defined by the first and last elements of 'angles' will contain. If no
            defined the number of points will be determined by the number of elements in 'angles'.
        view : bool, optional
            Option to visualize the grid with Plotly.
        Returns
        -------
        (N, 3) array containing the arc points and a dictionary containing the connectivity data.
        """
        if ref is None:
            center = self.centroid
        else:
            center = self.coords[ref, :].reshape(1, 3)

        points = {}
        if n is None:
            theta = [np.deg2rad(angle) for angle in angles]
        else:
            theta = np.linspace(np.deg2rad(angles[0]), np.deg2rad(angles[-1]), n)

        for i in range(len(theta)):
            x = center[0, 0] + radius * np.cos(theta[i])
            y = center[0, 1] + radius * np.sin(theta[i]) if axis != "y" else center[0, 2] + radius * np.sin(theta[i])
            z = center[0, 2] if axis != "y" else center[0, 1]

            if axis == "x":
                points[i] = np.array([z, y, x])
            if axis == "y":
                points[i] = np.array([x, z, y])
            if axis == "z":
                points[i] = np.array([x, y, z])

        arc = np.array(list(points.values()))
        if angles[0] == 0 and angles[1] == 360:
            arc = arc[:len(arc) - 1]

        points = dict(enumerate(arc.tolist(), 1))
        lines = {}
        a = 0
        for a in range(1, int(len(arc))):
            lines[a] = [a, a + 1]
        lines[list(lines.keys())[-1] + 1] = [a + 1, 1]
        surfaces = {1: list(lines.keys())}
        con = {"points": points,
               "lines": lines,
               "surfaces": surfaces
               }

        return arc, con

    def arc(self, radius=1, angles=(-30, 30), axis="z", center=None, n=None, flip=False):
        """
        Return an arc of points in the selected axis.

        Parameters
        ----------
        radius : int or float, optional
            Radius of the arc.
        angles : tuple or list, optional
            Defines how many points the arc will contain and at which angles in degrees.
        axis : str, optional
            Defines which plane the arc will be parallel to.
        ref : int, optional
            Index of what point in self.coords will be used as the pivot of the rotation. If not defined the centroid
            of 'coords' will be used.
        n : int, optional
            Number of points that the range defined by the first and last elements of 'angles' will contain. If no
            defined the number of points will be determined by the number of elements in 'angles'.
        view : bool, optional
            Option to visualize the grid with Plotly.

        Returns
        -------
        (N, 3) array containing the arc points and a dictionary containing the connectivity data.
        """
        axis_dict = {"x": 0, "y": 1, "z": 2}

        if center is None:
            center = np.asarray([0, 0, 0])


        points = {}
        if n is None:
            theta = [np.deg2rad(angle) for angle in angles]
        else:
            theta = np.linspace(np.deg2rad(angles[0]), np.deg2rad(angles[-1]), n)

        for i in range(len(theta)):
            x = center[0] + radius * np.cos(theta[i])
            y = center[1] + radius * np.sin(theta[i]) if axis != "y" else center[2] + radius * np.sin(theta[i])
            z = center[2] if axis != "y" else center[1]

            if axis == "x":
                points[i] = np.array([x, y, z])
            if axis == "y":
                points[i] = np.array([x, z, y])
            if axis == "z":
                points[i] = np.array([x, y, z])

        arc = np.array(list(points.values()))
        if angles[0] == 0 and angles[1] == 360:
            arc = arc[:len(arc) - 1]

        if flip:
            arc[:, axis_dict[axis]] = -arc[:,axis_dict[axis]]

        self.coord = arc
        self.theta = theta
        self.axis = axis
        return arc

    def arc_receivers(self, radius = 1.0, ns = 10, angle_span = (-90, 90), d = (0,0,0), axis = "x",add_perp_arc=False ):
        points = {}
        theta = np.arange(angle_span[0]*np.pi/180, (angle_span[1]+ns)*np.pi/180, ns*np.pi/180)
        for i in range(len(theta)):
            thetai = theta[i]
            # compute x1 and x2
            x1 = d[0] + radius*np.cos(thetai)
            x2 = d[1] + radius*np.sin(thetai)
            x3 = d[2]
            
            if axis == "x":
                points[i] = np.array([x3, x2, x1])
            if axis == "y":
                points[i] = np.array([x1, x3, x2])
            if axis == "z":
                points[i] = np.array([x1, x2, x3])
            
        self.coord = np.array([points[i] for i in points.keys()])
        self.theta = theta
        self.axis = axis
        if add_perp_arc == True:
            perp_arc = self.coord.copy()
            perp_arc[:,[2,0]] = perp_arc[:,[0,2]]

            self.coord = np.unique(np.concatenate((self.coord,perp_arc)),axis=0)

    def double_planar_array(self, x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, zr = 0.01, dz = 0.01):
        '''
        This method initializes a double planar array of receivers (z/xy plane)
        separated by z_dist. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            x_len - the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x - the number of receivers in the x direction
            y_len - the length of the y direction (array goes from -x_len/2 to +x_len/2).
            n_y - the number of receivers in the y direction
            zr - distance from the closest microphone layer to the sample
            dz - separation distance between the two layers
        '''
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        yc = np.linspace(-y_len/2, y_len/2, n_y)
        # meshgrid
        xv, yv = np.meshgrid(xc, yc)
        # initialize receiver list in memory
        self.coord = np.zeros((2 * n_x * n_y, 3), dtype = np.float32)
        self.coord[0:n_x*n_y, 0] = xv.flatten()
        self.coord[0:n_x*n_y, 1] = yv.flatten()
        self.coord[0:n_x*n_y, 2] = zr
        self.coord[n_x*n_y:, 0] = xv.flatten()
        self.coord[n_x*n_y:, 1] = yv.flatten()
        self.coord[n_x*n_y:, 2] = zr + dz

    def brick_array(self, x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, z_len = 1.0, n_z = 8, zr = 0.1):
        '''
        This method initializes a regular three dimensional array of receivers It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            x_len - the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x - the number of receivers in the x direction
            y_len - the length of the y direction (array goes from -x_len/2 to +x_len/2).
            n_y - the number of receivers in the y direction
            z_len - the length of the z direction (array goes from zr to zr+z_len).
            n_z - the number of receivers in the y direction
            zr - distance from the closest receiver to the sample's surface
        '''
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        yc = np.linspace(-y_len/2, y_len/2, n_y)
        zc = np.linspace(zr, zr+z_len, n_z)
        # print('sizes: xc {}, yc {}, zc {}'.format(xc.size, yc.size, zc.size))
        # meshgrid
        xv, yv, zv = np.meshgrid(xc, yc, zc)
        # print('sizes: xv {}, yv {}, zv {}'.format(xv.shape, yv.shape, zv.shape))
        # initialize receiver list in memory
        self.coord = np.zeros((n_x * n_y * n_z, 3), dtype = np.float32)
        self.coord[0:n_x*n_y*n_z, 0] = xv.flatten()
        self.coord[0:n_x*n_y*n_z, 1] = yv.flatten()
        self.coord[0:n_x*n_y*n_z, 2] = zv.flatten()
        # print(self.coord)

    def random_3d_array(self, x_len = 1.0, y_len = 1.0, z_len = 1.0, axis='z',zr = 0.1, n_total = 192, seed = 0):
        '''
        This method initializes a regular three dimensional array of receivers It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            x_len - the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x - the number of receivers in the x direction
            y_len - the length of the y direction (array goes from -x_len/2 to +x_len/2).
            n_y - the number of receivers in the y direction
            z_len - the length of the z direction (array goes from zr to zr+z_len).
            n_z - the number of receivers in the y direction
            zr - distance from the closest receiver to the sample's surface
        '''
        # x and y coordinates of the grid
        np.random.seed(seed)
        
        if axis =='z':
            
            xc = -x_len/2 + x_len * np.random.rand(n_total)#np.linspace(-x_len/2, x_len/2, n_x)
            yc = -y_len/2 + y_len * np.random.rand(n_total)
            zc = zr + z_len * np.random.rand(n_total)
        
        if axis =='y':           
            xc = -x_len/2 + x_len * np.random.rand(n_total)#np.linspace(-x_len/2, x_len/2, n_x)
            yc = zr + y_len * np.random.rand(n_total)
            zc = -z_len/2 + z_len * np.random.rand(n_total)       
            
        if axis =='x':
            xc = zr + x_len * np.random.rand(n_total)#np.linspace(-x_len/2, x_len/2, n_x)
            yc = -y_len/2 + y_len * np.random.rand(n_total)
            zc = -z_len/2 + z_len * np.random.rand(n_total)      
        # meshgrid
        # xv, yv, zv = np.meshgrid(xc, yc, zc)
        # initialize receiver list in memory
        self.coord = np.zeros((n_total, 3), dtype = np.float32)
        
        self.coord[0:n_total, 0] = xc.flatten()
        self.coord[0:n_total, 1] = yc.flatten()
        self.coord[0:n_total, 2] = zc.flatten()

    def spherical_array(self, radius = 0.1, n_rec = 32, center_dist = 0.5):
        '''
        This method initializes a spherical array of receivers. The array coordinates are
        separated by center_dist from the origin. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            radius - the radius of the sphere.
            n_rec - the number of receivers in the spherical array
            center_dist - center distance from the origin
        '''
        pass

    def plot(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.coord[:,0],self.coord[:,1],self.coord[:,2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
    def spherical_receivers(self, radius = 1.0, ns = 100, axis = 'x', random = False, plot=False):
        '''
        This method is used to generate an array of sound sources over a surface of a sphere
        Inputs:
            radius - radii of the source arc (how far from the sample they are)
            ns - the number of sound sources in the sphere
            random (bool) - if True, then the complex amplitudes are randomized
        '''
        # Define theta and phi discretization
        directions = RayInitialDirections()
        directions, n_waves = directions.isotropic_rays(Nrays = ns)
        print('The number of sources is: {}'.format(n_waves))
        if plot:
            directions.plot_points()
        r, theta, phi = cart2sph(directions[:,0], directions[:,1], directions[:,2])
        # print('theta: {}'.format(np.sort(np.unique(np.rad2deg(theta)))))
        # print('phi: {}'.format(np.sort(np.unique(np.rad2deg(phi)))))
        # theta_id = np.where(theta > -np.pi/2 and theta < np.pi/2)
        theta_id = np.where(np.logical_and(theta > 0, theta < np.pi/2))
        # coords = directions[theta_id[0]]
        # coords /= np.linalg.norm(coords, axis = 1)[:,None]
        # rr,tt,pp = cart2sph(coords[:,0],coords[:,1],coords[:,2])
        # x,y,z = sph2cart(radius*rr,tt,pp)
        # self.coord = np.vstack((x,y,z))
        self.coord = radius*directions[theta_id[0]]
        # print(theta_id)
        # phiv = np.linspace(start = 0, stop = 2*np.pi, num = int(np.sqrt(ns)))
        # thetav = np.linspace(start = -np.pi/2+np.deg2rad(5), stop = np.pi/2-np.deg2rad(5), num = int(np.sqrt(ns)))
        # phim, thetam = np.meshgrid(phiv, thetav)
        # xm, ym, zm = sph2cart(radius, phim, thetam)
        # self.coord = np.zeros((len(xm)**2, 3), dtype=np.float32)
        # self.coord[:,0] = xm.flatten()
        # self.coord[:,1] = ym.flatten()
        # self.coord[:,2] = zm.flatten()
        # self.coord /= np.linalg.norm(self.coord, axis = 1)[:,None]
        if axis == 'y':
            self.coord[:, 1], self.coord[:, 2] = self.coord[:, 2], self.coord[:, 1].copy()
        if axis == 'x':
            self.coord[:, 0], self.coord[:, 2] = self.coord[:, 2], self.coord[:, 0].copy()
            
    # def add_coord(self,coord):
        # 
# class Receivers():
#     def __init__(self, config_file):
#         '''
#         Set up the receivers
#         '''
#         config = load_cfg(config_file)
#         coord = []
#         orientation = []
#         for r in config['receivers']:
#             coord.append(r['position'])
#             orientation.append(r['orientation'])
#         self.coord = np.array(coord)
#         self.orientation = np.array(orientation)

    

# def setup_receivers(config_file):
#     '''
#     Set up the sound sources
#     '''
#     receivers = [] # An array of empty receiver objects
#     config = load_cfg(config_file) # toml file
#     for r in config['receivers']:
#         coord = np.array(r['position'], dtype=np.float32)
#         orientation = np.array(r['orientation'], dtype=np.float32)
#         ################### cpp receiver class #################
#         receivers.append(insitu_cpp.Receivercpp(coord, orientation)) # Append the source object
#         ################### py source class ################
#         # receivers.append(Receiver(coord, orientation))
#     return receivers

# # class Receiver from python side


