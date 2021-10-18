import numpy as np
from femder.controlsair import sph2cart, cart2sph
from femder.rayinidir import RayInitialDirections


class Source():
    '''
    A sound source class to initialize the following sound source properties.
    :
    Inputs:
        wavetype - Incident Pressure Field Type - "spherical" or "plane"
        cood - 3D coordinates of a single spherical source or wave direction for single plane wave
        q - volume velocity [m^3/s]
        
    '''
    def __init__(self, wavetype = "spherical" ,coord = [0.0, 0.0, 1.0], q = [1.0]):
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))
        self.q = np.array([q], dtype = np.float32)
        self.wavetype = wavetype
    
    def sym_pair(self, coord,axis='y'):
        if isinstance != (np.ndarray):
            coord = np.array(coord)
            coord_sym = coord
            
        if axis == 'x':
            
            coord_sym[0] = -coord_sym[0]
        if axis == 'y':
            
            coord_sym[1] = -coord_sym[1]
        if axis == 'z':
            
            coord_sym[2] = -coord_sym[2]
                                    
        self.coord =  np.append([self.coord,coord_sym],axis=1)
        # self.coord[
        
    def set_arc_sources(self, radius = 1.0, ns = 10, angle_span = (-90, 90), random = False):
        '''
        This method is used to generate an array of sound sources in an 2D arc
        Inputs:
            radius - radii of the source arc (how far from the sample they are)
            ns - the number of sound sources in the arc
            angle_span - tuple with the range for which the sources span
            random (bool) - if True, then the complex amplitudes are randomized
        '''
        pass

    # def set_sphdir_sources(self, ns = 100, random = False):
    #     '''
    #     This method is used to generate an array of sound sources directions for plane wave simulation
    #     Inputs:
    #         radius - radii of the source arc (how far from the sample they are)
    #         ns - the number of sound sources in the sphere
    #         random (bool) - if True, then the complex amplitudes are randomized
    #     '''
    #     pass
    def iso_17497_2(self,radius=5,axis='z',center=[0,0,0],survey=False):
        iso_table_elevation = np.deg2rad([0,30,30,30,30,30,30,60,60,60,60,60,60])
        iso_table_azimuth = np.deg2rad([0,0,60,120,180,240,300,0,60,120,180,240,300])
        
        coord = np.array(sph2cart(radius,iso_table_elevation,iso_table_azimuth)).T
        
        x = coord[:,0]
        y = coord[:,1]
        z = coord[:,2]
        if axis == 'x':
            coord[:,0] = y
            coord[:,1] = x
        if axis == 'y':
            coord[:,[2,1]] = coord[:,[1,2]]
            coord[0,:] = [coord[0][2],coord[0][0],coord[0][1]]
        self.coord = coord+np.ones_like(coord)*center
        
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
    
    def arc_sources(self, radius = 1.0, ns = 10, angle_span = (-90, 90), d = 0, axis = "x", random = False, plot=False,noise=False,noisescale = 1,seed=0 ):
        np.random.seed(seed)
        
        points = {}
        qi = {}
        theta = np.linspace(angle_span[0]*np.pi/180, angle_span[1]*np.pi/180, ns)
        for i in range(len(theta)):
            thetai = theta[i]
            # compute x1 and x2
            if noise == True:
                thetai = np.deg2rad(np.rad2deg(thetai) + np.float(noisescale*np.random.randn(1,1)))
                
            x1 = d + radius*np.cos(thetai)
            x2 = d + radius*np.sin(thetai)
            x3 = d
            
            if axis == "x":
                points[i] = np.array([x3, x2, x1])
            elif axis == "y":
                points[i] = np.array([x1, x3, x2])
            elif axis == "z":
                points[i] = np.array([x1, x2, x3])
                
            if random == True:
                qi[i] = np.random.rand() + 1j*np.random.rand()
            elif random == False:
                qi[i] = self.q
            

        self.coord= np.array([points[i] for i in points.keys()])
        self.q = np.array([qi[i] for i in qi.keys()])
        self.theta = theta
    def set_ssph_sources(self, radius = 1.0, ns = 100, random = False, plot=False):
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
        self.coord = directions[theta_id[0]]
        # print(theta_id)
        # phiv = np.linspace(start = 0, stop = 2*np.pi, num = int(np.sqrt(ns)))
        # thetav = np.linspace(start = -np.pi/2+np.deg2rad(5), stop = np.pi/2-np.deg2rad(5), num = int(np.sqrt(ns)))
        # phim, thetam = np.meshgrid(phiv, thetav)
        # xm, ym, zm = sph2cart(radius, phim, thetam)
        # self.coord = np.zeros((len(xm)**2, 3), dtype=np.float32)
        # self.coord[:,0] = xm.flatten()
        # self.coord[:,1] = ym.flatten()
        # self.coord[:,2] = zm.flatten()
        self.coord /= np.linalg.norm(self.coord, axis = 1)[:,None]
        
    def spherical_sources(self, radius = 1.0, ns = 100, axis = 'x', random = False, plot=False):
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
            
        if q == None:
            self.q = np.ones

    def set_vsph_sources(self, radii_span = (1.0, 10.0), ns = 100, random = False):
        '''
        This method is used to generate an array of sound sources over the volume of a sphere
        Inputs:
            radii_span - tuple with the range for which the sources span
            ns - the number of sound sources in the sphere
            random (bool) - if True, then the complex amplitudes are randomized
        '''
        pass
    
# def setup_sources(config_file, rays):
#     '''
#     Set up the sound sources
#     '''
#     sources = [] # An array of empty souce objects
#     config = load_cfg(config_file) # toml file
#     for s in config['sources']:
#         coord = np.array(s['position'], dtype=np.float32)
#         orientation = np.array(s['orientation'], dtype=np.float32)
#         power_dB = np.array(s['power_dB'], dtype=np.float32)
#         eq_dB = np.array(s['eq_dB'], dtype=np.float32)
#         power_lin = 10.0e-12 * 10**((power_dB + eq_dB) / 10.0)
#         delay = s['delay'] / 1000
#         ################### cpp source class #################
#         sources.append(insitu_cpp.Sourcecpp(coord, orientation,
#             power_dB, eq_dB, power_lin, delay)) # Append the source object
#         # sources.append([insitu_cpp.Sourcecpp(coord, orientation,
#         #     power_dB, eq_dB, power_lin, delay),
#         #     rays]) # Append the source object
#         ################### py source class ################
#         # sources.append(Source(coord, orientation,
#         #     power_dB, eq_dB, power_lin, delay)) # Append the source object
#     return sources
# # class Souces from python side