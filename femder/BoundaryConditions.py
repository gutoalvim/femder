import numpy as np
# import controlsair
class BC():
    def __init__(self,AC,AP):
        self.AP = AP
        self.AC = AC
        self.mu = {}
        self.v = {}
        self.rhoc = {}
        self.cc = {}
    def impedance(self,domain_index, impedance):
        """
        

        Parameters
        ----------
        domain_index : TYPE
            Physical group indexes assigned in gmsh for each surface.
        impedance : TYPE
            frequency x domain_index matrix with surface impedance values.

        Returns
        -------
        None.

        """
        
        for i in domain_index:
            self.mu[i] = np.array(1/impedance[:,i])

    def rigid(self,domain_index):
        """
        

        Parameters
        ----------
        domain_index : int
            Physical group index assigned in gmsh for rigid surfaces


        """
        self.mu[domain_index] = np.zeros_like(self.AC.freq)
                
    def admittance(self,domain_index, admittance):
        """
        

        Parameters
        ----------
        domain_index : TYPE
            Physical group indexes assigned in gmsh for each surface.
        impedance : TYPE
            frequency x domain_index matrix with surface impedance values.

        Returns
        -------
        None.

        """
        
        if type(admittance) == int or float:
            self.mu[domain_index] = np.ones_like(self.AC.freq)*admittance
        
        else:
            
            for i in domain_index:
                self.mu[i] = np.array(admittance[:,i])
    def normalized_admittance(self,domain_index, normalized_admittance):
        """
        

        Parameters
        ----------
        domain_index : TYPE
            Physical group indexes assigned in gmsh for each surface.
        impedance : TYPE
            frequency x domain_index matrix with surface impedance values.

        Returns
        -------
        None.

        """
        if type(domain_index) == int:
            if type(normalized_admittance) == int or float:
                self.mu[domain_index] = np.ones_like(self.AC.freq)*normalized_admittance/(self.AP.c0*self.AP.rho0)
            
        if type(domain_index) == list:
            for i in range(len(domain_index)):
                self.mu[domain_index[i]] = np.ones_like(self.AC.freq)*normalized_admittance/(self.AP.c0*self.AP.rho0)

        # elif isinstance(normalized_admittance, np.ndarray):
            
        #     # for i in domain_index:
        #     self.mu[domain_index] = np.array(normalized_admittance[:])/(self.AP.c0*self.AP.rho0)

            
    def velocity(self,domain_index, velocity):
        """
        

        Parameters
        ----------
        domain_index : TYPE
            Physical group indexes assigned in gmsh for each surface.
        impedance : TYPE
            frequency x domain_index matrix with surface impedance values.

        Returns
        -------
        None.

        """
        if type(velocity) == int or float:
            self.v[domain_index] = np.ones_like(self.AC.freq)*velocity
              
     
    def TMM(self,domain_index,TMM):
            
        from scipy import interpolate
        if len(TMM.freq) != len(self.AC.freq):
            f_real = interpolate.interp1d(TMM.freq.ravel(),np.real(TMM.y).ravel())
            f_imag = interpolate.interp1d(TMM.freq.ravel(),np.imag(TMM.y).ravel())
            mu_data = f_real(self.AC.freq).ravel() + 1j*f_imag(self.AC.freq).ravel()
        else:
            mu_data = TMM.y.ravel()
        self.mu[domain_index] = mu_data
        
        # if TMM.rhoc.any != None:
        #     f_real = interpolate.interp1d(TMM.freq.ravel(),np.real(TMM.rhoc).ravel())
        #     f_imag = interpolate.interp1d(TMM.freq.ravel(),np.imag(TMM.rhoc).ravel())
        #     rhoc_data = f_real(self.AC.freq).ravel() + 1j*f_imag(self.AC.freq).ravel()
        #     self.rhoc[domain_index] = rhoc_data
        # if TMM.cc.any != None:
        #     f_real = interpolate.interp1d(TMM.freq.ravel(),np.real(TMM.cc).ravel())
        #     f_imag = interpolate.interp1d(TMM.freq.ravel(),np.imag(TMM.cc).ravel())
        #     cc_data = f_real(self.AC.freq).ravel() + 1j*f_imag(self.AC.freq).ravel()
        #     self.cc[domain_index] = cc_data
            
    def fluid(self,domain_index,cc,rhoc):
        """
        

        Parameters
        ----------
        domain_index : int
            DESCRIPTION.
        cc : complex128
            DESCRIPTION.
        rhoc : complex128
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.rhoc[domain_index] = (np.ones_like(self.AC.freq,dtype = complex)*rhoc).ravel()
        self.cc[domain_index] = (np.ones_like(self.AC.freq,dtype = complex)*cc).ravel()
        
            
    def delany(self,domain_index=None,RF=10900,d=None,model='delany-bazley'):
    
        """
        This function implements th e Delany-Bazley-Miki model for a single porous layers.
        
        Input:
            RF: Flow Resistivity []
            d: Depth of porous layer [m]
            f_range: Frequency vector [Hz]
        
        Output:
            Zs: Surface Impedance [Pa*s/m]
        """
        f_range = self.AC.freq
        w = 2*np.pi*f_range
    
        if model == 'delany-bazley':
            C1=0.0978
            C2=0.7
            C3=0.189
            C4=0.595
            C5=0.0571
            C6=0.754
            C7=0.087
            C8=0.723
        elif model == 'miki':
            C1=0.122
            C2=0.618
            C3=0.18
            C4=0.618
            C5=0.079
            C6=0.632
            C7=0.12
            C8=0.632
        elif model == 'PET':
            C1=0.078
            C2=0.623
            C3=0.074
            C4=0.660
            C5=0.159
            C6=0.571
            C7=0.121
            C8=0.530

        X = f_range*self.AP.rho0/RF
        cc = (self.AP.c0/(1+C1*np.power(X,-C2) -1j*C3*np.power(X,-C4)))
        rhoc = ((self.AP.rho0*self.AP.c0/cc)*(1+C5*np.power(X,-C6)-1j*C7*np.power(X,-C8)))
        
        if d == None:
            self.cc = np.conj(cc)
            self.rhoc = np.conj(rhoc)
            return
        else:
            Zs = -1j*rhoc*cc/np.tan((w/cc)*d) 
            
            self.mu[domain_index] = np.array(1/Zs)

        

            