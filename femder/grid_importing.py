# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:23:08 2020

@author: gutoa
"""
import numpy as np
class GridImport():
    def __init__(self,AP,path_to_geo, fmax=1000, num_freq=6, plot=False,scale=1):
        self.path_to_geo = path_to_geo
        self.fmax = fmax
        self.num_freq = num_freq
        self.plot = plot
        self.scale = scale
        self.c0 = np.real(AP.c0)

        
    
        import meshio
        import gmsh
        import sys
        import os
        
        gmsh.initialize(sys.argv)
        gmsh.open(self.path_to_geo) # Open msh
        
        
        # dT = gmsh.model.getEntities()
        # gmsh.model.occ.dilate(dT,0,0,0,1/scale,1/scale,1/scale)
        gmsh.option.setNumber("Mesh.MeshSizeMax",(self.c0*self.scale)/self.fmax/self.num_freq)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5*(self.c0*self.scale)/self.fmax/self.num_freq)
        gmsh.model.occ.synchronize()
        
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.setOrder(1)

        
        
        path_name = os.path.dirname(self.path_to_geo)
        
        gmsh.write(path_name+'/current_mesh.msh')   
        if self.plot:
            gmsh.fltk.run()     
        gmsh.finalize() 
        
        msh = meshio.read(path_name+'/current_mesh.msh')
        
        self.nos = msh.points/self.scale
        self.elem = msh.cells_dict["line"]
        
        self.domain_index = msh.cell_data_dict["gmsh:physical"]["line"]
        self.nnos = len(self.nos)
        self.nelem = self.nnos -1
        os.remove(path_name+'/current_mesh.msh')
        
class GridImport3D():
    def __init__(self,AP,path_to_geo,S,R, fmax=1000, num_freq=6, plot=False,scale=1):
        
        self.path_to_geo = path_to_geo
        self.fmax = fmax
        self.num_freq = num_freq
        self.plot = plot
        self.scale = scale
        self.R = R
        self.S = S
        self.c0 = np.real(AP.c0)
    
        import meshio
        import gmsh
        import sys
        import os
        
        gmsh.initialize(sys.argv)
        gmsh.open(self.path_to_geo) # Open msh

        # dT = gmsh.model.getEntities()
        # gmsh.model.occ.dilate(dT,0,0,0,1/scale,1/scale,1/scale)
        gmsh.option.setNumber("Mesh.MeshSizeMax",(self.c0*self.scale)/self.fmax/self.num_freq)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5*(self.c0*self.scale)/self.fmax/self.num_freq)
        lc = (self.c0*self.scale)/self.fmax/self.num_freq
        tg = gmsh.model.occ.getEntities(3)
        for i in range(len(self.R.coord)):
            it = gmsh.model.geo.addPoint(self.R.coord[0,i], self.R.coord[1,i], self.R.coord[2,i], lc, -1)
            gmsh.model.mesh.embed(0, [it], 3, tg)
        for i in range(len(self.R.coord)):
            it = gmsh.model.geo.addPoint(self.S.coord[0,i], self.S.coord[1,i], self.S.coord[2,i], lc, -1)
            gmsh.model.mesh.embed(0, [it], 3, tg)
        
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.occ.synchronize()
        
        
        path_name = os.path.dirname(self.path_to_geo)
        
        gmsh.write(path_name+'/current_mesh.msh')   
        if self.plot:
            gmsh.fltk.run()     
        gmsh.finalize() 
        
        msh = meshio.read(path_name+'/current_mesh.msh')
        os.remove(path_name+'/current_mesh.msh')
        self.nos = msh.points/scale
        
        self.elem_surf = msh.cells_dict["triangle"]
        self.elem_vol = msh.cells_dict["tetra"]
        
        self.domain_index_surf = msh.cell_data_dict["gmsh:physical"]["triangle"]
        self.domain_index_vol = msh.cell_data_dict["gmsh:physical"]["tetra"]
        
        self.number_ID_faces = np.unique(self.domain_index_surf)
        self.number_ID_vol = np.unique(self.domain_index_vol)
        
        self.NumNosC = len(self.nos)
        self.NumElemC = len(self.elem_vol)
        
    def plot_mesh(self,onlySurface = True):
        import gmsh
        import sys
        gmsh.initialize(sys.argv)
        gmsh.open(self.path_to_geo)
        gmsh.option.setNumber("Mesh.MeshSizeMax",(self.c0*self.scale)/self.fmax/self.num_freq)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5*(self.c0*self.scale)/self.fmax/self.num_freq)
        gmsh.model.occ.synchronize()
        if onlySurface:
            gmsh.model.mesh.generate(2)
        else:
            gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        gmsh.fltk.run()
        gmsh.finalize()