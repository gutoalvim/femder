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
        
class GridImport3D:
    def __init__(self,AP,path_to_geo,S=None,R=None,fmax=1000, num_freq=6,scale=1,order=1,plot=False):
        
        self.R = R
        self.S = S
        self.path_to_geo = path_to_geo
        self.fmax = fmax
        self.num_freq = num_freq
        self.scale = scale
        self.order = order
    
        self.c0 = np.real(AP.c0)
    
        import meshio
        import gmsh
        import sys
        import os
        filename, file_extension = os.path.splitext(path_to_geo)
        if path_to_geo == '.geo' or '.brep':
            gmsh.initialize(sys.argv)
            gmsh.open(self.path_to_geo) # Open msh
    
            # dT = gmsh.model.getEntities()
            # gmsh.model.occ.dilate(dT,0,0,0,1/scale,1/scale,1/scale)
            gmsh.option.setNumber("Mesh.MeshSizeMax",(self.c0*self.scale)/self.fmax/self.num_freq)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1*(self.c0*self.scale)/self.fmax/self.num_freq)
            
    
            lc = 0#(self.c0*self.scale)/self.fmax/self.num_freq
            tg = gmsh.model.occ.getEntities(3)
            # tg2 = gmsh.model.occ.getEntities(2)
            if self.R != None:
                for i in range(len(self.R.coord)):
                    it = gmsh.model.occ.addPoint(self.R.coord[i,0], self.R.coord[i,1], self.R.coord[i,2], lc, -1)
                    gmsh.model.occ.synchronize()
                    gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
    
            if self.S != None:
                for i in range(len(self.S.coord)):
                    it = gmsh.model.occ.addPoint(self.S.coord[i,0], self.S.coord[i,1], self.S.coord[i,2], lc, -1)
                    gmsh.model.occ.synchronize()
                    gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
             

            # gmsh.model.mesh.embed(0, [15000], 3, tg[0][1])
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(self.order)
            # gmsh.model.mesh.optimize(method='Relocate3D',force=False)
            if self.order == 1:
                elemTy,elemTa,nodeTags = gmsh.model.mesh.getElements(3)
                self.elem_vol = np.array(nodeTags,dtype=int).reshape(-1,4)-1
                
                elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
                self.elem_surf = np.array(nodeTagss,dtype=int).reshape(-1,3)-1
                vtags, vxyz, _ = gmsh.model.mesh.getNodes()
                self.nos = vxyz.reshape((-1, 3))/scale
                
            if self.order == 2:
                elemTy,elemTa,nodeTags = gmsh.model.mesh.getElements(3)
                self.elem_vol = np.array(nodeTags,dtype=int).reshape(-1,10)-1
                
                elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
                self.elem_surf = np.array(nodeTagss,dtype=int).reshape(-1,6)-1
                vtags, vxyz, _ = gmsh.model.mesh.getNodes()
                self.nos = vxyz.reshape((-1, 3))/scale
                
                
            pg = gmsh.model.getPhysicalGroups(2)   
            va= []
            vpg = []
            for i in range(len(pg)):
                v = gmsh.model.getEntitiesForPhysicalGroup(2, pg[i][1])
                for ii in range(len(v)):
                    # print(v[ii])
                    vvv = gmsh.model.mesh.getElements(2,v[ii])[1][0]
                    pgones = np.ones_like(vvv)*pg[i][1]
                    va = np.hstack((va,vvv))
                    # print(pgones)
                    vpg = np.hstack((vpg,pgones))
            
            vas = np.argsort(va)
            self.domain_index_surf = vpg[vas] 
            
            pgv = gmsh.model.getPhysicalGroups(3)
            
            vav= []
            vpgv = []
            for i in range(len(pgv)):
                vv = gmsh.model.getEntitiesForPhysicalGroup(3, pgv[i][1])
                for ii in range(len(vv)):
                    # print(v[ii])
                    vvv = gmsh.model.mesh.getElements(3,vv[ii])[1][0]
                    pgones = np.ones_like(vvv)*pgv[i][1]
                    vavsv = np.hstack((vav,vvv))
                    # print(pgones)
                    vpgv = np.hstack((vpgv,pgones))
            
            vasv = np.argsort(vavsv)
            self.domain_index_vol = vpgv[vasv]
            # gmsh.model.mesh.optimize()
            gmsh.model.occ.synchronize()
            
            if plot:
                gmsh.fltk.run()                 
            # gmsh.fltk.run()
            path_name = os.path.dirname(self.path_to_geo)
            
            gmsh.write(path_name+'/current_mesh2.vtk')   
            self.model = gmsh.model
            gmsh.finalize() 
            
            msh = meshio.read(path_name+'/current_mesh2.vtk')
        elif file_extension=='.msh' or 'vtk':
            msh = meshio.read(path_to_geo)
            
            self.msh = msh
            # os.remove(path_name+'/current_mesh.msh')
            
            
            if order == 1:
                self.nos = msh.points/scale
                self.elem_surf = msh.cells_dict["triangle"]
                self.elem_vol = msh.cells_dict["tetra"]
                
                self.domain_index_surf = msh.cell_data_dict["CellEntityIds"]["triangle"]
                self.domain_index_vol = msh.cell_data_dict["CellEntityIds"]["tetra"]
                # self.domain_index_surf = msh.cell_data_dict["gmsh:physical"]["triangle"]
                # self.domain_index_vol = msh.cell_data_dict["gmsh:physical"]["tetra"]
            # elif order == 2:
            #     self.elem_surf = msh.cells_dict["triangle6"]
            #     self.elem_vol = msh.cells_dict["tetra10"]
                
            #     self.domain_index_surf = msh.cell_data_dict["gmsh:physical"]["triangle6"]
            #     self.domain_index_vol = msh.cell_data_dict["gmsh:physical"]["tetra10"]
                
            elif order == 2:
                # self.elem_surf = msh.cells_dict["triangle6"]
                # self.elem_vol = msh.cells_dict["tetra10"]
                
                self.domain_index_surf = msh.cell_data_dict["CellEntityIds"]["triangle6"]
                self.domain_index_vol = msh.cell_data_dict["CellEntityIds"]["tetra10"]
            
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
        gmsh.model.mesh.setOrder(self.order)
        gmsh.fltk.run()
        gmsh.finalize()