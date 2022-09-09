# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:23:08 2020

@author: gutoa
"""
import numpy as np
import os
import meshio

import gmsh
import sys
import tempfile

def write_and_extract_mesh_data(mesh_data):
    """
    Writes mesh into a temporary file and extracts mesh data with Meshio.

    Parameters
    ----------
    dim : int
        Mesh dimension.
    mesh_data : dict
        Dictionary that the mesh data will be written into.
    """
    out_data = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
    gmsh.write(out_data.name)
    mesh_file = meshio.read(out_data.name)
    mesh_data["vertices"] = mesh_file.points
    mesh_data["elem_surf"] = mesh_file.cells_dict["triangle"].astype("uint32")
    mesh_data["domain_index_surf"] = mesh_file.cell_data_dict["gmsh:physical"]["triangle"]

    try:
        mesh_data["domain_index_vol"] = mesh_file.cell_data_dict["gmsh:physical"]["tetra"]
        mesh_data["elem_vol"] = mesh_file.cells_dict["tetra"].astype("uint32")
    except:
        pass

    # except:
    #     mesh_data["domain_index_surf"] = None
    #     mesh_data["domain_index_vol"] = None
    out_data.close()
    os.remove(out_data.name)

class GridImport():
    def __init__(self,AP,path_to_geo, fmax=1000, num_freq=6, plot=False,scale=1):
        self.path_to_geo = path_to_geo
        self.fmax = fmax
        self.num_freq = num_freq
        self.plot = plot
        self.scale = scale
        self.c0 = np.real(AP.c0)
        

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
    def __init__(self,AP,path_to_geo,S=None,R=None,fmax=1000, num_freq=6,scale=1,order=1,
                 plot=False,meshDim=3,center_geom=False,add_rng=False, load_method="meshio", heal_shapes=False):
        
        self.R = R
        self.S = S
        self.path_to_geo = path_to_geo
        self.fmax = fmax
        self.num_freq = num_freq
        self.scale = scale
        self.order = order
        self.path_to_geo_unrolled = None
        self.c0 = np.real(AP.c0)

        try:
            gmsh.finalize()
        except:
            pass

        filename, file_extension = os.path.splitext(path_to_geo)
        # print(file_extension)
        file_list = ['.geo','.geo_unrolled','.brep','.igs','.iges','.stp','.step', '.IGS']
        if file_extension in file_list:
            try:
                gmsh.initialize()
            except:
                # gmsh.finalize()
                gmsh.initialize()
            gmsh.open(self.path_to_geo) # Open msh
            if heal_shapes:
                gmsh.model.occ.healShapes()
            gmsh.option.setNumber("Mesh.MeshSizeMax",(self.c0*self.scale)/self.fmax/self.num_freq)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1*(self.c0*self.scale)/self.fmax/self.num_freq)
            gmsh.option.setNumber("Mesh.Algorithm", 6)

            lc = 0
            tg = gmsh.model.getEntities(3)
            pgv = gmsh.model.getPhysicalGroups(3)

            if len(pgv) == 0:
                pg_start=1
                physical_groups = [i for i in range(pg_start, len(tg) + pg_start)]
                for entity, i in zip(tg, range(len(tg))):
                    gmsh.model.addPhysicalGroup(entity[0], [entity[1]], physical_groups[i])
                tg = gmsh.model.getEntities(3)

            pg = gmsh.model.getPhysicalGroups(2)
            tg2 = gmsh.model.getEntities(2)
            if len(pg) == 0:
                surfaces_entities = tg2
                pg_start = 2
                physical_groups = [i for i in range(pg_start, len(surfaces_entities) + pg_start)]
                for entity, i in zip(surfaces_entities, range(len(surfaces_entities))):
                    gmsh.model.addPhysicalGroup(entity[0], [entity[1]], physical_groups[i])
                tg2 = gmsh.model.occ.getEntities(2)

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

            # if self.S != None or self.R != None:
            #     gmsh.model.mesh.embed(0, [15000], 3, tg[0][1])
            if heal_shapes:
                gmsh.model.occ.healShapes()
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(meshDim)
            gmsh.model.mesh.setOrder(self.order)
            # gmsh.model.mesh.optimize(method='Relocate3D',force=False)
            if load_method == "meshio":
                mesh_data = {}
                write_and_extract_mesh_data(mesh_data)
                self.nos = mesh_data["vertices"]/scale
                self.elem_surf = mesh_data["elem_surf"]
                self.domain_index_surf = mesh_data["domain_index_surf"]
                try:
                    self.elem_vol = mesh_data["elem_vol"]
                    self.domain_index_vol = mesh_data["domain_index_vol"]
                except:
                    self.elem_vol = None
                    self.domain_index_vol = None
            if load_method != "meshio":
                if self.order == 1:
                    if meshDim == 3:
                        elemTy,elemTa,nodeTags = gmsh.model.mesh.getElements(3)
                        self.elem_vol = np.array(nodeTags,dtype=int).reshape(-1,4)-1
                    elif meshDim == 2:
                        self.elem_vol = []

                    elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
                    self.elem_surf = np.array(nodeTagss,dtype=int).reshape(-1,3)-1
                    vtags, vxyz, _ = gmsh.model.mesh.getNodes()
                    self.nos = vxyz.reshape((-1, 3))/scale

                if self.order == 2:
                    if meshDim == 3:
                        elemTy,elemTa,nodeTags = gmsh.model.mesh.getElements(3)
                        self.elem_vol = np.array(nodeTags,dtype=int).reshape(-1,10)-1
                    elif meshDim == 2:
                        self.elem_vol = []
                    elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
                    self.elem_surf = np.array(nodeTagss,dtype=int).reshape(-1,6)-1
                    vtags, vxyz, _ = gmsh.model.mesh.getNodes()
                    self.nos = vxyz.reshape((-1, 3))/scale



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
                # print(vas)
                if meshDim == 3:
                    pgv = gmsh.model.getPhysicalGroups(3)
                    # print(pgv)
                    vav= []
                    vpgv = []
                    for i in range(len(pgv)):
                        vv = gmsh.model.getEntitiesForPhysicalGroup(3, pgv[i][1])
                        for ii in range(len(vv)):
                            # print(v[ii])
                            vvv = gmsh.model.mesh.getElements(3,vv[ii])[1][0]
                            pgones = np.ones_like(vvv)*pgv[i][1]
                            vav = np.hstack((vav,vvv))
                            # print(pgones)
                            vpgv = np.hstack((vpgv,pgones))

                    vasv = np.argsort(vav)
                    self.domain_index_vol = vpgv[vasv]
                elif meshDim == 2:
                    self.domain_index_vol = []
            # gmsh.model.mesh.optimize()
            gmsh.model.occ.synchronize()
                    
            if center_geom == True:
                center_coord = np.mean(self.nos,axis=0)
                self.nos = self.nos - center_coord
        
            if plot:
                gmsh.fltk.run()
                import plotly.figure_factory as ff
                import plotly.graph_objs as go
                vertices = self.nos.T  # [np.unique(self.elem_surf)].T
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
                fig.show()

            # gmsh.fltk.run()
            elementTypes = gmsh.model.mesh.getElementTypes(2)

            for t in elementTypes:
                localCoords, weights = \
                    gmsh.model.mesh.getIntegrationPoints(t, "Gauss" + str(1))

                jacobians, determinants, coords = \
                    gmsh.model.mesh.getJacobians(t, localCoords)

            self.ja = jacobians
            path_name = os.path.dirname(self.path_to_geo)
            
            gmsh.write(path_name+'/current_mesh2.vtk')   
            gmsh.write(path_name+'/current_mesh2.brep')
            self.model = gmsh.model
            gmsh.finalize() 
            
            # msh = meshio.read(path_name+'/current_mesh2.vtk')
        elif file_extension=='.msh' or file_extension=='vtk':
            msh = meshio.read(path_to_geo)
            self.msh = msh
            # os.remove(path_name+'/current_mesh.msh')
            
            self.nos = msh.points/scale
            if self.order  == 1:
                
                self.elem_surf = msh.cells_dict["triangle"]

                # self.domain_index_surf = msh.cell_data_dict["CellEntityIds"]["triangle"]
                # self.domain_index_vol = msh.cell_data_dict["CellEntityIds"]["tetra"]
                try:
                    self.elem_vol = msh.cells_dict["tetra"]
                    self.domain_index_vol = msh.cell_data_dict["gmsh:physical"]["tetra"]
                    self.domain_index_surf = msh.cell_data_dict["gmsh:physical"]["triangle"]

                except:
                    self.elem_vol = None
                    self.domain_index_vol = None
                    self.domain_index_surf = None


            elif self.order  == 2 and file_extension=='.msh':
                
                self.elem_surf = msh.cells_dict["triangle6"]
                self.elem_vol = msh.cells_dict["tetra10"]
                
                self.domain_index_surf = msh.cell_data_dict["gmsh:physical"]["triangle6"]
                self.domain_index_vol = msh.cell_data_dict["gmsh:physical"]["tetra10"]
                
            # elif order == 2:
            #     # self.elem_surf = msh.cells_dict["triangle6"]
            #     # self.elem_vol = msh.cells_dict["tetra10"]
                
            #     self.domain_index_surf = msh.cell_data_dict["CellEntityIds"]["triangle6"]
            #     self.domain_index_vol = msh.cell_data_dict["CellEntityIds"]["tetra10"]
            
        self.number_ID_faces = np.unique(self.domain_index_surf)

        self.NumNosC = len(self.nos)

        if meshDim == 3:
            self.NumElemC = len(self.elem_vol)
            self.number_ID_vol = np.unique(self.domain_index_vol)
        else:
            self.NumElemC = None
            self.number_ID_vol = None
        if add_rng == True:
            rng_nos = 0.1*np.random.rand(len(self.nos[:,0]),len(self.nos[0,:]))
            self.nos = self.nos + rng_nos
        
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


if __name__ == "__main__":
    import femder as fd

    path_to_geo = r'C:\Users\gutoa\Documents\Room Acoustics\Cipriano_Rina_Gargel\Room Study\Rinaldi_Study\Geom\room_mesh_MG_treatments - Copy.geo'
    AP = fd.AirProperties(c0=343)
    AC = fd.AlgControls(AP, 20, 200, 1)

    # S = fd.Source("spherical")
    # S.coord = np.array([[1.53/2,2.7+1.32,1.14],[-1.53/2,2.7+1.32,1.14]])
    # S.q = np.array([[0.0001],[0.0001]])

    # R = fd.Receiver()
    # R.star([0,2.7,1.14],0.15)
    grid = fd.GridImport3D(AP, path_to_geo, S=None, R=None, fmax=200, num_freq=6, scale=1, order=1,
                           load_method="other", heal_shapes=False)
    # grid.plot_mesh(True)