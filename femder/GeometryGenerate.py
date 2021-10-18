# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:55:23 2020

@author: gutoa
"""
 
import gmsh
import numpy as np
import femder as fd
import sys
import os
class GeometryGenerator():
    def __init__(self,AP,fmax=1000, num_freq=6,scale=1,order=1,S=None,R=None,plot=False):
        self.R_ = R
        self.S_ = S
        self.fmax = fmax
        self.num_freq = num_freq
        self.S_cale = scale
        self.order = order
        self.AP = AP
        self.plot = plot
        
        self.c0 = AP.c0
    def generate_symmetric_polygon(self,pts,height):
        """
        Generates symmetric 3D polygonal geometry from a set of points in the plane. Points will be symmetrized in the Y axis

        Parameters
        ----------
        pts : Numpy Array
            Set of coordinates in the 2D plane.
        height : Float
            Height of the geometry in meters.

        Returns
        -------
        None.

        """
    
        # pts = np.array([[0,0],[2,0],[3,0],[3,2],[3,4],[2,4],[0,4]])
        # gmsh.finalize()
        gmsh.initialize()
        gmsh.model.add('gen_geom.geo')
        pts_sym = pts.copy()
        pts_sym[:,0] = -pts_sym[:,0]
        tagg = []
        for i in range(len(pts)):
            tag = 200+i
            tag2 = 1000+i 
            gmsh.model.occ.addPoint(pts[i,0], pts[i,1], 0, 0., tag)
            # print(tag)
            if i>0:
            #     # print(tag[i],tag[i-1])
                gmsh.model.occ.addLine(200+i, 199+i, tag2)
                tagg.append(tag2)
            
            
        tagg = []
        for i in range(len(pts_sym)):
            tag = 2000+i
            tag2 = 10000+i 
            gmsh.model.occ.addPoint(pts_sym[i,0], pts_sym[i,1], 0, 0., tag)
            # print(tag)
            if i>0:
            #     # print(tag[i],tag[i-1])
                gmsh.model.occ.addLine(2000+i, 1999+i, tag2)
                tagg.append(tag2)
        gmsh.model.occ.synchronize()
        # gmsh.model.occ.addLine(200, 200+len(pts)-1,1000+len(pts))
        tg = gmsh.model.occ.getEntities(1)
        # gmsh.model.occ.copy(tg)
        # gmsh.model.occ.mirror(tg,1,0,0,0)
        # tg = gmsh.model.occ.getEntities(1)
        
        tgg = []
        for i in range(len(tg)):
            
            tgg.append(tg[i][1])
            
        tgg.sort()
        it = gmsh.model.occ.addCurveLoop(tgg, -1)
        it = gmsh.model.occ.addPlaneSurface([it], -1)
        
        # gmsh.model.addPhysicalGroup(2, [15001], 15001)
        gmsh.model.occ.synchronize()
        # gmsh.model.geo.extrude([(3,15001)],dx=0,dy=0,dz = zmax)
        gmsh.model.occ.extrude([(2,it)],dx=0,dy=0,dz = height)
        gmsh.model.occ.synchronize()
        # tg3 = gmsh.model.occ.getEntities(2)
        # gmsh.model.occ.healShapes(tg3)

        tgv = gmsh.model.occ.getEntities(2)
        tggv = []
        for i in range(len(tgv)):
            
            tggv.append(tgv[i][1])
        # gmsh.model.occ.addVolume(tggv.sort,-1)
        gmsh.model.addPhysicalGroup(3, [1],-1)
        gmsh.model.addPhysicalGroup(2, tggv,-1)
        # 
        

        
        gmsh.option.setNumber("Mesh.MeshSizeMax",(self.c0*self.S_cale)/self.fmax/self.num_freq)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1*(self.c0*self.S_cale)/self.fmax/self.num_freq)
        gmsh.option.setNumber("Mesh.Algorithm", 1)

        # print((self.c0*self.S_cale)/self.fmax/self.num_freq)
        lc = 0#(self.c0*self.S_cale)/self.fmax/self.num_freq
        tg = gmsh.model.occ.getEntities(3)
        # tg2 = gmsh.model.occ.getEntities(2)
        if self.R_ != None:
            for i in range(len(self.R_.coord)):
                it = gmsh.model.occ.addPoint(self.R_.coord[i,0], self.R_.coord[i,1], self.R_.coord[i,2], lc, -1)
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
    
        if self.S_ != None:
            for i in range(len(self.S_.coord)):
                it = gmsh.model.occ.addPoint(self.S_.coord[i,0], self.S_.coord[i,1], self.S_.coord[i,2], lc, -1)
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
         

        # gmsh.model.mesh.embed(0, [15000], 3, tg[0][1])
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(self.order)    
        gmsh.write('current_mesh.msh')
        gmsh.write('current_mesh.vtk')

        gmsh.write('current_mesh.brep')
        if self.plot:
            gmsh.fltk.run()
        gmsh.model.occ.synchronize()
        gmsh.clear()


        # gmsh.write('current_geo2.geo_unrolled')
        Grid = fd.GridImport3D(self.AP, 'current_mesh.vtk',S=self.S_,R=self.R_,fmax=self.fmax,num_freq=self.num_freq,order = self.order,scale=self.S_cale,plot=self.plot)
        self.nos = Grid.nos
        self.elem_surf = Grid.elem_surf
        self.elem_vol =  Grid.elem_vol
        self.domain_index_surf =  Grid.domain_index_surf
        self.domain_index_vol =Grid.domain_index_vol
        self.number_ID_faces =Grid.number_ID_faces
        self.number_ID_vol = Grid.number_ID_vol
        self.NumNosC = Grid.NumNosC
        self.NumElemC = Grid.NumElemC
        self.order = Grid.order
        self.path_to_geo = Grid.path_to_geo
        
        current_path = os.getcwd()
        self.path_to_geo_unrolled = current_path+'\\current_mesh.brep'
        
        try:
            gmsh.finalize()
        except:
            pass
        
    def generate_symmetric_polygon_variheight(self,pts,height,angle):
        """
        Generates symmetric 3D polygonal geometry from a set of points in the plane. Points will be symmetrized in the Y axis

        Parameters
        ----------
        pts : Numpy Array
            Set of coordinates in the 2D plane.
        height : Float
            Height of the geometry in meters.

        Returns
        -------
        None.

        """
    
        # pts = np.array([[0,0],[2,0],[3,0],[3,2],[3,4],[2,4],[0,4]])
        # try:
        #     gmsh.finalize()
        # except:
        #     pass
        # time.sleep(0.0)
        gmsh.initialize()
        gmsh.model.add('gen_geom')
        pts_sym = pts.copy()
        pts_sym[:,0] = -pts_sym[:,0]
        tagg = []
        for i in range(len(pts)):
            tag = 200+i
            tag2 = 1000+i 
            gmsh.model.occ.addPoint(pts[i,0], pts[i,1], 0, 0., tag)
            # print(tag)
            if i>0:
            #     # print(tag[i],tag[i-1])
                gmsh.model.occ.addLine(200+i, 199+i, tag2)
                tagg.append(tag2)
            
            
        tagg = []
        for i in range(len(pts_sym)):
            tag = 2000+i
            tag2 = 10000+i 
            gmsh.model.occ.addPoint(pts_sym[i,0], pts_sym[i,1], 0, 0., tag)
            # print(tag)
            if i>0:
            #     # print(tag[i],tag[i-1])
                gmsh.model.occ.addLine(2000+i, 1999+i, tag2)
                tagg.append(tag2)
        gmsh.model.occ.synchronize()
        # gmsh.model.occ.addLine(200, 200+len(pts)-1,1000+len(pts))
        tg = gmsh.model.occ.getEntities(1)
        # gmsh.model.occ.copy(tg)
        # gmsh.model.occ.mirror(tg,1,0,0,0)
        # tg = gmsh.model.occ.getEntities(1)
        
        tgg = []
        for i in range(len(tg)):
            
            tgg.append(tg[i][1])
            
        tgg.sort()
        it = gmsh.model.occ.addCurveLoop(tgg, -1)
        it = gmsh.model.occ.addPlaneSurface([it], -1)
        
        # gmsh.model.addPhysicalGroup(2, [15001], 15001)
        gmsh.model.occ.synchronize()
        # gmsh.model.geo.extrude([(3,15001)],dx=0,dy=0,dz = zmax)
        gmsh.model.occ.extrude([(2,it)],dx=0,dy=0,dz = height)
        gmsh.model.occ.synchronize()
        # tg3 = gmsh.model.occ.getEntities(2)
        # gmsh.model.occ.healShapes(tg3)

        tgv = gmsh.model.occ.getEntities(2)
        tggv = []
        for i in range(len(tgv)):
            
            tggv.append(tgv[i][1])
        # gmsh.model.occ.addVolume(tggv.sort,-1)
        # gmsh.model.addPhysicalGroup(3, [1],-1)
        # gmsh.model.addPhysicalGroup(2, tggv,-1)
        # 
        gmsh.model.occ.addBox(2*np.amax(pts[:,0]), 2*np.amax(pts[:,1]), height*0.9, -4*np.amax(pts[:,0]),-4*np.amax(pts[:,1]), 2*height,tag=66666)
        gmsh.model.occ.rotate([(3,66666)], 0, 0,height, 1, 0, 0, -angle)
        gmsh.model.occ.synchronize()
        
        gmsh.model.occ.cut([(3,1)], [(3,66666)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1],-1)
        gmsh.model.addPhysicalGroup(2, tggv,-1)
        # gmsh.fltk.run()
        gmsh.option.setNumber("Mesh.MeshSizeMax",(self.c0*self.S_cale)/self.fmax/self.num_freq)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1*(self.c0*self.S_cale)/self.fmax/self.num_freq)
        
        # print((self.c0*self.S_cale)/self.fmax/self.num_freq)
        lc = 0#(self.c0*self.S_cale)/self.fmax/self.num_freq
        tg = gmsh.model.occ.getEntities(3)
        # tg2 = gmsh.model.occ.getEntities(2)
        if self.R_ != None:
            for i in range(len(self.R_.coord)):
                it = gmsh.model.occ.addPoint(self.R_.coord[i,0], self.R_.coord[i,1], self.R_.coord[i,2], lc, -1)
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
    
        if self.S_ != None:
            for i in range(len(self.S_.coord)):
                it = gmsh.model.occ.addPoint(self.S_.coord[i,0], self.S_.coord[i,1], self.S_.coord[i,2], lc, -1)
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
         

        # gmsh.model.mesh.embed(0, [15000], 3, tg[0][1])
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(self.order)
        gmsh.model.occ.synchronize()
        gmsh.write('current_mesh.msh')
        gmsh.write('current_mesh.brep')
        if self.plot:
            gmsh.fltk.run()
        gmsh.model.occ.synchronize()
        # gmsh.clear()


        # gmsh.write('current_geo2.geo_unrolled')
        Grid = fd.GridImport3D(self.AP, 'current_mesh.msh',S=self.S_,R=self.R_,fmax=self.fmax,num_freq=self.num_freq,order = self.order,scale=self.S_cale,plot=self.plot)
        self.nos = Grid.nos
        self.elem_surf = Grid.elem_surf
        self.elem_vol =  Grid.elem_vol
        self.domain_index_surf =  Grid.domain_index_surf
        self.domain_index_vol =Grid.domain_index_vol
        self.number_ID_faces =Grid.number_ID_faces
        self.number_ID_vol = Grid.number_ID_vol
        self.NumNosC = Grid.NumNosC
        self.NumElemC = Grid.NumElemC
        self.order = Grid.order
        self.path_to_geo = Grid.path_to_geo
        
        current_path = os.getcwd()
        self.path_to_geo_unrolled = current_path+'\\current_mesh.brep'
        gmsh.finalize()
        # try:
        #     gmsh.finalize()
        # except:
        #     pass
        


        # gmsh.model.mesh.optimize('Netgen')
        # # gmsh.model.mesh.optimize(method='Relocate3D',force=False)
        # if self.order == 1:
        #     elemTy,elemTa,nodeTags = gmsh.model.mesh.getElements(3)
        #     self.elem_vol = np.array(nodeTags,dtype=int).reshape(-1,4)-1
            
        #     elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
        #     self.elem_surf = np.array(nodeTagss,dtype=int).reshape(-1,3)-1
        #     vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        #     self.nos = vxyz.reshape((-1, 3))/self.S_cale
            
        # if self.order == 2:
        #     elemTy,elemTa,nodeTags = gmsh.model.mesh.getElements(3)
        #     self.elem_vol = np.array(nodeTags,dtype=int).reshape(-1,10)-1
            
        #     elemTys,elemTas,nodeTagss = gmsh.model.mesh.getElements(2)
        #     self.elem_surf = np.array(nodeTagss,dtype=int).reshape(-1,6)-1
        #     vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        #     self.nos = vxyz.reshape((-1, 3))/self.S_cale
            
            
        # pg = gmsh.model.getPhysicalGroups(2)   
        # va= []
        # vpg = []
        # for i in range(len(pg)):
        #     v = gmsh.model.getEntitiesForPhysicalGroup(2, pg[i][1])
        #     for ii in range(len(v)):
        #         # print(v[ii])
        #         vvv = gmsh.model.mesh.getElements(2,v[ii])[1][0]
        #         pgones = np.ones_like(vvv)*pg[i][1]
        #         va = np.hstack((va,vvv))
        #         # print(pgones)
        #         vpg = np.hstack((vpg,pgones))
        
        # vas = np.argsort(va)
        # self.domain_index_surf = vpg[vas] 
        # # print(vas)            
        # pgv = gmsh.model.getPhysicalGroups(3)
        # # print(pgv)
        # vav= []
        # vpgv = []
        # for i in range(len(pgv)):
        #     vv = gmsh.model.getEntitiesForPhysicalGroup(3, pgv[i][1])
        #     for ii in range(len(vv)):
        #         # print(v[ii])
        #         vvv = gmsh.model.mesh.getElements(3,vv[ii])[1][0]
        #         pgones = np.ones_like(vvv)*pgv[i][1]
        #         vav = np.hstack((vav,vvv))
        #         # print(pgones)
        #         vpgv = np.hstack((vpgv,pgones))
        
        # vasv = np.argsort(vav)
        # self.domain_index_vol = vpgv[vasv]
        # self.number_ID_faces = np.unique(self.domain_index_surf)
        # self.number_ID_vol = np.unique(self.domain_index_vol)
        
        # self.NumNosC = len(self.nos)
        # self.NumElemC = len(self.elem_vol)
        # if self.plot:s
        #     gmsh.fltk.run()  
            
        # gmsh.write('current_geo.geo_unrolled')
        
        # self.path_to_geo = 'current_geo.geo_unrolled'
        
