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
    def __init__(self, AP, fmax=1000, num_freq=6, scale=1, order=1, S=None, R=None, plot=False):
        self.R_ = R
        self.S_ = S
        self.fmax = fmax
        self.num_freq = num_freq
        self.S_cale = scale
        self.order = order
        self.AP = AP
        self.plot = plot

        self.c0 = AP.c0

    def generate_symmetric_polygon(self, pts, height):
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
        pts_sym[:, 0] = -pts_sym[:, 0]
        tagg = []
        for i in range(len(pts)):
            tag = 200 + i
            tag2 = 1000 + i
            gmsh.model.occ.addPoint(pts[i, 0], pts[i, 1], 0, 0., tag)
            # print(tag)
            if i > 0:
                #     # print(tag[i],tag[i-1])
                gmsh.model.occ.addLine(200 + i, 199 + i, tag2)
                tagg.append(tag2)

        tagg = []
        for i in range(len(pts_sym)):
            tag = 2000 + i
            tag2 = 10000 + i
            gmsh.model.occ.addPoint(pts_sym[i, 0], pts_sym[i, 1], 0, 0., tag)
            # print(tag)
            if i > 0:
                #     # print(tag[i],tag[i-1])
                gmsh.model.occ.addLine(2000 + i, 1999 + i, tag2)
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
        gmsh.model.occ.extrude([(2, it)], dx=0, dy=0, dz=height)
        gmsh.model.occ.synchronize()
        # tg3 = gmsh.model.occ.getEntities(2)
        # gmsh.model.occ.healShapes(tg3)

        tgv = gmsh.model.occ.getEntities(2)
        tggv = []
        for i in range(len(tgv)):
            tggv.append(tgv[i][1])
        # gmsh.model.occ.addVolume(tggv.sort,-1)
        gmsh.model.addPhysicalGroup(3, [1], -1)
        gmsh.model.addPhysicalGroup(2, tggv, -1)
        # 

        gmsh.option.setNumber("Mesh.MeshSizeMax", (self.c0 * self.S_cale) / self.fmax / self.num_freq)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1 * (self.c0 * self.S_cale) / self.fmax / self.num_freq)
        gmsh.option.setNumber("Mesh.Algorithm", 1)

        # print((self.c0*self.S_cale)/self.fmax/self.num_freq)
        lc = 0  # (self.c0*self.S_cale)/self.fmax/self.num_freq
        tg = gmsh.model.occ.getEntities(3)
        # tg2 = gmsh.model.occ.getEntities(2)
        if self.R_ != None:
            for i in range(len(self.R_.coord)):
                it = gmsh.model.occ.addPoint(self.R_.coord[i, 0], self.R_.coord[i, 1], self.R_.coord[i, 2], lc, -1)
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.embed(0, [it], 3, tg[0][1])

        if self.S_ != None:
            for i in range(len(self.S_.coord)):
                it = gmsh.model.occ.addPoint(self.S_.coord[i, 0], self.S_.coord[i, 1], self.S_.coord[i, 2], lc, -1)
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
        Grid = fd.GridImport3D(self.AP, 'current_mesh.brep', S=self.S_, R=self.R_, fmax=self.fmax,
                               num_freq=self.num_freq, order=self.order, scale=self.S_cale, plot=self.plot)
        self.nos = Grid.nos
        self.elem_surf = Grid.elem_surf
        self.elem_vol = Grid.elem_vol
        self.domain_index_surf = Grid.domain_index_surf
        self.domain_index_vol = Grid.domain_index_vol
        self.number_ID_faces = Grid.number_ID_faces
        self.number_ID_vol = Grid.number_ID_vol
        self.NumNosC = Grid.NumNosC
        self.NumElemC = Grid.NumElemC
        self.order = Grid.order
        self.path_to_geo = Grid.path_to_geo

        current_path = os.getcwd()
        self.path_to_geo_unrolled = current_path + '\\current_mesh.brep'

        try:
            gmsh.finalize()
        except:
            pass

    def build_surface_from_points(self, all_points, z):
        point_tags = []
        line_tags = []
        surface_tags = []
        for i in range(len(all_points)):
            temp_tag = gmsh.model.occ.addPoint(all_points[i, 0], all_points[i, 1], z, 0., -1)
            point_tags.append(temp_tag)
            # print(tag)
            if i > 0:
                temp_l_tag = gmsh.model.occ.addLine(temp_tag, temp_tag - 1, -1)
                line_tags.append(temp_l_tag)
                gmsh.model.occ.synchronize()

        point_tags = np.asarray(point_tags)
        temp_l_tag = gmsh.model.occ.addLine(point_tags[0], point_tags[-1])
        line_tags.append(temp_l_tag)
        _it = gmsh.model.occ.addCurveLoop(point_tags, -1)
        it = gmsh.model.occ.addSurfaceFilling(_it, -1)
        surface_tags.append(it)
        return point_tags, line_tags, surface_tags

    def generate_symmetric_polygon_variheight(self, pts, height, angle):
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
        gmsh.initialize(sys.argv)
        gmsh.clear()
        gmsh.option.setNumber("Mesh.MeshSizeMax", (self.c0 * self.S_cale) / self.fmax / self.num_freq)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1 * (self.c0 * self.S_cale) / self.fmax / self.num_freq)
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 6)
        gmsh.model.mesh.setOrder(self.order)
        # gmsh.model.add('gen_geom')
        pts = np.array([[2, 0], [2, 5]])
        pts_sym = pts.copy()
        pts_sym[:, 0] = -pts_sym[:, 0]

        all_points = np.concatenate((pts, np.flip(pts_sym[1:-1], axis=0)), axis=0)
        # all_points = all_points[1:-1]
        # pts_sym = pts_sym[1:]
        tagg = []
        tag1 = []
        tag_pts = []


        floor_tags, floor_line_tags, flor_surface = self.build_surface_from_points(all_points, 0)
        ceiling_tag, ceiling_line_tags, ceiling_surface = self.build_surface_from_points(all_points, height)

        line_tag = []
        surface_entities = flor_surface + ceiling_surface
        for i in range(len(floor_tags)+1):
            if i < len(floor_tags):
                _line_tag = gmsh.model.occ.addLine(floor_tags[i], ceiling_tag[i], -1)
                line_tag.append(_line_tag)
            gmsh.model.occ.synchronize()
            if i > 1:
                temp_tags = [floor_line_tags[i-2],line_tag[i-1], -ceiling_line_tags[i-2],line_tag[i-2]]
                _it = gmsh.model.occ.addCurveLoop(temp_tags, -1)
                it = gmsh.model.occ.addSurfaceFilling(_it, -1)
                surface_entities.append(it)
                gmsh.model.occ.synchronize()
        temp_tags = [floor_line_tags[-1], line_tag[-1], -ceiling_line_tags[-1], line_tag[0]]
        _it = gmsh.model.occ.addCurveLoop(temp_tags, -1)
        it = gmsh.model.occ.addSurfaceFilling(_it, -1)
        surface_entities.append(it)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        surface_loop = gmsh.model.occ.addSurfaceLoop(surface_entities, sewing=True)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        _surface_tags = gmsh.model.occ.getEntities(2)
        # gmsh.model.occ.healShapes()

        gmsh.model.occ.addVolume([surface_loop])
        # gmsh.model.occ.healShapes(sewFaces=False)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        # tagg = []
        # for i in range(len(pts_sym)):
        #     tag = 2000+i
        #     tag2 = 10000+i
        #     gmsh.model.occ.addPoint(pts_sym[i,0], pts_sym[i,1], 0, 0., tag)
        #     # print(tag)
        #     if i>0:
        #     #     # print(tag[i],tag[i-1])
        #         gmsh.model.occ.addLine(2000+i, 1999+i, tag2)
        #         tagg.append(tag2)

        # gmsh.model.occ.addLine(200, 200+len(pts)-1,1000+len(pts))
        tg = gmsh.model.occ.getEntities(1)
        # gmsh.model.occ.copy(tg)
        # gmsh.model.occ.mirror(tg,1,0,0,0)
        # tg = gmsh.model.occ.getEntities(1)

        tgg = []
        for i in range(len(tg)):
            tgg.append(tg[i][1])

        # tgg.sort()
        _it = gmsh.model.occ.addCurveLoop(tgg, 20000)
        it = gmsh.model.occ.addSurfaceFilling(20000, 1454356)
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(2)

        # gmsh.model.addPhysicalGroup(2, [15001], 15001)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        # gmsh.model.occ.removeAllDuplicates()
        # gmsh.model.geo.extrude([(3,15001)],dx=0,dy=0,dz = zmax)
        a = gmsh.model.occ.extrude([(2, 1454356)], dx=0, dy=0, dz=height, recombine=False)
        gmsh.model.occ.synchronize()

        # gmsh.model.occ.remove([(3,1)])

        gmsh.model.occ.synchronize()
        # gmsh.model.occ.addVolume(a)
        # gmsh.model.occ.synchronize()

        gmsh.fltk.run()
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
        # gmsh.model.occ.addBox(2*np.amax(pts[:,0]), 2*np.amax(pts[:,1]), height*0.9, -4*np.amax(pts[:,0]),-4*np.amax(pts[:,1]), 2*height,tag=66666)
        # gmsh.model.occ.rotate([(3,66666)], 0, 0,height, 1, 0, 0, -angle)
        # gmsh.model.occ.synchronize()
        #
        # gmsh.model.occ.cut([(3,1)], [(3,66666)])
        # gmsh.model.occ.addVolume(tggv)

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], -1)
        gmsh.model.addPhysicalGroup(2, tggv, -1)
        gmsh.model.occ.synchronize()

        gmsh.fltk.run()

        # # print((self.c0*self.S_cale)/self.fmax/self.num_freq)
        # lc = 0#(self.c0*self.S_cale)/self.fmax/self.num_freq
        # tg = gmsh.model.occ.getEntities(3)
        # gmsh.model.occ.healShapes(tgv,sewFaces=True)
        # gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        # tg2 = gmsh.model.occ.getEntities(2)
        # if self.R_ != None:
        #     for i in range(len(self.R_.coord)):
        #         it = gmsh.model.occ.addPoint(self.R_.coord[i,0], self.R_.coord[i,1], self.R_.coord[i,2], lc, -1)
        #         gmsh.model.occ.synchronize()
        #         gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
        #
        # if self.S_ != None:
        #     for i in range(len(self.S_.coord)):
        #         it = gmsh.model.occ.addPoint(self.S_.coord[i,0], self.S_.coord[i,1], self.S_.coord[i,2], lc, -1)
        #         gmsh.model.occ.synchronize()
        #         gmsh.model.mesh.embed(0, [it], 3, tg[0][1])
        #

        # gmsh.model.mesh.embed(0, [15000], 3, tg[0][1])
        gmsh.model.mesh.generate(3)

        gmsh.model.occ.synchronize()
        # gmsh.write('current_mesh.msh')
        # gmsh.write('current_mesh.brep')
        # if self.plot:
        #     gmsh.fltk.run()
        # gmsh.model.occ.synchronize()
        # # gmsh.clear()
        #
        #
        # # gmsh.write('current_geo2.geo_unrolled')
        # Grid = fd.GridImport3D(self.AP, 'current_mesh.msh',S=self.S_,R=self.R_,fmax=self.fmax,num_freq=self.num_freq,order = self.order,scale=self.S_cale,plot=True)
        # self.nos = Grid.nos
        # self.elem_surf = Grid.elem_surf
        # self.elem_vol =  Grid.elem_vol
        # self.domain_index_surf =  Grid.domain_index_surf
        # self.domain_index_vol =Grid.domain_index_vol
        # self.number_ID_faces =Grid.number_ID_faces
        # self.number_ID_vol = Grid.number_ID_vol
        # self.NumNosC = Grid.NumNosC
        # self.NumElemC = Grid.NumElemC
        # self.order = Grid.order
        # self.path_to_geo = Grid.path_to_geo
        #
        # current_path = os.getcwd()
        # self.path_to_geo_unrolled = current_path+'\\current_mesh.brep'
        # try:
        #     gmsh.finalize()
        # except:
        #     pass

        # gmsh.model.mesh.optimize('Netgen')
        # # gmsh.model.mesh.optimize(method='Relocate3D',force=False)
        if self.order == 1:
            elemTy, elemTa, nodeTags = gmsh.model.mesh.getElements(3)
            self.elem_vol = np.array(nodeTags, dtype=int).reshape(-1, 4) - 1

            elemTys, elemTas, nodeTagss = gmsh.model.mesh.getElements(2)
            self.elem_surf = np.array(nodeTagss, dtype=int).reshape(-1, 3) - 1
            vtags, vxyz, _ = gmsh.model.mesh.getNodes()
            self.nos = vxyz.reshape((-1, 3)) / self.S_cale

        if self.order == 2:
            elemTy, elemTa, nodeTags = gmsh.model.mesh.getElements(3)
            self.elem_vol = np.array(nodeTags, dtype=int).reshape(-1, 10) - 1

            elemTys, elemTas, nodeTagss = gmsh.model.mesh.getElements(2)
            self.elem_surf = np.array(nodeTagss, dtype=int).reshape(-1, 6) - 1
            vtags, vxyz, _ = gmsh.model.mesh.getNodes()
            self.nos = vxyz.reshape((-1, 3)) / self.S_cale

        pg = gmsh.model.getPhysicalGroups(2)
        va = []
        vpg = []
        for i in range(len(pg)):
            v = gmsh.model.getEntitiesForPhysicalGroup(2, pg[i][1])
            for ii in range(len(v)):
                # print(v[ii])
                vvv = gmsh.model.mesh.getElements(2, v[ii])[1][0]
                pgones = np.ones_like(vvv) * pg[i][1]
                va = np.hstack((va, vvv))
                # print(pgones)
                vpg = np.hstack((vpg, pgones))

        vas = np.argsort(va)
        self.domain_index_surf = vpg[vas]
        # print(vas)
        pgv = gmsh.model.getPhysicalGroups(3)
        # print(pgv)
        vav = []
        vpgv = []
        for i in range(len(pgv)):
            vv = gmsh.model.getEntitiesForPhysicalGroup(3, pgv[i][1])
            for ii in range(len(vv)):
                # print(v[ii])
                vvv = gmsh.model.mesh.getElements(3, vv[ii])[1][0]
                pgones = np.ones_like(vvv) * pgv[i][1]
                vav = np.hstack((vav, vvv))
                # print(pgones)
                vpgv = np.hstack((vpgv, pgones))

        vasv = np.argsort(vav)
        self.domain_index_vol = vpgv[vasv]
        self.number_ID_faces = np.unique(self.domain_index_surf)
        self.number_ID_vol = np.unique(self.domain_index_vol)

        self.NumNosC = len(self.nos)
        self.NumElemC = len(self.elem_vol)
        import plotly.io as pio
        pio.renderers.default = "browser"
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
        gmsh.fltk.run()
        gmsh.finalize()

        # if self.plot:s
        #     gmsh.fltk.run()  

        # gmsh.write('current_geo.geo_unrolled')

        # self.path_to_geo = 'current_geo.geo_unrolled'


def embed_points_in_mesh(factory, embed_points):
    lc = 0
    for i in range(len(embed_points)):
        it = gmsh.model.occ.addPoint(embed_points[i, 0], embed_points[i, 1], embed_points[i, 2], lc, -1)
        factory.synchronize()
        gmsh.model.mesh.embed(0, [it], 3, gmsh.model.occ.getEntities(3)[0][1])

    factory.synchronize()


def extract_mesh_data_from_gmsh(dim, order, mesh_data):
    """
    Extracts mesh data from Gmsh object.

    Parameters
    ----------
    dim : int
        Mesh dimension.
    order : int
        Mesh order.
    mesh_data : dict
        Dictionary that the mesh data will be written into.
    """
    # gmsh.fltk.run()
    gmsh.model.mesh.generate(dim=dim)
    gmsh.model.mesh.setOrder(order)
    mesh_data['order'] = order
    if order == 1:
        order_volume_dimension = 4
        order_surface_dimension = 3
    elif order == 2:
        order_volume_dimension = 10
        order_surface_dimension = 6

    elif order != 1 or order != 2:
        raise ValueError(f"Order must be 1 or 2, linear or quadractic")

    if dim == 3:
        _, _, node_tags = gmsh.model.mesh.getElements(3)
        mesh_data['volume_elements'] = np.array(node_tags, dtype=int).reshape(-1, order_volume_dimension) - 1
    elif dim == 2:
        mesh_data['volume_elements'] = []
    elif dim != 2 or dim != 3:
        raise ValueError(f"Dim must be 2 or 3")

    _, _, node_tags_surface = gmsh.model.mesh.getElements(2)
    mesh_data['elements'] = (np.array(node_tags_surface, dtype=int).reshape(-1, order_surface_dimension) - 1).T.astype(
        "uint32")
    _, vertices_xyz, _ = gmsh.model.mesh.getNodes()
    mesh_data['vertices'] = vertices_xyz.reshape((-1, 3)).T
    pg = gmsh.model.getPhysicalGroups(2)
    tg2 = gmsh.model.occ.getEntities(2)

    va = []
    vpg = []
    for i in range(len(pg)):
        v = gmsh.model.getEntitiesForPhysicalGroup(2, pg[i][1])
        for ii in range(len(v)):
            vvv = gmsh.model.mesh.getElements(2, v[ii])[1][0]
            pgones = np.ones_like(vvv) * pg[i][1]
            va = np.hstack((va, vvv))
            vpg = np.hstack((vpg, pgones))
    vas = np.argsort(va)
    mesh_data['domain_indices'] = vpg[vas]
    # print(vas)
    if dim == 3:
        pgv = gmsh.model.getPhysicalGroups(3)
        vav = []
        vpgv = []
        for i in range(len(pgv)):
            vv = gmsh.model.getEntitiesForPhysicalGroup(3, pgv[i][1])
            for ii in range(len(vv)):
                vvv = gmsh.model.mesh.getElements(3, vv[ii])[1][0]
                pgones = np.ones_like(vvv) * pgv[i][1]
                vav = np.hstack((vav, vvv))
                vpgv = np.hstack((vpgv, pgones))

        vasv = np.argsort(vav)
        mesh_data['domain_indices_volume'] = vpgv[vasv]
    elif dim == 2:
        mesh_data['domain_indices_volume'] = []


if __name__ == "__main__":
    import femder as fd
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    import femder as fd
    import time

    then = time.time()
    from pymoo.problems.functional import FunctionalProblem

    AP = fd.AirProperties()
    AC = fd.AlgControls(AP, 20, 150, 1)
    BC = fd.BC(AC, AP)
    BC.normalized_admittance(2, 0.02)


    def optim_fun(X):
        if X[1] < (X[0] + X[3]) / 2:
            y = 10e6
            print("Geometry fails constraints")
            return y
        try:
            X = X / d_change  # Divide pela escala
            X[-1] = X[-1] / 10
            # try:
            pts = np.array([[0, 0], [X[0], 0], [X[1], X[2]], [X[3], X[4]],
                            [0, X[4]]])  # Montando os vértices da geometria com varíaveis da função
            height = X[5]
            angle = X[6]
            grid = fd.GeometryGenerator(AP, fmax=150, num_freq=7, plot=False)  # Gera objeto da geometria
            # # grid.generate_symmetric_polygon(pts, height) #Criando geometria a partir dos vétrices.
            grid.generate_symmetric_polygon_variheight(pts, height, angle)

            obj = fd.FEM3D(grid, S=None, R=None, AP=AP, AC=AC, BC=BC)  # Cria objeto de simulação

            # #Otimiza posição de fonte-receptor
            obj.optimize_source_receiver_pos([3, 3], fmin=20, fmax=150, star_average=True,
                                             minimum_distance_between_speakers=1.5,
                                             max_distance_from_wall=0.9, speaker_receiver_height=1.24,
                                             min_distance_from_backwall=0.7, max_distance_from_backwall=1,
                                             method='direct', neigs=200, plot_geom=False, renderer='notebook',
                                             print_info=False, saveFig=False, camera_angles=['diagonal_front'],
                                             plot_evaluate=False, plotBest=False)

            y = obj.bestMetric
            print(f'Fitness Metric: {y}')
            fitness.append(y)
            present_X.append(X)
            np.save('checkpoint_current.npy', algorithm)
            np.savetxt('fitness_current.txt', fitness)
            np.savetxt('present_X_current.txt', present_X)
        except Exception as e:
            y = 10e6
            print(e)
        return y


    L = np.array([1.55, 1.85, 1]) * 2.96  # Vértices do retângulo
    X = np.array([L[0] / 2, L[0] / 2, L[1] / 2, L[0] / 2, L[1], L[2] * 1.2, 0])

    d_change = 100
    Xmin = X - 0.5  # Limite inferior das varíaveis
    Xmin[-1] = 0
    Xmax = X + 0.5  # Limite superior das varíaveis
    Xmax[-1] = np.pi / 20

    varbound = np.vstack((Xmin, Xmax)).T  # Colocando a váriveis em colunas
    varbound = varbound * d_change  # Multiplica varíaveis por uma escala, determinando a casa decimal em que mudanças ocorrerão nos parâmetros da sala
    varbound[-1] = 10 * varbound[-1]
    # varbound=np.int64(varbound)

    fitness = []
    present_X = []
    functional_problem = FunctionalProblem(n_var=7,
                                           objs=optim_fun,
                                           xl=varbound[:, 0],
                                           xu=varbound[:, 1],
                                           type_var=np.int16)


    def optim_plot(X):
        X = X / d_change  # Divide pela escala
        X[-1] = X[-1] / 10
        # try:
        pts = np.array([[0, 0], [X[0], 0], [X[1], X[2]], [X[3], X[4]],
                        [0, X[4]]])  # Montando os vértices da geometria com varíaveis da função
        height = X[5]
        angle = X[6]
        # grid = fd.GeometryGenerator(AP, fmax=150, num_freq=12, plot=False)  # Gera objeto da geometria
        # # # grid.generate_symmetric_polygon(pts, height) #Criando geometria a partir dos vétrices.
        # grid.generate_symmetric_polygon_variheight(pts, height, angle)

        obj = fd.FEM3D(None, S=None, R=None, AP=AP, AC=AC, BC=BC)  # Cria objeto de simulação
        # obj.plot_problem()
        # #Otimiza posição de fonte-receptor
        obj.optimize_source_receiver_pos(pts, height, angle, 6, [3, 3], fmin=20, fmax=150, star_average=True,
                                         minimum_distance_between_speakers=1.15,
                                         max_distance_from_wall=0.9, speaker_receiver_height=1.24,
                                         min_distance_from_backwall=0.5, max_distance_from_backwall=0.9,
                                         method='direct', neigs=200, plot_geom=False, renderer='notebook',
                                         print_info=False, saveFig=False, camera_angles=['diagonal_front'],
                                         plot_evaluate=False, plotBest=False)

        y = obj.bestMetric
        print(f'Fitness Metric: {y}')
        return obj


    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.factory import get_sampling, get_crossover, get_mutation

    from pymoo.factory import get_termination

    termination = get_termination("n_gen", 1)

    from pymoo.optimize import minimize

    # np.savetxt(f'time_taken_{time.strftime("%Y%m%d-%H%M%S")}_FINAL.txt',then-time.time())
    dir_ = r'D:\Documents\UFSM\MNAV\Applied Acoustics\Optimizer\checkpoint_20211021-115238.npy'
    algorithm, = np.load(dir_, allow_pickle=True).flatten()

    algorithm.has_terminated = True
    res = minimize(functional_problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   copy_algorithm=False,
                   verbose=True)

    optim_plot(res.X)
