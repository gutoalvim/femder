# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:24:04 2020

@author: gutoa
"""
from femder.controlsair import AirProperties, AlgControls, sph2cart, cart2sph
from femder.FEM_1D import FEM1D
from femder.grid_importing import GridImport, GridImport3D
from femder.FEM_3D import FEM3D, fem_load, p2SPL
from femder.BoundaryConditions import BC
# from femder.TMM_rina_improved import TMM
from femder.sources import Source
from femder.receivers import Receiver
from femder.GeometryGenerate import GeometryGenerator
from femder.optimization_helpers import r_s_for_room, r_s_positions,r_s_from_grid,fitness_metric
from femder.utils import IR, SBIR
from femder.BEM_3D import BEM3D
from tmm.tmm import TMM
from femder import plot_tools
from femder import fem_numerical

plot_tools.set_plotly_renderer()