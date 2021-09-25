# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:19:55 2020

@author: gutoa
"""
import femder as fd
import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def r_s_coord_pair(y_value,distance_s_r):
        
    r_coord = np.array([0,y_value,1.2])
    
    hyp = distance_s_r
    
    rad30 = np.deg2rad(30)
    s_x = np.sin(rad30)*hyp
    s_y = np.cos(rad30)*hyp + y_value
    
    s_coord = np.array([[s_x,s_y,1.2],[-s_x,s_y,1.2]])
    
    R = fd.Receiver()
    R.star(r_coord,0.15)
    S = fd.Source()
    S.coord = s_coord
    S.q = np.array([[0.0001],[0.0001]])
    
    return R,S


def r_s_for_room(param,num_options):
    
    xmax = np.amax(param[:,0])
    ymax = np.amax(param[:,1])
    
    Ro = []
    So = []
    
    y_space = np.arange(ymax/3,(2*ymax/3)+0.15,0.15)
    distance_space = np.linspace(1.2,xmax-1,len(y_space))
    for i in range(len(y_space)):
        y_value = y_space[i]
        distance_s_r = distance_space[i]
        R,S = r_s_coord_pair(y_value, distance_s_r)
        Ro.append(R)
        So.append(S)
        
    return Ro,So


def r_s_positions(grid,grid_pts,bias):
        
    xmax = np.amax(grid.nos[:,0])
    ymax = np.amax(grid.nos[:,1])
    
    x_start = xmax*0.8+bias[0]
    x_end = xmax*0.5+bias[0]
    
    
    
    y_start = ymax*0.7 + bias[1]
    y_end = ymax*0.9+bias[1]
    
    y_space = np.linspace(y_start,y_end,grid_pts[1])
    x_space = np.linspace(x_start,x_end,grid_pts[0])
    s_coords_L = []#np.zeros(grid_pts)
    r_coords = []
    # print(s_coords_L.shape)
    # print(np.array([x_space[0],y_space[0],1.2]))
    for i in range(grid_pts[0]):
        for ii in range(grid_pts[1]):
            s_coords_L.append([x_space[i],y_space[ii],1.2])
            r_coords.append([0,(1/y_space[ii])*np.cos(np.deg2rad(30))*x_space[i]*2,1.21])
        
    s_coords_L = np.array(s_coords_L)
    s_coords_R = s_coords_L.copy()
    s_coords_R[:,0] = -s_coords_R[:,0]
    
    r_coords = np.array(r_coords)
    
    Ro = []
    So = []
    for i in range(len(r_coords)):
        
        R = fd.Receiver(coord = [r_coords[i]])
        Ro.append(R)
        S = fd.Source()
        S.coord = np.array([[s_coords_L[i,:]],[s_coords_R[i,:]]])
        S.q = np.array([[0.0001],[0.0001]])
        So.append(S)
    return So,Ro

def r_s_from_grid(grid,grid_pts,star_average=True,minimum_distance_between_speakers=1.2,
                  max_distance_from_wall=0.6,speaker_receiver_height=1.2,
                  min_distance_from_backwall=0.6,max_distance_from_backwall=1.5):
    
    ymax = np.amax(grid.nos[:,1])
    
    y_start = ymax - max_distance_from_backwall
    y_end = ymax-min_distance_from_backwall#*0.85
    
    y_space = np.linspace(y_start,y_end,grid_pts[1])
    
    y_height = speaker_receiver_height
    a = grid.nos
    xmax = []
    for i in y_space:
        
        b = a[(a[:,2]>y_height*.8) & (a[:,2]<y_height*1.2)]
        b = b[(b[:,1]>i*.8) & (b[:,1]<i*speaker_receiver_height)]
        xmax.append(np.amax(b[:,0]))
    
    xmax = np.array(xmax)
    
    x_end = xmax-max_distance_from_wall
    x_start = np.ones_like(x_end)*(minimum_distance_between_speakers/2)
    x_space = []
    for i in range(len(x_end)):
        x_space.append(np.linspace(x_start[i],x_end[i],grid_pts[0]))
    
    x_space = np.array(x_space)
    s_coords_L = []#np.zeros(grid_pts)
    r_coords = []
    # print(s_coords_L.shape)
    # print(np.array([x_space[0],y_space[0],1.2]))
    for i in range(grid_pts[0]):
        x_spacei = x_space[:,i]
        for ii in range(grid_pts[1]):
            s_coords_L.append([x_spacei[i],y_space[ii],speaker_receiver_height])
            r_coords.append([0,(y_space[ii])-np.cos(np.deg2rad(30))*x_spacei[i]*2,speaker_receiver_height])
     
        
    s_coords_L = np.array(s_coords_L)
    s_coords_R = s_coords_L.copy()
    s_coords_R[:,0] = -s_coords_R[:,0]
    
    r_coords = np.array(r_coords)
    
    Ro = []
    So = []
    if star_average:
        for i in range(len(r_coords)):
            
            R = fd.Receiver()
            R.star(r_coords[i], 0.15)
            Ro.append(R)
            S = fd.Source()
            S.coord = np.array([[s_coords_L[i,:]],[s_coords_R[i,:]]])
            S.q = np.array([[0.0001],[0.0001]])
            So.append(S)
    else:
        for i in range(len(r_coords)):
            
            R = fd.Receiver(coord = [r_coords[i]])
            Ro.append(R)
            S = fd.Source()
            S.coord = np.array([[s_coords_L[i,:]],[s_coords_R[i,:]]])
            S.q = np.array([[0.0001],[0.0001]])
            So.append(S)
    return So,Ro


def fitness_metric(complex_pressure,AC,fmin,fmax):
    
    fs = 44100
    
    fmin_indx = np.argwhere(AC.freq==fmin)[0][0]
    fmax_indx = np.argwhere(AC.freq==fmax)[0][0]
    
    w_sbir = 1/2
    w_modal = 1/2
    
    df = (AC.freq[-1]-AC.freq[0])/len(AC.freq)
    
    ir_duration = 1/df
    
    ir = fd.IR(fs,ir_duration,fmin,fmax).compute_room_impulse_response(complex_pressure.ravel())
    t_ir = np.linspace(0,ir_duration,len(ir))
    sbir = fd.SBIR(ir,t_ir,AC.freq[0],AC.freq[-1],method='peak')
    sbir_freq = sbir[1]
    sbir_SPL = fd.p2SPL(sbir_freq)[fmin_indx:fmax_indx]
    modal_SPL = fd.p2SPL(complex_pressure.ravel()).ravel()[fmin_indx:fmax_indx]
    
    std_sbir = w_sbir*np.std(sbir_SPL)
    std_modal = w_modal*np.std(modal_SPL)
    
    fm = std_sbir + std_modal
    
    return fm    
    




    
    