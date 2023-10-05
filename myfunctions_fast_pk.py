# Import external modules
import plotly
import seaborn as sns
sns.set(style="darkgrid", palette="colorblind")#, font_scale = 2)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import numba
from sklearn.utils import check_array

import warnings
# warnings.filterwarnings('ignore')
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import pandas as pandas

import random
import pymp
import time
import math
import os
import matplotlib.pyplot as plt
from pyDOE import lhs

from scipy import signal
from scipy import stats
from scipy.stats import norm


# Generate the paraview files that serve to create the result figures for the prediction error from BSP data

def testRootTimeEffect(threadsNum_val, meshName_val, conduction_speeds, rootNodeResolution, healthy_val, has_endocardial_layer_val):
    global is_healthy
    is_healthy = healthy_val
    global has_endocardial_layer
    has_endocardial_layer = has_endocardial_layer_val
    global meshName
    meshName = meshName_val
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            nlhsParam = 2
        else:
            nlhsParam = 4
    else:
        nlhsParam = 3
    global experiment_output
    experiment_output = 'atm'

    
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'

    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # rv endocardium triangles
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int) # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int)
    
    global rootNodeActivationIndexes
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    nRootLocations=rootNodeActivationIndexes.shape[0]
    nparam = nlhsParam + nRootLocations
    
    rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
    rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    
    global target_rootNodes
    target_rootNodes = nodesXYZ[rootNodesIndexes_true, :]

    # Set endocardial edges aside
    global isEndocardial
    if has_endocardial_layer:
        isEndocardial=np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                              np.all(np.isin(edges, rvface), axis=1))
    else:
        isEndocardial = np.zeros((edges.shape[0])).astype(bool)
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None # Clear Memory

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)

    # Generate baseline
    target_output = eikonal_ecg(np.array([conduction_speeds])/1000, rootNodesIndexes_true)
    target_output = target_output[0, :]
    
    # Fibre x 2
    print('Fibre x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[0] = fibre_conduction_speeds[0] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Fibre / 2
    print('Fibre / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[0] = fibre_conduction_speeds[0] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    
    print()
    print('---------------------')
    
    # Sheet x 2
    print('Sheet x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[1] = fibre_conduction_speeds[1] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Sheet / 2
    print('Sheet / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[1] = fibre_conduction_speeds[1] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    
    print()
    print('---------------------')
    
    # Sheet-normal x 2
    print('Sheet-normal x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[2] = fibre_conduction_speeds[2] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Sheet-normal / 2
    print('Sheet-normal / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[2] = fibre_conduction_speeds[2] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    
    print()
    print('---------------------')
    
    # Endocardial x 2
    print('Endocardial x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[3] = fibre_conduction_speeds[3] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Endocardial / 2
    print('Endocardial / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[3] = fibre_conduction_speeds[3] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])


def testSpeedEffect(threadsNum_val, meshName_val, conduction_speeds, rootNodeResolution, healthy_val, has_endocardial_layer_val):
    global is_healthy
    is_healthy = healthy_val
    global has_endocardial_layer
    has_endocardial_layer = has_endocardial_layer_val
    global meshName
    meshName = meshName_val
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            nlhsParam = 2
        else:
            nlhsParam = 4
    else:
        nlhsParam = 3
    global experiment_output
    experiment_output = 'atm'

    
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'

    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # rv endocardium triangles
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int) # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int)
    
    global rootNodeActivationIndexes
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    nRootLocations=rootNodeActivationIndexes.shape[0]
    nparam = nlhsParam + nRootLocations
    
    rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
    rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    
    global target_rootNodes
    target_rootNodes = nodesXYZ[rootNodesIndexes_true, :]

    # Set endocardial edges aside
    global isEndocardial
    if has_endocardial_layer:
        isEndocardial=np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                              np.all(np.isin(edges, rvface), axis=1))
    else:
        isEndocardial = np.zeros((edges.shape[0])).astype(bool)
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None # Clear Memory

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)

    # Generate baseline
    target_output = eikonal_ecg(np.array([conduction_speeds])/1000, rootNodesIndexes_true)
    target_output = target_output[0, :]
    
    # Fibre x 2
    print('Fibre x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[0] = fibre_conduction_speeds[0] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Fibre / 2
    print('Fibre / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[0] = fibre_conduction_speeds[0] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    
    print()
    print('---------------------')
    
    # Sheet x 2
    print('Sheet x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[1] = fibre_conduction_speeds[1] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Sheet / 2
    print('Sheet / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[1] = fibre_conduction_speeds[1] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    
    print()
    print('---------------------')
    
    # Sheet-normal x 2
    print('Sheet-normal x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[2] = fibre_conduction_speeds[2] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Sheet-normal / 2
    print('Sheet-normal / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[2] = fibre_conduction_speeds[2] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    
    print()
    print('---------------------')
    
    # Endocardial x 2
    print('Endocardial x 2')
    print(conduction_speeds)
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[3] = fibre_conduction_speeds[3] * 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])
    # Endocardial / 2
    print('Endocardial / 2')
    fibre_conduction_speeds = conduction_speeds.copy()
    fibre_conduction_speeds[3] = fibre_conduction_speeds[3] / 2
    print(fibre_conduction_speeds)
    fibre_moved = eikonal_ecg(np.array([fibre_conduction_speeds])/1000, rootNodesIndexes_true)
    print(np.corrcoef(target_output, fibre_moved[0, :])[0, 1])


def recalculateError(threadsNum_val, meshName_val, pred_fileName_list, target_fileName, conduction_speeds, rootNodeResolution,
                     previousResultsPath, atmapPath, target_type, healthy_val,  load_target, has_endocardial_layer_val,
                     is_ECGi):
    time_s = time.time()
    global is_healthy
    is_healthy = healthy_val
    global has_endocardial_layer
    has_endocardial_layer = has_endocardial_layer_val
    global meshName
    meshName = meshName_val
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = [ 'I' , 'II' , 'V1' , 'V2' , 'V3' , 'V4' , 'V5' , 'V6', 'lead_prog' ]
#     electrodeNames = ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_leads
    global nb_bsp
#     nb_roots_range = [2, 5] # range of number of root nodes per endocardial chamber
    frequency = 1000 # Hz
    freq_cut= 150 # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2) # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            nlhsParam = 2
        else:
            nlhsParam = 4
    else:
        nlhsParam = 3
    global experiment_output
    if 'atm' in target_type:
        experiment_output = 'atm'
    elif 'ecg' in target_type:
        experiment_output = 'ecg'
    elif 'bsp' in target_type:
        experiment_output = 'bsp'
        
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'

    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # rv endocardium triangles
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int) # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int)
    
    global rootNodeActivationIndexes
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    nRootLocations=rootNodeActivationIndexes.shape[0]
    nparam = nlhsParam + nRootLocations
    
    rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
    rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    
    global target_rootNodes
    target_rootNodes = nodesXYZ[rootNodesIndexes_true, :]

    if experiment_output == 'ecg' or experiment_output == 'bsp':
        global tetrahedrons
        tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
        tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
        ecg_electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
        if experiment_output == 'bsp':
            ECGi_electrodePositions = np.loadtxt(dataPath + meshName + '_ECGiElectrodePositions.csv', delimiter=',')
            nb_leads = 8 + ECGi_electrodePositions.shape[0] # All leads from the ECGi are calculated like the precordial leads
            electrodePositions = np.concatenate((ecg_electrodePositions, ECGi_electrodePositions), axis=0)
        elif experiment_output == 'ecg':
            nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad  # 8 + lead progression (or 12)
            electrodePositions = ecg_electrodePositions
        nb_bsp = electrodePositions.shape[0]

        aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
        for i in range(0, tetrahedrons.shape[0], 1):
            aux[tetrahedrons[i, 0]].append(i)
            aux[tetrahedrons[i, 1]].append(i)
            aux[tetrahedrons[i, 2]].append(i)
            aux[tetrahedrons[i, 3]].append(i)
        global elements
        elements = [np.array(n) for n in aux]
        aux = None # Clear Memory

        # Precompute PseudoECG stuff
        # Calculate the tetrahedrons volumes
        D = nodesXYZ[tetrahedrons[:, 3], :] # RECYCLED
        A = nodesXYZ[tetrahedrons[:, 0], :]-D # RECYCLED
        B = nodesXYZ[tetrahedrons[:, 1], :]-D # RECYCLED
        C = nodesXYZ[tetrahedrons[:, 2], :]-D # RECYCLED
        D = None # Clear Memory

        global tVolumes
        tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1
                                               ), (np.cross(B, C)[:, :, np.newaxis]))),
                              tetrahedrons.shape[0]) #Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
        tVolumes = tVolumes/np.sum(tVolumes)

        # Calculate the tetrahedron (temporal) voltage gradients
        Mg = np.stack((A, B, C), axis=-1)
        A = None # Clear Memory
        B = None # Clear Memory
        C = None # Clear Memory

        # Calculate the gradients
        global G_pseudo
        G_pseudo = np.zeros(Mg.shape)
        for i in range(Mg.shape[0]):
            G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
        G_pseudo = np.moveaxis(G_pseudo, 1, 2)
        Mg = None # clear memory

        # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
        r=np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
               (tetrahedronCenters.shape[0],
                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1)-electrodePositions

        global d_r
        d_r= np.moveaxis(np.multiply(
            np.moveaxis(r, [0, 1], [-1, -2]),
            np.multiply(np.moveaxis(np.sqrt(np.sum(r**2, axis=2))**(-3), 0, 1), tVolumes)), 0, -1)

    # Set endocardial edges aside
    global isEndocardial
    if has_endocardial_layer:
        isEndocardial=np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                              np.all(np.isin(edges, rvface), axis=1))
    else:
        isEndocardial = np.zeros((edges.shape[0])).astype(bool)
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None # Clear Memory

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)

    # Generate target data
    if load_target:
        target_output = np.loadtxt(atmapPath + target_fileName, delimiter=',')
        if is_ECGi:
            target_output = target_output[epiface]
        # Load and compile Numba
        if has_endocardial_layer:
            if is_healthy:
                compilation_params = np.array([np.concatenate((np.array([0.1, 0.1]),
                                           np.ones(rootNodeActivationIndexes.shape).astype(int)))])
            else:
                compilation_params = np.array([np.concatenate((np.array([0.1, 0.1, 0.1, 0.1]),
                                           np.ones(rootNodeActivationIndexes.shape).astype(int)))])
        else:
            compilation_params = np.array([np.concatenate((np.array([0.1, 0.1, 0.1]),
                                           np.ones(rootNodeActivationIndexes.shape).astype(int)))])
        eikonal_ecg(compilation_params, rootNodeActivationIndexes)
    else:
        # Load and compile Numba
        target_output = eikonal_ecg(np.array([conduction_speeds])/1000, rootNodesIndexes_true)
        ind=[slice(None)]*target_output.ndim
        ind[0] = 0
        # if experiment_output == 'atm':
        target_output = target_output[tuple(ind)]
        # else:
        #     target_output = target_output[0, :, :]
            # target_output = target_output[:, np.logical_not(np.isnan(target_output[0, :]))]
    
    # Iterate over population files
    print(previousResultsPath)
    print(pred_fileName_list)
    population = np.loadtxt(previousResultsPath + pred_fileName_list[0], delimiter=',')
    for i in range(1, len(pred_fileName_list)):
        population = np.concatenate((population, np.loadtxt(previousResultsPath + pred_fileName_list[i],
                                     delimiter=',')), axis=0)
    population, unique_particles = np.unique(population, return_inverse=True, axis=0)
    prediction_1 = eikonal_ecg(population[0:1, :], rootNodeActivationIndexes)
    shape_1 = np.array(prediction_1.shape) # shape returns a tuple which is a static class that cannot be modified
    shape_1[0] = population.shape[0]
    prediction_list = np.full(shape_1, np.nan)
    for i in range(0, population.shape[0], threadsNum):
        prediction_list[i:min(i+threadsNum, population.shape[0])] = eikonal_ecg(population[i:min(i+threadsNum, population.shape[0])], rootNodeActivationIndexes)
    if experiment_output == 'atm':
        correlation_list = np.zeros((prediction_list.shape[0]))
    else:
        correlation_list = np.zeros((prediction_list.shape[0], nb_leads))
    for i in range(prediction_list.shape[0]):
        if experiment_output == 'atm':
            correlation_list[i] = np.corrcoef(target_output, prediction_list[i, :])[0, 1]
        else:
            signal_length = max(np.sum(np.logical_not(np.isnan(target_output[0, :]))), np.sum(np.logical_not(np.isnan(prediction_list[i, 0, :]))))
            aux_target_output = np.zeros((nb_leads, signal_length))
            aux_target_output[:, :np.sum(np.logical_not(np.isnan(target_output[0, :])))] = target_output[:, :np.sum(np.logical_not(np.isnan(target_output[0, :])))]
            aux_prediction_output = np.zeros((nb_leads, signal_length))
            aux_prediction_output[:, :np.sum(np.logical_not(np.isnan(prediction_list[i, 0, :])))] = prediction_list[i, :, :np.sum(np.logical_not(np.isnan(prediction_list[i, 0, :])))]
            for i_lead in range(nb_leads):
                correlation_list[i, i_lead] = np.corrcoef(aux_target_output[i_lead, :], aux_prediction_output[i_lead, :])[0, 1]
    # if experiment_output == 'atm':
    ind=[slice(None)]*correlation_list.ndim
    ind[0] = unique_particles
    correlation_list = correlation_list[tuple(ind)]
    # else:
    #     correlation_list = correlation_list[unique_particles, :]
    correlation_list = correlation_list.flatten()
    
    print('Time: ' + str(round(time.time()-time_s)))
    return np.mean(correlation_list), np.std(correlation_list), correlation_list



def bspPredictionError(threadsNum_val, meshName_val, conduction_speeds, rootNodeResolution, previousResultsPath, healthy_val,
                       targetType, figuresPath, is_ECGi, has_endocardial_layer_val):
    global is_healthy
    is_healthy = healthy_val
    global has_endocardial_layer
    has_endocardial_layer = has_endocardial_layer_val
    global meshName
    meshName = meshName_val
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = [ 'I' , 'II' , 'V1' , 'V2' , 'V3' , 'V4' , 'V5' , 'V6', 'lead_prog' ]
#     electrodeNames = ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_leads
    global nb_bsp
#     nb_roots_range = [2, 5] # range of number of root nodes per endocardial chamber
    frequency = 1000 # Hz
    freq_cut= 150 # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2) # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            nlhsParam = 2
        else:
            nlhsParam = 4
    else:
        nlhsParam = 3
    global experiment_output
    if 'atm' in targetType:
        experiment_output = 'atm'
    elif 'ecg' in targetType:
        experiment_output = 'ecg'
    elif 'bsp' in targetType:
        experiment_output = 'bsp'
        
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'

    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
                                   delimiter=',') - 1).astype(int)) # rv endocardium triangles
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int) # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int)
    #print(lvActivationIndexes.shape)
    #print(rvActivationIndexes.shape)
    
    global rootNodeActivationIndexes
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    nRootLocations=rootNodeActivationIndexes.shape[0]
    nparam = nlhsParam + nRootLocations
    
    if is_ECGi:
        rootNodesIndexes_true = rootNodeActivationIndexes
    else:
        rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
        rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
        
    global target_rootNodes
    target_rootNodes = nodesXYZ[rootNodesIndexes_true, :]

    if experiment_output == 'ecg' or experiment_output == 'bsp':
        global tetrahedrons
        tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
        tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
        ecg_electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
        if experiment_output == 'bsp':
            ECGi_electrodePositions = np.loadtxt(dataPath + meshName + '_ECGiElectrodePositions.csv', delimiter=',')
            nb_leads = 8 + ECGi_electrodePositions.shape[0] # All leads from the ECGi are calculated like the precordial leads
            electrodePositions = np.concatenate((ecg_electrodePositions, ECGi_electrodePositions), axis=0)
        elif experiment_output == 'ecg':
            nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad  # 8 + lead progression (or 12)
            electrodePositions = ecg_electrodePositions
        nb_bsp = electrodePositions.shape[0]

        aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
        for i in range(0, tetrahedrons.shape[0], 1):
            aux[tetrahedrons[i, 0]].append(i)
            aux[tetrahedrons[i, 1]].append(i)
            aux[tetrahedrons[i, 2]].append(i)
            aux[tetrahedrons[i, 3]].append(i)
        global elements
        elements = [np.array(n) for n in aux]
        aux = None # Clear Memory

        # Precompute PseudoECG stuff
        # Calculate the tetrahedrons volumes
        D = nodesXYZ[tetrahedrons[:, 3], :] # RECYCLED
        A = nodesXYZ[tetrahedrons[:, 0], :]-D # RECYCLED
        B = nodesXYZ[tetrahedrons[:, 1], :]-D # RECYCLED
        C = nodesXYZ[tetrahedrons[:, 2], :]-D # RECYCLED
        D = None # Clear Memory

        global tVolumes
        tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1
                                               ), (np.cross(B, C)[:, :, np.newaxis]))),
                              tetrahedrons.shape[0]) #Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
        tVolumes = tVolumes/np.sum(tVolumes)

        # Calculate the tetrahedron (temporal) voltage gradients
        Mg = np.stack((A, B, C), axis=-1)
        A = None # Clear Memory
        B = None # Clear Memory
        C = None # Clear Memory

        # Calculate the gradients
        global G_pseudo
        G_pseudo = np.zeros(Mg.shape)
        for i in range(Mg.shape[0]):
            G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
        G_pseudo = np.moveaxis(G_pseudo, 1, 2)
        Mg = None # clear memory

        # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
        r=np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
               (tetrahedronCenters.shape[0],
                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1)-electrodePositions

        global d_r
        d_r= np.moveaxis(np.multiply(
            np.moveaxis(r, [0, 1], [-1, -2]),
            np.multiply(np.moveaxis(np.sqrt(np.sum(r**2, axis=2))**(-3), 0, 1), tVolumes)), 0, -1)

    # Set endocardial edges aside
    global isEndocardial
    if has_endocardial_layer:
        isEndocardial=np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                              np.all(np.isin(edges, rvface), axis=1))
    else:
        isEndocardial = np.zeros((edges.shape[0])).astype(bool)
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None # Clear Memory

    # Generate target data
    target_output = eikonal_ecg(np.array([conduction_speeds])/1000, rootNodesIndexes_true)
    target_output = target_output[0, :, :]
    target_output = target_output[:, np.logical_not(np.isnan(target_output[0, :]))]

    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_' + rootNodeResolution + 'Res.csv', delimiter=',') - 1).astype(int)
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)

    # Load an example result
    for i in range(0, 5, 1):
        fileName = (previousResultsPath + meshName + '_' + rootNodeResolution + '_' + str(conduction_speeds) + '_' \
                    + rootNodeResolution + '_' + targetType + '_' + str(i) + '_population.csv')
        if os.path.isfile(fileName):
            break
    print(fileName)
    population = np.loadtxt(fileName, delimiter=',')
    discrepancy = np.loadtxt(fileName.replace('population', 'discrepancy'), delimiter=',')
    best_ind = np.argmin(discrepancy)
    print(discrepancy[0])
    print(discrepancy[best_ind])
    print(best_ind)
    print(np.amax(discrepancy))
    best_particle = population[best_ind:best_ind+1, :] # not really random because I take the one with minimum discrepancy
    prediction = eikonal_ecg(best_particle, rootNodeActivationIndexes)[0, :nb_leads, :]
    signal_length = min(np.sum(np.logical_not(np.isnan(target_output[0, :]))), np.sum(np.logical_not(np.isnan(prediction[0, :]))))
    correlation_list = np.zeros((nb_leads))
    for i in range(nb_leads):
        correlation_list[i] = np.corrcoef(target_output[i, :signal_length], prediction[i, :signal_length])[0, 1]
    # print(correlation_list)

    print ('Export mesh to ensight format')
    torso_face = (np.loadtxt(dataPath + meshName + '_coarse_torsoface.csv', delimiter=',') - 1).astype(int)
    torso_xyz = (np.loadtxt(dataPath + meshName + '_coarse_torsoxyz.csv', delimiter=',') - 1).astype(int)
    # aux_elems = torso_face + 1    # Make indexing Paraview and Matlab friendly
    # with open(figuresPath + meshName+'.ensi.geo', 'w') as f:
    #     f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(torso_xyz.shape[0])+'\n')
    #     for i in range(0, torso_xyz.shape[0]):
    #         f.write(str(i+1)+'\n')
    #     for c in [0,1,2]:
    #         for i in range(0, torso_xyz.shape[0]):
    #             f.write(str(torso_xyz[i,c])+'\n')
    #     print('Write tria3...')
    #     f.write('tria3\n  '+str(len(aux_elems))+'\n')
    #     for i in range(0, len(aux_elems)):
    #         f.write('  '+str(i+1)+'\n')
    #     for i in range(0, len(aux_elems)):
    #         f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\n')
    # with open(figuresPath+meshName+'.ensi.case', 'w') as f:
    #     f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
    #     f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
    #     f.write('scalar per node: 1	corr_'+rootNodeResolution+'	'+meshName+'_'+targetType+'.ensi.corr_'+rootNodeResolution+'\n')
    
    # Save Spatio-temporal correlation between true bidomain and predicted eikonal
    # print(electrodePositions.shape)
    # print(correlation_list.shape)
    correlation_list = correlation_list[2:]
    electrodePositions = electrodePositions[4:, :]
    
    torso_surface_node_IDs = np.unique(torso_face.flatten(order='C'))
    torso_surface_nodes = torso_xyz[torso_surface_node_IDs, :]
    distances = np.linalg.norm(torso_surface_nodes[:, np.newaxis, :]-electrodePositions, ord=2, axis=2)
    # print(distances.shape)
    closest_electrode = np.argmin(distances, axis=1)
    # print(closest_electrode.shape)
    
    fig, axs = plt.subplots(nrows=1, ncols=6, constrained_layout=True, figsize=(8, 3), sharey='all')
    max_count = signal_length + 5
    for i in range(6):
        leadName = leadNames[i+2]
        # Print out Pearson's correlation coefficients for each lead
        print(leadName + ': ' + str(correlation_list[i]))
        # print(signal_length)
        # print(target_output.shape)
        # print(prediction.shape)
        # print(correlation_list.shape)
        # print(np.corrcoef(target_output[i+2, :signal_length], prediction[i+2, :signal_length])[0, 1])
        axs[i].plot(target_output[i+2, :signal_length], 'k-', label='target', linewidth=1.5)
        axs[i].plot(prediction[i+2, :signal_length], 'g-', label='prediction', linewidth=1.5)

        # decorate figure
        axs[i].set_xlim(0, max_count)
        axs[i].xaxis.set_major_locator(MultipleLocator(40))
        axs[i].yaxis.set_major_locator(MultipleLocator(2))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.grid(True, which='major')
        axs[i].yaxis.grid(True, which='major')
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        axs[i].set_title('Lead ' + leadName, fontsize=14)
        axs[i].set_xlabel('ms', fontsize=14)
    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show()
    
    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8, 3), sharey='all')
    max_count = signal_length + 5
    id_1 = np.argmax(correlation_list)
    id_2 = np.argmin(correlation_list)
    print('max: '+ str(id_1) + ' , ' + str(correlation_list[id_1]))
    print('min: '+ str(id_2) + ' , ' + str(correlation_list[id_2]))
    # Print out Pearson's correlation coefficients for each lead
    axs[0].plot(target_output[id_1+2, :signal_length], 'k-', label='target', linewidth=1.5)
    axs[0].plot(prediction[id_1+2, :signal_length], 'g-', label='prediction', linewidth=1.5)
    # decorate figure
    axs[0].set_xlim(0, max_count)
    axs[0].xaxis.set_major_locator(MultipleLocator(40))
    axs[0].yaxis.set_major_locator(MultipleLocator(2))
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[0].xaxis.grid(True, which='major')
    axs[0].yaxis.grid(True, which='major')
    for tick in axs[0].xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    for tick in axs[0].yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    axs[0].set_xlabel('ms', fontsize=14)
    axs[0].set_ylabel('Max correlation', fontsize=14)
    
    # Print out Pearson's correlation coefficients for each lead
    axs[1].plot(target_output[id_2+2, :signal_length], 'k-', label='target', linewidth=1.5)
    axs[1].plot(prediction[id_2+2, :signal_length], 'g-', label='prediction', linewidth=1.5)
    # decorate figure
    axs[1].set_xlim(0, max_count)
    axs[1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1].yaxis.set_major_locator(MultipleLocator(2))
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[1].xaxis.grid(True, which='major')
    axs[1].yaxis.grid(True, which='major')
    for tick in axs[1].xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    for tick in axs[1].yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    axs[1].set_xlabel('ms', fontsize=14)
    axs[1].set_ylabel('Min correlation', fontsize=14)
    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show()
    
    correlation_map = np.zeros((torso_xyz.shape[0]))
    correlation_map[torso_surface_node_IDs] = correlation_list[closest_electrode]
    
    np.savetxt(figuresPath + meshName + '_torso_electrodes.csv', electrodePositions, delimiter=',')
    np.savetxt(figuresPath + meshName + '_torso_correlations.csv', correlation_map, delimiter=',')

    
    # with open(figuresPath + meshName+'_'+targetType+'.ensi.corr_'+rootNodeResolution, 'w') as f:
    #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #     for i in range(0, len(correlation_map)):
    #         f.write(str(correlation_map[i]) + '\n')
    

    #
    #
    #     lv_rootNodes = None
    #     rv_rootNodes = None
    #     lv_rootNodes_list = None
    #     rv_rootNodes_list = None
    #     rootNodes_list = None
    #     # new_lv_roots_ids_list = []
    #     # new_lv_roots_weights_list = []
    #     # new_rv_roots_ids_list = []
    #     # new_rv_roots_weights_list = []
    #     new_lv_roots_list = []
    #     new_rv_roots_list = []
    #     data_list = []
    #     disc_roots_list = []
    #     count = 0.
    #
    #             # Calculate Djikstra distances in the LV endocardium
    #             #lvdistance_mat = np.zeros((lvnodes.shape[0], lvActivationIndexes.shape[0]))
    #     lvnodesXYZ = nodesXYZ[lvnodes, :]
    #     # Set endocardial edges aside
    #     lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    #     lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    #     lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    #     lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :] # edge vectors
    #     lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    #     aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    #     for i in range(0, len(lvunfoldedEdges), 1):
    #         aux[lvunfoldedEdges[i, 0]].append(i)
    #     lvneighbours = [np.array(n) for n in aux]
    #     aux = None # Clear Memory
    #     lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
    #     lvdistance_mat = djikstra(lvActnode_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours)
    #
    #             # Calculate Djikstra distances in the RV endocardium
    #             #rvdistance_mat = np.zeros((rvnodes.shape[0], rvActivationIndexes.shape[0]))
    #             rvnodesXYZ = nodesXYZ[rvnodes, :]
    #             # Set endocardial edges aside
    #             rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    #             rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    #             rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    #             rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :] # edge vectors
    #             rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    #             aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    #             for i in range(0, len(rvunfoldedEdges), 1):
    #                 aux[rvunfoldedEdges[i, 0]].append(i)
    #             rvneighbours = [np.array(n) for n in aux]
    #             aux = None # Clear Memory
    #             rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
    #             rvdistance_mat = djikstra(rvActnode_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours)
    #
    #             for fileName in population_files:
    #                 if meshName in fileName and targetType in fileName and discretisation_resolution in fileName:
    #                     population, particles_index = np.unique(np.loadtxt(previousResultsPath + fileName, delimiter=','), axis=0, return_index=True)
    #                     rootNodes = np.round(population[:, 4:]).astype(int) # Non-repeated particles
    #                     count = count + rootNodes.shape[0]
    #                     # Number of Root nodes percentage-error
    #                     lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
    #                     rv_rootNodes_part = rootNodes[:, len(lvActivationIndexes):]
    #
    #                     # K-means for root nodes
    #                     # Choose the number of clusters 'k'
    #                     lv_num_nodes = np.sum(lv_rootNodes_part, axis=1)
    #                     lv_num_nodes, lv_num_nodes_counts = np.unique(lv_num_nodes, return_counts=True, axis=0)
    #                     lv_num_clusters = lv_num_nodes[np.argmax(lv_num_nodes_counts)]
    #
    #                     rv_num_nodes = np.sum(rv_rootNodes_part, axis=1)
    #                     rv_num_nodes, rv_num_nodes_counts = np.unique(rv_num_nodes, return_counts=True, axis=0)
    #                     rv_num_clusters = rv_num_nodes[np.argmax(rv_num_nodes_counts)]
    #
    #                     # Choose the initial centroides
    #                     k_lv_rootNodes_list, lvunique_counts = np.unique(lv_rootNodes_part, return_counts=True, axis=0)
    #                     lv_rootNodes_list_index = np.sum(k_lv_rootNodes_list, axis=1) == lv_num_clusters
    #                     k_lv_rootNodes_list = k_lv_rootNodes_list[lv_rootNodes_list_index, :]
    #                     lvunique_counts = lvunique_counts[lv_rootNodes_list_index]
    #                     lv_centroid_ids = lvActnode_ids[k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)].astype(int)
    #         #             print('lv: ' +str(lv_centroid_ids))
    #
    #                     k_rv_rootNodes_list, rvunique_counts = np.unique(rv_rootNodes_part, return_counts=True, axis=0)
    #                     rv_rootNodes_list_index = np.sum(k_rv_rootNodes_list, axis=1) == rv_num_clusters
    #                     k_rv_rootNodes_list = k_rv_rootNodes_list[rv_rootNodes_list_index, :]
    #                     rvunique_counts = rvunique_counts[rv_rootNodes_list_index]
    #                     rv_centroid_ids = rvActnode_ids[k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)].astype(int)
    #         #             print('rv: ' +str(rv_centroid_ids))
    #         #             raise
    #
    #                     # Check that everything is OK
    #         #             print(np.all(lvnodesXYZ[lv_centroid_ids, :] == nodesXYZ[lvActivationIndexes[k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)], :]))
    #         #             print(np.all(rvnodesXYZ[rv_centroid_ids, :] == nodesXYZ[rvActivationIndexes[k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)], :]))
    #
    #                     # Transform the root nodes predicted data into k-means data
    #                     lvdata_ids = np.concatenate(([lvActivationIndexes[(lv_rootNodes_part[i, :]).astype(bool)] for i in range(lv_rootNodes_part.shape[0])]), axis=0)
    #                     lvdata_ids = np.asarray([np.flatnonzero(lvActivationIndexes == node_id)[0] for node_id in lvdata_ids]).astype(int)
    #
    #                     rvdata_ids = np.concatenate(([rvActivationIndexes[(rv_rootNodes_part[i, :]).astype(bool)] for i in range(rv_rootNodes_part.shape[0])]), axis=0)
    #                     rvdata_ids = np.asarray([np.flatnonzero(rvActivationIndexes == node_id)[0] for node_id in rvdata_ids]).astype(int)
    #
    #                     # K-means algorithm
    #                     # LV
    #                     k_means_lv_labels, k_means_lv_centroids = k_means(lvnodesXYZ[lvActnode_ids, :], lvdata_ids, lv_num_clusters, lv_centroid_ids, lvdistance_mat, lvnodesXYZ, max_iter=10)
    #
    #                     # Any node in the endocardium can be a result
    #                     new_lv_roots = np.unique(lvnodes[k_means_lv_centroids])
    #
    #                     # # We project the centroids to potential root nodes using distance as a weight for a better visualisation
    #                     # nb_neighbours = 4
    #                     # new_lv_roots_ids = np.zeros((k_means_lv_centroids.shape[0], nb_neighbours)).astype(int)
    #                     # new_lv_roots_weights = np.zeros((k_means_lv_centroids.shape[0], nb_neighbours)).astype(float)
    #                     # distAux = np.linalg.norm(lvnodesXYZ[lvActnode_ids, np.newaxis, :] - lvnodesXYZ[k_means_lv_centroids, :], ord=2, axis=2)
    #                     # # Take the nb_neighbours closest potential root nodes and weight them based on distance to the k-means returned position
    #                     # for i in range(distAux.shape[1]):
    #                     #     lvind = np.argsort(distAux[:, i], axis=0)[:nb_neighbours].astype(int)
    #                     #     new_lv_roots_ids[i, :] = lvActivationIndexes[lvind]
    #                     #     new_lv_roots_weights[i, :] = np.maximum(distAux[lvind, i]**2, 10e-17)**(-1)
    #                     #     new_lv_roots_weights[i, :] = new_lv_roots_weights[i, :] / np.sum(new_lv_roots_weights[i, :])
    #                     #     new_lv_roots_weights[i, new_lv_roots_weights[i, :] < 10e-2] = 0.
    #                     #     new_lv_roots_weights[i, :] = new_lv_roots_weights[i, :] / np.sum(new_lv_roots_weights[i, :])
    #
    #                     # RV
    #                     k_means_rv_labels, k_means_rv_centroids = k_means(rvnodesXYZ[rvActnode_ids, :], rvdata_ids, rv_num_clusters, rv_centroid_ids, rvdistance_mat, rvnodesXYZ, max_iter=10)
    #
    #                     # We make that only preselected potential root nodes can be a result to make for a better visualisation
    #         #             rvind = np.asarray([np.argmin(np.linalg.norm(rvnodesXYZ[rvActnode_ids, :] - rvnodesXYZ[k_means_rv_centroids[i], :], ord=2, axis=1)).astype(int) for i in range(k_means_rv_centroids.shape[0])])
    #         #             new_rv_roots = rvActivationIndexes[rvind]
    #
    #                     # Any node in the endocardium can be a result
    #                     new_rv_roots = np.unique(rvnodes[k_means_rv_centroids])
    #
    #                     if new_lv_roots.shape[0] != lv_num_clusters:
    #                         print('LV')
    #                         print(fileName)
    #                         print(lv_centroid_ids)
    #                         print(k_means_lv_centroids)
    #                         print(lvnodes[k_means_lv_centroids])
    #                         print(lv_centroid_ids)
    #                         raise
    #
    #                     if new_rv_roots.shape[0] != rv_num_clusters:
    #                         print('RV')
    #                         print(fileName)
    #                         print(rv_centroid_ids)
    #                         print(k_means_rv_centroids)
    #                         print(rvnodes[k_means_rv_centroids])
    #                         print(rv_centroid_ids)
    #                         raise
    #
    #                     # # We project the centroids to potential root nodes using distance as a weight for a better visualisation
    #                     # new_rv_roots_ids = np.zeros((k_means_rv_centroids.shape[0], nb_neighbours)).astype(int)
    #                     # new_rv_roots_weights = np.zeros((k_means_rv_centroids.shape[0], nb_neighbours)).astype(float)
    #                     # distAux = np.linalg.norm(rvnodesXYZ[rvActnode_ids, np.newaxis, :] - rvnodesXYZ[k_means_rv_centroids, :], ord=2, axis=2)
    #                     # # Take the nb_neighbours closest potential root nodes and weight them based on distance to the k-means returned position
    #                     # for i in range(distAux.shape[1]):
    #                     #     rvind = np.argsort(distAux[:, i], axis=0)[:nb_neighbours].astype(int)
    #                     #     new_rv_roots_ids[i, :] = rvActivationIndexes[rvind]
    #                     #     new_rv_roots_weights[i, :] = np.maximum(distAux[rvind, i]**2, 10e-17)**(-1)
    #                     #     new_rv_roots_weights[i, :] = new_rv_roots_weights[i, :] / np.sum(new_rv_roots_weights[i, :])
    #                     #     new_rv_roots_weights[i, new_rv_roots_weights[i, :] < 10e-2] = 0.
    #                     #     new_rv_roots_weights[i, :] = new_rv_roots_weights[i, :] / np.sum(new_rv_roots_weights[i, :])
    #
    #                     if lv_rootNodes is None:
    #                         lv_rootNodes = np.sum(lv_rootNodes_part, axis=0)
    #                         rv_rootNodes = np.sum(rv_rootNodes_part, axis=0)
    #                         lv_rootNodes_list = lv_rootNodes_part
    #                         rv_rootNodes_list = rv_rootNodes_part
    #                         rootNodes_list = rootNodes
    #                     else:
    #                         lv_rootNodes = lv_rootNodes + np.sum(lv_rootNodes_part, axis=0)
    #                         rv_rootNodes = rv_rootNodes + np.sum(rv_rootNodes_part, axis=0)
    #                         lv_rootNodes_list = np.concatenate((lv_rootNodes_list, lv_rootNodes_part), axis=0)
    #                         rv_rootNodes_list = np.concatenate((rv_rootNodes_list, rv_rootNodes_part), axis=0)
    #                         rootNodes_list = np.concatenate((rootNodes_list, rootNodes), axis=0)
    #                     # K-means weighted and projected back to potential root nodes
    #                     #new_lv_roots_ids_list.append(new_lv_roots_ids)
    #                     #new_lv_roots_weights_list.append(new_lv_roots_weights)
    #                     #new_rv_roots_ids_list.append(new_rv_roots_ids)
    #                     #new_rv_roots_weights_list.append(new_rv_roots_weights)
    #                     # K-means raw centroids
    #                     new_lv_roots_list.append(new_lv_roots)
    #                     new_rv_roots_list.append(new_rv_roots)
    #                     # Root nodes with the best discrepancy score
    #                     #discrepancy = np.loadtxt(previousResultsPath + fileName.replace('population', 'discrepancy'), delimiter=',')[particles_index]
    #                     #discrepancy_root_meta_indexes = np.round(population[np.argmin(discrepancy), 4:]).astype(int)
    #                     #disc_roots_list.append(activationIndexes[discrepancy_root_meta_indexes.astype(bool)])
    #
    #                     if save_roots:
    #                         data_list.append({
    #                             'fileName': fileName,
    #                             'lv_roots': new_lv_roots,
    #                             'rv_roots': new_rv_roots
    #                             #,
    #                             #'params': population[np.argmin(discrepancy), :]
    #                         })
    #
    #             # K-means CENTROIDS
    #             atmap = np.zeros((nodesXYZ.shape[0]))
    #             if not is_ECGi:
    #                 atmap[lv_rootNodesIndexes_true] = -1000
    #                 atmap[rv_rootNodesIndexes_true] = -1000
    #
    #             new_lv_roots, new_lv_roots_count = np.unique(np.concatenate((new_lv_roots_list), axis=0), return_counts=True)
    #             atmap[new_lv_roots] = np.round(100*new_lv_roots_count/len(new_lv_roots_list), 2)
    #
    #             new_rv_roots, new_rv_roots_count = np.unique(np.concatenate((new_rv_roots_list), axis=0), return_counts=True)
    #             atmap[new_rv_roots] = np.round(100*new_rv_roots_count/len(new_rv_roots_list), 2)
    #
    #             # Translate the atmap to be an element-wise map
    #             elem_atmap = np.zeros((elems.shape[0]))
    #             for i in range(elems.shape[0]):
    #                 elem_atmap[i] = np.sum(atmap[elems[i]])
    #
    #             with open(figuresPath+meshName+'_'+targetType+'_'
    #                         +discretisation_resolution+'.ensi.kMeans_centroids', 'w') as f:
    #                 f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
    #                 for i in range(0, len(elem_atmap)):
    #                     f.write(str(elem_atmap[i])+'\n')
    #
    #             # K-means PROJECTED BACK TO POENTIAL ROOT NODES USING DISTANCE FROM CENTROIDS AS WEIGHTS
    #             # atmap = np.zeros((nodesXYZ.shape[0]))
    #             # atmap[lv_rootNodesIndexes_true] = -1000
    #             # atmap[rv_rootNodesIndexes_true] = -1000
    #             #
    #             # new_lv_roots = np.unique(np.concatenate((new_lv_roots_ids_list)))
    #             # new_lv_roots_count = np.zeros((new_lv_roots.shape[0]))
    #             # for i in range(len(new_lv_roots_ids_list)):
    #             #     new_lv_roots_ids = new_lv_roots_ids_list[i]
    #             #     new_lv_roots_weights = new_lv_roots_weights_list[i]
    #             #     for j in range(new_lv_roots_ids.shape[0]):
    #             #         indexes = np.asarray([np.flatnonzero(new_lv_roots == node_id)[0] for node_id in new_lv_roots_ids[j, :]]).astype(int)
    #             #         new_lv_roots_count[indexes] = new_lv_roots_count[indexes] + new_lv_roots_weights[j, :]
    #             # atmap[new_lv_roots] = np.round(100*new_lv_roots_count/len(new_lv_roots_ids_list), 2)
    #             #
    #             #
    #             # new_rv_roots = np.unique(np.concatenate((new_rv_roots_ids_list)))
    #             # new_rv_roots_count = np.zeros((new_rv_roots.shape[0]))
    #             # for i in range(len(new_rv_roots_ids_list)):
    #             #     new_rv_roots_ids = new_rv_roots_ids_list[i]
    #             #     new_rv_roots_weights = new_rv_roots_weights_list[i]
    #             #     for j in range(new_rv_roots_ids.shape[0]):
    #             #         indexes = np.asarray([np.flatnonzero(new_rv_roots == node_id)[0] for node_id in new_rv_roots_ids[j, :]]).astype(int)
    #             #         new_rv_roots_count[indexes] = new_rv_roots_count[indexes] + new_rv_roots_weights[j, :]
    #             # atmap[new_rv_roots] = np.round(100*new_rv_roots_count/len(new_rv_roots_ids_list), 2)
    #
    #
    #             # Translate the atmap to be an element-wise map
    #             # elem_atmap = np.zeros((elems.shape[0]))
    #             # for i in range(elems.shape[0]):
    #             #    elem_atmap[i] = np.sum(atmap[elems[i]])
    #
    #             # with open(figuresPath+meshName+'_'+targetType+'.ensi.kMeans_projected', 'w') as f:
    #             #     f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
    #             #     for i in range(0, len(elem_atmap)):
    #             #         f.write(str(elem_atmap[i])+'\n')
    #             pass
    #
    #             # CUMM ROOT NODES TOGETHER
    #             atmap = np.zeros((nodesXYZ.shape[0]))
    #             if not is_ECGi:
    #                 atmap[lv_rootNodesIndexes_true] = -1000
    #                 atmap[rv_rootNodesIndexes_true] = -1000
    #             atmap[lvActivationIndexes] = np.round(100*lv_rootNodes/count, 2)#-6 # substracting the baseline value
    #             atmap[rvActivationIndexes] = np.round(100*rv_rootNodes/count, 2)#-6 # substracting the baseline value
    #
    #             # Translate the atmap to be an element-wise map
    #             elem_atmap = np.zeros((elems.shape[0]))
    #             for i in range(elems.shape[0]):
    #                 elem_atmap[i] = np.sum(atmap[elems[i]])
    #
    #             with open(figuresPath+meshName+'_'+targetType+'_'
    #                         +discretisation_resolution+'.ensi.cummrootNodes', 'w') as f:
    #                 f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
    #                 for i in range(0, len(elem_atmap)):
    #                     f.write(str(elem_atmap[i])+'\n')
    #
    #             if not is_ECGi:
    #                 # GROUND TRUTH ROOT NODES
    #                 atmap = np.zeros((nodesXYZ.shape[0]))
    #                 atmap[lv_rootNodesIndexes_true] = -1000
    #                 atmap[rv_rootNodesIndexes_true] = -1000
    #
    #                 # Translate the atmap to be an element-wise map
    #                 elem_atmap = np.zeros((elems.shape[0]))
    #                 for i in range(elems.shape[0]):
    #                     elem_atmap[i] = np.sum(atmap[elems[i]])
    #
    #                 with open(figuresPath+meshName+'_'+targetType+'_'
    #                             +discretisation_resolution+'.ensi.trueNodes', 'w') as f:
    #                     f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
    #                     for i in range(0, len(elem_atmap)):
    #                         f.write(str(elem_atmap[i])+'\n')
    #
    #             # # DISCREPANCY ROOT NODES
    #             # atmap = np.zeros((nodesXYZ.shape[0]))
    #             # atmap[lv_rootNodesIndexes_true] = -1000
    #             # atmap[rv_rootNodesIndexes_true] = -1000
    #             #
    #             # disc_roots, disc_roots_count = np.unique(np.concatenate((disc_roots_list), axis=0), return_counts=True)
    #             # atmap[disc_roots] = np.round(100*disc_roots_count/len(disc_roots_list), 2)
    #             #
    #             # # Translate the atmap to be an element-wise map
    #             # elem_atmap = np.zeros((elems.shape[0]))
    #             # for i in range(elems.shape[0]):
    #             #     elem_atmap[i] = np.sum(atmap[elems[i]])
    #             #
    #             # with open(figuresPath+meshName+'_'+targetType+'.ensi.discrepancy', 'w') as f:
    #             #     f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
    #             #     for i in range(0, len(elem_atmap)):
    #             #         f.write(str(elem_atmap[i])+'\n')
    #
    #             # # Save Atmaps for true bidomain and predicted eikonal
    #             # if 'atm' in target_type:
    #             #     atmap = np.loadtxt('metaData/ATMaps/' + meshName + '_coarse_true_ATMap_120_1x.csv', delimiter=',')
    #             #     atmap1 = np.loadtxt(previousResultsPath  + meshName +  '_coarse_120_1x_low_'+target_type+'_0_prediction.csv', delimiter=',')
    #             #     atmap2 = np.loadtxt(previousResultsPath  + meshName +  '_coarse_120_1x_high_'+target_type+'_0_prediction.csv', delimiter=',')
    #             #     max_val = np.amax(np.vstack((atmap, atmap1, atmap2)))
    #             #     atmap[0] = max_val
    #             #     atmap1[0] = max_val
    #             #     atmap2[0] = max_val
    #             #     with open(figuresPath + meshName + '_' + target_type + '.ensi.true', 'w') as f:
    #             #         f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #             #         for i in range(0, len(atmap)):
    #             #             f.write(str(atmap[i]) + '\n')
    #             #     with open(figuresPath + meshName + '_' + target_type + '.ensi.pred_low', 'w') as f:
    #             #         f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #             #         for i in range(0, len(atmap1)):
    #             #             f.write(str(atmap1[i]) + '\n')
    #             #     with open(figuresPath + meshName + '_' + target_type + '.ensi.pred_high', 'w') as f:
    #             #         f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #             #         for i in range(0, len(atmap2)):
    #             #             f.write(str(atmap2[i]) + '\n')
    #             #pass
    #
    #             # SAVE RESULTS TO REUSE BY OTHER SCRIPTS
    #             if save_roots:
    #                 df_roots = pandas.DataFrame(data_list)
    #                 df_roots.to_pickle(path=resultsPath+meshName+'_coarse_'+discretisation_resolution+'_'+targetType+'_rootNodes.gz', compression='infer')


# Started modifying this function on the 2021/05/24 but it's uncomplete
def makeSpeedPredFigures(threadsNum_val, previousResultsPath, meshName, dataType, figPath, conduction_speeds,
                         resolution, is_ECGi, is_healthy_val, endocardial_layer, load_target, target_data_path):
    global has_endocardial_layer
    has_endocardial_layer = endocardial_layer
    global is_healthy
    is_healthy = is_healthy_val

    # global gf_factor # 27/02/2021 TODO changed
    # gf_factor = 1.3
    # global gn_factor
    # gn_factor = 0.9
    
    population_files = []
    for fileName in os.listdir(previousResultsPath):
        if ('population' in fileName and meshName in fileName and dataType in fileName and resolution in fileName
        and str(conduction_speeds) in fileName):
            population_files.append(fileName)
            
    print(population_files)
    
    global experiment_output
    experiment_output = 'atm'
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_bsp
    global nb_leads
    nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad   # 8 + lead progression (or 12)
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    global nlhsParam
    if is_healthy:
        nlhsParam = 2
    else:
        nlhsParam = 4

    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'

    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvtri = (np.loadtxt(dataPath + meshName + '_lvnodes.csv', delimiter=',') - 1).astype(int)               # lv endocardium triangles
    lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
    rvtri =(np.loadtxt(dataPath + meshName + '_rvnodes.csv', delimiter=',') - 1).astype(int)                # rv endocardium triangles
    rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv',
                             delimiter=',')  # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :]  # edge vectors

    if not is_ECGi:
        # True
        rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
        rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
        lv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in lvnodes])
        rv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in rvnodes])

    global tetrahedrons
    tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
    tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
    electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    nb_bsp = electrodePositions.shape[0]

    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, tetrahedrons.shape[0], 1):
        aux[tetrahedrons[i, 0]].append(i)
        aux[tetrahedrons[i, 1]].append(i)
        aux[tetrahedrons[i, 2]].append(i)
        aux[tetrahedrons[i, 3]].append(i)
    global elements
    elements = [np.array(n) for n in aux]
    aux = None  # Clear Memory

    # Precompute PseudoECG stuff
    # Calculate the tetrahedrons volumes
    D = nodesXYZ[tetrahedrons[:, 3], :]  # RECYCLED
    A = nodesXYZ[tetrahedrons[:, 0], :] - D  # RECYCLED
    B = nodesXYZ[tetrahedrons[:, 1], :] - D  # RECYCLED
    C = nodesXYZ[tetrahedrons[:, 2], :] - D  # RECYCLED
    D = None  # Clear Memory

    global tVolumes
    tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1),
                                           (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0])  # Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
    tVolumes = tVolumes / np.sum(tVolumes)

    # Calculate the tetrahedron (temporal) voltage gradients
    Mg = np.stack((A, B, C), axis=-1)
    A = None  # Clear Memory
    B = None  # Clear Memory
    C = None  # Clear Memory

    # Calculate the gradients
    global G_pseudo
    G_pseudo = np.zeros(Mg.shape)
    for i in range(Mg.shape[0]):
        G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
    G_pseudo = np.moveaxis(G_pseudo, 1, 2)
    Mg = None  # clear memory

    # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
    r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                               (tetrahedronCenters.shape[0],
                                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1) - electrodePositions

    global d_r
    d_r = np.moveaxis(np.multiply(
        np.moveaxis(r, [0, 1], [-1, -2]),
        np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0, -1)

    # Set endocardial edges aside
    global isEndocardial
    isEndocardial = np.logical_or(np.all(np.isin(edges, lvnodes), axis=1),
                                  np.all(np.isin(edges, rvnodes), axis=1))
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan,
                         np.int32)  # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    
    global epiface_tri
    epiface_tri = (np.loadtxt(dataPath + meshName + '_coarse_epiface.csv', delimiter=',') - 1).astype(int) # epicardium nodes
    global epiface
    epiface = np.unique(epiface_tri) # epicardium nodes
    print(epiface.shape)
    print(epiface_tri.shape)
    
    # True atmap
    atmap = np.zeros((nodesXYZ.shape[0]))
    if load_target:
        atmap_true = np.loadtxt(target_data_path, delimiter=',')
    else:
        atmap_true = eikonal_ecg(np.array([conduction_speeds])/1000, rootNodesIndexes_true)[0, :]
    print(atmap_true.shape)
    atmap[epiface] = atmap_true[epiface]
    with open(figPath+'/'+meshName+'_atm_'+resolution+'.ensi.true', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i])+'\n')
    
    print ('Export mesh to ensight format')
    aux_elems = tetrahedrons+1    # Make indexing Paraview and Matlab friendly
    with open(figPath+'/'+meshName+'.ensi.geo', 'w') as f:
        f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
        for i in range(0, nodesXYZ.shape[0]):
            f.write(str(i+1)+'\n')
        for c in [0,1,2]:
            for i in range(0, nodesXYZ.shape[0]):
                f.write(str(nodesXYZ[i,c])+'\n')
        print('Write tetra4...')
        f.write('tetra4\n  '+str(len(aux_elems))+'\n')
        for i in range(0, len(aux_elems)):
            f.write('  '+str(i+1)+'\n')
        for i in range(0, len(aux_elems)):
            f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\t'+str(aux_elems[i,3])+'\n')
    with open(figPath+'/'+meshName+'.ensi.case', 'w') as f:
        f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
        f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
        f.write('scalar per element: 1	total_cummrootNodes_atm_'+resolution+'	'+meshName+'_atm_'+resolution+'.ensi.total_cummrootNodes\n')
        for fileName in population_files:
            for i in range(10):
                if '_'+str(i)+'_' in fileName:
                    consistency_i = i
                    f.write('scalar per element: 1	kMeans_centroids_atm_'+resolution+'_'+str(consistency_i)+'	'+meshName+'_atm_'+resolution+'_'+str(consistency_i)+'.ensi.kMeans_centroids\n')
                    f.write('scalar per element: 1	cummrootNodes_atm_'+resolution+'_'+str(consistency_i)+'	'+meshName+'_atm_'+resolution+'_'+str(consistency_i)+'.ensi.cummrootNodes\n')
                    f.write('scalar per node: 1	bestPred_atm_'+resolution+'_'+str(consistency_i)+'	'+meshName+'_atm_'+resolution+'_'+str(consistency_i)+'.ensi.bestPred\n')
        f.write('scalar per node: 1	true_atm_'+resolution+'	'+meshName+'_atm_'+resolution+'.ensi.true\n')

    # Predictions
    lvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_lv_activationIndexes_'+resolution+'Res.csv', delimiter=',') - 1).astype(int)
    rvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rv_activationIndexes_'+resolution+'Res.csv', delimiter=',') - 1).astype(int)

    for i in range(lvActivationIndexes.shape[0]):
        if lvActivationIndexes[i] not in lvnodes:
            a = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
            #print('diff ' + str(np.round(np.linalg.norm(nodesXYZ[a, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=0)*10, 2)) + ' mm')
            lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    for i in range(rvActivationIndexes.shape[0]):
        if rvActivationIndexes[i] not in rvnodes:
            b = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
            #print('diff ' + str(np.round(np.linalg.norm(nodesXYZ[b, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=0)*10, 2)) + ' mm')
            rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    activationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    
    # Set LV endocardial edges aside
    lvnodesXYZ = nodesXYZ[lvnodes, :]
    lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :] # edge vectors
    lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    for i in range(0, len(lvunfoldedEdges), 1):
        aux[lvunfoldedEdges[i, 0]].append(i)
    lvneighbours = [np.array(n).astype(int) for n in aux]
    aux = None # Clear Memory
    
    # Calculate Djikstra distances in the LV endocardium
    lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
    lvdistance_mat = djikstra(lvActnode_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours)
    
    # Set RV endocardial edges aside
    rvnodesXYZ = nodesXYZ[rvnodes, :]
    rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :] # edge vectors
    rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    for i in range(0, len(rvunfoldedEdges), 1):
        aux[rvunfoldedEdges[i, 0]].append(i)
    rvneighbours = [np.array(n) for n in aux]
    aux = None # Clear Memory
    
    # Calculate Djikstra distances in the RV endocardium
    rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
    rvdistance_mat = djikstra(rvActnode_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours)
    
    # Compute centroids and speeds for each experiment
    # CUMM TOTAL ROOT NODES TOGETHER
    cumm_atmap = np.zeros((nodesXYZ.shape[0]))
    for fileName in population_files:
        consistency_i = 0
        for i in range(10):
            s = '_'+str(i)+'_'
            if s in fileName:
                consistency_i = i
        
        # Load population
        speeds = []
        lv_rootNodes = None
        rv_rootNodes = None
        lv_rootNodes_list = None
        rv_rootNodes_list = None
        rootNodes_list = None
        new_lv_roots_list = []
        new_rv_roots_list = []
        data_list = []
        disc_roots_list = []
        count = 0.
        population = np.loadtxt(previousResultsPath + fileName, delimiter=',')
        
        # Best atmap
        print(previousResultsPath + fileName.replace('population', 'discrepancy'))
        # CAUTION! THIS IS A WORKAROUND BECAUSE THERE ARE NO COMAS ADDED WHEN SAVING THE DISCREPANCY FILE, NUMPY STUPIDITY THAT IF YOU SAVE A LIST IT ADDS NO DELIMITTER, ONLY IF IT IS A LIST OF LISTS
        discrepancies = np.genfromtxt(previousResultsPath + fileName.replace('population', 'discrepancy')) # should have delimiter set to ',' but it doesnt have them
        bestInd = np.argsort(discrepancies)[0]
        print('discrepancy')
        print(discrepancies[bestInd])
        print(np.amax(discrepancies))
        bestParams = population[bestInd, :]
        print(bestParams[:4])
        atmap = eikonal_ecg(np.array([bestParams]), activationIndexes)[0, :]
        with open(figPath+'/'+meshName+'_atm_'+resolution+'_'+str(consistency_i)+'.ensi.bestPred', 'w') as f:
            f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            for i in range(0, len(atmap)):
                f.write(str(atmap[i])+'\n')
        
        # Extract parameter values from population
        speeds.append(np.round(population[:, :nlhsParam], 2)) # Repeated particles
        population_unique, particles_index = np.unique(population, axis=0, return_index=True)
        rootNodes = np.round(population_unique[:, nlhsParam:]).astype(int) # Non-repeated particles
        count = count + rootNodes.shape[0]
        # Number of Root nodes percentage-error
        lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
        rv_rootNodes_part = rootNodes[:, len(lvActivationIndexes):]
        #rv_rootNodes_part = rootNodes[:, -len(rvActivationIndexes):len(rvActivationIndexes)]


        # K-means for root nodes
        # Choose the number of clusters 'k'
        lv_num_nodes = np.sum(lv_rootNodes_part, axis=1)
        lv_num_nodes, lv_num_nodes_counts = np.unique(lv_num_nodes, return_counts=True, axis=0)
        lv_num_clusters = lv_num_nodes[np.argmax(lv_num_nodes_counts)]

        rv_num_nodes = np.sum(rv_rootNodes_part, axis=1)
        rv_num_nodes, rv_num_nodes_counts = np.unique(rv_num_nodes, return_counts=True, axis=0)
        rv_num_clusters = rv_num_nodes[np.argmax(rv_num_nodes_counts)]

        # Choose the initial centroides
        k_lv_rootNodes_list, lvunique_counts = np.unique(lv_rootNodes_part, return_counts=True, axis=0)
        lv_rootNodes_list_index = np.sum(k_lv_rootNodes_list, axis=1) == lv_num_clusters
        k_lv_rootNodes_list = k_lv_rootNodes_list[lv_rootNodes_list_index, :]
        lvunique_counts = lvunique_counts[lv_rootNodes_list_index]
        lv_centroid_ids = lvActnode_ids[k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)].astype(int)
#             print('lv: ' +str(lv_centroid_ids))

        k_rv_rootNodes_list, rvunique_counts = np.unique(rv_rootNodes_part, return_counts=True, axis=0)
        rv_rootNodes_list_index = np.sum(k_rv_rootNodes_list, axis=1) == rv_num_clusters
        k_rv_rootNodes_list = k_rv_rootNodes_list[rv_rootNodes_list_index, :]
        rvunique_counts = rvunique_counts[rv_rootNodes_list_index]
        rv_centroid_ids = rvActnode_ids[k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)].astype(int)

        # Transform the root nodes predicted data into k-means data
        lvdata_ids = np.concatenate(([lvActivationIndexes[(lv_rootNodes_part[i, :]).astype(bool)] for i in range(lv_rootNodes_part.shape[0])]), axis=0)
        lvdata_ids = np.asarray([np.flatnonzero(lvActivationIndexes == node_id)[0] for node_id in lvdata_ids]).astype(int)

        rvdata_ids = np.concatenate(([rvActivationIndexes[(rv_rootNodes_part[i, :]).astype(bool)] for i in range(rv_rootNodes_part.shape[0])]), axis=0)
        rvdata_ids = np.asarray([np.flatnonzero(rvActivationIndexes == node_id)[0] for node_id in rvdata_ids]).astype(int)

        # K-means algorithm
        # LV
        k_means_lv_labels, k_means_lv_centroids = k_means(lvnodesXYZ[lvActnode_ids, :], lvdata_ids, lv_num_clusters, lv_centroid_ids, lvdistance_mat, lvnodesXYZ, max_iter=10)

        # Any node in the endocardium can be a result
        new_lv_roots = np.unique(lvnodes[k_means_lv_centroids])

        # RV
        k_means_rv_labels, k_means_rv_centroids = k_means(rvnodesXYZ[rvActnode_ids, :], rvdata_ids, rv_num_clusters, rv_centroid_ids, rvdistance_mat, rvnodesXYZ, max_iter=10)

        # Any node in the endocardium can be a result
        new_rv_roots = np.unique(rvnodes[k_means_rv_centroids])

        if new_lv_roots.shape[0] != lv_num_clusters:
            print('LV')
            print(fileName)
            print(lv_centroid_ids)
            print(k_means_lv_centroids)
            print(lvnodes[k_means_lv_centroids])
            print(lv_centroid_ids)
            raise

        if new_rv_roots.shape[0] != rv_num_clusters:
            print('RV')
            print(fileName)
            print(rv_centroid_ids)
            print(k_means_rv_centroids)
            print(rvnodes[k_means_rv_centroids])
            print(rv_centroid_ids)
            raise

        if lv_rootNodes is None:
            lv_rootNodes = np.sum(lv_rootNodes_part, axis=0)
            rv_rootNodes = np.sum(rv_rootNodes_part, axis=0)
            lv_rootNodes_list = lv_rootNodes_part
            rv_rootNodes_list = rv_rootNodes_part
            rootNodes_list = rootNodes
        else:
            lv_rootNodes = lv_rootNodes + np.sum(lv_rootNodes_part, axis=0)
            rv_rootNodes = rv_rootNodes + np.sum(rv_rootNodes_part, axis=0)
            lv_rootNodes_list = np.concatenate((lv_rootNodes_list, lv_rootNodes_part), axis=0)
            rv_rootNodes_list = np.concatenate((rv_rootNodes_list, rv_rootNodes_part), axis=0)
            rootNodes_list = np.concatenate((rootNodes_list, rootNodes), axis=0)
        # K-means raw centroids
        new_lv_roots_list.append(new_lv_roots)
        new_rv_roots_list.append(new_rv_roots)
        
        # PLOT SPEED distribution - graphical abstract small version
        #speeds = np.concatenate(speeds, axis=0)*1000
        speeds_scaled = np.array(speeds[0])*1000
        # Check std because if zero then the kde plot will fail
        kde_plot = True
        for i_speed in range(speeds_scaled.shape[1]):
            if np.std(speeds_scaled[:, i_speed]) == 0:
                kde_plot = False
        if is_healthy:
            speed_dict = ([{'speed': 'endocardial', 'cm/s': speeds_scaled[i, 1]} for i in range(speeds_scaled.shape[0])]
                        + [{'speed': 'sheet', 'cm/s': speeds_scaled[i, 0]} for i in range(speeds_scaled.shape[0])])
        else:
            speed_dict = ([{'speed': 'endocardial', 'cm/s': speeds_scaled[i, 3]} for i in range(speeds_scaled.shape[0])]
                        + [{'speed': 'fibre', 'cm/s': speeds_scaled[i, 0]} for i in range(speeds_scaled.shape[0])]
                        + [{'speed': 'sheet', 'cm/s': speeds_scaled[i, 1]} for i in range(speeds_scaled.shape[0])]
                        + [{'speed': 'sheet-normal', 'cm/s': speeds_scaled[i, 2]} for i in range(speeds_scaled.shape[0])])
        df = pandas.DataFrame(speed_dict)
        
        # Speeds distribution plot
        if kde_plot:
            fig = sns.displot(data=df, x='cm/s', hue='speed', multiple='layer', kind="kde", cut=0, fill=True,
                common_norm=False, facet_kws={'legend_out': False}, height=6, aspect=1)
        else:
            fig = sns.displot(data=df, x='cm/s', hue='speed', multiple='layer', kind="hist", fill=True,
                common_norm=False, facet_kws={'legend_out': False}, height=6, aspect=1)
        print(fig.axes[0])
        fig.axes[0][0].set_xticks(np.arange(0, 200, step=20))
        fig.savefig(figPath+'/'+ meshName + '_speeds_'+str(consistency_i)+'.png', dpi=300)
        print(figPath+'/'+ meshName + '_speeds_'+str(consistency_i)+'.png')
        
        # K-means CENTROIDS
        atmap = np.zeros((nodesXYZ.shape[0]))
    
        new_lv_roots, new_lv_roots_count = np.unique(np.concatenate((new_lv_roots_list), axis=0), return_counts=True)
        atmap[new_lv_roots] = np.round(100*new_lv_roots_count/len(new_lv_roots_list), 2)
    
        new_rv_roots, new_rv_roots_count = np.unique(np.concatenate((new_rv_roots_list), axis=0), return_counts=True)
        atmap[new_rv_roots] = np.round(100*new_rv_roots_count/len(new_rv_roots_list), 2)
    
        # Translate the atmap to be an element-wise map
        elem_atmap = np.zeros((tetrahedrons.shape[0]))
        for i in range(tetrahedrons.shape[0]):
            elem_atmap[i] = np.sum(atmap[tetrahedrons[i]])
    
        with open(figPath+'/'+meshName+'_atm_'+resolution+'_'+str(consistency_i)+'.ensi.kMeans_centroids', 'w') as f:
            f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
            for i in range(0, len(elem_atmap)):
                f.write(str(elem_atmap[i])+'\n')
    
        # CUMM ROOT NODES TOGETHER
        atmap = np.zeros((nodesXYZ.shape[0]))
        atmap[lvActivationIndexes] = np.round(100*lv_rootNodes/count, 2)#-6 # substracting the baseline value
        atmap[rvActivationIndexes] = np.round(100*rv_rootNodes/count, 2)#-6 # substracting the baseline value
        # Update total
        cumm_atmap[lvActivationIndexes] = cumm_atmap[lvActivationIndexes] + np.round(100*lv_rootNodes/count, 2)
        cumm_atmap[rvActivationIndexes] = cumm_atmap[rvActivationIndexes] + np.round(100*rv_rootNodes/count, 2)
        
        # Translate the atmap to be an element-wise map
        elem_atmap = np.zeros((tetrahedrons.shape[0]))
        for i in range(tetrahedrons.shape[0]):
            elem_atmap[i] = np.sum(atmap[tetrahedrons[i]])
    
        with open(figPath+'/'+meshName+'_atm_'+resolution+'_'+str(consistency_i)+'.ensi.cummrootNodes', 'w') as f:
            f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
            for i in range(0, len(elem_atmap)):
                f.write(str(elem_atmap[i])+'\n')
    
    # Translate the atmap to be an element-wise map
    elem_atmap = np.zeros((tetrahedrons.shape[0]))
    for i in range(tetrahedrons.shape[0]):
        elem_atmap[i] = np.sum(cumm_atmap[tetrahedrons[i]])/len(population_files)

    with open(figPath+'/'+meshName+'_atm_'+resolution+'.ensi.total_cummrootNodes', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
        for i in range(0, len(elem_atmap)):
            f.write(str(elem_atmap[i])+'\n')
            

def makeAbstractFigures(threadsNum_val, previousResultsPath, meshName, dataType, is_healthy_val):
    population_files = []
    global is_healthy
    is_healthy = is_healthy_val
    for fileName in os.listdir(previousResultsPath):
        if ('population' in fileName and meshName in fileName and dataType in fileName and 'high' in fileName
        and '120' in fileName and '1x' in fileName and '_1_' in fileName):
            population_files.append(fileName)
    
    global experiment_output
    experiment_output = 'atm'
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_bsp
    global nb_leads
    nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad   # 8 + lead progression (or 12)
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    global nlhsParam
    if is_healthy:
        nlhsParam = 2
    else:
        nlhsParam = 4

    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'

    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvtri = (np.loadtxt(dataPath + meshName + '_lvnodes.csv', delimiter=',') - 1).astype(int)               # lv endocardium triangles
    lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
    rvtri =(np.loadtxt(dataPath + meshName + '_rvnodes.csv', delimiter=',') - 1).astype(int)                # rv endocardium triangles
    rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv',
                             delimiter=',')  # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :]  # edge vectors

    # True
    rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
    rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    lv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in lvnodes])
    rv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in rvnodes])

    global tetrahedrons
    tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
    tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
    electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    nb_bsp = electrodePositions.shape[0]

    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, tetrahedrons.shape[0], 1):
        aux[tetrahedrons[i, 0]].append(i)
        aux[tetrahedrons[i, 1]].append(i)
        aux[tetrahedrons[i, 2]].append(i)
        aux[tetrahedrons[i, 3]].append(i)
    global elements
    elements = [np.array(n) for n in aux]
    aux = None  # Clear Memory

    # Precompute PseudoECG stuff
    # Calculate the tetrahedrons volumes
    D = nodesXYZ[tetrahedrons[:, 3], :]  # RECYCLED
    A = nodesXYZ[tetrahedrons[:, 0], :] - D  # RECYCLED
    B = nodesXYZ[tetrahedrons[:, 1], :] - D  # RECYCLED
    C = nodesXYZ[tetrahedrons[:, 2], :] - D  # RECYCLED
    D = None  # Clear Memory

    global tVolumes
    tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1),
                                           (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0])  # Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
    tVolumes = tVolumes / np.sum(tVolumes)

    # Calculate the tetrahedron (temporal) voltage gradients
    Mg = np.stack((A, B, C), axis=-1)
    A = None  # Clear Memory
    B = None  # Clear Memory
    C = None  # Clear Memory

    # Calculate the gradients
    global G_pseudo
    G_pseudo = np.zeros(Mg.shape)
    for i in range(Mg.shape[0]):
        G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
    G_pseudo = np.moveaxis(G_pseudo, 1, 2)
    Mg = None  # clear memory

    # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
    r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                               (tetrahedronCenters.shape[0],
                                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1) - electrodePositions

    global d_r
    d_r = np.moveaxis(np.multiply(
        np.moveaxis(r, [0, 1], [-1, -2]),
        np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0, -1)

    # Set endocardial edges aside
    global isEndocardial
    isEndocardial = np.logical_or(np.all(np.isin(edges, lvnodes), axis=1),
                                  np.all(np.isin(edges, rvnodes), axis=1))
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan,
                         np.int32)  # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    
    global epiface
    epiface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_epiface.csv', delimiter=',') - 1).astype(int)) # epicardium nodes
    global epiface_tri
    epiface_tri = (np.loadtxt(dataPath + meshName + '_coarse_epiface.csv', delimiter=',') - 1).astype(int) # epicardium nodes
    
    print ('Export mesh to ensight format')
    aux_elems = tetrahedrons+1    # Make indexing Paraview and Matlab friendly
    with open('Abstract/'+meshName+'.ensi.geo', 'w') as f:
        f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
        for i in range(0, nodesXYZ.shape[0]):
            f.write(str(i+1)+'\n')
        for c in [0,1,2]:
            for i in range(0, nodesXYZ.shape[0]):
                f.write(str(nodesXYZ[i,c])+'\n')
        print('Write tetra4...')
        f.write('tetra4\n  '+str(len(aux_elems))+'\n')
        for i in range(0, len(aux_elems)):
            f.write('  '+str(i+1)+'\n')
        for i in range(0, len(aux_elems)):
            f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\t'+str(aux_elems[i,3])+'\n')
    with open('Abstract/'+meshName+'.ensi.case', 'w') as f:
        f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
        f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
        f.write('scalar per element: 1	kMeans_centroids_atm_high	'+meshName+'_atm_high.ensi.kMeans_centroids\n')
        f.write('scalar per element: 1	cummrootNodes_atm_high	'+meshName+'_atm_high.ensi.cummrootNodes\n')
        f.write('scalar per element: 1	trueNodes_atm_high	'+meshName+'_atm_high.ensi.trueNodes\n')
        f.write('scalar per node: 1	root1	'+meshName+'_atm_high.ensi.root1\n')
        f.write('scalar per node: 1	root2	'+meshName+'_atm_high.ensi.root2\n')
        f.write('scalar per node: 1	root3	'+meshName+'_atm_high.ensi.root3\n')
        f.write('scalar per node: 1	root4	'+meshName+'_atm_high.ensi.root4\n')
        f.write('scalar per node: 1	root5	'+meshName+'_atm_high.ensi.root5\n')
        f.write('scalar per node: 1	rootTrue	'+meshName+'_atm_high.ensi.rootTrue\n')

    # Predictions
    lvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_lv_activationIndexes_highRes.csv', delimiter=',') - 1).astype(int)
    rvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rv_activationIndexes_highRes.csv', delimiter=',') - 1).astype(int)

    #activationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    for i in range(lvActivationIndexes.shape[0]):
        if lvActivationIndexes[i] not in lvnodes:
            a = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
            #print('diff ' + str(np.round(np.linalg.norm(nodesXYZ[a, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=0)*10, 2)) + ' mm')
            lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    for i in range(rvActivationIndexes.shape[0]):
        if rvActivationIndexes[i] not in rvnodes:
            b = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
            #print('diff ' + str(np.round(np.linalg.norm(nodesXYZ[b, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=0)*10, 2)) + ' mm')
            rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    
    lv_rootNodes = None
    rv_rootNodes = None
    lv_rootNodes_list = None
    rv_rootNodes_list = None
    rootNodes_list = None
    new_lv_roots_list = []
    new_rv_roots_list = []
    data_list = []
    disc_roots_list = []
    count = 0.
    
    # Calculate Djikstra distances in the LV endocardium
    lvnodesXYZ = nodesXYZ[lvnodes, :]
    # Set endocardial edges aside
    lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :] # edge vectors
    lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    for i in range(0, len(lvunfoldedEdges), 1):
        aux[lvunfoldedEdges[i, 0]].append(i)
    lvneighbours = [np.array(n) for n in aux]
    aux = None # Clear Memory
    lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
    lvdistance_mat = djikstra(lvActnode_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours)
    
    # Calculate Djikstra distances in the RV endocardium
    rvnodesXYZ = nodesXYZ[rvnodes, :]
    # Set endocardial edges aside
    rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :] # edge vectors
    rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    for i in range(0, len(rvunfoldedEdges), 1):
        aux[rvunfoldedEdges[i, 0]].append(i)
    rvneighbours = [np.array(n) for n in aux]
    aux = None # Clear Memory
    rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
    rvdistance_mat = djikstra(rvActnode_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours)
    
    speeds = []
    for fileName in population_files:
        population = np.loadtxt(previousResultsPath + fileName, delimiter=',')
        speeds.append(np.round(population[:, :4], 2)) # Repeated particles
        population, particles_index = np.unique(population, axis=0, return_index=True)
        rootNodes = np.round(population[:, 4:]).astype(int) # Non-repeated particles
        count = count + rootNodes.shape[0]
        # Number of Root nodes percentage-error
        lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
        rv_rootNodes_part = rootNodes[:, len(lvActivationIndexes):]

        # K-means for root nodes
        # Choose the number of clusters 'k'
        lv_num_nodes = np.sum(lv_rootNodes_part, axis=1)
        lv_num_nodes, lv_num_nodes_counts = np.unique(lv_num_nodes, return_counts=True, axis=0)
        lv_num_clusters = lv_num_nodes[np.argmax(lv_num_nodes_counts)]

        rv_num_nodes = np.sum(rv_rootNodes_part, axis=1)
        rv_num_nodes, rv_num_nodes_counts = np.unique(rv_num_nodes, return_counts=True, axis=0)
        rv_num_clusters = rv_num_nodes[np.argmax(rv_num_nodes_counts)]

        # Choose the initial centroides
        k_lv_rootNodes_list, lvunique_counts = np.unique(lv_rootNodes_part, return_counts=True, axis=0)
        lv_rootNodes_list_index = np.sum(k_lv_rootNodes_list, axis=1) == lv_num_clusters
        k_lv_rootNodes_list = k_lv_rootNodes_list[lv_rootNodes_list_index, :]
        lvunique_counts = lvunique_counts[lv_rootNodes_list_index]
        lv_centroid_ids = lvActnode_ids[k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)].astype(int)
#             print('lv: ' +str(lv_centroid_ids))

        k_rv_rootNodes_list, rvunique_counts = np.unique(rv_rootNodes_part, return_counts=True, axis=0)
        rv_rootNodes_list_index = np.sum(k_rv_rootNodes_list, axis=1) == rv_num_clusters
        k_rv_rootNodes_list = k_rv_rootNodes_list[rv_rootNodes_list_index, :]
        rvunique_counts = rvunique_counts[rv_rootNodes_list_index]
        rv_centroid_ids = rvActnode_ids[k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)].astype(int)

        # Transform the root nodes predicted data into k-means data
        lvdata_ids = np.concatenate(([lvActivationIndexes[(lv_rootNodes_part[i, :]).astype(bool)] for i in range(lv_rootNodes_part.shape[0])]), axis=0)
        lvdata_ids = np.asarray([np.flatnonzero(lvActivationIndexes == node_id)[0] for node_id in lvdata_ids]).astype(int)

        rvdata_ids = np.concatenate(([rvActivationIndexes[(rv_rootNodes_part[i, :]).astype(bool)] for i in range(rv_rootNodes_part.shape[0])]), axis=0)
        rvdata_ids = np.asarray([np.flatnonzero(rvActivationIndexes == node_id)[0] for node_id in rvdata_ids]).astype(int)

        # K-means algorithm
        # LV
        k_means_lv_labels, k_means_lv_centroids = k_means(lvnodesXYZ[lvActnode_ids, :], lvdata_ids, lv_num_clusters, lv_centroid_ids, lvdistance_mat, lvnodesXYZ, max_iter=10)

        # Any node in the endocardium can be a result
        new_lv_roots = np.unique(lvnodes[k_means_lv_centroids])

        # RV
        k_means_rv_labels, k_means_rv_centroids = k_means(rvnodesXYZ[rvActnode_ids, :], rvdata_ids, rv_num_clusters, rv_centroid_ids, rvdistance_mat, rvnodesXYZ, max_iter=10)

        # Any node in the endocardium can be a result
        new_rv_roots = np.unique(rvnodes[k_means_rv_centroids])

        if new_lv_roots.shape[0] != lv_num_clusters:
            print('LV')
            print(fileName)
            print(lv_centroid_ids)
            print(k_means_lv_centroids)
            print(lvnodes[k_means_lv_centroids])
            print(lv_centroid_ids)
            raise

        if new_rv_roots.shape[0] != rv_num_clusters:
            print('RV')
            print(fileName)
            print(rv_centroid_ids)
            print(k_means_rv_centroids)
            print(rvnodes[k_means_rv_centroids])
            print(rv_centroid_ids)
            raise

        if lv_rootNodes is None:
            lv_rootNodes = np.sum(lv_rootNodes_part, axis=0)
            rv_rootNodes = np.sum(rv_rootNodes_part, axis=0)
            lv_rootNodes_list = lv_rootNodes_part
            rv_rootNodes_list = rv_rootNodes_part
            rootNodes_list = rootNodes
        else:
            lv_rootNodes = lv_rootNodes + np.sum(lv_rootNodes_part, axis=0)
            rv_rootNodes = rv_rootNodes + np.sum(rv_rootNodes_part, axis=0)
            lv_rootNodes_list = np.concatenate((lv_rootNodes_list, lv_rootNodes_part), axis=0)
            rv_rootNodes_list = np.concatenate((rv_rootNodes_list, rv_rootNodes_part), axis=0)
            rootNodes_list = np.concatenate((rootNodes_list, rootNodes), axis=0)
        # K-means raw centroids
        new_lv_roots_list.append(new_lv_roots)
        new_rv_roots_list.append(new_rv_roots)
        
    
    # PLOT SPEED distribution - graphical abstract small version
    sns.set(style="darkgrid", palette="colorblind", font_scale = 1.4)
    speeds = np.concatenate(speeds, axis=0)*1000
    if is_healthy:
        speed_dict = ([{'speed': 'endocardial', 'cm/s': speeds[i, 1]} for i in range(speeds.shape[0])]
                    + [{'speed': 'sheet', 'cm/s': speeds[i, 0]} for i in range(speeds.shape[0])])
    else:
        speed_dict = ([{'speed': 'endocardial', 'cm/s': speeds[i, 3]} for i in range(speeds.shape[0])]
                    + [{'speed': 'fibre', 'cm/s': speeds[i, 0]} for i in range(speeds.shape[0])]
                    + [{'speed': 'sheet', 'cm/s': speeds[i, 1]} for i in range(speeds.shape[0])]
                    + [{'speed': 'sheet-normal', 'cm/s': speeds[i, 2]} for i in range(speeds.shape[0])])
    df = pandas.DataFrame(speed_dict)
    fig = sns.displot(data=df, x='cm/s', hue='speed',
    #hue_norm=[0,100],
    multiple='layer', kind="kde", cut=0, fill=True, common_norm=False
   , facet_kws={'legend_out': True}, height=2, aspect=2.1
                )
    fig.savefig('Abstract/' + meshName + '_speeds_graphical_abstract.png', dpi=300)
    
    # 2nd larger plot for main manuscript
    sns.set(style="darkgrid", palette="colorblind", font_scale = 1.5)
    fig = sns.displot(data=df, x='cm/s', hue='speed',
    #hue_norm=[0,100],
    multiple='layer', kind="kde", cut=0, fill=True, common_norm=False
   , facet_kws={'legend_out': False}, height=5.1, aspect=0.9
                )
    fig.savefig('Abstract/' + meshName + '_speeds_large.png', dpi=300)
    
    #df = pandas.DataFrame([{'endocardial': speeds[i, 3], 'fibre': speeds[i, 0], 'sheet': speeds[i, 1],
    #                        'sheet-normal': speeds[i, 2]} for i in range(speeds.shape[0])])
    #sns.pairplot(df)
    # speedNames = ['endocardial', 'fibre', 'sheet', 'sheet-normal']
    # for speed_name in speedNames:
    #     subset = df[df['speed'] == speed_name]
    #     sns.distplot(subset['val'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = speed_name)
    #
    # # Plot formatting
    # plt.legend(prop={'size': 16}, title = 'Speeds')
    # plt.xlabel('cm/s')
    # plt.ylabel('density')

    #sns.distplot(df, x='val', hue='speed', kind='kde', fill=True)
    

    #RANDOM ROOT NODES
    # From matlab plot_abstract_figures
    conf_1 = np.array([1, 8, 15])-1
    conf_2 = np.array([1, 5, 10])-1
    conf_3 = np.array([4, 7, 17])-1
    conf_4 = np.array([6, 14, 13])-1
    conf_5 = np.array([2, 3, 17])-1
    conf_6 = np.array([11, 19, 20])-1
    
    # Root nodes 1
    atmap = eikonal_ecg(np.array([[0.05, 0.04, 0.03, 0.15]]),
                        np.concatenate((lvActivationIndexes[conf_4], rvActivationIndexes[conf_2])))[0, :]
    with open('Abstract/'+meshName+'_atm_high.ensi.root1', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i])+'\n')
    ecg1 = pseudoBSP(atmap)
    
    # Root nodes 2
    atmap = eikonal_ecg(np.array([[0.05, 0.08, 0.03, 0.15]]),
                        np.concatenate((lvActivationIndexes[conf_3], rvActivationIndexes[conf_1])))[0, :]
    with open('Abstract/'+meshName+'_atm_high.ensi.root2', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i])+'\n')
    ecg2 = pseudoBSP(atmap)
    
    # Root nodes 3
    atmap = eikonal_ecg(np.array([[0.05, 0.02, 0.03, 0.1]]),
                        np.concatenate((lvActivationIndexes[conf_2], rvActivationIndexes[conf_3])))[0, :]
    with open('Abstract/'+meshName+'_atm_high.ensi.root3', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i])+'\n')
    ecg3 = pseudoBSP(atmap)
    
    # Root nodes 4
    atmap = eikonal_ecg(np.array([[0.05, 0.07, 0.03, 0.18]]),
                        np.concatenate((lvActivationIndexes[conf_1], rvActivationIndexes[conf_4])))[0, :]
    with open('Abstract/'+meshName+'_atm_high.ensi.root4', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i])+'\n')
    ecg4 = pseudoBSP(atmap)
    
    # Root nodes 5
    atmap = eikonal_ecg(np.array([[0.1, 0.04, 0.1, 0.15]]),
                        np.concatenate((lvActivationIndexes[conf_5], rvActivationIndexes[conf_6])))[0, :]
    with open('Abstract/'+meshName+'_atm_high.ensi.root5', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i])+'\n')
    ecg5 = pseudoBSP(atmap)
    
    # True atmap
    atmap = eikonal_ecg(np.array([[0.05, 0.04, 0.03, 0.15]]), rootNodesIndexes_true)[0, :]
    with open('Abstract/'+meshName+'_atm_high.ensi.rootTrue', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i])+'\n')
    ecgTrue = pseudoBSP(atmap)
    
    # Save ECGs
    # Calculate figure ms width
    max_count = np.amax(np.vstack((np.sum(np.logical_not(np.isnan(ecgTrue[0, :]))),
                                  np.sum(np.logical_not(np.isnan(ecg1[0, :]))),
                                  np.sum(np.logical_not(np.isnan(ecg2[0, :]))),
                                  np.sum(np.logical_not(np.isnan(ecg3[0, :])))#,
                                  #np.sum(np.logical_not(np.isnan(ecg4[0, :]))),
                                  #np.sum(np.logical_not(np.isnan(ecg5[0, :])))
                                  ))) + 5
    # Create True figure - graphical abstract small version
    fig, axs = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(9.2, 2.15), sharey='all')
    for i in range(nb_leads):
        axs[i].plot(ecgTrue[i, :], 'k-', label='Target', linewidth=2)
        axs[i].set_xlim(0, max_count)
        axs[i].xaxis.set_major_locator(MultipleLocator(80))
        axs[i].xaxis.set_minor_locator(MultipleLocator(40))
        axs[i].yaxis.set_major_locator(MultipleLocator(2))
        #axs[i].yaxis.set_minor_locator(MultipleLocator(1))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.grid(True, which='major')
        axs[i].xaxis.grid(True, which='minor')
        axs[i].yaxis.grid(True, which='major')
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        axs[i].set_xlabel('ms', fontsize=16)
        axs[i].set_title(leadNames[i], fontsize=16)
    axs[0].set_ylabel('standardised volt', fontsize=16)
    plt.savefig('Abstract/' + meshName + '_ecg_true_abstract.png', dpi=300)
    plt.show()
    # Create Prediction figure
    fig, axs = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(9.2, 2.2), sharey='all')
    for i in range(nb_leads):
        axs[i].plot(ecg1[i, :], linewidth=2)
        axs[i].plot(ecg2[i, :], linewidth=2)
        axs[i].plot(ecg3[i, :], linewidth=2)
        #axs[i].plot(ecg4[i, :], linewidth=1.5)
        #axs[i].plot(ecg5[i, :], linewidth=1.5)
        axs[i].set_xlim(0, max_count)
        axs[i].xaxis.set_major_locator(MultipleLocator(80))
        axs[i].xaxis.set_minor_locator(MultipleLocator(40))
        axs[i].yaxis.set_major_locator(MultipleLocator(2))
        #axs[i].yaxis.set_minor_locator(MultipleLocator(1))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.grid(True, which='major')
        axs[i].xaxis.grid(True, which='minor')
        axs[i].yaxis.grid(True, which='major')
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        axs[i].set_xlabel('ms', fontsize=16)
        axs[i].set_title(leadNames[i], fontsize=16)
    axs[0].set_ylabel('standardised volt', fontsize=14)
    plt.savefig('Abstract/' + meshName + '_ecg_pred_abstract.png', dpi=300)
    plt.show()
    
    # Second larger plot for main manuscript
    # Create True figure
    fig, axs = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(6, 1.8), sharey='all')
    for i in range(nb_leads):
        axs[i].plot(ecgTrue[i, :], 'k-', label='Target', linewidth=1.5)
        axs[i].set_xlim(0, max_count)
        axs[i].xaxis.set_major_locator(MultipleLocator(80))
        axs[i].xaxis.set_minor_locator(MultipleLocator(40))
        axs[i].yaxis.set_major_locator(MultipleLocator(2))
        axs[i].yaxis.set_minor_locator(MultipleLocator(1))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.grid(True, which='major')
        axs[i].xaxis.grid(True, which='minor')
        axs[i].yaxis.grid(True, which='major')
        axs[i].yaxis.grid(True, which='minor')
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        axs[i].set_xlabel('ms', fontsize=12)
        axs[i].set_title(leadNames[i], fontsize=12)
    axs[0].set_ylabel('standardised volts', fontsize=12)
    plt.savefig('Abstract/' + meshName + '_ecg_true_manuscript.png', dpi=300)
    plt.show()
    # Create Prediction figure
    fig, axs = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(6, 1.8), sharey='all')
    for i in range(nb_leads):
        axs[i].plot(ecg1[i, :], linewidth=1.5)
        axs[i].plot(ecg2[i, :], linewidth=1.5)
        axs[i].plot(ecg3[i, :], linewidth=1.5)
        #axs[i].plot(ecg4[i, :], linewidth=1.5)
        #axs[i].plot(ecg5[i, :], linewidth=1.5)
        axs[i].set_xlim(0, max_count)
        axs[i].xaxis.set_major_locator(MultipleLocator(80))
        axs[i].xaxis.set_minor_locator(MultipleLocator(40))
        axs[i].yaxis.set_major_locator(MultipleLocator(2))
        axs[i].yaxis.set_minor_locator(MultipleLocator(1))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.grid(True, which='major')
        axs[i].xaxis.grid(True, which='minor')
        axs[i].yaxis.grid(True, which='major')
        axs[i].yaxis.grid(True, which='minor')
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        axs[i].set_xlabel('ms', fontsize=12)
        axs[i].set_title(leadNames[i], fontsize=12)
    axs[0].set_ylabel('standardised volts', fontsize=12)
    plt.savefig('Abstract/' + meshName + '_ecg_pred_manuscript.png', dpi=300)
    plt.show()
    
    ############# Normal stuff for saving results from root ndoes ##############
    
    # K-means CENTROIDS
    atmap = np.zeros((nodesXYZ.shape[0]))
    atmap[lv_rootNodesIndexes_true] = -1000
    atmap[rv_rootNodesIndexes_true] = -1000

    new_lv_roots, new_lv_roots_count = np.unique(np.concatenate((new_lv_roots_list), axis=0), return_counts=True)
    atmap[new_lv_roots] = np.round(100*new_lv_roots_count/len(new_lv_roots_list), 2)

    new_rv_roots, new_rv_roots_count = np.unique(np.concatenate((new_rv_roots_list), axis=0), return_counts=True)
    atmap[new_rv_roots] = np.round(100*new_rv_roots_count/len(new_rv_roots_list), 2)

    # Translate the atmap to be an element-wise map
    elem_atmap = np.zeros((tetrahedrons.shape[0]))
    for i in range(tetrahedrons.shape[0]):
        elem_atmap[i] = np.sum(atmap[tetrahedrons[i]])

    with open('Abstract/'+meshName+'_atm_high.ensi.kMeans_centroids', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
        for i in range(0, len(elem_atmap)):
            f.write(str(elem_atmap[i])+'\n')

    # CUMM ROOT NODES TOGETHER
    atmap = np.zeros((nodesXYZ.shape[0]))
    atmap[lv_rootNodesIndexes_true] = -1000
    atmap[rv_rootNodesIndexes_true] = -1000
    atmap[lvActivationIndexes] = np.round(100*lv_rootNodes/count, 2)#-6 # substracting the baseline value
    atmap[rvActivationIndexes] = np.round(100*rv_rootNodes/count, 2)#-6 # substracting the baseline value

    # Translate the atmap to be an element-wise map
    elem_atmap = np.zeros((tetrahedrons.shape[0]))
    for i in range(tetrahedrons.shape[0]):
        elem_atmap[i] = np.sum(atmap[tetrahedrons[i]])

    with open('Abstract/'+meshName+'_atm_high.ensi.cummrootNodes', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
        for i in range(0, len(elem_atmap)):
            f.write(str(elem_atmap[i])+'\n')
            
    # GROUND TRUTH ROOT NODES
    atmap = np.zeros((nodesXYZ.shape[0]))
    atmap[lv_rootNodesIndexes_true] = -1000
    atmap[rv_rootNodesIndexes_true] = -1000
    
    # Translate the atmap to be an element-wise map
    elem_atmap = np.zeros((tetrahedrons.shape[0]))
    for i in range(tetrahedrons.shape[0]):
        elem_atmap[i] = np.sum(atmap[tetrahedrons[i]])

    with open('Abstract/'+meshName+'_atm_high.ensi.trueNodes', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
        for i in range(0, len(elem_atmap)):
            f.write(str(elem_atmap[i])+'\n')


def test_dtw(population, population_2, meshName_val, meshVolume_val, threadsNum_val, experiment_output_val):
    global meshName
    meshName = meshName_val
    global meshVolume
    meshName = meshVolume_val
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    global experiment_output
    experiment_output = experiment_output_val
    # Eikonal configuration
    global leadNames
    leadNames = [ 'I' , 'II' , 'V1' , 'V2' , 'V3' , 'V4' , 'V5' , 'V6', 'lead_prog' ]
    global nb_leads
    global nb_bsp
    frequency = 1000 # Hz
    freq_cut= 150 # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2) # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')

    # SMC-ABC configuration
    global nlhsParam
    nlhsParam = 4

    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'
    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_lvface.csv',
                                   delimiter=',') - 1).astype(int)) # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_rvface.csv',
                                   delimiter=',') - 1).astype(int)) # rv endocardium triangles
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors

    # global rootNodeActivationIndexes
    rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
    rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
#     rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
#     nRootLocations=rootNodeActivationIndexes.shape[0]
#     nparam = nlhsParam + nRootLocations
# #         scale_param=np.zeros((nparam,)).astype(bool)
# # #             param_ranges = np.concatenate((np.array([gtRange, gtRange, gtRange, endoRange]), np.array([[0, 1] for i in range(nRootLocations)])), axis=0)
#     param_boundaries = np.concatenate((np.array([gtRange, gtRange, gtRange, endoRange]), np.array([[0, 1] for i in range(nRootLocations)])))

    global tetrahedrons
    tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
    tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
    ecg_electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    if experiment_output == 'bsp':
        ECGi_electrodePositions = np.loadtxt(dataPath + meshName + '_ECGiElectrodePositions.csv', delimiter=',')
        nb_leads = 8 + ECGi_electrodePositions.shape[0] # All leads from the ECGi are calculated like the precordial leads
        electrodePositions = np.concatenate((ecg_electrodePositions, ECGi_electrodePositions), axis=0)
    else:
        nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad  # 8 + lead progression (or 12)
        electrodePositions = ecg_electrodePositions
    nb_bsp = electrodePositions.shape[0]

    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, tetrahedrons.shape[0], 1):
        aux[tetrahedrons[i, 0]].append(i)
        aux[tetrahedrons[i, 1]].append(i)
        aux[tetrahedrons[i, 2]].append(i)
        aux[tetrahedrons[i, 3]].append(i)
    global elements
    elements = [np.array(n) for n in aux]
    aux = None # Clear Memory

    # Precompute PseudoECG stuff
    # Calculate the tetrahedrons volumes
    D = nodesXYZ[tetrahedrons[:, 3], :] # RECYCLED
    A = nodesXYZ[tetrahedrons[:, 0], :]-D # RECYCLED
    B = nodesXYZ[tetrahedrons[:, 1], :]-D # RECYCLED
    C = nodesXYZ[tetrahedrons[:, 2], :]-D # RECYCLED
    D = None # Clear Memory

    global tVolumes
    tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1
                                           ), (np.cross(B, C)[:, :, np.newaxis]))),
                          tetrahedrons.shape[0]) #Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
    tVolumes = tVolumes/np.sum(tVolumes)

    # Calculate the tetrahedron (temporal) voltage gradients
    Mg = np.stack((A, B, C), axis=-1)
    A = None # Clear Memory
    B = None # Clear Memory
    C = None # Clear Memory

    # Calculate the gradients
    global G_pseudo
    G_pseudo = np.zeros(Mg.shape)
    for i in range(Mg.shape[0]):
        G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
    G_pseudo = np.moveaxis(G_pseudo, 1, 2)
    Mg = None # clear memory

    # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
    r=np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
           (tetrahedronCenters.shape[0],
            tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1)-electrodePositions

    global d_r
    d_r= np.moveaxis(np.multiply(
        np.moveaxis(r, [0, 1], [-1, -2]),
        np.multiply(np.moveaxis(np.sqrt(np.sum(r**2, axis=2))**(-3), 0, 1), tVolumes)), 0, -1)

    # Set endocardial edges aside
    global isEndocardial
    isEndocardial=np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                          np.all(np.isin(edges, rvface), axis=1))
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None # Clear Memory

    # Generate ECGs
    pseudo_ecg = eikonal_ecg(population, rootNodesIndexes_true, rootNodesTimes)
    pseudo_ecg_2 = eikonal_ecg(population_2, rootNodesIndexes_true, rootNodesTimes)
    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(40, 5), sharey='all')
    fig.suptitle(meshName, fontsize=24)

    # Calculate figure ms width
    max_count = max(np.sum(np.logical_not(np.isnan(pseudo_ecg[0, 0, :]))),
                    np.sum(np.logical_not(np.isnan(pseudo_ecg_2[0, 0, :])))) + 5
    for i in range(nb_leads):
        leadName = leadNames[i]
        axs[i].plot(pseudo_ecg[0, i, :], 'b-', label='Baseline', linewidth=3.)
        axs[i].plot(pseudo_ecg_2[0, i, :], 'g-', label='faster Endo', linewidth=3.)

        # decorate figure
        axs[i].set_xlim(0, max_count)
        axs[i].xaxis.set_major_locator(MultipleLocator(10))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.grid(True, which='major')
        axs[i].yaxis.grid(True, which='major')
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        axs[i].set_title('Lead ' + leadName, fontsize=20)
        axs[i].set_xlabel('ms', fontsize=20)
        total_cost, dtw_cost = dtw_trianglorgram(x=pseudo_ecg[0, i, np.logical_not(np.isnan(pseudo_ecg[0, 0, :]))],
                                                 y=pseudo_ecg_2[0, i, np.logical_not(np.isnan(pseudo_ecg_2[0, 0, :]))])
        axs[i].set_title('Lead ' + leadName + '\nD: ' + str(round(total_cost-dtw_cost, 1))
                         + '   -   ' + str(round(dtw_cost, 1))
                         + '\nL: ' + str(np.sum(np.logical_not(np.isnan(pseudo_ecg[0, 0, :]))))
                         + '   -   ' + str(np.sum(np.logical_not(np.isnan(pseudo_ecg_2[0, 0, :]))))
                         + '  ==  '+ str(abs(np.sum(np.logical_not(np.isnan(pseudo_ecg[0, 0, :])))
                                             - np.sum(np.logical_not(np.isnan(pseudo_ecg_2[0, 0, :]))))), fontsize=16)
    axs[0].set_ylabel('standardised voltage', fontsize=16)
    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show()


# ------------------------------------------- MAIN RESULTS FUNCTIONS ----------------------------------------------------

# Show pseudoECG vs diffusion
def makeFigure_Compare_PseudoECG_vs_Diffusion():
    # Definitions
    figuresPath = 'Figures/'
    meshName = 'DTI003'
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_bsp
    global nb_leads
    nb_leads = 8
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt, a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    dataPath = 'metaData/' + meshName + '/'

    # Pseudo ECG
    atmap = np.loadtxt('metaData/ATMaps/' + meshName + '_coarse_true_ATMap_120_1x.csv', delimiter=',')

    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global tetrahedrons
    tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
    tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
    electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    nb_bsp = electrodePositions.shape[0]

    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, tetrahedrons.shape[0], 1):
        aux[tetrahedrons[i, 0]].append(i)
        aux[tetrahedrons[i, 1]].append(i)
        aux[tetrahedrons[i, 2]].append(i)
        aux[tetrahedrons[i, 3]].append(i)
    global elements
    elements = [np.array(n) for n in aux]

    # Calculate the tetrahedrons volumes
    D = nodesXYZ[tetrahedrons[:, 3], :]
    A = nodesXYZ[tetrahedrons[:, 0], :] - D
    B = nodesXYZ[tetrahedrons[:, 1], :] - D
    C = nodesXYZ[tetrahedrons[:, 2], :] - D

    tVolumes = np.reshape(
        np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1), (np.cross(B, C)[:, :, np.newaxis]))),
        tetrahedrons.shape[0])
    tVolumes = tVolumes / np.sum(tVolumes)

    # Calculate the tetrahedron (temporal) voltage gradients
    Mg = np.stack((A, B, C), axis=-1)

    # Calculate the gradients
    global G_pseudo
    G_pseudo = np.zeros(Mg.shape)
    for i in range(Mg.shape[0]):
        G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
    G_pseudo = np.moveaxis(G_pseudo, 1, 2)

    # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
    r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                               (tetrahedronCenters.shape[0], tetrahedronCenters.shape[1], electrodePositions.shape[0])),
                    1, -1) - electrodePositions
    global d_r
    d_r = np.moveaxis(np.multiply(np.moveaxis(r, [0, 1], [-1, -2]),
                                  np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0,
                      -1)

    pseudo_ecg = pseudoBSP(atmap)

    # Bidomain ECG
    LA = np.loadtxt('metaData/ECGs/trace_LA_120_1x.txt')[:, 2]
    LL = np.loadtxt('metaData/ECGs/trace_LL_120_1x.txt')[:, 2]
    RA = np.loadtxt('metaData/ECGs/trace_RA_120_1x.txt')[:, 2]
    RL = np.loadtxt('metaData/ECGs/trace_RL_120_1x.txt')[:, 2]
    V1 = np.loadtxt('metaData/ECGs/trace_V1_120_1x.txt')[:, 2]
    V2 = np.loadtxt('metaData/ECGs/trace_V2_120_1x.txt')[:, 2]
    V3 = np.loadtxt('metaData/ECGs/trace_V3_120_1x.txt')[:, 2]
    V4 = np.loadtxt('metaData/ECGs/trace_V4_120_1x.txt')[:, 2]
    V5 = np.loadtxt('metaData/ECGs/trace_V5_120_1x.txt')[:, 2]
    V6 = np.loadtxt('metaData/ECGs/trace_V6_120_1x.txt')[:, 2]

    BSP = np.zeros((10, LA.shape[0]))  # order is: ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    BSP[0, :] = LA
    BSP[1, :] = RA
    BSP[2, :] = LL
    BSP[3, :] = RL
    BSP[4, :] = V1
    BSP[5, :] = V2
    BSP[6, :] = V3
    BSP[7, :] = V4
    BSP[8, :] = V5
    BSP[9, :] = V6

    ECG = np.zeros((nb_leads, LA.shape[0]))
    ECG[0, :] = (BSP[0, :] - BSP[1, :])
    ECG[1, :] = (BSP[2, :] - BSP[1, :])
    BSPecg = BSP - np.mean(BSP[0:2, :], axis=0)
    BSP = None  # Clear Memory
    ECG[2:8, :] = BSPecg[4:10, :]
    ECG = signal.filtfilt(b_filtfilt, a_filtfilt, ECG)
    ECG = (ECG - np.mean(ECG[:, :pseudo_ecg.shape[1]], axis=1)[:, np.newaxis])
    ECG = (ECG / np.std(ECG[:, :pseudo_ecg.shape[1]], axis=1)[:, np.newaxis])
    ECG = ECG - ECG[:, 0][:, np.newaxis]  # align at zero
    
    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(8, 3), sharey='all')
    # fig.suptitle('12-lead ECG-QRS Bidomain versus PseudoECG', fontsize=24)

    # Calculate figure ms width
    max_count = np.sum(np.logical_not(np.isnan(pseudo_ecg[0, :]))) + 5
    for i in range(nb_leads):
        leadName = leadNames[i]
        # Print out Pearson's correlation coefficients for each lead
        print(leadName + ': ' + str(np.corrcoef(ECG[i, :pseudo_ecg.shape[1]], pseudo_ecg[i, :])[0,1]))
        axs[i].plot(ECG[i, :pseudo_ecg.shape[1]], 'k-', label='Diffusion', linewidth=1.5)
        axs[i].plot(pseudo_ecg[i, :], 'g-', label='Pseudo', linewidth=1.5)

        # decorate figure
        axs[i].set_xlim(0, max_count)
        axs[i].xaxis.set_major_locator(MultipleLocator(40))
        axs[i].yaxis.set_major_locator(MultipleLocator(2))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.grid(True, which='major')
        axs[i].yaxis.grid(True, which='major')
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        axs[i].set_title('Lead ' + leadName, fontsize=14)
        axs[i].set_xlabel('ms', fontsize=14)
    axs[0].set_ylabel('standardised voltage', fontsize=14)
    #axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.savefig(figuresPath + meshName + '_coarse_120_1x_pseudoECG_vs_diffusion_comparison.png', dpi=300)
    plt.show()


# AUXILIAR FUNCTION FOR MAKESPEEDFIGURES FUNCTION
def makeSpeed(pred_values, true_value, x_d, bandwidths):
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
    grid.fit(pred_values[:, None]);
    aux_bandwidth = grid.best_params_['bandwidth']
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=aux_bandwidth, kernel='gaussian')
    kde.fit(pred_values[:, None])
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    probabilities = np.exp(logprob)

    x_range = []
    y_range = []
    for xi in pred_values:
        x_range.append(xi)
        x_range.append(xi)
        x_range.append(None)
        y_range.append(-np.amax(probabilities) * 0.05)
        y_range.append(0.)
        y_range.append(None)

    yaxis_range = [-np.amax(probabilities) * 0.05, np.amax(probabilities) + np.amax(probabilities) * 0.1]
    xaxis_range = [np.amin(pred_values) - (np.amax(pred_values) - np.amin(pred_values)) * 0.1,
                   np.amax(pred_values) + (np.amax(pred_values) - np.amin(pred_values)) * 0.1]

    map_height = int(probabilities.shape[0])
    x_heat_range = np.array(x_d, copy=True)
    y_heat_range = np.linspace(0., yaxis_range[1], map_height)
    z_heat_range = np.zeros((map_height, probabilities.shape[0]))
    for i in range(probabilities.shape[0]):
        for j in range(map_height):
            if y_heat_range[j] > probabilities[i]:
                z_heat_range[j, i] = None
            else:
                z_heat_range[j, i] = np.abs(100 * (x_d[i] - true_value) / true_value)
    fig = None
    #     fig = graph_objects.Figure(data=[
    #         graph_objects.Heatmap(z=z_heat_range,x=x_heat_range,y=y_heat_range,hoverongaps = False, hoverinfo='skip'),
    #         graph_objects.Scatter(x=x_d, y=probabilities,# fill='tozeroy', #line=dict(width=0.5, color='rgba(0., 1., 0.5, 0.5)'),
    #                mode="lines", line_color ='rgba(0.3, 0.3, 1., 1.)'),
    #         graph_objects.Scatter(x=x_range, y=y_range, line_color='orange')
    #     ])
    #     fig.update_layout(width=1500, height=500, showlegend=False, yaxis_range=yaxis_range, xaxis_range=xaxis_range)

    mean_error = np.round(100 * (np.mean(pred_values) - true_value) / true_value, 2)
    median_error = np.round(100 * (np.median(pred_values) - true_value) / true_value, 2)
    mode, count = stats.mode(np.round(pred_values, 0))  # Lose some definition to gain prediction power
    mode_error = np.round(100 * (mode[0] - true_value) / true_value, 2)
    kde_pred = x_d[np.argmax(probabilities)]
    kde_error = np.round(100 * (kde_pred - true_value) / true_value, 2)
    # return fig, mean_error, median_error, mode_error, kde_error, aux_bandwidth
    return fig, mean_error, median_error, mode_error, kde_error, aux_bandwidth


# GENERATE SPEED ERROR HISTOGRAMS AND DATAFRAME WITH SPEED AND ROOT NODE ERRORS 04/02/2021
def makeSpeedFigures(mesh_name_list, group_labels, conduction_speed_names_plot, conduction_speed_names_data,
                        previousResultsPath, colors, target_type, discretisation, figPath, do_root_error):
    population_files = []
    for fileName in os.listdir(previousResultsPath):
        if ('population' in fileName and discretisation in fileName and target_type in fileName
                and np.any([meshName in fileName for meshName in mesh_name_list])):
            population_files.append(fileName)
    file_list = []
    mesh_list = []
    conduction_median_error_list_dic = {
        conduction_speed_names_data[i]: [] for i in range(len(conduction_speed_names_data))}
    conduction_median_error_list_dic['lv_loc_error'] = []
    conduction_median_error_list_dic['rv_loc_error'] = []
    conduction_median_error_list_dic['lv_nb_error'] = []
    conduction_median_error_list_dic['rv_nb_error'] = []
    for meshName in mesh_name_list:
        if do_root_error:
            nodesXYZ = np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_xyz.csv', delimiter=',')
            edges = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_edges.csv',
                                delimiter=',') - 1).astype(int)
            lvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_lvnodes.csv',
                                delimiter=',') - 1).astype(int)  # lv endocardium triangles
            lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
            rvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_rvnodes.csv',
                                delimiter=',') - 1).astype(int)  # rv endocardium triangles
            rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
    
            rootNodesIndexes_true = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
            rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
            lv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in lvnodes])
            rv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in rvnodes])
            lv_rootNodes_true = nodesXYZ[lv_rootNodesIndexes_true, :]
            rv_rootNodes_true = nodesXYZ[rv_rootNodesIndexes_true, :]
            # Predictions
            lvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_lv_activationIndexes_' + discretisation + 'Res.csv', delimiter=',') - 1).astype(int)
            rvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rv_activationIndexes_' + discretisation + 'Res.csv', delimiter=',') - 1).astype(int)
    
            for i in range(lvActivationIndexes.shape[0]):
                if lvActivationIndexes[i] not in lvnodes:
                    a = lvnodes[np.argmin(
                        np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2,
                                       axis=1)).astype(int)]
                    # print('diff ' + str(np.round(
                    #     np.linalg.norm(nodesXYZ[a, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=0) * 10,
                    #     2)) + ' mm')
                    lvActivationIndexes[i] = lvnodes[np.argmin(
                        np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2,
                                       axis=1)).astype(int)]
            for i in range(rvActivationIndexes.shape[0]):
                if rvActivationIndexes[i] not in rvnodes:
                    b = rvnodes[np.argmin(
                        np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2,
                                       axis=1)).astype(int)]
                    # print('diff ' + str(np.round(
                    #     np.linalg.norm(nodesXYZ[b, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=0) * 10,
                    #     2)) + ' mm')
                    rvActivationIndexes[i] = rvnodes[np.argmin(
                        np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2,
                                       axis=1)).astype(int)]
    
            # Calculate Djikstra distances in the LV endocardium
            lvnodesXYZ = nodesXYZ[lvnodes, :]
            # Set endocardial edges aside
            lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
            lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
            lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
            lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
            lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
            aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
            for i in range(0, len(lvunfoldedEdges), 1):
                aux[lvunfoldedEdges[i, 0]].append(i)
            lvneighbours = [np.array(n) for n in aux]
            aux = None  # Clear Memory
            lvActnode_ids = np.asarray(
                [np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
            lvdistance_mat = djikstra(lvActnode_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours)
    
            # Calculate Djikstra distances in the RV endocardium
            rvnodesXYZ = nodesXYZ[rvnodes, :]
            # Set endocardial edges aside
            rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
            rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
            rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
            rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
            rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
            aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
            for i in range(0, len(rvunfoldedEdges), 1):
                aux[rvunfoldedEdges[i, 0]].append(i)
            rvneighbours = [np.array(n) for n in aux]
            aux = None  # Clear Memory
            rvActnode_ids = np.asarray(
                [np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
            rvdistance_mat = djikstra(rvActnode_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours)

        # Iterate over files and do each separately and store the results into the dictionary
        for fileName in population_files:
            if meshName in fileName:
                if '1x' in fileName or '2x' in fileName:
                    if '120' in fileName:
                                endoSpeedTag = '120'
                                endoSpeed_true = 120
                    elif '179' in fileName:
                        endoSpeedTag = '179'
                        endoSpeed_true = 179.0
                    elif '150' in fileName:
                        endoSpeedTag = '150'
                        endoSpeed_true = 150
                    else:
                        raise
                    if '1x' in fileName:
                        epiSpeedTag = '1x'
                    else:
                        epiSpeedTag = '2x'
                    myocardial_true = 1000 * np.median(np.loadtxt(
                                    'metaData/MyocardialSpeeds/' + meshName + '_coarse_true_MyocardialSpeed_' + endoSpeedTag + '_' + epiSpeedTag + '.csv',
                                    delimiter=','), axis=0)
                    conduction_speeds_true = [myocardial_true[0], myocardial_true[1], myocardial_true[2], endoSpeed_true]
                elif '[' in fileName and ']' in fileName:
                    conduction_speeds_true = list(map(float, fileName[fileName.find("[")+1 : fileName.find("]")].split(', ')))
                ## General file metadata
                file_list.append(fileName)
                mesh_list.append(meshName)
                # Remove duplicates since these are 'randomly' generated to terminate the inference, even from different but equivalent root node locations
                population = np.unique(np.round(np.loadtxt(previousResultsPath + fileName, delimiter=','), 3)[:, :4], axis=0)# TODO: change back to median 22/02/2021
                # population = np.unique(np.loadtxt(previousResultsPath + fileName, delimiter=','), axis=0)

                # Conduction speeds part
                for cond_i in range(len(conduction_speed_names_data)):
                    key_name = conduction_speed_names_data[cond_i]
                    # Sheet speed predictions
                    # Use kde # TODO: change back to median 22/02/2021
                    # if 'atm' in target_type:
                    #     bandwidths = np.unique(np.round(10 ** np.linspace(-1, 0.5, 32), 2))
                    # elif 'ecg' in target_type:
                    #     bandwidths = np.unique(np.round(10 ** np.linspace(-1, 1., 128), 2))
                    # else:
                    #     bandwidths = np.unique(np.round(10 ** np.linspace(-1, 1., 128), 2))
                    # fig, mean_error, median_error, mode_error, kde_error, aux_bandwidth = makeSpeed(
                    #     pred_values=1000 * population[:, cond_i], true_value=conduction_speeds_true[cond_i],
                    #     x_d=np.linspace(6, 100, 1000), bandwidths=bandwidths)
                    # conduction_median_error_list_dic[key_name].append(kde_error)
                    median_error = np.round(100 * ((np.median(1000 * population[:, cond_i])) - # TODO: -10. from median
                            conduction_speeds_true[cond_i]) / conduction_speeds_true[cond_i], 2)
                    # median_error = np.round(np.median(1000 * population[:, cond_i]) # TODO: change back to percentage 22/02/2021
                    #         - conduction_speeds_true[cond_i], 2)
                    conduction_median_error_list_dic[key_name].append(median_error)
                
                if do_root_error:
                    # Root nodes part, remove duplicates with different speed values
                    population = np.unique(np.loadtxt(previousResultsPath + fileName, delimiter=',')[:, 4:], axis=0)
                    rootNodes = np.round(population).astype(int)  # Non-repeated particles
                    # Number of Root nodes percentage-error
                    lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
                    rv_rootNodes_part = rootNodes[:, len(lvActivationIndexes):]
    
                    # K-means for root nodes: Choose the number of clusters 'k'
                    lv_num_nodes = np.sum(lv_rootNodes_part, axis=1)
                    lv_num_nodes, lv_num_nodes_counts = np.unique(lv_num_nodes, return_counts=True, axis=0)
                    lv_num_clusters = lv_num_nodes[np.argmax(lv_num_nodes_counts)]
    
                    rv_num_nodes = np.sum(rv_rootNodes_part, axis=1)
                    rv_num_nodes, rv_num_nodes_counts = np.unique(rv_num_nodes, return_counts=True, axis=0)
                    rv_num_clusters = rv_num_nodes[np.argmax(rv_num_nodes_counts)]
    
                    # Choose the initial centroides
                    k_lv_rootNodes_list, lvunique_counts = np.unique(lv_rootNodes_part, return_counts=True, axis=0)
                    lv_rootNodes_list_index = np.sum(k_lv_rootNodes_list, axis=1) == lv_num_clusters
                    k_lv_rootNodes_list = k_lv_rootNodes_list[lv_rootNodes_list_index, :]
                    lvunique_counts = lvunique_counts[lv_rootNodes_list_index]
                    lv_centroid_ids = lvActnode_ids[
                        k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)].astype(int)
    
                    k_rv_rootNodes_list, rvunique_counts = np.unique(rv_rootNodes_part, return_counts=True, axis=0)
                    rv_rootNodes_list_index = np.sum(k_rv_rootNodes_list, axis=1) == rv_num_clusters
                    k_rv_rootNodes_list = k_rv_rootNodes_list[rv_rootNodes_list_index, :]
                    rvunique_counts = rvunique_counts[rv_rootNodes_list_index]
                    rv_centroid_ids = rvActnode_ids[
                        k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)].astype(int)
                    
                    # Transform the root nodes predicted data into k-means data
                    lvdata_ids = np.concatenate(([lvActivationIndexes[(lv_rootNodes_part[i, :]).astype(bool)] for i in
                                                  range(lv_rootNodes_part.shape[0])]), axis=0)
                    lvdata_ids = np.asarray(
                        [np.flatnonzero(lvActivationIndexes == node_id)[0] for node_id in lvdata_ids]).astype(int)
    
                    rvdata_ids = np.concatenate(([rvActivationIndexes[(rv_rootNodes_part[i, :]).astype(bool)] for i in
                                                  range(rv_rootNodes_part.shape[0])]), axis=0)
                    rvdata_ids = np.asarray(
                        [np.flatnonzero(rvActivationIndexes == node_id)[0] for node_id in rvdata_ids]).astype(int)
    
                    # K-means algorithm: LV
                    k_means_lv_labels, k_means_lv_centroids = k_means(lvnodesXYZ[lvActnode_ids, :], lvdata_ids,
                                                                      lv_num_clusters, lv_centroid_ids, lvdistance_mat,
                                                                      lvnodesXYZ, max_iter=10)
                    # Any node in the endocardium can be a result
                    new_lv_roots = np.unique(lvnodes[k_means_lv_centroids])
    
                    # RV
                    k_means_rv_labels, k_means_rv_centroids = k_means(rvnodesXYZ[rvActnode_ids, :], rvdata_ids,
                                                                      rv_num_clusters, rv_centroid_ids, rvdistance_mat,
                                                                      rvnodesXYZ, max_iter=10)
                    # Any node in the endocardium can be a result
                    new_rv_roots = np.unique(rvnodes[k_means_rv_centroids])
                    
                    # Check if something went wrong
                    if new_lv_roots.shape[0] != lv_num_clusters:
                        print('LV')
                        raise
    
                    if new_rv_roots.shape[0] != rv_num_clusters:
                        print('RV')
                        print(fileName)
                        print(rv_centroid_ids)
                        print(k_means_rv_centroids)
                        print(rvnodes[k_means_rv_centroids])
                        print(lv_centroid_ids)
                        raise
    
                    # ROOT NODE ERROR: LV
                    lv_dist = np.linalg.norm(lv_rootNodes_true[:, np.newaxis, :] - nodesXYZ[new_lv_roots, :], ord=2, axis=2)
                    conduction_median_error_list_dic['lv_loc_error'].append(np.mean(np.amin(lv_dist, axis=1)))
                    conduction_median_error_list_dic['lv_nb_error'].append(abs(new_lv_roots.shape[0] - lv_rootNodes_true.shape[0]))
                    # RV
                    rv_dist = np.linalg.norm(rv_rootNodes_true[:, np.newaxis, :] - nodesXYZ[new_rv_roots, :], ord=2, axis=2)
                    conduction_median_error_list_dic['rv_loc_error'].append(np.mean(np.amin(rv_dist, axis=1)))
                    conduction_median_error_list_dic['rv_nb_error'].append(abs(new_rv_roots.shape[0] - rv_rootNodes_true.shape[0]))
        
    #     for fileName in population_files:
    #         if meshName in fileName and target_type in fileName:
    #             population, particles_index = np.unique(np.loadtxt(previousResultsPath + fileName, delimiter=','),
    #                                                     axis=0, return_index=True)
    #             fileNameList.append(fileName)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #     data_list.append({
    #         'fileName': fileNameList,
    #         'meshName': meshName,
    #         'lv_roots': new_lv_roots_list,
    #         'rv_roots': new_rv_roots_list,
    #         'lv_roots_location_error': lv_location_error,
    #         'rv_roots_location_error': rv_location_error,
    #         'lv_roots_number_error': lv_number_error,
    #         'rv_roots_number_error': rv_number_error,
    #     })
    #
    # df_roots = pandas.DataFrame(data_list)
    # for error_name in ['lv_roots_location_error', 'lv_roots_number_error', 'rv_roots_location_error', 'rv_roots_number_error']:
    #     hist_data = [df_roots[df_roots['meshName'] == meshNameList[i]][error_name].values[0] for i in range(len(meshNameList))]
    #     m = np.round(np.mean([np.mean(x) for x in hist_data]), 2)
    #     s = np.round(np.mean([np.std(x) for x in hist_data]), 2)
    #
    #     print(error_name + ' :' + str(m) + ' + ' + str(s) + ' cm')
    #
    # df_roots.to_pickle(path=resultsPath + target_type + '_' + discretisation_resolution + '_rootNodes_error.gz', compression='infer')
    # return df_roots
    
    
    # Save and check that has saved correctly
    if do_root_error:
        df = pandas.DataFrame([{'fileName': file_list[i], 'meshName': mesh_list[i],
                                conduction_speed_names_data[0]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[0]][i],
                                conduction_speed_names_data[1]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[1]][i],
                                conduction_speed_names_data[2]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[2]][i],
                                conduction_speed_names_data[3]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[3]][i],
                                'lv_loc_error': conduction_median_error_list_dic['lv_loc_error'][i],
                                'rv_loc_error': conduction_median_error_list_dic['rv_loc_error'][i],
                                'lv_nb_error': conduction_median_error_list_dic['lv_nb_error'][i],
                                'rv_nb_error': conduction_median_error_list_dic['rv_nb_error'][i]
                                }
                                for i in range(len(file_list))])
    else:
        df = pandas.DataFrame([{'fileName': file_list[i], 'meshName': mesh_list[i],
                                conduction_speed_names_data[0]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[0]][i],
                                conduction_speed_names_data[1]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[1]][i],
                                conduction_speed_names_data[2]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[2]][i],
                                conduction_speed_names_data[3]+'-median-error': conduction_median_error_list_dic[conduction_speed_names_data[3]][i]
                                }
                                for i in range(len(file_list))])
    
    dataFrame_path = figPath + target_type + '_' + discretisation + '_dataFrame.gz'
    df.to_pickle(path=dataFrame_path, compression='infer')
    # Reduntant attempts to save the dataframe
    attemp_count = 4
    while (not df.equals(pandas.read_pickle(filepath_or_buffer=dataFrame_path, compression='infer'))) and (
            attemp_count > 0):
        print('Sleeping')
        time.sleep(1.)
        df.to_pickle(path=dataFrame_path, compression='infer')
        time.sleep(1.)
        attemp_count = attemp_count - 1
    
    for speed_i in range(len(conduction_speed_names_plot), 0, -1):
        speed = conduction_speed_names_plot[speed_i-1]
        # print('Median-Error for ' + speed + ' speed: ' + str(np.median(np.abs(df[speed + '-median-error'].values))))
        # print(df.keys())
        print('Mean-Error for ' + speed + ' speed: ' + str(np.mean(np.abs(df[speed + '-median-error'].values))))
        print('STD-Error for ' + speed + ' speed: ' + str(np.std(np.abs(df[speed + '-median-error'].values))))
        print()
    
    speed_str_list = []
    for speed_i in range(len(conduction_speed_names_plot), 0, -1):
        speed = conduction_speed_names_plot[speed_i-1]
        speed_str = (str(round(np.mean(np.abs(df[speed + '-median-error'].values)), 1)) + ' ' + str(u"\u00B1")
                + ' ' + str(round(np.std(np.abs(df[speed + '-median-error'].values)), 1)))
        print(speed_str)
        speed_str_list.append(speed_str)
    print()
    print(speed_str_list)
    print()
    
    # Make figure using horizontal box plots
    max_width = 100. # TODO: 02/03/2021 it was zero
    fig = plotly.graph_objects.Figure()
    for mesh_i in range(len(mesh_name_list)):
        groupName = group_labels[mesh_i]
        meshName = mesh_name_list[mesh_i]
        y = []
        x = []
        hist_data = df[df['meshName'] == meshName]
        for speed_i in range(len(conduction_speed_names_plot)):
            speed = conduction_speed_names_plot[speed_i]
            value_list = hist_data[speed + '-median-error'].values
            for i in range(len(value_list)):
                y.append(speed)
                x.append(value_list[i])
        fig.add_trace(plotly.graph_objects.Box(
            x=x,
            y=y,
            name=groupName,
            marker_color=colors[mesh_i]
        ))
        max_width = max(max_width, np.amax(np.abs(np.asarray(x))))
    max_width = math.ceil(max_width)
    if 'atm' in target_type:
        split_tick = 20
    else:
        split_tick = 20
    max_width = int(math.ceil(max_width/split_tick) * split_tick)
    fig.update_layout(xaxis={'autorange': False, 'fixedrange': True, 'range': [max_width*(-1), max_width], 'tickfont': {'size': 20},
                             'tickvals': list(range(max_width*(-1), max_width+split_tick, split_tick))},
                      width=int(1000),
                      height=int(800),
                      yaxis={'tickfont': {'size': 20}},
                      legend={'font': {'size': 20}},
                      boxmode='group', legend_traceorder="reversed")
    fig.update_traces(orientation='h') # horizontal box plots
    fig.show()
    fig.write_image(file=figPath+'/'+ target_type + '_' + discretisation + '_speed_errors.png', format='png', engine="kaleido")
    return df



# CALCULATE ERROR OF ROOT NODES 18/12/2020
def calcRootError(criteria_names, previousResultsPath, exclude_names, discretisation_resolution, target_type, resultsPath):
    population_files = []
    for fileName in os.listdir(previousResultsPath):
        if 'population' in fileName and discretisation_resolution in fileName and target_type in fileName and np.all(
                [criteria in fileName for criteria in criteria_names]) and (
        not np.any([exclude in fileName for exclude in exclude_names])):
            population_files.append(fileName)
    meshNameList = [meshName for meshName in ['DTI003', 'DTI001', 'DTI004', 'DTI024'] if
                    np.any(np.array([meshName in fileName for fileName in population_files]))]
                    
    data_list = []
    for meshName in meshNameList:
        lv_location_error = []
        lv_number_error = []
        rv_location_error = []
        rv_number_error = []
        fileNameList = []
        # lv_rootNodes = None
        # rv_rootNodes = None
        # lv_rootNodes_list = None
        # rv_rootNodes_list = None
        # rootNodes_list = None
        # new_lv_roots_ids_list = []
        # new_lv_roots_weights_list = []
        # new_rv_roots_ids_list = []
        # new_rv_roots_weights_list = []
        new_lv_roots_list = []
        new_rv_roots_list = []

        # disc_roots_list = []
        # count = 0.
        nodesXYZ = np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_xyz.csv', delimiter=',')
        # elems = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
        edges = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_edges.csv',
                            delimiter=',') - 1).astype(int)
        lvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_lvnodes.csv',
                            delimiter=',') - 1).astype(int)  # lv endocardium triangles
        lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
        rvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_rvnodes.csv',
                            delimiter=',') - 1).astype(int)  # rv endocardium triangles
        rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
        # True
        rootNodesIndexes_true = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
        rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
        lv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in lvnodes])
        rv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in rvnodes])
        lv_rootNodes_true = nodesXYZ[lv_rootNodesIndexes_true, :]
        rv_rootNodes_true = nodesXYZ[rv_rootNodesIndexes_true, :]
        # Predictions
        lvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_lv_activationIndexes_' + discretisation_resolution + 'Res.csv', delimiter=',') - 1).astype(int)
        rvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rv_activationIndexes_' + discretisation_resolution + 'Res.csv', delimiter=',') - 1).astype(int)

        # activationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
        print(meshName)
        for i in range(lvActivationIndexes.shape[0]):
            if lvActivationIndexes[i] not in lvnodes:
                a = lvnodes[np.argmin(
                    np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2,
                                   axis=1)).astype(int)]
                # print('diff ' + str(np.round(
                #     np.linalg.norm(nodesXYZ[a, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=0) * 10,
                #     2)) + ' mm')
                lvActivationIndexes[i] = lvnodes[np.argmin(
                    np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2,
                                   axis=1)).astype(int)]
        for i in range(rvActivationIndexes.shape[0]):
            if rvActivationIndexes[i] not in rvnodes:
                b = rvnodes[np.argmin(
                    np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2,
                                   axis=1)).astype(int)]
                # print('diff ' + str(np.round(
                #     np.linalg.norm(nodesXYZ[b, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=0) * 10,
                #     2)) + ' mm')
                rvActivationIndexes[i] = rvnodes[np.argmin(
                    np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2,
                                   axis=1)).astype(int)]

        # Calculate Djikstra distances in the LV endocardium
        # lvdistance_mat = np.zeros((lvnodes.shape[0], lvActivationIndexes.shape[0]))
        lvnodesXYZ = nodesXYZ[lvnodes, :]
        # Set endocardial edges aside
        lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
        lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
        lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
        lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
        lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
        aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
        for i in range(0, len(lvunfoldedEdges), 1):
            aux[lvunfoldedEdges[i, 0]].append(i)
        lvneighbours = [np.array(n) for n in aux]
        aux = None  # Clear Memory
        lvActnode_ids = np.asarray(
            [np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
        lvdistance_mat = djikstra(lvActnode_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours)

        # Calculate Djikstra distances in the RV endocardium
        # rvdistance_mat = np.zeros((rvnodes.shape[0], rvActivationIndexes.shape[0]))
        rvnodesXYZ = nodesXYZ[rvnodes, :]
        # Set endocardial edges aside
        rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
        rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
        rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
        rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
        rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
        aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
        for i in range(0, len(rvunfoldedEdges), 1):
            aux[rvunfoldedEdges[i, 0]].append(i)
        rvneighbours = [np.array(n) for n in aux]
        aux = None  # Clear Memory
        rvActnode_ids = np.asarray(
            [np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
        rvdistance_mat = djikstra(rvActnode_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours)
        
        for fileName in population_files:
            if meshName in fileName and target_type in fileName:
                population, particles_index = np.unique(np.loadtxt(previousResultsPath + fileName, delimiter=','),
                                                        axis=0, return_index=True)
                fileNameList.append(fileName)
                rootNodes = np.round(population[:, 4:]).astype(int)  # Non-repeated particles
                # count = count + rootNodes.shape[0]
                # Number of Root nodes percentage-error
                lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
                rv_rootNodes_part = rootNodes[:, len(lvActivationIndexes):]

                # K-means for root nodes
                # Choose the number of clusters 'k'
                lv_num_nodes = np.sum(lv_rootNodes_part, axis=1)
                lv_num_nodes, lv_num_nodes_counts = np.unique(lv_num_nodes, return_counts=True, axis=0)
                lv_num_clusters = lv_num_nodes[np.argmax(lv_num_nodes_counts)]

                rv_num_nodes = np.sum(rv_rootNodes_part, axis=1)
                rv_num_nodes, rv_num_nodes_counts = np.unique(rv_num_nodes, return_counts=True, axis=0)
                rv_num_clusters = rv_num_nodes[np.argmax(rv_num_nodes_counts)]

                # Choose the initial centroides
                k_lv_rootNodes_list, lvunique_counts = np.unique(lv_rootNodes_part, return_counts=True, axis=0)
                lv_rootNodes_list_index = np.sum(k_lv_rootNodes_list, axis=1) == lv_num_clusters
                k_lv_rootNodes_list = k_lv_rootNodes_list[lv_rootNodes_list_index, :]
                lvunique_counts = lvunique_counts[lv_rootNodes_list_index]
                lv_centroid_ids = lvActnode_ids[
                    k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)].astype(int)

                k_rv_rootNodes_list, rvunique_counts = np.unique(rv_rootNodes_part, return_counts=True, axis=0)
                rv_rootNodes_list_index = np.sum(k_rv_rootNodes_list, axis=1) == rv_num_clusters
                k_rv_rootNodes_list = k_rv_rootNodes_list[rv_rootNodes_list_index, :]
                rvunique_counts = rvunique_counts[rv_rootNodes_list_index]
                rv_centroid_ids = rvActnode_ids[
                    k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)].astype(int)


                # Transform the root nodes predicted data into k-means data
                lvdata_ids = np.concatenate(([lvActivationIndexes[(lv_rootNodes_part[i, :]).astype(bool)] for i in
                                              range(lv_rootNodes_part.shape[0])]), axis=0)
                lvdata_ids = np.asarray(
                    [np.flatnonzero(lvActivationIndexes == node_id)[0] for node_id in lvdata_ids]).astype(int)

                rvdata_ids = np.concatenate(([rvActivationIndexes[(rv_rootNodes_part[i, :]).astype(bool)] for i in
                                              range(rv_rootNodes_part.shape[0])]), axis=0)
                rvdata_ids = np.asarray(
                    [np.flatnonzero(rvActivationIndexes == node_id)[0] for node_id in rvdata_ids]).astype(int)

                # K-means algorithm
                # LV
                k_means_lv_labels, k_means_lv_centroids = k_means(lvnodesXYZ[lvActnode_ids, :], lvdata_ids,
                                                                  lv_num_clusters, lv_centroid_ids, lvdistance_mat,
                                                                  lvnodesXYZ, max_iter=10)
                # Any node in the endocardium can be a result
                new_lv_roots = np.unique(lvnodes[k_means_lv_centroids])

                # RV
                k_means_rv_labels, k_means_rv_centroids = k_means(rvnodesXYZ[rvActnode_ids, :], rvdata_ids,
                                                                  rv_num_clusters, rv_centroid_ids, rvdistance_mat,
                                                                  rvnodesXYZ, max_iter=10)
                # Any node in the endocardium can be a result
                new_rv_roots = np.unique(rvnodes[k_means_rv_centroids])

                if new_lv_roots.shape[0] != lv_num_clusters:
                    print('LV')
                    raise

                if new_rv_roots.shape[0] != rv_num_clusters:
                    print('RV')
                    print(fileName)
                    print(rv_centroid_ids)
                    print(k_means_rv_centroids)
                    print(rvnodes[k_means_rv_centroids])
                    print(lv_centroid_ids)
                    raise

                # K-means raw centroids
                new_lv_roots_list.append(new_lv_roots)
                new_rv_roots_list.append(new_rv_roots)


                # ROOT NODE ERROR
                lv_dist = np.linalg.norm(lv_rootNodes_true[:, np.newaxis, :] - nodesXYZ[new_lv_roots, :], ord=2, axis=2)
                lv_location_error.append(np.mean(np.amin(lv_dist, axis=1)))
                lv_number_error.append(abs(new_lv_roots.shape[0] - lv_rootNodes_true.shape[0]))

                rv_dist = np.linalg.norm(rv_rootNodes_true[:, np.newaxis, :] - nodesXYZ[new_rv_roots, :], ord=2, axis=2)
                rv_location_error.append(np.mean(np.amin(rv_dist, axis=1)))
                rv_number_error.append(abs(new_rv_roots.shape[0] - rv_rootNodes_true.shape[0]))

        data_list.append({
            'fileName': fileNameList,
            'meshName': meshName,
            'lv_roots': new_lv_roots_list,
            'rv_roots': new_rv_roots_list,
            'lv_roots_location_error': lv_location_error,
            'rv_roots_location_error': rv_location_error,
            'lv_roots_number_error': lv_number_error,
            'rv_roots_number_error': rv_number_error,
        })

    df_roots = pandas.DataFrame(data_list)
    for error_name in ['lv_roots_location_error', 'rv_roots_location_error', 'lv_roots_number_error', 'rv_roots_number_error']:
        hist_data = [df_roots[df_roots['meshName'] == meshNameList[i]][error_name].values[0] for i in range(len(meshNameList))]
        m = np.round(np.mean([np.mean(x) for x in hist_data]), 1)
        s = np.round(np.mean([np.std(x) for x in hist_data]), 1)

        print(str(m) + ' ' + str(u"\u00B1") + ' ' + str(s))

    df_roots.to_pickle(path=resultsPath + target_type + '_' + discretisation_resolution + '_rootNodes_error.gz', compression='infer')
    return df_roots




# Group results into pandas dataframes
# NEEDS FIXING!!!! 14/08/2020
def generateResultsDataFrames(threadsNum_val, experiments_list, do_pred_ecgs, do_target_ecgs, resultsPath, atmapPath,
                              exclude_names=['nothing'], only_complete=False):
    global experiment_output
    experiment_output = 'ecg'
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = [ 'I' , 'II' , 'V1' , 'V2' , 'V3' , 'V4' , 'V5' , 'V6', 'lead_prog' ]
    global nb_leads
    nb_leads = 8 # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad   # 8 + lead progression (or 12)
#     nb_roots_range = [4, 7] # range of number of root nodes per endocardial chamber
    frequency = 1000 # Hz
    freq_cut= 150 # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2) # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    global nlhsParam
    nlhsParam = 4
    
    population_fileName_list = []
    for experiment in experiments_list:
        meshNameList = experiment['meshes']
        outputTypeList = experiment['output']
#         experiment_metric = experiment['metric']
        resolutionList = experiment['nodeRes']

        # Smart iteration to minimise data-reads
        count = 0
        for meshName in meshNameList:
            # Paths and tags
            dataPath = 'metaData/' + meshName + '/'
            #smcResultsPath = dataPath + 'SMC_ABC/results/'
            # Load mesh
            global nodesXYZ
            nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
            global edges
            edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
            lvface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_lvface.csv',
                                           delimiter=',') - 1).astype(int)) # lv endocardium triangles
            rvface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_rvface.csv',
                                           delimiter=',') - 1).astype(int)) # rv endocardium triangles
            global tetraFibers
            tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions

            tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
            global edgeVEC
            edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors

            for resolution in resolutionList:
                lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_'+resolution+'Res.csv', delimiter=',') - 1).astype(int) # possible root nodes for the chosen mesh
                rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_'+resolution+'Res.csv', delimiter=',') - 1).astype(int)

                rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)

                rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
                rootNodesIndexes_true = np.unique(rootNodesIndexes_true)

                global tetrahedrons
                tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
                tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
                electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')

                aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
                for i in range(0, tetrahedrons.shape[0], 1):
                    aux[tetrahedrons[i, 0]].append(i)
                    aux[tetrahedrons[i, 1]].append(i)
                    aux[tetrahedrons[i, 2]].append(i)
                    aux[tetrahedrons[i, 3]].append(i)
                global elements
                elements = [np.array(n) for n in aux]
                aux = None # Clear Memory

                # Precompute PseudoECG stuff
                # Calculate the tetrahedrons volumes
                D = nodesXYZ[tetrahedrons[:, 3], :] # RECYCLED
                A = nodesXYZ[tetrahedrons[:, 0], :]-D # RECYCLED
                B = nodesXYZ[tetrahedrons[:, 1], :]-D # RECYCLED
                C = nodesXYZ[tetrahedrons[:, 2], :]-D # RECYCLED
                D = None # Clear Memory

                global tVolumes
                tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1),
                                                       (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0]) #Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
                tVolumes = tVolumes/np.sum(tVolumes)

                # Calculate the tetrahedron (temporal) voltage gradients
                Mg = np.stack((A, B, C), axis=-1)
                A = None # Clear Memory
                B = None # Clear Memory
                C = None # Clear Memory

                # Calculate the gradients
                global G_pseudo
                G_pseudo = np.zeros(Mg.shape)
                for i in range(Mg.shape[0]):
                    G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
                G_pseudo = np.moveaxis(G_pseudo, 1, 2)
                Mg = None # clear memory

                # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
                r=np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                       (tetrahedronCenters.shape[0],
                        tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1)-electrodePositions

                global d_r
                d_r= np.moveaxis(np.multiply(
                    np.moveaxis(r, [0, 1], [-1, -2]),
                    np.multiply(np.moveaxis(np.sqrt(np.sum(r**2, axis=2))**(-3), 0, 1), tVolumes)), 0, -1)



                # Set endocardial edges aside
                global isEndocardial
                isEndocardial=np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                                      np.all(np.isin(edges, rvface), axis=1))
                # Build adjacentcies
                global unfoldedEdges
                unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
                aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
                for i in range(0, len(unfoldedEdges), 1):
                    aux[unfoldedEdges[i, 0]].append(i)
                global neighbours
                # make neighbours Numba friendly
                neighbours_aux = [np.array(n) for n in aux]
                aux = None # Clear Memory
                m = 0
                for n in neighbours_aux:
                    m = max(m, len(n))
                neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
                for i in range(len(neighbours_aux)):
                    n = neighbours_aux[i]
                    neighbours[i, :n.shape[0]] = n
                neighbours_aux = None

                # neighbours_original
                aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
                for i in range(0, len(unfoldedEdges), 1):
                    aux[unfoldedEdges[i, 0]].append(i)
                global neighbours_original
                neighbours_original = [np.array(n) for n in aux]
                aux = None # Clear Memory


                # Load and compile Numba
                eikonal_ecg(np.array([[0.1, 0.1, 0.1, 0.1]]), rootNodesIndexes_true, rootNodesTimes_true)
                
                for outputType in outputTypeList:
                    target_fileName = resultsPath+meshName+'_coarse_'+resolution+'_'+outputType+'_target'+'.gz'
                    pred_fileName = resultsPath+meshName+'_coarse_'+resolution+'_'+outputType+'_pred'+'.gz'


                    # Check if it has been done already
                    do_this = True
                    if only_complete:
                        do_this = (not os.path.isfile(target_fileName))
                    if do_target_ecgs and do_this:
                        # List to store the target egs if there is need to compute them
                        target_data = []
                        # Target dataframe
                        print('Computing targets ...')
#                         for aux_endoSpeedTag in ['120', '150', '179']:
                        for aux_endoSpeedTag in ['120', '179']:
                            if aux_endoSpeedTag == '120':
                                endo = 0.12
                            elif aux_endoSpeedTag == '179':
                                endo = 0.179
                            elif aux_endoSpeedTag == '150':
                                endo = 0.15
                            else:
                                raise
                            for aux_epiSpeedTag in ['1x', '2x']:
                                aux_fileName = (atmapPath + meshName + '_coarse_true_ATMap_' + aux_endoSpeedTag + '_'
                                                + aux_epiSpeedTag +  '.csv')
                                if os.path.isfile(aux_fileName):
#                                     if aux_epiSpeedTag=='1x' and aux_endoSpeedTag=='120':
#                                         atm1 = np.loadtxt(aux_fileName, delimiter=',')
#                                     if aux_epiSpeedTag=='2x' and aux_endoSpeedTag=='120':
#                                         atm2 = np.loadtxt(aux_fileName, delimiter=',')

                                    true_ecg = pseudoBSP(np.loadtxt(aux_fileName, delimiter=','))
                                    target_data.append({'ecg': pseudoBSP(np.loadtxt(aux_fileName, delimiter=',')),
                                                        'model': 'bidomain', 'origin': 'true', 'source': 'atm',
                                                        'endoTag': aux_endoSpeedTag, 'epiTag': aux_epiSpeedTag,
                                                        'endo': endo, 'trans': -1, 'run': -1})

                                    aux_fileName = (atmapPath + meshName + '_coarse_true_ecg_' + aux_endoSpeedTag
                                                    + '_' + aux_epiSpeedTag +  '.csv')

                                    myoSpeed_true = np.median(np.loadtxt('metaData/MyocardialSpeeds/'+ meshName +
                                                                         '_coarse_true_MyocardialSpeed_' + aux_endoSpeedTag
                                                                         + '_' + aux_epiSpeedTag +  '.csv', delimiter=','), axis=0)
                                    
                                    for endo_mult in [0.5, 1., 2]:
                                        for trans_mult in [0.5, 1., 2]:
                                            target_data.append({'ecg': eikonal_ecg(
                                                np.array([np.array([myoSpeed_true[0], myoSpeed_true[1]*trans_mult, myoSpeed_true[2],
                                                                    endo*endo_mult])]), rootNodesIndexes_true, rootNodesTimes_true)[0],
                                                                'model': 'eikonal', 'origin': 'true', 'source': 'ecg',
                                                                'endoTag': aux_endoSpeedTag, 'epiTag': aux_epiSpeedTag,
                                                                'endo': endo_mult, 'trans': trans_mult, 'run': -1})
            
                                    for i in range(5):
                                        # Store predictions with lowest discrepancies for atm as target
                                        if outputType == 'atm' or outputType == 'all':
                                            aux_fileName = (meshName + '_coarse_' + aux_endoSpeedTag + '_' + aux_epiSpeedTag
                                                        + '_' + resolution +'_atm_euclidean_'+ str(i)+ '_population.csv')
                                            if os.path.isfile(previousResultsPath + aux_fileName):
                                                df_roots = pandas.read_pickle(
                                                    filepath_or_buffer=resultsPath + meshName
                                                    + '_coarse_'+resolution+'_atm_rootNodes.gz', compression='infer')
                                                params = df_roots[df_roots['fileName']==aux_fileName]['params'].to_numpy()[0]
                                                target_data.append({'ecg': eikonal_ecg(np.array([params]), rootNodeActivationIndexes, rootNodesTimes)[0],
                                                                    'model': 'eikonal', 'origin': 'pred', 'source': 'atm',
                                                                    'endoTag': aux_endoSpeedTag, 'epiTag': aux_epiSpeedTag,
                                                                    'endo': params[3], 'trans': params[1], 'run': i})
                                            elif i==0:
                                                print('atm')
                                                raise
                                            
                                            
                                            # Store predictions using the ensamble infered parameters from the atm results
                                            aux_fileName = (meshName + '_coarse_' + aux_endoSpeedTag + '_' + aux_epiSpeedTag + '_'
                                                            + resolution +'_atm_euclidean_'+ str(i)+ '_population.csv')
                                            if os.path.isfile(previousResultsPath + aux_fileName):
                                                population = np.unique(np.loadtxt(previousResultsPath
                                                                                  + aux_fileName, delimiter=','), axis=0)
                                                df_roots = pandas.read_pickle(filepath_or_buffer=resultsPath
                                                                              + meshName + '_coarse_'+resolution+'_atm_rootNodes.gz',
                                                                              compression='infer')
                                                rootNodes = np.concatenate(
                                                    (df_roots[df_roots['fileName']==aux_fileName]['lv_roots'].to_numpy()[0],
                                                     df_roots[df_roots['fileName']==aux_fileName]['rv_roots'].to_numpy()[0]), axis=0)
                                                pred_speeds = np.median(population[:, :4], axis=0)
                                                target_data.append({'ecg': eikonal_ecg(np.array([pred_speeds]), rootNodes, rootNodesTimes)[0],
                                                                    'model': 'eikonal', 'origin': 'median-pred', 'source': 'atm',
                                                                    'endoTag': aux_endoSpeedTag, 'epiTag': aux_epiSpeedTag,
                                                                    'endo': pred_speeds[3], 'trans': pred_speeds[1], 'run': i})
                                            elif i==0:
                                                print('atm')
                                                raise
                                                
                                        # Store predictions with lowest discrepancies for ecg as target
                                        if outputType == 'ecg' or outputType == 'all':
                                            aux_fileName = (meshName + '_coarse_' + aux_endoSpeedTag + '_' + aux_epiSpeedTag
                                                            + '_' + resolution +'_ecg_dtw_'+ str(i)+ '_population.csv')
                                            if os.path.isfile(previousResultsPath + aux_fileName):
                                                df_roots = pandas.read_pickle(filepath_or_buffer=resultsPath
                                                                              + meshName
                                                                              + '_coarse_'+resolution+'_ecg_dtw_rootNodes.gz', compression='infer')
                                                params = df_roots[df_roots['fileName']==aux_fileName]['params'].to_numpy()[0]
                                                target_data.append({'ecg': eikonal_ecg(np.array([params]), rootNodeActivationIndexes, rootNodesTimes)[0],
                                                                    'model': 'eikonal', 'origin': 'pred', 'source': 'ecg',
                                                                    'endoTag': aux_endoSpeedTag, 'epiTag': aux_epiSpeedTag,
                                                                    'endo': params[3], 'trans': params[1], 'run': i})
                                            elif i==0:
                                                print(previousResultsPath + aux_fileName)
                                                print('ecg')
                                                raise

                                        

                                            # Store predictions using the ensamble infered parameters from the ecg results
                                            aux_fileName = (meshName + '_coarse_' + aux_endoSpeedTag + '_' + aux_epiSpeedTag + '_'
                                                            + resolution +'_ecg_dtw_'+ str(i)+ '_population.csv')
                                            if os.path.isfile(previousResultsPath + aux_fileName):
                                                population = np.unique(np.loadtxt(previousResultsPath
                                                                                  + aux_fileName, delimiter=','), axis=0)
                                                df_roots = pandas.read_pickle(filepath_or_buffer=resultsPath
                                                                              + meshName + '_coarse_'+resolution+'_ecg_dtw_rootNodes.gz',
                                                                              compression='infer')
                                                rootNodes = np.concatenate(
                                                    (df_roots[df_roots['fileName']==aux_fileName]['lv_roots'].to_numpy()[0],
                                                     df_roots[df_roots['fileName']==aux_fileName]['rv_roots'].to_numpy()[0]), axis=0)
                                                pred_speeds = np.median(population[:, :4], axis=0)
                                                target_data.append({'ecg': eikonal_ecg(np.array([pred_speeds]), rootNodes, rootNodesTimes)[0],
                                                                    'model': 'eikonal', 'origin': 'median-pred', 'source': 'ecg',
                                                                    'endoTag': aux_endoSpeedTag, 'epiTag': aux_epiSpeedTag,
                                                                    'endo': pred_speeds[3], 'trans': pred_speeds[1], 'run': i})

                        df_target = pandas.DataFrame(target_data)
                        df_target.to_pickle(path=target_fileName, compression='infer')
                        print('... done computing targets!')
                    else:
                        print('Not doing target-ecgs')


                    # Check if it has been done already
                    do_this = True
                    if only_complete:
                        do_this = (not os.path.isfile(pred_fileName))
                    if do_pred_ecgs and do_this:
                        criteria_names = ['population', meshName, resolution, outputType, 'dtw']
                        population_files = []
                        for fileName in os.listdir(previousResultsPath):
                            if (not np.any([exclude in fileName for exclude in exclude_names])
                                and np.all([criteria in fileName for criteria in criteria_names])):
                                population_files.append(fileName)
                                
                        # List to store the predicted egs if there is need to compute them
                        pred_data = []
                        # Predicted dataframe
                        print('Computing predictions ...')
                        for fileName in population_files:
                            if meshName in fileName and resolution in fileName and outputType in fileName:
                                t_start = time.time()
                                population = np.loadtxt(previousResultsPath + fileName, delimiter=',')
                                # Re-Compute predictions
                                prediction_list = eikonal_ecg(population[:, :], rootNodeActivationIndexes, rootNodesTimes)
                                pred_data.append({
                                    'fileName': fileName,
                                    'prediction_list': prediction_list
                                })
                                count = count + 1
                                print('Progress ' + str(count) + '/' + str(len(population_files))
                                      + '\nTime spent: '+str(time.time()-t_start))
                        df_pred = pandas.DataFrame(pred_data)
                        df_pred.to_pickle(path=pred_fileName, compression='infer')
                        print('... done computing predictions!')
                    else:
                        print('Not doing pred-ecgs')
    print('Done')


# Plot grouped results from the pandas dataframes
# NEEDS FIXING!!!! 14/08/2020
def makePDFfigures(conf_list, pdf_path, resultsPath, leadNames, threadsNum, line_names_all, configurations_all,
                   meshName_list_all, redoAll, resolution):
    numConf = len(conf_list)
    count = pymp.shared.array((numConf), dtype=np.int32)
    with pymp.Parallel(min(threadsNum, numConf)) as p3:
        for conf_i in p3.range(numConf):
            t_start = time.time()

            do_lines = conf_list[conf_i]['do_lines']
            lines_per_conf = int(np.sum(do_lines))
            line_names = [line_names_all[i] for i in range(len(do_lines)) if do_lines[i]]
            do_eikonal_line = do_lines[0]
            do_atm_pred_line = do_lines[1]
            do_ecg_pred_line = do_lines[2]
            do_atm_inf_line = do_lines[3]
            do_ecg_inf_line = do_lines[4]

            configurations = conf_list[conf_i]['configurations']
            configurations = [configurations_all[i] for i in range(len(configurations)) if configurations[i]]

            meshName_list = conf_list[conf_i]['meshName_list']
            meshName_list = [meshName_list_all[i] for i in range(len(meshName_list)) if meshName_list[i]]

            figName = pdf_path + '_'.join(meshName_list)+'_'+'_'.join(line_names)+'_'+resolution+'.pdf'

            if os.path.isfile(figName) and not redoAll:
                p3.print('Skiping: '+figName)
            else:
                max_count = 0
                if 'Eikonal-True' in line_names:
                    plot_width = 6
                else:
                    plot_width = 4
                # Check overall ECG value tendencies as a function of the endocardial and myocardial speeds
                fig, axs = plt.subplots(len(configurations)*lines_per_conf*len(meshName_list), len(leadNames), sharey='all', sharex=False, figsize = (40,plot_width*len(configurations)*lines_per_conf*len(meshName_list)), constrained_layout=True)
                fig.suptitle(' '.join(meshName_list) + ': Bidomain vs '+' '.join(line_names), fontsize=24)
                for meshName_i in range(len(meshName_list)):
                    meshName = meshName_list[meshName_i]
                    df_target = pandas.read_pickle(filepath_or_buffer=resultsPath+meshName+'_coarse_'+resolution+'_ecg_target'+'.gz', compression='infer')
                    for i in range(meshName_i*len(configurations), (meshName_i+1)*len(configurations)):
                        endoTag = configurations[i%len(configurations)][0]
                        epiTag = configurations[i%len(configurations)][1]

                        x1 = df_target['endoTag']==endoTag
                        x2 = df_target['epiTag']==epiTag
                        x1 = np.logical_and(x1, x2)
                        auxdf = df_target[x1]

                        x1 = auxdf['model']=='bidomain'
                        x2 = auxdf['origin']=='true'
                        x1 = np.logical_and(x1, x2)
                        x2 = auxdf['source']=='atm'
                        x1 = np.logical_and(x1, x2)
                        bidoEcg = auxdf[x1]['ecg'].to_numpy()[0]

                        # Eikonal True
                        if do_eikonal_line:
                            eikoEcg = []
                            for endo_mult in [1., 0.5, 2]:
                                for trans_mult in [1., 0.5, 2]:
                                    x1 = auxdf['model']=='eikonal'
                                    x2 = auxdf['origin']=='true'
                                    x1 = np.logical_and(x1, x2)
                                    x2 = auxdf['source']=='ecg'
                                    x1 = np.logical_and(x1, x2)
                                    x2 = auxdf['endo']==endo_mult
                                    x1 = np.logical_and(x1, x2)
                                    x2 = auxdf['trans']==trans_mult
                                    x1 = np.logical_and(x1, x2)
                                    eikoEcg.append(auxdf[x1]['ecg'].to_numpy()[0])

                        # ATM pred
                        if do_atm_pred_line:
                            x1 = auxdf['model']=='eikonal'
                            x2 = auxdf['origin']=='pred'
                            x1 = np.logical_and(x1, x2)
                            x2 = auxdf['source']=='atm'
                            x1 = np.logical_and(x1, x2)
                            eikoEcg_atm_pred = auxdf[x1]['ecg'].to_numpy()
                #             print(len(eikoEcg_atm_pred))

                        # ECG pred
                        if do_ecg_pred_line:
                            x1 = auxdf['model']=='eikonal'
                            x2 = auxdf['origin']=='pred'
                            x1 = np.logical_and(x1, x2)
                            x2 = auxdf['source']=='ecg'
                            x1 = np.logical_and(x1, x2)
                            eikoEcg_ecg_pred = auxdf[x1]['ecg'].to_numpy()
                #             print(len(eikoEcg_ecg_pred))

                        # ATM inf
                        if do_atm_inf_line:
                            x1 = auxdf['model']=='eikonal'
                            x2 = auxdf['origin']=='median-pred'
                            x1 = np.logical_and(x1, x2)
                            x2 = auxdf['source']=='atm'
                            x1 = np.logical_and(x1, x2)
                            eikoEcg_atm_inf = auxdf[x1]['ecg'].to_numpy()
                #             print(len(eikoEcg_atm_inf))

                        # ECG inf
                        if do_ecg_inf_line:
                            x1 = auxdf['model']=='eikonal'
                            x2 = auxdf['origin']=='median-pred'
                            x1 = np.logical_and(x1, x2)
                            x2 = auxdf['source']=='ecg'
                            x1 = np.logical_and(x1, x2)
                            eikoEcg_ecg_inf = auxdf[x1]['ecg'].to_numpy()
                #             print(len(eikoEcg_ecg_inf))

                        colors = ['r-', 'c-', 'b-', 'g-', 'y-']
                        for j in range(len(leadNames)):
                #             print('j: '+ str(j))
                            max_count = max(max_count, len(bidoEcg[j, :]))

                            offset_i = -1

                            vline_min = -3.5
                            vline_max = -1
                            vline_offset = 0.5
                            # Eikonal True
                            if do_eikonal_line:
                                offset_i = offset_i + 1
                                new_i = i*lines_per_conf+offset_i

                                ecg = bidoEcg[j, :]
                                axs[new_i,j].plot(ecg, 'k-', linewidth=5, label='Bidomain')
                                axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max, colors='k', linestyles='solid', linewidth=3)

                                iterator = 0
                                color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                                for endo_mult in [1., 0.5, 2]:
                                    for trans_mult in [1., 0.5, 2]:
                                        ecg = eikoEcg[iterator][j, :]
                                        if endo_mult == 1 and trans_mult == 1:
                                            axs[new_i,j].plot(ecg, color=color_list[iterator], linestyle ='-', linewidth=3.5, label=line_names[offset_i])
                                            axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max-vline_offset*iterator, colors=color_list[iterator], linestyles='solid', linewidth=3)
                                            max_count = max(max_count, np.sum(np.logical_not(np.isnan(ecg))))
                                        else:
                                            if endo_mult != trans_mult and endo_mult==1:
                                                axs[new_i,j].plot(ecg, color=color_list[iterator], linestyle ='-', linewidth=2., label='endo: ' + str(endo_mult)+', trans: ' + str(trans_mult))
                                                axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max-vline_offset*iterator, colors=color_list[iterator], linestyles='solid', linewidth=2.)
                                                max_count = max(max_count, np.sum(np.logical_not(np.isnan(ecg))))
                                        iterator = iterator + 1

                            # atm_pred
                            if do_atm_pred_line:
                                offset_i = offset_i + 1
                                new_i = i*lines_per_conf+offset_i
                                ecg = bidoEcg[j, :]
                                axs[new_i,j].plot(ecg, 'k-', linewidth=5, label='Bidomain')
                                axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max, colors='k', linestyles='solid', linewidth=3)
                                for k in range(len(eikoEcg_atm_pred)):
                                    ecg = eikoEcg_atm_pred[k][j, :]
                                    axs[new_i,j].plot(ecg, colors[k], linewidth=3.5-0.2*k, label=line_names[offset_i]+'-'+str(k))
                                    axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max-vline_offset*k, colors=colors[k][0], linestyles='solid', linewidth=4.)
                                    max_count = max(max_count, np.sum(np.logical_not(np.isnan(ecg))))

                            # ecg_pred
                            if do_ecg_pred_line:
                                offset_i = offset_i + 1
                                new_i = i*lines_per_conf+offset_i
                                ecg = bidoEcg[j, :]
                                axs[new_i,j].plot(ecg, 'k-', linewidth=5, label='Bidomain')
                                axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max, colors='k', linestyles='solid', linewidth=3)
                                for k in range(len(eikoEcg_ecg_pred)):
                                    ecg = eikoEcg_ecg_pred[k][j, :]
                                    axs[new_i,j].plot(ecg, colors[k], linewidth=3.5-0.2*k, label=line_names[offset_i]+'-'+str(k))
                                    axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max-vline_offset*k, colors=colors[k][0], linestyles='solid', linewidth=4.)
                                    max_count = max(max_count, np.sum(np.logical_not(np.isnan(ecg))))

                            # atm_inf
                            if do_atm_inf_line:
                                offset_i = offset_i + 1
                                new_i = i*lines_per_conf+offset_i
                                ecg = bidoEcg[j, :]
                                axs[new_i,j].plot(ecg, 'k-', linewidth=5, label='Bidomain')
                                axs[new_i,j].vlines(len(ecg), vline_min, vline_max, colors='k', linestyles='solid', linewidth=3)
                                for k in range(len(eikoEcg_atm_inf)):
                                    ecg = eikoEcg_atm_inf[k][j, :]
                                    axs[new_i,j].plot(ecg, colors[k], linewidth=3.5-0.2*k, label=line_names[offset_i]+'-'+str(k))
                                    axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max-vline_offset*k, colors=colors[k][0], linestyles='solid', linewidth=4.)
                                    max_count = max(max_count, np.sum(np.logical_not(np.isnan(ecg))))

                            # ecg_inf
                            if do_ecg_inf_line:
                                offset_i = offset_i + 1
                                new_i = i*lines_per_conf+offset_i
                                ecg = bidoEcg[j, :]
                                axs[new_i,j].plot(ecg, 'k-', linewidth=5, label='Bidomain')
                                axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max, colors='k', linestyles='solid', linewidth=3)
                                for k in range(len(eikoEcg_ecg_inf)):
                                    ecg = eikoEcg_ecg_inf[k][j, :]
                                    axs[new_i,j].plot(ecg, colors[k], linewidth=3.5-0.2*k, label=line_names[offset_i]+'-'+str(k))
                                    axs[new_i,j].vlines(np.sum(np.logical_not(np.isnan(ecg))), vline_min, vline_max-vline_offset*k, colors=colors[k][0], linestyles='solid', linewidth=4.)
                                    max_count = max(max_count, np.sum(np.logical_not(np.isnan(ecg))))

                        offset_i = -1

                        # eikonal true
                        if do_eikonal_line:
                            offset_i = offset_i + 1
                            axs[i*lines_per_conf+offset_i,0].set_ylabel(str(configurations[i%len(configurations)]), fontsize=20)
                            axs[i*lines_per_conf,j].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

                        # atm_pred
                        if do_atm_pred_line:
                            offset_i = offset_i + 1
                            axs[i*lines_per_conf+offset_i,0].set_ylabel(str(configurations[i%len(configurations)]), fontsize=20)
                            axs[i*lines_per_conf+offset_i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

                        # ecg_pred
                        if do_ecg_pred_line:
                            offset_i = offset_i + 1
                            axs[i*lines_per_conf+offset_i,0].set_ylabel(str(configurations[i%len(configurations)]), fontsize=20)
                            axs[i*lines_per_conf+offset_i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

                        # atm_inf
                        if do_atm_inf_line:
                            offset_i = offset_i + 1
                            axs[i*lines_per_conf+offset_i,0].set_ylabel(str(configurations[i%len(configurations)]), fontsize=20)
                            axs[i*lines_per_conf+offset_i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

                        # ecg_inf
                        if do_ecg_inf_line:
                            offset_i = offset_i + 1
                #             axs[i*lines_per_conf+offset_i,0].set_ylabel(meshName+'\n'+line_names[offset_i]+'\n'+str(configurations[i%len(configurations)]), fontsize=20)
                            axs[i*lines_per_conf+offset_i,0].set_ylabel(str(configurations[i%len(configurations)]), fontsize=20)
                            axs[i*lines_per_conf+offset_i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

                for meshName_i in range(len(meshName_list)):
                    for j in range(len(leadNames)):
                        axs[meshName_i*len(configurations), j].set_title(meshName, fontsize=20)
                    for i in range(meshName_i*len(configurations), (meshName_i+1)*len(configurations)):
                            for k in range(lines_per_conf):
                                for j in range(len(leadNames)):
                                    new_i = i*lines_per_conf+k
                                    axs[new_i,j].set_xlabel('Lead ' + leadNames[j], fontsize=20)
                                    axs[new_i,j].xaxis.grid(True, which='major')
                                    axs[new_i,j].yaxis.grid(True, which='major')
                                    axs[new_i,j].xaxis.set_major_locator(MultipleLocator(10))
#                                     axs[new_i,j].xaxis.set_major_locator(MultipleLocator(20))
                                    axs[new_i,j].yaxis.set_major_locator(MultipleLocator(1))
                                    axs[new_i,j].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                #                     axs[new_i,j].xaxis.set_minor_locator(MultipleLocator(5))
                                    axs[new_i,j].set_xlim(0, max_count+5)
                #                     axs[new_i,j].tick_params(which='both', width=2)
                                    axs[new_i,j].tick_params(which='major', width=2, length=7, color='k')
                #                     axs[new_i,j].tick_params(which='minor', width=1.5, length=4, color='k')
                                    for tick in axs[new_i,j].xaxis.get_major_ticks():
                                        tick.label.set_fontsize(16)
                                    for tick in axs[new_i,j].yaxis.get_major_ticks():
                                        tick.label.set_fontsize(16)

                p3.print('Saving figure: '+figName)
                plt.savefig(figName, dpi=150)
                p3.print('Saved figure: '+figName)
            count[conf_i] = 1
            p3.print('Progress ' + str(np.sum(count)) + '/' + str(numConf) + '\nTime spent: '+str(time.time()-t_start))
    print('Done!')


# Generate the paraview files that serve to create the result figures for the root nodes inference
def makeRootFigures(resultsPath, meshNames_list, target_type_list, discretisation_list,
                    save_roots, figuresPath, is_ECGi, meshName_pred, load_target,
                    conduction_speeds_pred, healthy_val, threadsNum_val, endocardial_layer):
    global is_healthy
    is_healthy = healthy_val
    global experiment_output
    experiment_output = 'atm'
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_bsp
    global nb_leads
    nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad   # 8 + lead progression (or 12)
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    global nlhsParam
    nlhsParam = 4
    
    population_files = []
    for discretisation_resolution in discretisation_list:
        for target_type in target_type_list:
            if 'atm' in target_type:
                data_type = 'atm'
            elif 'ecg' in target_type:
                data_type = 'ecg'
            elif 'bsp' in target_type:
                data_type = 'bsp'
            previousResultsPath = resultsPath+data_type+"Noise_new_code_"+discretisation_resolution+"/"
            for fileName in os.listdir(previousResultsPath):
                if ('population' in fileName and np.any([meshName in fileName for meshName in meshNames_list])
                        and np.any([target_type in fileName for target_type in target_type_list])
                        and np.any([discretisation_resolution in fileName for discretisation_resolution in discretisation_list]) ):
                    population_files.append(fileName)
            
    for meshName in meshNames_list:
        global nodesXYZ
        nodesXYZ = np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_xyz.csv', delimiter=',')
        elems = (np.loadtxt('metaData/' + meshName+'/'+ meshName + '_coarse_tri.csv', delimiter=',')-1).astype(int)
        global edges
        edges = (np.loadtxt('metaData/'  + meshName + '/' + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
        lvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_lvnodes.csv', delimiter=',') - 1).astype(int)               # lv endocardium triangles
        lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
        rvtri =(np.loadtxt('metaData/' + meshName + '/' + meshName + '_rvnodes.csv', delimiter=',') - 1).astype(int)                # rv endocardium triangles
        rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
        if not is_ECGi:
            # True
            rootNodesIndexes_true = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
            rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
            lv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in lvnodes])
            rv_rootNodesIndexes_true = np.array([node_i for node_i in rootNodesIndexes_true if node_i in rvnodes])
        #lv_rootNodes_true = nodesXYZ[lv_rootNodesIndexes_true, :]
        #rv_rootNodes_true = nodesXYZ[rv_rootNodesIndexes_true, :]
        
        print ('Export mesh to ensight format')
        aux_elems = elems+1    # Make indexing Paraview and Matlab friendly
        with open(figuresPath+meshName+'.ensi.geo', 'w') as f:
            f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
            for i in range(0, nodesXYZ.shape[0]):
                f.write(str(i+1)+'\n')
            for c in [0,1,2]:
                for i in range(0, nodesXYZ.shape[0]):
                    f.write(str(nodesXYZ[i,c])+'\n')
            print('Write tetra4...')
            f.write('tetra4\n  '+str(len(aux_elems))+'\n')
            for i in range(0, len(aux_elems)):
                f.write('  '+str(i+1)+'\n')
            for i in range(0, len(aux_elems)):
                f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\t'+str(aux_elems[i,3])+'\n')

        for discretisation_resolution in discretisation_list:
            # Predictions
            lvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_lv_activationIndexes_'+discretisation_resolution+'Res.csv', delimiter=',') - 1).astype(int)
            rvActivationIndexes = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_coarse_rv_activationIndexes_'+discretisation_resolution+'Res.csv', delimiter=',') - 1).astype(int)

            activationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
            #print(meshName)
            # for i in range(lvActivationIndexes.shape[0]):
            #     if lvActivationIndexes[i] not in lvnodes:
            #         a = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
            #         #print('diff ' + str(np.round(np.linalg.norm(nodesXYZ[a, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=0)*10, 2)) + ' mm')
            #         lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
            # for i in range(rvActivationIndexes.shape[0]):
            #     if rvActivationIndexes[i] not in rvnodes:
            #         b = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
            #         #print('diff ' + str(np.round(np.linalg.norm(nodesXYZ[b, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=0)*10, 2)) + ' mm')
            #         rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    
            for target_type in target_type_list:
                if 'atm' in target_type:
                    data_type = 'atm'
                elif 'ecg' in target_type:
                    data_type = 'ecg'
                elif 'bsp' in target_type:
                    data_type = 'bsp'
                previousResultsPath = resultsPath+data_type+"Noise_new_code_"+discretisation_resolution+"/"
            
                lv_rootNodes = None
                rv_rootNodes = None
                lv_rootNodes_list = None
                rv_rootNodes_list = None
                rootNodes_list = None
                # new_lv_roots_ids_list = []
                # new_lv_roots_weights_list = []
                # new_rv_roots_ids_list = []
                # new_rv_roots_weights_list = []
                new_lv_roots_list = []
                new_rv_roots_list = []
                data_list = []
                disc_roots_list = []
                count = 0.
                
                # Calculate Djikstra distances in the LV endocardium
                #lvdistance_mat = np.zeros((lvnodes.shape[0], lvActivationIndexes.shape[0]))
                lvnodesXYZ = nodesXYZ[lvnodes, :]
                # Set endocardial edges aside
                lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
                lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
                lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
                lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :] # edge vectors
                lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
                aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
                for i in range(0, len(lvunfoldedEdges), 1):
                    aux[lvunfoldedEdges[i, 0]].append(i)
                lvneighbours = [np.array(n) for n in aux]
                aux = None # Clear Memory
                lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
                lvdistance_mat = djikstra(lvActnode_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours)
    
                # Calculate Djikstra distances in the RV endocardium
                #rvdistance_mat = np.zeros((rvnodes.shape[0], rvActivationIndexes.shape[0]))
                rvnodesXYZ = nodesXYZ[rvnodes, :]
                # Set endocardial edges aside
                rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
                rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
                rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
                rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :] # edge vectors
                rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
                aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
                for i in range(0, len(rvunfoldedEdges), 1):
                    aux[rvunfoldedEdges[i, 0]].append(i)
                rvneighbours = [np.array(n) for n in aux]
                aux = None # Clear Memory
                rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
                rvdistance_mat = djikstra(rvActnode_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours)
    
                for fileName in population_files:
                    if meshName in fileName and target_type in fileName and discretisation_resolution in fileName:
                        population, particles_index = np.unique(np.loadtxt(previousResultsPath + fileName, delimiter=','), axis=0, return_index=True)
                        rootNodes = np.round(population[:, 4:]).astype(int) # Non-repeated particles
                        count = count + rootNodes.shape[0]
                        # Number of Root nodes percentage-error
                        lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
                        rv_rootNodes_part = rootNodes[:, len(lvActivationIndexes):]
    
                        # K-means for root nodes
                        # Choose the number of clusters 'k'
                        lv_num_nodes = np.sum(lv_rootNodes_part, axis=1)
                        lv_num_nodes, lv_num_nodes_counts = np.unique(lv_num_nodes, return_counts=True, axis=0)
                        lv_num_clusters = lv_num_nodes[np.argmax(lv_num_nodes_counts)]
    
                        rv_num_nodes = np.sum(rv_rootNodes_part, axis=1)
                        rv_num_nodes, rv_num_nodes_counts = np.unique(rv_num_nodes, return_counts=True, axis=0)
                        rv_num_clusters = rv_num_nodes[np.argmax(rv_num_nodes_counts)]
    
                        # Choose the initial centroides
                        k_lv_rootNodes_list, lvunique_counts = np.unique(lv_rootNodes_part, return_counts=True, axis=0)
                        lv_rootNodes_list_index = np.sum(k_lv_rootNodes_list, axis=1) == lv_num_clusters
                        k_lv_rootNodes_list = k_lv_rootNodes_list[lv_rootNodes_list_index, :]
                        lvunique_counts = lvunique_counts[lv_rootNodes_list_index]
                        lv_centroid_ids = lvActnode_ids[k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)].astype(int)
            #             print('lv: ' +str(lv_centroid_ids))
    
                        k_rv_rootNodes_list, rvunique_counts = np.unique(rv_rootNodes_part, return_counts=True, axis=0)
                        rv_rootNodes_list_index = np.sum(k_rv_rootNodes_list, axis=1) == rv_num_clusters
                        k_rv_rootNodes_list = k_rv_rootNodes_list[rv_rootNodes_list_index, :]
                        rvunique_counts = rvunique_counts[rv_rootNodes_list_index]
                        rv_centroid_ids = rvActnode_ids[k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)].astype(int)
            #             print('rv: ' +str(rv_centroid_ids))
            #             raise
    
                        # Check that everything is OK
            #             print(np.all(lvnodesXYZ[lv_centroid_ids, :] == nodesXYZ[lvActivationIndexes[k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)], :]))
            #             print(np.all(rvnodesXYZ[rv_centroid_ids, :] == nodesXYZ[rvActivationIndexes[k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)], :]))
    
                        # Transform the root nodes predicted data into k-means data
                        lvdata_ids = np.concatenate(([lvActivationIndexes[(lv_rootNodes_part[i, :]).astype(bool)] for i in range(lv_rootNodes_part.shape[0])]), axis=0)
                        lvdata_ids = np.asarray([np.flatnonzero(lvActivationIndexes == node_id)[0] for node_id in lvdata_ids]).astype(int)
    
                        rvdata_ids = np.concatenate(([rvActivationIndexes[(rv_rootNodes_part[i, :]).astype(bool)] for i in range(rv_rootNodes_part.shape[0])]), axis=0)
                        rvdata_ids = np.asarray([np.flatnonzero(rvActivationIndexes == node_id)[0] for node_id in rvdata_ids]).astype(int)
    
                        # K-means algorithm
                        # LV
                        k_means_lv_labels, k_means_lv_centroids = k_means(lvnodesXYZ[lvActnode_ids, :], lvdata_ids, lv_num_clusters, lv_centroid_ids, lvdistance_mat, lvnodesXYZ, max_iter=10)
    
                        # Any node in the endocardium can be a result
                        new_lv_roots = np.unique(lvnodes[k_means_lv_centroids])
    
                        # # We project the centroids to potential root nodes using distance as a weight for a better visualisation
                        # nb_neighbours = 4
                        # new_lv_roots_ids = np.zeros((k_means_lv_centroids.shape[0], nb_neighbours)).astype(int)
                        # new_lv_roots_weights = np.zeros((k_means_lv_centroids.shape[0], nb_neighbours)).astype(float)
                        # distAux = np.linalg.norm(lvnodesXYZ[lvActnode_ids, np.newaxis, :] - lvnodesXYZ[k_means_lv_centroids, :], ord=2, axis=2)
                        # # Take the nb_neighbours closest potential root nodes and weight them based on distance to the k-means returned position
                        # for i in range(distAux.shape[1]):
                        #     lvind = np.argsort(distAux[:, i], axis=0)[:nb_neighbours].astype(int)
                        #     new_lv_roots_ids[i, :] = lvActivationIndexes[lvind]
                        #     new_lv_roots_weights[i, :] = np.maximum(distAux[lvind, i]**2, 10e-17)**(-1)
                        #     new_lv_roots_weights[i, :] = new_lv_roots_weights[i, :] / np.sum(new_lv_roots_weights[i, :])
                        #     new_lv_roots_weights[i, new_lv_roots_weights[i, :] < 10e-2] = 0.
                        #     new_lv_roots_weights[i, :] = new_lv_roots_weights[i, :] / np.sum(new_lv_roots_weights[i, :])
    
                        # RV
                        k_means_rv_labels, k_means_rv_centroids = k_means(rvnodesXYZ[rvActnode_ids, :], rvdata_ids, rv_num_clusters, rv_centroid_ids, rvdistance_mat, rvnodesXYZ, max_iter=10)
    
                        # We make that only preselected potential root nodes can be a result to make for a better visualisation
            #             rvind = np.asarray([np.argmin(np.linalg.norm(rvnodesXYZ[rvActnode_ids, :] - rvnodesXYZ[k_means_rv_centroids[i], :], ord=2, axis=1)).astype(int) for i in range(k_means_rv_centroids.shape[0])])
            #             new_rv_roots = rvActivationIndexes[rvind]
    
                        # Any node in the endocardium can be a result
                        new_rv_roots = np.unique(rvnodes[k_means_rv_centroids])
    
                        if new_lv_roots.shape[0] != lv_num_clusters:
                            print('LV')
                            print(fileName)
                            print(lv_centroid_ids)
                            print(k_means_lv_centroids)
                            print(lvnodes[k_means_lv_centroids])
                            print(lv_centroid_ids)
                            raise
    
                        if new_rv_roots.shape[0] != rv_num_clusters:
                            print('RV')
                            print(fileName)
                            print(rv_centroid_ids)
                            print(k_means_rv_centroids)
                            print(rvnodes[k_means_rv_centroids])
                            print(rv_centroid_ids)
                            raise
    
                        # # We project the centroids to potential root nodes using distance as a weight for a better visualisation
                        # new_rv_roots_ids = np.zeros((k_means_rv_centroids.shape[0], nb_neighbours)).astype(int)
                        # new_rv_roots_weights = np.zeros((k_means_rv_centroids.shape[0], nb_neighbours)).astype(float)
                        # distAux = np.linalg.norm(rvnodesXYZ[rvActnode_ids, np.newaxis, :] - rvnodesXYZ[k_means_rv_centroids, :], ord=2, axis=2)
                        # # Take the nb_neighbours closest potential root nodes and weight them based on distance to the k-means returned position
                        # for i in range(distAux.shape[1]):
                        #     rvind = np.argsort(distAux[:, i], axis=0)[:nb_neighbours].astype(int)
                        #     new_rv_roots_ids[i, :] = rvActivationIndexes[rvind]
                        #     new_rv_roots_weights[i, :] = np.maximum(distAux[rvind, i]**2, 10e-17)**(-1)
                        #     new_rv_roots_weights[i, :] = new_rv_roots_weights[i, :] / np.sum(new_rv_roots_weights[i, :])
                        #     new_rv_roots_weights[i, new_rv_roots_weights[i, :] < 10e-2] = 0.
                        #     new_rv_roots_weights[i, :] = new_rv_roots_weights[i, :] / np.sum(new_rv_roots_weights[i, :])
    
                        if lv_rootNodes is None:
                            lv_rootNodes = np.sum(lv_rootNodes_part, axis=0)
                            rv_rootNodes = np.sum(rv_rootNodes_part, axis=0)
                            lv_rootNodes_list = lv_rootNodes_part
                            rv_rootNodes_list = rv_rootNodes_part
                            rootNodes_list = rootNodes
                        else:
                            lv_rootNodes = lv_rootNodes + np.sum(lv_rootNodes_part, axis=0)
                            rv_rootNodes = rv_rootNodes + np.sum(rv_rootNodes_part, axis=0)
                            lv_rootNodes_list = np.concatenate((lv_rootNodes_list, lv_rootNodes_part), axis=0)
                            rv_rootNodes_list = np.concatenate((rv_rootNodes_list, rv_rootNodes_part), axis=0)
                            rootNodes_list = np.concatenate((rootNodes_list, rootNodes), axis=0)
                        # K-means weighted and projected back to potential root nodes
                        #new_lv_roots_ids_list.append(new_lv_roots_ids)
                        #new_lv_roots_weights_list.append(new_lv_roots_weights)
                        #new_rv_roots_ids_list.append(new_rv_roots_ids)
                        #new_rv_roots_weights_list.append(new_rv_roots_weights)
                        # K-means raw centroids
                        new_lv_roots_list.append(new_lv_roots)
                        new_rv_roots_list.append(new_rv_roots)
                        # Root nodes with the best discrepancy score
                        #discrepancy = np.loadtxt(previousResultsPath + fileName.replace('population', 'discrepancy'), delimiter=',')[particles_index]
                        #discrepancy_root_meta_indexes = np.round(population[np.argmin(discrepancy), 4:]).astype(int)
                        #disc_roots_list.append(activationIndexes[discrepancy_root_meta_indexes.astype(bool)])
    
                        if save_roots:
                            data_list.append({
                                'fileName': fileName,
                                'lv_roots': new_lv_roots,
                                'rv_roots': new_rv_roots
                                #,
                                #'params': population[np.argmin(discrepancy), :]
                            })
    
                # K-means CENTROIDS
                atmap = np.zeros((nodesXYZ.shape[0]))
                # if not is_ECGi:
                #     atmap[lv_rootNodesIndexes_true] = -1000
                #     atmap[rv_rootNodesIndexes_true] = -1000
    
                new_lv_roots, new_lv_roots_count = np.unique(np.concatenate((new_lv_roots_list), axis=0), return_counts=True)
                atmap[new_lv_roots] = np.round(100*new_lv_roots_count/len(new_lv_roots_list), 2)
    
                new_rv_roots, new_rv_roots_count = np.unique(np.concatenate((new_rv_roots_list), axis=0), return_counts=True)
                atmap[new_rv_roots] = np.round(100*new_rv_roots_count/len(new_rv_roots_list), 2)
    
                # Translate the atmap to be an element-wise map
                elem_atmap = np.zeros((elems.shape[0]))
                for i in range(elems.shape[0]):
                    elem_atmap[i] = np.sum(atmap[elems[i]])
    
                with open(figuresPath+meshName+'_'+target_type+'_'
                            +discretisation_resolution+'.ensi.kMeans_centroids', 'w') as f:
                    f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
                    for i in range(0, len(elem_atmap)):
                        f.write(str(elem_atmap[i])+'\n')
    
                # K-means PROJECTED BACK TO POENTIAL ROOT NODES USING DISTANCE FROM CENTROIDS AS WEIGHTS
                # atmap = np.zeros((nodesXYZ.shape[0]))
                # atmap[lv_rootNodesIndexes_true] = -1000
                # atmap[rv_rootNodesIndexes_true] = -1000
                #
                # new_lv_roots = np.unique(np.concatenate((new_lv_roots_ids_list)))
                # new_lv_roots_count = np.zeros((new_lv_roots.shape[0]))
                # for i in range(len(new_lv_roots_ids_list)):
                #     new_lv_roots_ids = new_lv_roots_ids_list[i]
                #     new_lv_roots_weights = new_lv_roots_weights_list[i]
                #     for j in range(new_lv_roots_ids.shape[0]):
                #         indexes = np.asarray([np.flatnonzero(new_lv_roots == node_id)[0] for node_id in new_lv_roots_ids[j, :]]).astype(int)
                #         new_lv_roots_count[indexes] = new_lv_roots_count[indexes] + new_lv_roots_weights[j, :]
                # atmap[new_lv_roots] = np.round(100*new_lv_roots_count/len(new_lv_roots_ids_list), 2)
                #
                #
                # new_rv_roots = np.unique(np.concatenate((new_rv_roots_ids_list)))
                # new_rv_roots_count = np.zeros((new_rv_roots.shape[0]))
                # for i in range(len(new_rv_roots_ids_list)):
                #     new_rv_roots_ids = new_rv_roots_ids_list[i]
                #     new_rv_roots_weights = new_rv_roots_weights_list[i]
                #     for j in range(new_rv_roots_ids.shape[0]):
                #         indexes = np.asarray([np.flatnonzero(new_rv_roots == node_id)[0] for node_id in new_rv_roots_ids[j, :]]).astype(int)
                #         new_rv_roots_count[indexes] = new_rv_roots_count[indexes] + new_rv_roots_weights[j, :]
                # atmap[new_rv_roots] = np.round(100*new_rv_roots_count/len(new_rv_roots_ids_list), 2)
    
    
                # Translate the atmap to be an element-wise map
                # elem_atmap = np.zeros((elems.shape[0]))
                # for i in range(elems.shape[0]):
                #    elem_atmap[i] = np.sum(atmap[elems[i]])
    
                # with open(figuresPath+meshName+'_'+targetType+'.ensi.kMeans_projected', 'w') as f:
                #     f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
                #     for i in range(0, len(elem_atmap)):
                #         f.write(str(elem_atmap[i])+'\n')
                pass
    
                # CUMM ROOT NODES TOGETHER
                atmap = np.zeros((nodesXYZ.shape[0]))
                if not is_ECGi:
                    atmap[lv_rootNodesIndexes_true] = -1000
                    atmap[rv_rootNodesIndexes_true] = -1000
                atmap[lvActivationIndexes] = np.round(100*lv_rootNodes/count, 2)#-6 # substracting the baseline value
                atmap[rvActivationIndexes] = np.round(100*rv_rootNodes/count, 2)#-6 # substracting the baseline value
    
                # Translate the atmap to be an element-wise map
                elem_atmap = np.zeros((elems.shape[0]))
                for i in range(elems.shape[0]):
                    elem_atmap[i] = np.sum(atmap[elems[i]])
    
                with open(figuresPath+meshName+'_'+target_type+'_'
                            +discretisation_resolution+'.ensi.cummrootNodes', 'w') as f:
                    f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
                    for i in range(0, len(elem_atmap)):
                        f.write(str(elem_atmap[i])+'\n')
                       
                if not is_ECGi:
                    # GROUND TRUTH ROOT NODES
                    atmap = np.zeros((nodesXYZ.shape[0]))
                    atmap[lv_rootNodesIndexes_true] = -1000
                    atmap[rv_rootNodesIndexes_true] = -1000
                    
                    # Translate the atmap to be an element-wise map
                    elem_atmap = np.zeros((elems.shape[0]))
                    for i in range(elems.shape[0]):
                        elem_atmap[i] = np.sum(atmap[elems[i]])
        
                    with open(figuresPath+meshName+'_'+target_type+'_'
                                +discretisation_resolution+'.ensi.trueNodes', 'w') as f:
                        f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
                        for i in range(0, len(elem_atmap)):
                            f.write(str(elem_atmap[i])+'\n')
    
                # # DISCREPANCY ROOT NODES
                # atmap = np.zeros((nodesXYZ.shape[0]))
                # atmap[lv_rootNodesIndexes_true] = -1000
                # atmap[rv_rootNodesIndexes_true] = -1000
                #
                # disc_roots, disc_roots_count = np.unique(np.concatenate((disc_roots_list), axis=0), return_counts=True)
                # atmap[disc_roots] = np.round(100*disc_roots_count/len(disc_roots_list), 2)
                #
                # # Translate the atmap to be an element-wise map
                # elem_atmap = np.zeros((elems.shape[0]))
                # for i in range(elems.shape[0]):
                #     elem_atmap[i] = np.sum(atmap[elems[i]])
                #
                # with open(figuresPath+meshName+'_'+targetType+'.ensi.discrepancy', 'w') as f:
                #     f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
                #     for i in range(0, len(elem_atmap)):
                #         f.write(str(elem_atmap[i])+'\n')
    
                # SAVE RESULTS TO REUSE BY OTHER SCRIPTS
                if save_roots:
                    df_roots = pandas.DataFrame(data_list)
                    df_roots.to_pickle(path=figuresPath+meshName+'_coarse_'+discretisation_resolution+'_'+target_type+'_rootNodes.gz', compression='infer')

        
        with open(figuresPath+meshName+'.ensi.case', 'w') as f:
            f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
            f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
            
            #f.write('scalar per element: 1	kMeans_projected	'+meshName+'_'+target_type+'.ensi.kMeans_projected\n')
            #f.write('scalar per element: 1	discrepancy	'+meshName+'_'+target_type+'.ensi.discrepancy\n')
            
            #f.write('scalar per node: 1	pred_low	'+meshName+'_'+target_type+'.ensi.pred_low\n')
            #f.write('scalar per node: 1	pred_high	'+meshName+'_'+target_type+'.ensi.pred_high\n')
            
            for discretisation_resolution in discretisation_list:
                for target_type in target_type_list:
                    f.write('scalar per element: 1	kMeans_centroids_'+target_type+'_'
                            +discretisation_resolution+'	'+meshName+'_'+target_type+'_'
                            +discretisation_resolution+'.ensi.kMeans_centroids\n')
                    f.write('scalar per element: 1	cummrootNodes_'+target_type+'_'
                            +discretisation_resolution+'	'+meshName+'_'+target_type+'_'
                            +discretisation_resolution+'.ensi.cummrootNodes\n')
                    if not is_ECGi:
                        f.write('scalar per element: 1	trueNodes_'+target_type+'_'
                                +discretisation_resolution+'	'+meshName+'_'+target_type+'_'
                                +discretisation_resolution+'.ensi.trueNodes\n')
            if meshName == meshName_pred and np.any(np.asarray(['atm' in target_type for target_type in target_type_list])):
                f.write('scalar per node: 1	true	'+meshName+'.ensi.true\n')
                for discretisation_resolution in discretisation_list:
                    f.write('scalar per node: 1	pred_'+discretisation_resolution+'	'+meshName+'_'+discretisation_resolution+'.ensi.pred_'+discretisation_resolution+'\n')
                
        
        # Save Atmaps for true bidomain and predicted eikonal
        if meshName == meshName_pred and np.any(np.asarray(['atm' in target_type for target_type in target_type_list])):
            for target_type_aux in target_type_list:
                if 'atm' in target_type_aux:
                    target_type = target_type_aux
            # Paths and tags
            dataPath = 'metaData/' + meshName + '/'
        
            # Load mesh
            # global nodesXYZ
            # nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
            # global edges
            # edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
            lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
                                           delimiter=',') - 1).astype(int))  # lv endocardium triangles
            rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
                                           delimiter=',') - 1).astype(int))  # rv endocardium triangles
            global tetraFibers
            tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv',
                                     delimiter=',')  # tetrahedron fiber directions
        
            tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
            global edgeVEC
            edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :]  # edge vectors
        
            rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
            rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
        
            global tetrahedrons
            tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
            tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
            electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
            nb_bsp = electrodePositions.shape[0]
        
            aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
            for i in range(0, tetrahedrons.shape[0], 1):
                aux[tetrahedrons[i, 0]].append(i)
                aux[tetrahedrons[i, 1]].append(i)
                aux[tetrahedrons[i, 2]].append(i)
                aux[tetrahedrons[i, 3]].append(i)
            global elements
            elements = [np.array(n) for n in aux]
            aux = None  # Clear Memory
        
            # Precompute PseudoECG stuff
            # Calculate the tetrahedrons volumes
            D = nodesXYZ[tetrahedrons[:, 3], :]  # RECYCLED
            A = nodesXYZ[tetrahedrons[:, 0], :] - D  # RECYCLED
            B = nodesXYZ[tetrahedrons[:, 1], :] - D  # RECYCLED
            C = nodesXYZ[tetrahedrons[:, 2], :] - D  # RECYCLED
            D = None  # Clear Memory
        
            global tVolumes
            tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1),
                                                   (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0])  # Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
            tVolumes = tVolumes / np.sum(tVolumes)
        
            # Calculate the tetrahedron (temporal) voltage gradients
            Mg = np.stack((A, B, C), axis=-1)
            A = None  # Clear Memory
            B = None  # Clear Memory
            C = None  # Clear Memory
        
            # Calculate the gradients
            global G_pseudo
            G_pseudo = np.zeros(Mg.shape)
            for i in range(Mg.shape[0]):
                G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
            G_pseudo = np.moveaxis(G_pseudo, 1, 2)
            Mg = None  # clear memory
        
            # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
            r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                                       (tetrahedronCenters.shape[0],
                                        tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1) - electrodePositions
        
            global d_r
            d_r = np.moveaxis(np.multiply(
                np.moveaxis(r, [0, 1], [-1, -2]),
                np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0, -1)
        
            # Set endocardial edges aside
            global isEndocardial
            isEndocardial = np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                                          np.all(np.isin(edges, rvface), axis=1))
            # Build adjacentcies
            global unfoldedEdges
            unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
            aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
            for i in range(0, len(unfoldedEdges), 1):
                aux[unfoldedEdges[i, 0]].append(i)
            global neighbours
            # make neighbours Numba friendly
            neighbours_aux = [np.array(n) for n in aux]
            aux = None  # Clear Memory
            m = 0
            for n in neighbours_aux:
                m = max(m, len(n))
            neighbours = np.full((len(neighbours_aux), m), np.nan,
                                 np.int32)  # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
            for i in range(len(neighbours_aux)):
                n = neighbours_aux[i]
                neighbours[i, :n.shape[0]] = n
            neighbours_aux = None
        
            # neighbours_original
            aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
            for i in range(0, len(unfoldedEdges), 1):
                aux[unfoldedEdges[i, 0]].append(i)
            global neighbours_original
            neighbours_original = [np.array(n) for n in aux]
            aux = None  # Clear Memory
            
            if load_target:
                atmap = np.loadtxt('metaData/ATMaps/' + meshName + '_coarse_true_ATMap_120_1x.csv', delimiter=',')
            else:
                atmap = eikonal_ecg(np.array([conduction_speeds_pred])/1000., rootNodesIndexes_true, rootNodesTimes)[0, :]
            with open(figuresPath + meshName + '.ensi.true', 'w') as f:
                f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
                for i in range(0, len(atmap)):
                    f.write(str(atmap[i]) + '\n')
            atm_pred_list = []
            for discretisation_resolution in discretisation_list:
                if 'atm' in target_type:
                    data_type = 'atm'
                elif 'ecg' in target_type:
                    data_type = 'ecg'
                elif 'bsp' in target_type:
                    data_type = 'bsp'
                previousResultsPath = resultsPath+data_type+"Noise_new_code_"+discretisation_resolution+"/"
                new_pred_atm = np.loadtxt(previousResultsPath  + meshName + '_' + discretisation_resolution
                    + '_' + str(conduction_speeds_pred) + '_' + discretisation_resolution + '_' + target_type
                    + '_0_prediction.csv', delimiter=',')
                print(discretisation_resolution)
                print(np.corrcoef(atmap, new_pred_atm)[0,1])
                atm_pred_list.append(new_pred_atm)
            # max_val = np.amax(np.vstack((atm_pred_list)))
            for atmap_i in range(len(atm_pred_list)):
                # atm_pred_list[atmap_i][0] = max_val
                with open(figuresPath + meshName + '_' + discretisation_list[atmap_i] + '.ensi.pred_'+discretisation_list[atmap_i], 'w') as f:
                    f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
                    for i in range(0, len(atm_pred_list[atmap_i])):
                        f.write(str(atm_pred_list[atmap_i][i]) + '\n')


# Create equally spaced potential root-node locations in the mesh before 28/06/2021
def generatePotentialRoots(meshNameList, resName):
    # Make the potential root nodes
    for meshName in meshNameList:
        print(meshName)
        # Paths and tags
        dataPath = 'metaData/' + meshName + '/'
        # Load mesh
        nodesXYZ = np.loadtxt(dataPath + meshName + '_xyz.csv', delimiter=',')
        edges = (np.loadtxt(dataPath + meshName + '_edges.csv', delimiter=',') - 1).astype(int)
        lvnodes = np.unique((np.loadtxt(dataPath + meshName + '_lvface.csv', delimiter=',') - 1).astype(int)) # lv endocardium nodes
        rvnodes = np.unique((np.loadtxt(dataPath + meshName + '_rvface.csv', delimiter=',') - 1).astype(int)) # rv endocardium nodes

        # Check integrity of the ventricle surfaces
        # Check integrity of LV
        lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
        lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
        lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
        lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
        aux = [[] for i in range(0, lvnodes.shape[0], 1)]
        for i in range(0, len(lvunfoldedEdges), 1):
            aux[lvunfoldedEdges[i, 0]].append(i)
        lvneighbours = [np.array(n).astype(int) for n in aux]
        aux = None # Clear Memory
        removeNodes = np.zeros((lvnodes.shape[0]), dtype=np.bool_)
        if meshName == 'C2':
            nextNodes = [0] # This could fail if the first root node of the endocardial surface is not connected to the rest
        else:
            nextNodes = [100] # for mesh C1
        while np.sum(removeNodes) < len(removeNodes):
            if len(nextNodes) == 0:
                break
            for i in range(len(nextNodes)):
                act_node = nextNodes.pop()
                if not removeNodes[act_node]:
                    removeNodes[act_node] = True
                    for edge_i in lvneighbours[act_node]:
                        node = lvunfoldedEdges[edge_i, 1]
                        if not removeNodes[node]:
                            nextNodes.append(node)
            # print('-----')
            # print(len(removeNodes)-np.sum(removeNodes))
            # print(len(nextNodes))
            nextNodes = np.sort(np.unique(nextNodes)).tolist()
        print('LV Remomve ' + str(len(removeNodes)-np.sum(removeNodes)))
        lvnodes = lvnodes[removeNodes]
        # Save revised version of the endocardial faces
        np.savetxt(dataPath + meshName + '_lvnodes.csv', lvnodes.astype(int)+1, delimiter=',')

        # # Revise integrity of fixed LV
        # lvnodes = np.unique(lvnodes) # lv endocardium nodes
        # lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
        # lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
        # lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
        # lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
        # aux = [[] for i in range(0, lvnodes.shape[0], 1)]
        # for i in range(0, len(lvunfoldedEdges), 1):
        #     aux[lvunfoldedEdges[i, 0]].append(i)
        # lvneighbours = [np.array(n).astype(int) for n in aux]
        # aux = None # Clear Memory
        # removeNodes = np.zeros((lvnodes.shape[0]), dtype=np.bool_)
        # nextNodes = [0] # This could fail if the first root node of the endocardial surface is not connected to the rest
        # while np.sum(removeNodes) < len(removeNodes):
        #     if len(nextNodes) == 0:
        #         break
        #     for i in range(len(nextNodes)):
        #         act_node = nextNodes.pop()
        #         if not removeNodes[act_node]:
        #             removeNodes[act_node] = True
        #             for edge_i in lvneighbours[act_node]:
        #                 node = lvunfoldedEdges[edge_i, 1]
        #                 if not removeNodes[node]:
        #                     nextNodes.append(node)
        #     nextNodes = np.sort(np.unique(nextNodes)).tolist()
        # print('2nd LV attempt to Remomve ' + str(len(removeNodes)-np.sum(removeNodes)))
        
        # Check integrity of RV
        rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
        rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
        rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
        rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
        aux = [[] for i in range(0, rvnodes.shape[0], 1)]
        for i in range(0, len(rvunfoldedEdges), 1):
            aux[rvunfoldedEdges[i, 0]].append(i)
        rvneighbours = [np.array(n).astype(int) for n in aux]
        aux = None # Clear Memory
        removeNodes = np.zeros((rvnodes.shape[0]), dtype=np.bool_)
        #nextNodes = [0] # This could fail if the first root node of the endocardial surface is not connected to the rest
        # Zero fails for mesh C2
        nextNodes = [1000]
        while np.sum(removeNodes) < len(removeNodes):
            if len(nextNodes) == 0:
                break
            for i in range(len(nextNodes)):
                act_node = nextNodes.pop()
                if not removeNodes[act_node]:
                    removeNodes[act_node] = True
                    for edge_i in rvneighbours[act_node]:
                        node = rvunfoldedEdges[edge_i, 1]
                        if not removeNodes[node]:
                            nextNodes.append(node)
            # print('-----')
            # print(len(removeNodes)-np.sum(removeNodes))
            # print(len(nextNodes))
            nextNodes = np.sort(np.unique(nextNodes)).tolist()
        print('RV Remomve ' + str(len(removeNodes)-np.sum(removeNodes)))
        rvnodes = rvnodes[removeNodes]
        # Save revised version of the endocardial faces
        np.savetxt(dataPath + meshName + '_rvnodes.csv', rvnodes.astype(int)+1, delimiter=',')
        
        # # Revise integrity of fixed RV
        # rvnodes = np.unique(rvnodes) # rv endocardium nodes
        # rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
        # rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
        # rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
        # rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
        # aux = [[] for i in range(0, rvnodes.shape[0], 1)]
        # for i in range(0, len(rvunfoldedEdges), 1):
        #     aux[rvunfoldedEdges[i, 0]].append(i)
        # rvneighbours = [np.array(n).astype(int) for n in aux]
        # aux = None # Clear Memory
        # removeNodes = np.zeros((rvnodes.shape[0]), dtype=np.bool_)
        # nextNodes = [0] # This could fail if the first root node of the endocardial surface is not connected to the rest
        # while np.sum(removeNodes) < len(removeNodes):
        #     if len(nextNodes) == 0:
        #         break
        #     for i in range(len(nextNodes)):
        #         act_node = nextNodes.pop()
        #         if not removeNodes[act_node]:
        #             removeNodes[act_node] = True
        #             for edge_i in rvneighbours[act_node]:
        #                 node = rvunfoldedEdges[edge_i, 1]
        #                 if not removeNodes[node]:
        #                     nextNodes.append(node)
        #     nextNodes = np.sort(np.unique(nextNodes)).tolist()
        # print('2nd RV attempt to Remomve ' + str(len(removeNodes)-np.sum(removeNodes)))

        # # Determine potential root nodes in a less sofisticated way to reduce the computational cost
        # lvRoots = np.zeros(lvnodes.shape)
        # lvIndexes = np.random.permutation(lvnodes.shape[0])
        # lvRoots[lvIndexes[0]]
        # for i in lvIndexes:
        # #TODO TODO HERE HERE AQUI AKI HOLA 2021
        # # TODO September 2021: finally decided to not use the HCM meshes for the time being, thus, not requiering to generate new coarse meshes yet.
        
        
        # Determine potential root nodes in a sofisticated way
        rvRoots = np.zeros(rvnodes.shape)
        rvIndexes = np.random.permutation(rvnodes.shape[0])
        
        isLV_Endocardial = np.all(np.isin(edges, lvnodes), axis=1)
        lvEndoEdges = np.array([edges[i, :] for i in range(edges.shape[0]) if isLV_Endocardial[i]])
        lvUnfoldedEdges = np.concatenate((lvEndoEdges, np.flip(lvEndoEdges, axis=1))).astype(int)
        lvinverseIndexing = np.zeros((nodesXYZ.shape[0])).astype(int)
        lvinverseIndexing[lvnodes] = np.arange(0, lvnodes.shape[0], 1).astype(int)
        aux = [[] for i in range(0, lvnodes.shape[0], 1)]
        for i in range(0, len(lvUnfoldedEdges), 1):
            aux[lvinverseIndexing[lvUnfoldedEdges[i, 0]]].append(lvinverseIndexing[lvUnfoldedEdges[i, 1]])
        lvNeighbours = [np.array(n) for n in aux]
        aux = [[] for i in range(0, lvnodes.shape[0], 1)]
        for i in range(0, len(lvNeighbours), 1):
            for j in lvNeighbours[i]:
                aux[i].append(j)
                for k in lvNeighbours[j]:
                    aux[i].append(k)
        lvNeighbours = [np.unique(np.array(n)) for n in aux]
        aux = [[] for i in range(0, lvnodes.shape[0], 1)]
        for i in range(0, len(lvNeighbours), 1):
            for j in lvNeighbours[i]:
                aux[i].append(j)
                for k in lvNeighbours[j]:
                    aux[i].append(k)
        lvNeighbours = [np.unique(np.array(n)) for n in aux]
        aux = [[] for i in range(0, lvnodes.shape[0], 1)]
        for i in range(0, len(lvNeighbours), 1):
            for j in lvNeighbours[i]:
                aux[i].append(j)
                for k in lvNeighbours[j]:
                    aux[i].append(k)
        lvNeighbours = [np.unique(np.array(n)) for n in aux]
        if resName == "newLow":
            aux = [[] for i in range(0, lvnodes.shape[0], 1)] # TODO: added on 17/02/2021
            for i in range(0, len(lvNeighbours), 1): # TODO: added on 17/02/2021
                for j in lvNeighbours[i]: # TODO: added on 17/02/2021
                    aux[i].append(j) # TODO: added on 17/02/2021
                    for k in lvNeighbours[j]: # TODO: added on 17/02/2021
                        aux[i].append(k) # TODO: added on 17/02/2021
            lvNeighbours = [np.unique(np.array(n)) for n in aux] # TODO: added on 17/02/2021

        isRV_Endocardial = np.all(np.isin(edges, rvnodes), axis=1)
        rvEndoEdges = np.array([edges[i, :] for i in range(edges.shape[0]) if isRV_Endocardial[i]])
        rvUnfoldedEdges = np.concatenate((rvEndoEdges, np.flip(rvEndoEdges, axis=1))).astype(int)
        rvinverseIndexing = np.zeros((nodesXYZ.shape[0])).astype(int)
        rvinverseIndexing[rvnodes] = np.arange(0, rvnodes.shape[0], 1).astype(int)
        aux = [[] for i in range(0, rvnodes.shape[0], 1)]
        for i in range(0, len(rvUnfoldedEdges), 1):
            aux[rvinverseIndexing[rvUnfoldedEdges[i, 0]]].append(rvinverseIndexing[rvUnfoldedEdges[i, 1]])
        rvNeighbours = [np.unique(np.array(n)) for n in aux]
        aux = [[] for i in range(0, rvnodes.shape[0], 1)]
        for i in range(0, len(rvNeighbours), 1):
            for j in rvNeighbours[i]:
                aux[i].append(j)
                for k in rvNeighbours[j]:
                    aux[i].append(k)
        rvNeighbours = [np.unique(np.array(n)) for n in aux]
        aux = [[] for i in range(0, rvnodes.shape[0], 1)]
        for i in range(0, len(rvNeighbours), 1):
            for j in rvNeighbours[i]:
                aux[i].append(j)
                for k in rvNeighbours[j]:
                    aux[i].append(k)
        rvNeighbours = [np.unique(np.array(n)) for n in aux]
        aux = [[] for i in range(0, rvnodes.shape[0], 1)]
        for i in range(0, len(rvNeighbours), 1):
            for j in rvNeighbours[i]:
                aux[i].append(j)
                for k in rvNeighbours[j]:
                    aux[i].append(k)
        rvNeighbours = [np.unique(np.array(n)) for n in aux]
        if resName == "newRV" or resName == "newLow":
            aux = [[] for i in range(0, rvnodes.shape[0], 1)] # TODO: added on 17/02/2021
            for i in range(0, len(rvNeighbours), 1): # TODO: added on 17/02/2021
                for j in rvNeighbours[i]: # TODO: added on 17/02/2021
                    aux[i].append(j) # TODO: added on 17/02/2021
                    for k in rvNeighbours[j]: # TODO: added on 17/02/2021
                        aux[i].append(k) # TODO: added on 17/02/2021
            rvNeighbours = [np.unique(np.array(n)) for n in aux] # TODO: added on 17/02/2021

        lvRoots = np.zeros((lvnodes.shape[0])).astype(bool)
        for i in range(lvRoots.shape[0]):
            if not np.any([lvRoots[neighbour] for neighbour in lvNeighbours[i]]):
                lvRoots[i] = 1

        rvRoots = np.zeros((rvnodes.shape[0])).astype(bool)
        for i in range(rvRoots.shape[0]):
            if not np.any([rvRoots[neighbour] for neighbour in rvNeighbours[i]]):
                rvRoots[i] = 1
        
        print(len(lvnodes[lvRoots].astype(int)+1))
        print(len(rvnodes[rvRoots].astype(int)+1))
        
        if resName == "newRV":  # TODO: added on 17/02/2021
            lv_spread_distance = 1.5 # cm
            rv_spread_distance = 2.5 # cm
            # rv_spread_distance = 1.5 # cm # TODO: CHANGE THIS URGENTLY
        elif resName == "newLow":
            lv_spread_distance = 2.5 # cm
            rv_spread_distance = 2.5 # cm
        elif resName == "newHigh":
            lv_spread_distance = 1.5 # cm
            rv_spread_distance = 1.5 # cm
        lv_spread_distance = 2. # cm
        rv_spread_distance = 2. # cm
        # LV: Add root nodes that are NOT too close # TODO: added on 17/02/2021
        do_next = True
        while do_next:
            distances = np.amin(np.sqrt(np.sum((nodesXYZ[lvnodes[lvRoots == 0].astype(int), np.newaxis, :]
                                        - nodesXYZ[lvnodes[lvRoots.astype(bool)].astype(int), :])**2, axis=2)), axis=1)
            non_root_node_indexes = np.nonzero(lvRoots == 0)[0]
            min_iter = np.argmax(distances)
            min_distance = distances[min_iter]
            do_next = min_distance > lv_spread_distance
            if do_next:
                lvRoots[non_root_node_indexes[min_iter]] = 1

        # # RV: Add root nodes that are NOT too close # TODO: added on 17/02/2021
        do_next = True
        while do_next:
            distances = np.amin(np.sqrt(np.sum((nodesXYZ[rvnodes[rvRoots == 0].astype(int), np.newaxis, :]
                                        - nodesXYZ[rvnodes[rvRoots.astype(bool)].astype(int), :])**2, axis=2)), axis=1)
            non_root_node_indexes = np.nonzero(rvRoots == 0)[0]
            min_iter = np.argmax(distances)
            min_distance = distances[min_iter]
            do_next = min_distance > rv_spread_distance
            if do_next:
                rvRoots[non_root_node_indexes[min_iter]] = 1
        
        print()
        print(len(lvnodes[lvRoots].astype(int)+1))
        print(len(rvnodes[rvRoots].astype(int)+1))
        print('saving!')
        np.savetxt(dataPath + meshName + '_lv_activationIndexes_'+resName+'Res.csv', lvnodes[lvRoots].astype(int)+1, delimiter=',')
        np.savetxt(dataPath + meshName + '_rv_activationIndexes_'+resName+'Res.csv', rvnodes[rvRoots].astype(int)+1, delimiter=',')

    return None


# Use the torso mesh as in the code for generating the new root nodes from the endocardia submeshes. Then register a realistic number of electrodes.
def generateVirtualElectrodes(meshNameList):
    # Make the potential root nodes
    for meshName in meshNameList:
        # Paths and tags
        dataPath = 'metaData/' + meshName + '/'
        # Load mesh
        nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_torsoxyz.csv', delimiter=',')
        edges = (np.loadtxt(dataPath + meshName + '_coarse_torsoedges.csv', delimiter=',') - 1).astype(int)
        torsonodes = np.unique((np.loadtxt(dataPath + meshName + '_coarse_torsoface.csv', delimiter=',') - 1).astype(
            int))  # torso-surface nodes
        electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')

        # Build adjacentcies
        isTorso = np.all(np.isin(edges, torsonodes), axis=1)
        torsoEdges = np.array([edges[i, :] for i in range(edges.shape[0]) if isTorso[i]])
        torsoUnfoldedEdges = np.concatenate((torsoEdges, np.flip(torsoEdges, axis=1))).astype(int)
        torsoinverseIndexing = np.zeros((nodesXYZ.shape[0])).astype(int)
        torsoinverseIndexing[torsonodes] = np.arange(0, torsonodes.shape[0], 1).astype(int)
        aux = [[] for i in range(0, torsonodes.shape[0], 1)]
        for i in range(0, len(torsoUnfoldedEdges), 1):
            aux[torsoinverseIndexing[torsoUnfoldedEdges[i, 0]]].append(torsoinverseIndexing[torsoUnfoldedEdges[i, 1]])
        torsoNeighbours = [np.array(n) for n in aux]
        aux = [[] for i in range(0, torsonodes.shape[0], 1)]
        for i in range(0, len(torsoNeighbours), 1):
            for j in torsoNeighbours[i]:
                aux[i].append(j)
                for k in torsoNeighbours[j]:
                    aux[i].append(k)
        torsoNeighbours = [np.unique(np.array(n)) for n in aux]
        aux = [[] for i in range(0, torsonodes.shape[0], 1)]
        for i in range(0, len(torsoNeighbours), 1):
            for j in torsoNeighbours[i]:
                aux[i].append(j)
                for k in torsoNeighbours[j]:
                    aux[i].append(k)
        torsoNeighbours = [np.unique(np.array(n)) for n in aux]

        torsoElectrodes = np.zeros((torsonodes.shape[0])).astype(bool)
        for i in range(torsoElectrodes.shape[0]):
            if not np.any([torsoElectrodes[neighbour] for neighbour in torsoNeighbours[i]]):
                torsoElectrodes[i] = 1

        torsoNeighbours = None  # Clear
        edges = None  # Clear
        isTorso = None  # Clear
        torsoUnfoldedEdges = None  # Clear
        torsoinverseIndexing = None  # Clear
        torsoElectrodePositions = nodesXYZ[torsonodes[torsoElectrodes].astype(int), :]
        nodesXYZ = None  # Clear
        torsonodes = None  # Clear
        torsoElectrodes = None  # Clear
        # Filter electrodes out of clinical-vest area
        vestArea = np.ones((torsoElectrodePositions.shape[0]))
        print('Abans: ' + str(np.sum(vestArea)))
        if meshName == 'DTI024':
            # taller and a bit wider on the left
            # height
            vestArea[torsoElectrodePositions[:, 2] < np.amin(electrodePositions[:, 2]) - 2] = 0
            vestArea[torsoElectrodePositions[:, 2] > np.amax(electrodePositions[:, 2]) + 4.5] = 0
            # width
            vestArea[torsoElectrodePositions[:, 0] < np.amin(electrodePositions[:, 0]) - 4.3] = 0
            vestArea[torsoElectrodePositions[:, 0] > np.amax(electrodePositions[:, 0]) + 1.7] = 0
        elif meshName == 'DTI004':
            # shrink it in width
            # height
            vestArea[torsoElectrodePositions[:, 2] < np.amin(electrodePositions[:, 2]) - 2] = 0
            vestArea[torsoElectrodePositions[:, 2] > np.amax(electrodePositions[:, 2]) + 4] = 0
            # width
            vestArea[torsoElectrodePositions[:, 0] < np.amin(electrodePositions[:, 0]) - 1.8] = 0
            vestArea[torsoElectrodePositions[:, 0] > np.amax(electrodePositions[:, 0]) + 0.2] = 0
        elif meshName == 'DTI003':
            # a tinny bit taller
            # height
            vestArea[torsoElectrodePositions[:, 2] < np.amin(electrodePositions[:, 2]) - 2] = 0
            vestArea[torsoElectrodePositions[:, 2] > np.amax(electrodePositions[:, 2]) + 4.5] = 0
            # width
            vestArea[torsoElectrodePositions[:, 0] < np.amin(electrodePositions[:, 0]) - 3] = 0
            vestArea[torsoElectrodePositions[:, 0] > np.amax(electrodePositions[:, 0]) + 2] = 0
        else:
            # default
            # height
            vestArea[torsoElectrodePositions[:, 2] < np.amin(electrodePositions[:, 2]) - 2] = 0
            vestArea[torsoElectrodePositions[:, 2] > np.amax(electrodePositions[:, 2]) + 4] = 0
            # width
            vestArea[torsoElectrodePositions[:, 0] < np.amin(electrodePositions[:, 0]) - 3] = 0
            vestArea[torsoElectrodePositions[:, 0] > np.amax(electrodePositions[:, 0]) + 2] = 0
        # depth
        # No restrictions for depth
        torsoElectrodePositions = torsoElectrodePositions[vestArea.astype(bool), :]
        # Filter out electrodes that are too close to each other
        # vestArea = np.ones((torsoElectrodePositions.shape[0]))
        # Number of electrodes for body surface potential recording
        # Medtronics vest has 252 electrodes -> https://europe.medtronic.com/xd-en/healthcare-professionals/products/cardiac-rhythm/cardiac-mapping/cardioinsight-mapping-vest.html
        # Sophie Giffard-Rosin used 205 electrodes for BSP recording
        # Yoram used 200 in his paper
        # desired_nb_electrodes = 252 # We consider the full 252 electrodes plus
        # Minimal inter-electrode distance
        # CAUTION! COMMENTED OUT OF THE METHOD BECAUSE IT WAS COMPUTATIONALLY INTRACTABLE!!
        # min_inter_dist = 1 # 1 cm for us, Yoram did 1.2 cm between pairs of 5 mm separated electrodes, others don't specify
        # # aux = np.copy(torsoElectrodePositions)
        # # while filter_more:
        #
        # #     dist = np.sqrt(np.sum((aux[vestArea.astype(bool), None, :]
        # #                            - aux[None, vestArea.astype(bool), :]) ** 2, axis=2))
        # # print(np.sqrt(np.sum((torsoElectrodePositions[vestArea.astype(bool), :]
        # #                            - torsoElectrodePositions[i, :]) ** 2, axis=1)).shape)
        # for i in range(torsoElectrodePositions.shape[0]):
        #     aux = torsoElectrodePositions[i+1:, :]
        #     aux_vestArea = vestArea[i+1:]
        #     if np.amin(np.sqrt(np.sum((aux[aux_vestArea.astype(bool), :]
        #                            - torsoElectrodePositions[i, :]) ** 2, axis=1))) < min_inter_dist: # 1.2 cm apart double electrodes in Yoram's paper
        #
        #         print(np.sqrt(np.sum((torsoElectrodePositions[vestArea.astype(bool), :]
        #                                                     - torsoElectrodePositions[i, :]) ** 2, axis=1)).shape)
        #         vestArea[i] = 0
        #
        # torsoElectrodePositions = torsoElectrodePositions[vestArea.astype(bool), :]
        # print('Despres: ' + str(np.sum(vestArea)))
        #
        #
        # # Save
        # torsoElectrodePositions = torsoElectrodePositions[vestArea.astype(bool), :]
        np.savetxt(dataPath + meshName + '_coarse_torso_electrodePositions.csv', torsoElectrodePositions, delimiter=',')
        print('Done with ' + meshName + ', with nb_virtual_electrodes = ' + str(np.sum(vestArea))) # Sophie Giffard-Rosin used 205 electrodes for BSP recording
        torsoElectrodePositions = None  # Clear
    return None


def makeECGirecording(meshName, atmap, target_fileName):
    # Definitions
    global nb_leads
    global nb_bsp
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt, a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    dataPath = 'metaData/' + meshName + '/'

    ecg_electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    ECGi_electrodePositions = np.loadtxt(dataPath + meshName + '_ECGiElectrodePositions.csv', delimiter=',')
    nb_leads = 8 + ECGi_electrodePositions.shape[0]  # All leads from the ECGi are calculated like the precordial leads
    electrodePositions = np.concatenate((ecg_electrodePositions, ECGi_electrodePositions), axis=0)
    nb_bsp = electrodePositions.shape[0]

    # if os.path.isfile(target_fileName):
    #     pseudo_ecg = np.loadtxt(target_fileName, delimiter=',')
    # else:
    if True:
        global nodesXYZ
        nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
        global tetrahedrons
        tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
        tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')


        aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
        for i in range(0, tetrahedrons.shape[0], 1):
            aux[tetrahedrons[i, 0]].append(i)
            aux[tetrahedrons[i, 1]].append(i)
            aux[tetrahedrons[i, 2]].append(i)
            aux[tetrahedrons[i, 3]].append(i)
        global elements
        elements = [np.array(n) for n in aux]

        # Calculate the tetrahedrons volumes
        D = nodesXYZ[tetrahedrons[:, 3], :]
        A = nodesXYZ[tetrahedrons[:, 0], :] - D
        B = nodesXYZ[tetrahedrons[:, 1], :] - D
        C = nodesXYZ[tetrahedrons[:, 2], :] - D

        tVolumes = np.reshape(
            np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1), (np.cross(B, C)[:, :, np.newaxis]))),
            tetrahedrons.shape[0])
        tVolumes = tVolumes / np.sum(tVolumes)

        # Calculate the tetrahedron (temporal) voltage gradients
        Mg = np.stack((A, B, C), axis=-1)

        # Calculate the gradients
        global G_pseudo
        G_pseudo = np.zeros(Mg.shape)
        for i in range(Mg.shape[0]):
            G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
        G_pseudo = np.moveaxis(G_pseudo, 1, 2)

        # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
        r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                                   (tetrahedronCenters.shape[0], tetrahedronCenters.shape[1],
                                    electrodePositions.shape[0])),
                        1, -1) - electrodePositions
        global d_r
        d_r = np.moveaxis(np.multiply(np.moveaxis(r, [0, 1], [-1, -2]),
                                      np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1),
                                                  tVolumes)), 0,
                          -1)

        t_start = time.time()
        pseudo_ecg = pseudoBSP(atmap)
        print('Time elapsed: '+str(round(time.time()-t_start)) + ' sec')
        # Save ECGi
        np.savetxt(target_fileName, pseudo_ecg, delimiter=',')
    return pseudo_ecg


def makeECGiPlot(figName, fileName):

    # Definitions
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    if os.path.isfile(fileName):
        pseudo_ecg = np.loadtxt(fileName, delimiter=',')
        nb_leads = pseudo_ecg.shape[0]
        # Create figure
        nb_rows = math.ceil(nb_leads / 8)
        fig, axs = plt.subplots(nrows=nb_rows, ncols=8, constrained_layout=True, figsize=(40, nb_rows * 5),
                                sharey='all')
        fig.suptitle('ECGi recording', fontsize=24)

        # Calculate figure ms width
        max_count = np.sum(np.logical_not(np.isnan(pseudo_ecg[0, :]))) + 5
        for j in range(nb_rows):
            for k in range(8):
                i = j * 8 + k
                if i < nb_leads:
                    if i < 8:
                        leadName = leadNames[i]
                        axs[j, k].set_title('Lead ' + leadName, fontsize=20)
                    axs[j, k].plot(pseudo_ecg[i, :], 'g-', label='Pseudo', linewidth=3.)

                    # decorate figure
                    axs[j, k].set_xlim(0, max_count)
                    axs[j, k].xaxis.set_major_locator(MultipleLocator(10))
                    axs[j, k].yaxis.set_major_locator(MultipleLocator(1))
                    axs[j, k].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    axs[j, k].xaxis.grid(True, which='major')
                    axs[j, k].yaxis.grid(True, which='major')
                    for tick in axs[j, k].xaxis.get_major_ticks():
                        tick.label.set_fontsize(16)
                    for tick in axs[j, k].yaxis.get_major_ticks():
                        tick.label.set_fontsize(16)

                    axs[j, k].set_xlabel('ms', fontsize=20)
            axs[j, 0].set_ylabel('standardised voltage', fontsize=16)
        axs[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

        plt.savefig(figName, dpi=150)
        plt.close()
        print('Done: ' + figName)
    else:
        print('No file: ' + fileName)


def remove_file(path):
    if os.path.isfile(path):
        os.remove(path)


def clearPreviousOutputs(figPath, meshName, target_type):
    remove_file(figPath + meshName + '_' +  target_type + '.ensi.PKtimes')
    remove_file(figPath + meshName + '.ensi.geo')
    remove_file(figPath + meshName + '.ensi.case')
    remove_file(figPath + meshName + '_available_LV_PKnetwork.vtk')
    remove_file(figPath + meshName + '_available_RV_PKnetwork.vtk')
    remove_file(figPath + meshName + '_newRVRes_available_root_nodes.csv')
    

def mapAndGeneratePKTree(threadsNum_val, meshName, conduction_speeds, resolution_list, figPath, healthy_val,
                load_target, data_type, metric, target_data_path, endocardial_layer, is_clinical, experiment_path_tag,
                min_possible_consistency_count=0, max_possible_consistency_count = 100):
    target_type = data_type+'_'+metric
    clearPreviousOutputs(figPath, meshName, target_type)
    
    # mappWithCobiveco(meshName_ref='DTI003', meshName_target='DTI4586_2_coarse', points_to_map_filenames=['_lvhisbundle_xyz.csv',
    #     '_rvhisbundle_xyz.csv', '_lvdense_xyz.csv', '_rvdense_xyz.csv', '_lvseptalwall_xyz.csv',  '_lvfreewallnavigationpoints_xyz.csv',
    #     '_lvfreewallextended_xyz.csv', '_rvfreewall_xyz.csv', '_rvlowerthird_xyz.csv', '_lvhistop_xyz.csv', '_rvhistop_xyz.csv',
    #     '_lvhisbundleConnected_xyz.csv', '_rvhisbundleConnected_xyz.csv', '_lvapexnavigation_xyz.csv', '_lv_activationIndexes_newRVRes.csv',
    #     '_rv_activationIndexes_newRVRes.csv'], points_are_indexes=[False, False, False, False, False, False, False, False, False, False,
    #     False, False, False, False, True, True])
    purkinje_speed = 0.2 # cm/ms # the final purkinje network will use 0.19 but we don't need to use a slightly slower speed to
#allow for curved branches in the networks, because the times are here computed on a jiggsaw-like coarse mesh which will already
#add a little of buffer time
    target_type = data_type+'_'+metric
    # generatePotentialRoots([meshName], "newRV")
    #TODO
    # Fix this function, not working on 23/12/2020
    global has_endocardial_layer
    has_endocardial_layer = endocardial_layer
    # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
    global is_healthy
    is_healthy = healthy_val
    global experiment_output
    experiment_output = 'ecg'
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_bsp
    global nb_leads
    global nb_limb_leads
    nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad   # 8 + lead progression (or 12)
    nb_limb_leads = 2
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    # Only used if is_helathy
    global gf_factor
    # gf_factor = 1.5
    gf_factor = 0.065 # 10/12/2021 - Taggart et al. (2000)
    # gf_factor = 0.067 # 10/12/2021 - Caldwell et al. (2009)
    global gn_factor
    # gn_factor = 0.7
    gn_factor = 0.048 # 10/12/2021 - Taggart et al. (2000)
    # gn_factor = 0.017 # 10/12/2021 - Caldwell et al. (2009)
    # print('gn_factor')
    # print(gn_factor)
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            # nlhsParam = 2 # t, e
            nlhsParam = 3 # 2022/01/18 # t, e
        else:
            # nlhsParam = 4 # f, t, n, e
            nlhsParam = 5 # 2022/01/18 # f, t, n, e
    else:
        if is_healthy:
            nlhsParam = 1 # t
        else:
            nlhsParam = 3 # f, t, n
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'
    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
                                   delimiter=',') - 1).astype(int))  # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
                                   delimiter=',') - 1).astype(int))  # rv endocardium triangles
    global lvnodes
    lvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_lvnodes.csv', delimiter=',') - 1).astype(int)               # lv endocardium triangles
    lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
    global rvnodes
    rvtri =(np.loadtxt('metaData/' + meshName + '/' + meshName + '_rvnodes.csv', delimiter=',') - 1).astype(int)                # rv endocardium triangles
    rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
    
    # Generate pseudo-Purkinje structure
    lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, lv_dense_indexes, rv_dense_indexes, nodesCobiveco = generatePurkinjeWithCobiveco(dataPath=dataPath, meshName=meshName,
        figPath=figPath)
    
    # print('YES')
    # # raise
    #
    # # Start of new Purkinje code # 2022/01/17
    # # lv_his_bundle_nodes = np.loadtxt(dataPath + meshName + '_lvhisbundle_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
    # # rv_his_bundle_nodes = np.loadtxt(dataPath + meshName + '_rvhisbundle_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
    # # lv_dense_nodes = np.loadtxt(dataPath + meshName + '_lvdense_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # # rv_dense_nodes = np.loadtxt(dataPath + meshName + '_rvdense_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # # lv_septalwall_nodes = np.loadtxt(dataPath + meshName + '_lvseptalwall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # # lv_freewallnavigation_nodes = np.loadtxt(dataPath + meshName + '_lvfreewallnavigationpoints_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # # lv_freewall_nodes = np.loadtxt(dataPath + meshName + '_lvfreewallextended_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/04/05
    # # # lv_interpapillary_freewall_nodes = np.loadtxt(dataPath + meshName + '_lvfreewall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # TODO 2022/05/05 - Adding redundancy through the LV PK ring
    # # # LV nodes that aren't freewall, septal or dense are considered to be paraseptal
    # # rv_freewall_nodes = np.loadtxt(dataPath + meshName + '_rvfreewall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # # rv_lowerthird_nodes = np.loadtxt(dataPath + meshName + '_rvlowerthird_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/04/06
    # # lv_histop_nodes = np.loadtxt(dataPath + meshName + '_lvhistop_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # # rv_histop_nodes = np.loadtxt(dataPath + meshName + '_rvhistop_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # # lv_hisbundleConnected_nodes = np.loadtxt(dataPath + meshName + '_lvhisbundleConnected_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # # rv_hisbundleConnected_nodes = np.loadtxt(dataPath + meshName + '_rvhisbundleConnected_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # # lv_apexnavigation_nodes = np.loadtxt(dataPath + meshName + '_lvapexnavigation_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # # # Project xyz points to nodes in the endocardial layer
    # # if True:
    # #     lv_his_bundle_indexes = np.zeros((lv_his_bundle_nodes.shape[0])).astype(int)
    # #     for i in range(lv_his_bundle_nodes.shape[0]):
    # #         lv_his_bundle_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_his_bundle_indexes, axis=0, return_index=True)[1] # use the unique function without sorting the contents of the array
    # #     lv_his_bundle_indexes = lv_his_bundle_indexes[sorted(indexes)]
    # #
    # #     rv_his_bundle_indexes = np.zeros((rv_his_bundle_nodes.shape[0])).astype(int)
    # #     for i in range(rv_his_bundle_nodes.shape[0]):
    # #         rv_his_bundle_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(rv_his_bundle_indexes, axis=0, return_index=True)[1]
    # #     rv_his_bundle_indexes = rv_his_bundle_indexes[sorted(indexes)]
    # #
    # #     lv_dense_indexes = np.zeros((lv_dense_nodes.shape[0])).astype(int)
    # #     for i in range(lv_dense_nodes.shape[0]):
    # #         lv_dense_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_dense_indexes, axis=0, return_index=True)[1]
    # #     lv_dense_indexes = lv_dense_indexes[sorted(indexes)]
    # #
    # #     rv_dense_indexes = np.zeros((rv_dense_nodes.shape[0])).astype(int)
    # #     for i in range(rv_dense_nodes.shape[0]):
    # #         rv_dense_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(rv_dense_indexes, axis=0, return_index=True)[1]
    # #     rv_dense_indexes = rv_dense_indexes[sorted(indexes)]
    # #
    # #     lv_septalwall_indexes = np.zeros((lv_septalwall_nodes.shape[0])).astype(int)
    # #     for i in range(lv_septalwall_nodes.shape[0]):
    # #         lv_septalwall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_septalwall_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_septalwall_indexes, axis=0, return_index=True)[1]
    # #     lv_septalwall_indexes = lv_septalwall_indexes[sorted(indexes)]
    # #
    # #     lv_freewallnavigation_indexes = np.zeros((lv_freewallnavigation_nodes.shape[0])).astype(int)
    # #     for i in range(lv_freewallnavigation_nodes.shape[0]):
    # #         lv_freewallnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewallnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_freewallnavigation_indexes, axis=0, return_index=True)[1]
    # #     lv_freewallnavigation_indexes = lv_freewallnavigation_indexes[sorted(indexes)]
    # #
    # #     lv_freewall_indexes = np.zeros((lv_freewall_nodes.shape[0])).astype(int)
    # #     for i in range(lv_freewall_nodes.shape[0]):
    # #         lv_freewall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_freewall_indexes, axis=0, return_index=True)[1]
    # #     lv_freewall_indexes = lv_freewall_indexes[sorted(indexes)]
    # #
    # #     #TODO
    # #     # lv_interpapillary_freewall_indexes = np.zeros((lv_interpapillary_freewall_nodes.shape[0])).astype(int) # TODO 2022/05/05 - Adding redundancy through the LV PK ring
    # #     # for i in range(lv_interpapillary_freewall_nodes.shape[0]):
    # #     #     lv_interpapillary_freewall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_interpapillary_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     # indexes = np.unique(lv_interpapillary_freewall_indexes, axis=0, return_index=True)[1]
    # #     # lv_interpapillary_freewall_indexes = lv_interpapillary_freewall_indexes[sorted(indexes)]
    # #
    # #     rv_freewall_indexes = np.zeros((rv_freewall_nodes.shape[0])).astype(int)
    # #     for i in range(rv_freewall_nodes.shape[0]):
    # #         rv_freewall_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(rv_freewall_indexes, axis=0, return_index=True)[1]
    # #     rv_freewall_indexes = rv_freewall_indexes[sorted(indexes)]
    # #
    # #     rv_lowerthird_indexes = np.zeros((rv_lowerthird_nodes.shape[0])).astype(int)
    # #     for i in range(rv_lowerthird_nodes.shape[0]):
    # #         rv_lowerthird_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_lowerthird_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(rv_lowerthird_indexes, axis=0, return_index=True)[1]
    # #     rv_lowerthird_indexes = rv_lowerthird_indexes[sorted(indexes)]
    # #
    # #     lv_histop_indexes = np.zeros((lv_histop_nodes.shape[0])).astype(int)
    # #     for i in range(lv_histop_nodes.shape[0]):
    # #         lv_histop_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_histop_indexes, axis=0, return_index=True)[1]
    # #     lv_histop_indexes = lv_histop_indexes[sorted(indexes)]
    # #
    # #     rv_histop_indexes = np.zeros((rv_histop_nodes.shape[0])).astype(int)
    # #     for i in range(rv_histop_nodes.shape[0]):
    # #         rv_histop_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(rv_histop_indexes, axis=0, return_index=True)[1]
    # #     rv_histop_indexes = rv_histop_indexes[sorted(indexes)]
    # #
    # #     lv_hisbundleConnected_indexes = np.zeros((lv_hisbundleConnected_nodes.shape[0])).astype(int)
    # #     for i in range(lv_hisbundleConnected_nodes.shape[0]):
    # #         lv_hisbundleConnected_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    # #     lv_hisbundleConnected_indexes = lv_hisbundleConnected_indexes[sorted(indexes)]
    # #
    # #     rv_hisbundleConnected_indexes = np.zeros((rv_hisbundleConnected_nodes.shape[0])).astype(int)
    # #     for i in range(rv_hisbundleConnected_nodes.shape[0]):
    # #         rv_hisbundleConnected_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(rv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    # #     rv_hisbundleConnected_indexes = rv_hisbundleConnected_indexes[sorted(indexes)]
    # #
    # #     lv_apexnavigation_indexes = np.zeros((lv_apexnavigation_nodes.shape[0])).astype(int)
    # #     for i in range(lv_apexnavigation_nodes.shape[0]):
    # #         lv_apexnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_apexnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    # #     indexes = np.unique(lv_apexnavigation_indexes, axis=0, return_index=True)[1]
    # #     lv_apexnavigation_indexes = lv_apexnavigation_indexes[sorted(indexes)]
    # #
    # # # Set LV endocardial edges aside
    # # lvnodesXYZ = nodesXYZ[lvnodes, :]
    # # lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    # # lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    # # lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    # # lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
    # # lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    # # aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    # # for i in range(0, len(lvunfoldedEdges), 1):
    # #     aux[lvunfoldedEdges[i, 0]].append(i)
    # # lvneighbours = [np.array(n) for n in aux]
    # # aux = None  # Clear Memory
    # # # Calculate Djikstra distances from the LV connected (branching) part in the his-Bundle (meshes are in cm)
    # # lv_hisbundleConnected_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_hisbundleConnected_indexes]).astype(int)
    # # lvHisBundledistance_mat, lvHisBundlepath_mat = djikstra(lv_hisbundleConnected_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # # Calculate Djikstra distances from the LV freewall (meshes are in cm)
    # # lv_freewallnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_freewallnavigation_indexes]).astype(int)
    # # lvFreewalldistance_mat, lvFreewallpath_mat = djikstra(lv_freewallnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # # Calculate Djikstra distances from the paraseptalwall (meshes are in cm)
    # # lv_apexnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_apexnavigation_indexes]).astype(int)
    # # lvParaseptalwalldistance_mat, lvParaseptalwallpath_mat = djikstra(lv_apexnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/04/05
    # # # Calculate the offsets to the top of the LV his bundle for the LV his bundle connected nodes
    # # lv_histop_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_histop_indexes]).astype(int)
    # # lvHisBundledistance_offset = lvHisBundledistance_mat[lv_histop_ids[0], :] # offset to the top of the his-bundle
    # # # Calculate within the his-bundle for plotting purposes
    # # lvHis_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_his_bundle_indexes]).astype(int)
    # # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[lvnodes[lvHis_ids], :] - nodesXYZ[lvnodes[lv_histop_ids], :], 2), axis=1))
    # # sorted_indexes = np.argsort(distances_to_top)
    # # sorted_lvHis_ids = lvHis_ids[sorted_indexes]
    # # lv_edges_his = np.array([np.array([lvnodes[sorted_lvHis_ids[i]], lvnodes[sorted_lvHis_ids[i+1]]]) for i in range(0, sorted_lvHis_ids.shape[0]-1, 1)])
    # # # Calculate the offset to the apex navigation point
    # # apex_navigation_reference_his_node = np.argmin(lvHisBundledistance_mat[lv_apexnavigation_ids[0], :])
    # # lvapexdistance_offset = lvHisBundledistance_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node] + lvHisBundledistance_offset[apex_navigation_reference_his_node] # offset of the apex node itself
    # # lvapexpath_offset = lvHisBundlepath_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node, :]
    # # # Make the paths on the lateral walls go to the free wall and then down the apex
    # # lvFreeWalldistance_offset = (lvFreewalldistance_mat[lv_apexnavigation_ids[0], :] + lvapexdistance_offset) # offset to the pass-through node in the apex + offset of the apex node itself
    # # lvFreeWallpath_offset = np.concatenate((lvFreewallpath_mat[lv_apexnavigation_ids[0], :, :], np.tile(lvapexpath_offset, (lvFreewallpath_mat.shape[1], 1))), axis=1)
    # # # Make the paths on the paraseptal walls go down the apex
    # # lvParaseptalWalldistance_offset = lvapexdistance_offset # offset of the apex node itself
    # # lvParaseptalWallpath_offset = np.tile(lvapexpath_offset, (lvParaseptalwallpath_mat.shape[1], 1))
    # # # Add the pieces of itinerary to the so called "free wall paths", mostly for plotting purposes
    # # lvFreewallpath_mat = np.concatenate((lvFreewallpath_mat, np.tile(lvFreeWallpath_offset, (lvFreewallpath_mat.shape[0], 1, 1))), axis=2)
    # # # Add the pieces of itinerary to the so called "paraseptal wall paths", mostly for plotting purposes
    # # lvParaseptalwallpath_mat = np.concatenate((lvParaseptalwallpath_mat, np.tile(lvParaseptalWallpath_offset, (lvParaseptalwallpath_mat.shape[0], 1, 1))), axis=2)
    # # # For each endocardial node, chose which path it should take: directly to the his bundle, or through the free wall
    # # lvHisBundledistance_vec_indexes = np.argmin(lvHisBundledistance_mat, axis=1)
    # # lvFreewalldistance_vec_indexes = np.argmin(lvFreewalldistance_mat, axis=1)
    # # lvParaseptaldistance_vec_indexes = np.argmin(lvParaseptalwalldistance_mat, axis=1) # if there is only one routing node in the apex, all indexes will be zero
    # # # Initialise data structures
    # # lv_PK_distance_mat = np.full((lvnodes.shape[0]), np.nan, np.float64)
    # # lv_PK_path_mat = np.full((lvnodes.shape[0], max(lvHisBundlepath_mat.shape[2], lvFreewallpath_mat.shape[2])), np.nan, np.int32)
    # # # Assign paths to each subgroup of root nodes
    # # for endo_node_i in range(lvnodes.shape[0]):
    # #     if lvnodes[endo_node_i] in lv_dense_indexes or lvnodes[endo_node_i] in lv_septalwall_indexes: # Septal-wall and dense region
    # #         lv_PK_distance_mat[endo_node_i] = lvHisBundledistance_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i]] + lvHisBundledistance_offset[lvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    # #         lv_PK_path_mat[endo_node_i, :lvHisBundlepath_mat.shape[2]] = lvHisBundlepath_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i], :]
    # #     elif lvnodes[endo_node_i] in lv_freewall_indexes: # FreeWall
    # #         # aux_0 = aux_0 + 1
    # #         lv_PK_distance_mat[endo_node_i] = lvFreewalldistance_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i]] + lvFreeWalldistance_offset[lvFreewalldistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    # #         lv_PK_path_mat[endo_node_i, :lvFreewallpath_mat.shape[2]] = lvFreewallpath_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i], :]
    # #     else: # Paraseptal
    # #         lv_PK_distance_mat[endo_node_i] = lvParaseptalwalldistance_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i]] + lvParaseptalWalldistance_offset #[lvParaseptaldistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    # #         lv_PK_path_mat[endo_node_i, :lvParaseptalwallpath_mat.shape[2]] = lvParaseptalwallpath_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i], :]
    # # # Take care of redundant paths
    # # # TODO 2022/05/05 - Adding redundancy through the LV PK ring - Postponed!
    # # # Time cost from each point in the left his-bundle to every point in the LV endocardium
    # # lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    # #
    # # # Set RV endocardial edges aside
    # # rvnodesXYZ = nodesXYZ[rvnodes, :]
    # # rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    # # rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    # # rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    # # rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
    # # rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    # # aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    # # for i in range(0, len(rvunfoldedEdges), 1):
    # #     aux[rvunfoldedEdges[i, 0]].append(i)
    # # rvneighbours = [np.array(n) for n in aux]
    # # aux = None  # Clear Memory
    # # # Calculate Djikstra distances from the RV connected (branching) part in the his-Bundle (meshes are in cm)
    # # rv_hisbundleConnected_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_hisbundleConnected_indexes]).astype(int)
    # # rvHisBundledistance_mat, rvHisBundlepath_mat = djikstra(rv_hisbundleConnected_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # 2022/01/10
    # # # Calculate the offsets to the top of the RV his bundle for the RV his bundle connected nodes
    # # rv_histop_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_histop_indexes]).astype(int)
    # # rvHisBundledistance_offset = rvHisBundledistance_mat[rv_histop_ids[0], :] # offset to the top of the his-bundle
    # # # Calculate within the his-bundle for plotting purposes
    # # rvHis_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_his_bundle_indexes]).astype(int)
    # # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[rvnodes[rvHis_ids], :] - nodesXYZ[rvnodes[rv_histop_ids], :], 2), axis=1))
    # # sorted_indexes = np.argsort(distances_to_top)
    # # sorted_rvHis_ids = rvHis_ids[sorted_indexes]
    # # rv_edges_his = np.array([np.array([rvnodes[sorted_rvHis_ids[i]], rvnodes[sorted_rvHis_ids[i+1]]]) for i in range(0, sorted_rvHis_ids.shape[0]-1, 1)])
    # # # Calculate Crossing Distances RV freewall (meshes are in cm)
    # # rvCrossingHisBundledistance_mat = np.sqrt(np.sum(np.power(rvnodesXYZ[:, np.newaxis, :] - rvnodesXYZ[rvHis_ids, :], 2), axis=2))
    # # rvCrossingHisBundlepath_mat = np.full((rvnodes.shape[0], rvHis_ids.shape[0], 2), np.nan, np.int32)
    # # for i in range(rvnodesXYZ.shape[0]):
    # #     for j in range(rvHis_ids.shape[0]):
    # #         rvCrossingHisBundlepath_mat[i, j, :] = np.array([i, rvHis_ids[j]])
    # # # Calculate the offsets to the top of the RV his bundle for the RV his bundle NOT-connected nodes
    # # rv_histopdistance_mat, rv_histoppath_mat = djikstra(rv_histop_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # offets to the top of the his 2022/01/10
    # # rvCrossingHisBundledistance_offset = np.squeeze(rv_histopdistance_mat[rvHis_ids, :])
    # # # For each endocardial node, chose which path it should take: directly to the his bundle following the endocardial wall or as a false tendon from the his bundle and crossing the cavity
    # # rvHisBundledistance_vec_indexes = np.argmin(rvHisBundledistance_mat, axis=1)
    # # rvCrossingHisBundledistance_vec_indexes = np.argmin(rvCrossingHisBundledistance_mat, axis=1)
    # # rv_edges_crossing = []
    # # rv_PK_distance_mat = np.full((rvnodes.shape[0]), np.nan, np.float64)
    # # rv_PK_path_mat = np.full((rvnodes.shape[0], max(rvHisBundlepath_mat.shape[2], rvCrossingHisBundlepath_mat.shape[2])), np.nan, np.int32)
    # # # rv_freewall_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_freewall_indexes]).astype(int)
    # # for endo_node_i in range(rvnodes.shape[0]):
    # #     # if rvnodes[endo_node_i] in rv_freewall_indexes:
    # #     if rvnodes[endo_node_i] in rv_lowerthird_indexes: # Restrain false tendons to only the lower 1/3 in the RV 2022/04/06
    # #         rv_PK_distance_mat[endo_node_i] = rvCrossingHisBundledistance_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i]] + rvCrossingHisBundledistance_offset[rvCrossingHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    # #         rv_PK_path_mat[endo_node_i, :rvCrossingHisBundlepath_mat.shape[2]] = rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :]
    # #         rv_edges_crossing.append(rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :])
    # #     else:
    # #         rv_PK_distance_mat[endo_node_i] = rvHisBundledistance_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i]] + rvHisBundledistance_offset[rvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    # #         rv_PK_path_mat[endo_node_i, :rvHisBundlepath_mat.shape[2]] = rvHisBundlepath_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i], :]
    # #         rv_edges_crossing.append(np.array([nan_value, nan_value], dtype=np.int32))
    # # # Time cost from each point in the left his-bundle to every point in the RV endocardium (for plotting purposes)
    # # rv_edges_crossing = np.asarray(rv_edges_crossing, dtype=np.int32)
    # # rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms
    # # # Save the times that each point would be activated by the Purkinje fibres
    # # atmap = np.zeros((nodesXYZ.shape[0]))
    # # atmap[lvnodes] = lv_PK_time_mat
    # # atmap[rvnodes] = rv_PK_time_mat
    # # with open(figPath + meshName + '_' + target_type + '.ensi.PKtimes', 'w') as f:
    # #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    # #     for i in range(0, len(atmap)):
    # #         f.write(str(atmap[i]) + '\n')
    #
    # Time cost from each point in the his-bundle to every point in the endocardium
    lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms
    # # Save the times that each point would be activated by the Purkinje fibres
    # atmap = np.zeros((nodesXYZ.shape[0]))
    # atmap[lvnodes] = lv_PK_time_mat
    # atmap[rvnodes] = rv_PK_time_mat
    # with open(figPath + meshName  + '_' +  target_type + '.ensi.PKtimes', 'w') as f:
    #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #     for i in range(0, len(atmap)):
    #         f.write(str(atmap[i]) + '\n')
    # # # End of new Purkinje code # 2022/01/17
    #
    #
    #
    # global tetraFibers
    # tetraFibers = np.loadtxt(dataPath + meshName + '_tetrahedronFibers.csv', delimiter=',')  # tetrahedron fiber directions
    #
    # tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    # global edgeVEC
    # edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :]  # edge vectors
    #
    # # Code to plot the root nodes used by Ana in her bidomain simulations for the paper in Frontiers 2019
    # rootNodesIndexes_true = None
    # rootNodeIndexes_true_file_path = dataPath + meshName + '_rootNodes.csv'
    # if os.path.isfile(rootNodeIndexes_true_file_path):
    #     rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_rootNodes.csv') - 1).astype(int)
    #     rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    #
    # global tetrahedrons
    tetrahedrons = (np.loadtxt(dataPath + meshName + '_tri.csv', delimiter=',') - 1).astype(int)
    elems = tetrahedrons
    # tetrahedronCenters = np.loadtxt(dataPath + meshName + '_tetrahedronCenters.csv', delimiter=',')
    # electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    # nb_bsp = electrodePositions.shape[0]
    #
    # create heart.ensi.geo file
    print ('Export mesh to ensight format')
    aux_elems = elems+1    # Make indexing Paraview and Matlab friendly
    save_cobiveco(figPath, meshName, nodesXYZ, aux_elems, nodesCobiveco) # Save cobiveco projection
    if not os.path.isfile(figPath + meshName+'.ensi.geo'):
        with open(figPath + meshName+'.ensi.geo', 'w') as f:
            f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
            for i in range(0, nodesXYZ.shape[0]):
                f.write(str(i+1)+'\n')
            for c in [0,1,2]:
                for i in range(0, nodesXYZ.shape[0]):
                    f.write(str(nodesXYZ[i,c])+'\n')
            # print('Write tetra4...')
            f.write('tetra4\n  '+str(len(aux_elems))+'\n')
            for i in range(0, len(aux_elems)):
                f.write('  '+str(i+1)+'\n')
            for i in range(0, len(aux_elems)):
                f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\t'+str(aux_elems[i,3])+'\n')
    # create heart.ensi.case file
    if os.path.isfile(figPath+meshName+'.ensi.case'):
        with open(figPath+meshName+'.ensi.case', 'a') as f:
            # f.write('scalar per element: 1	trueNodes_' + target_type + '	'+ meshName + '_' + target_type + '.ensi.trueNodes\n')
            f.write('scalar per node: 1	PK_times	' + meshName + '_' + target_type + '.ensi.PKtimes\n')
    else:
        with open(figPath+meshName+'.ensi.case', 'w') as f:
            f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
            f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
            # f.write('scalar per element: 1	trueNodes_' + target_type + '	'+ meshName + '_' + target_type + '.ensi.trueNodes\n')
            f.write('scalar per node: 1	PK_times	' + meshName + '_' + target_type + '.ensi.PKtimes\n') # 2022/01/11
    # resolution = 'newRV' # TODO: refactor this
    #
    # aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    # for i in range(0, tetrahedrons.shape[0], 1):
    #     aux[tetrahedrons[i, 0]].append(i)
    #     aux[tetrahedrons[i, 1]].append(i)
    #     aux[tetrahedrons[i, 2]].append(i)
    #     aux[tetrahedrons[i, 3]].append(i)
    # global elements
    # elements = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    #
    # # Precompute PseudoECG stuff - Calculate the tetrahedrons volumes
    # D = nodesXYZ[tetrahedrons[:, 3], :]  # RECYCLED
    # A = nodesXYZ[tetrahedrons[:, 0], :] - D  # RECYCLED
    # B = nodesXYZ[tetrahedrons[:, 1], :] - D  # RECYCLED
    # C = nodesXYZ[tetrahedrons[:, 2], :] - D  # RECYCLED
    # D = None  # Clear Memory
    #
    # global tVolumes
    # tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1),
    #                                        (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0])  # Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
    # tVolumes = tVolumes / np.sum(tVolumes)
    #
    # # Calculate the tetrahedron (temporal) voltage gradients
    # Mg = np.stack((A, B, C), axis=-1)
    # A = None  # Clear Memory
    # B = None  # Clear Memory
    # C = None  # Clear Memory
    #
    # # Calculate the gradients
    # global G_pseudo
    # G_pseudo = np.zeros(Mg.shape)
    # for i in range(Mg.shape[0]):
    #     G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
    # G_pseudo = np.moveaxis(G_pseudo, 1, 2)
    # Mg = None  # clear memory
    #
    # # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
    # r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
    #                            (tetrahedronCenters.shape[0],
    #                             tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1) - electrodePositions
    #
    # global d_r
    # d_r = np.moveaxis(np.multiply(
    #     np.moveaxis(r, [0, 1], [-1, -2]),
    #     np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0, -1)
    #
    # # Set endocardial edges aside # 2022/01/18 - Split endocardium into two parts, a fast, and a slower one, namely, a dense and a sparse sub-endocardial Purkinje network
    # global isEndocardial
    # global isDenseEndocardial # 2022/01/18
    # if has_endocardial_layer:
    #     isEndocardial=np.logical_or(np.all(np.isin(edges, lvnodes), axis=1), np.all(np.isin(edges, rvnodes), axis=1)) # 2022/01/18
    #     isDenseEndocardial=np.logical_or(np.all(np.isin(edges, lv_dense_indexes), axis=1), np.all(np.isin(edges, rv_dense_indexes), axis=1)) # 2022/01/18
    #     # print('isEndocardial.shape')
    #     # print(isEndocardial.shape)
    #     # print(np.sum(isEndocardial))
    #     # print(isDenseEndocardial.shape)
    #     # print(np.sum(isDenseEndocardial))
    #     # print(np.sum(np.logical_or(isEndocardial, isDenseEndocardial)))
    # else:
    #     isEndocardial = np.zeros((edges.shape[0])).astype(bool)
    #     isDenseEndocardial = np.zeros((edges.shape[0])).astype(bool) # 2022/01/18
    # # Build adjacentcies
    # global unfoldedEdges
    # unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    # aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    # for i in range(0, len(unfoldedEdges), 1):
    #     aux[unfoldedEdges[i, 0]].append(i)
    # global neighbours
    # # make neighbours Numba friendly
    # neighbours_aux = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    # m = 0
    # for n in neighbours_aux:
    #     m = max(m, len(n))
    # neighbours = np.full((len(neighbours_aux), m), np.nan,
    #                      np.int32)  # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    # for i in range(len(neighbours_aux)):
    #     n = neighbours_aux[i]
    #     neighbours[i, :n.shape[0]] = n
    # neighbours_aux = None
    #
    # # neighbours_original
    # aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    # for i in range(0, len(unfoldedEdges), 1):
    #     aux[unfoldedEdges[i, 0]].append(i)
    # global neighbours_original
    # neighbours_original = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    #
    # # Target result
    # global reference_lead_is_max # 22/05/03 - Normalise also the limb leads to give them relative importance
    # # Load and generate target data
    # if load_target:
    #     target_output = np.genfromtxt(target_data_path, delimiter=',')
    #     target_output = target_output - (target_output[:, 0:1]+target_output[:, -2:-1])/2 # align at zero # Re-added on 22/05/03 after it was worse without alingment
    #     reference_lead_max = np.amax(target_output, axis=1) # 22/05/03
    #     reference_lead_min = np.absolute(np.amin(target_output, axis=1)) # 22/05/03
    #     reference_lead_is_max_aux = reference_lead_max >= reference_lead_min
    #     reference_amplitudes = np.zeros(shape=(nb_leads), dtype=np.float64) # 2022/05/04
    #     reference_amplitudes[reference_lead_is_max_aux] = reference_lead_max[reference_lead_is_max_aux]
    #     reference_amplitudes[np.logical_not(reference_lead_is_max_aux)] = reference_lead_min[np.logical_not(reference_lead_is_max_aux)]
    #     target_output[:2, :] = target_output[:2, :] / np.mean(reference_amplitudes[:2]) # 22/05/03 TODO: Uncomment
    #     target_output[2:nb_leads, :] = target_output[2:nb_leads, :] / np.mean(reference_amplitudes[2:nb_leads]) # 22/05/03 TODO: Uncomment
    # else:
    #     reference_lead_is_max_aux = None # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
    #     print('TODO: add normalisation rules for synthetic data')
    #     # TODO: add modifications for new Purkinje strategy # 2022/01/17
    #     print("TODO: add modifications for new Purkinje strategy # 2022/01/17")
    #     raise()
    #
    # reference_lead_is_max = reference_lead_is_max_aux # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
    #
    #
    # load root nodes with the current resolution
    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_lv_activationIndexes_newRVRes.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_rv_activationIndexes_newRVRes.csv', delimiter=',') - 1).astype(int)
    
    for i in range(lvActivationIndexes.shape[0]):
        if lvActivationIndexes[i] not in lvnodes:
            lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
    for i in range(rvActivationIndexes.shape[0]):
        if rvActivationIndexes[i] not in rvnodes:
            rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
    # Save the pseudo-Purkinje networks considered during the inference for visulalisation and plotting purposes - LV
    lvedges_indexes = np.all(np.isin(edges, lvnodes), axis=1)
    lv_edges = edges[lvedges_indexes, :]
    lvPK_edges_indexes = np.zeros(lv_edges.shape[0], dtype=np.bool_)
    lv_root_to_PKpath_mat = lv_PK_path_mat[lvActnode_ids, :]
    for i in range(0, lv_edges.shape[0], 1):
        for j in range(0, lv_root_to_PKpath_mat.shape[0], 1):
            path = lvnodes[lv_root_to_PKpath_mat[j, :][lv_root_to_PKpath_mat[j, :] != nan_value]]
            for k in range(0, path.shape[0]-1, 1):
                new_edge = path[k:k+2]
                if np.all(np.isin(lv_edges[i, :], new_edge)):
                    lvPK_edges_indexes[i] = 1
                    break
    LV_PK_edges = lv_edges[lvPK_edges_indexes, :]
    # Save the available LV Purkinje network
    with open(figPath + meshName + '_available_LV_PKnetwork.vtk', 'w') as f:
        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
        for i in range(0, nodesXYZ.shape[0], 1):
            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
        f.write('LINES '+str(LV_PK_edges.shape[0])+' '+str(LV_PK_edges.shape[0]*3)+'\n')
        for i in range(0, LV_PK_edges.shape[0], 1):
            f.write('2 ' + str(LV_PK_edges[i, 0]) + ' ' + str(LV_PK_edges[i, 1]) + '\n')
        f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(LV_PK_edges.shape[0]) + '\n')
    # RV
    rvedges_indexes = np.all(np.isin(edges, rvnodes), axis=1)
    rv_edges = edges[rvedges_indexes, :]
    rvPK_edges_indexes = np.zeros(rv_edges.shape[0], dtype=np.bool_)
    rv_root_to_PKpath_mat = rv_PK_path_mat[rvActnode_ids, :]
    for i in range(0, rv_edges.shape[0], 1):
        for j in range(0, rv_root_to_PKpath_mat.shape[0], 1):
            path = rvnodes[rv_root_to_PKpath_mat[j, :][rv_root_to_PKpath_mat[j, :] != nan_value]]
            for k in range(0, path.shape[0]-1, 1):
                new_edge = path[k:k+2]
                if np.all(np.isin(rv_edges[i, :], new_edge)):
                    rvPK_edges_indexes[i] = 1
                    break
            
    RV_PK_edges = rv_edges[rvPK_edges_indexes, :]
    # Save the available RV Purkinje network
    with open(figPath + meshName + '_available_RV_PKnetwork.vtk', 'w') as f:
        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
        for i in range(0, nodesXYZ.shape[0], 1):
            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
        f.write('LINES '+str(RV_PK_edges.shape[0])+' '+str(RV_PK_edges.shape[0]*3)+'\n')
        for i in range(0, RV_PK_edges.shape[0], 1):
            f.write('2 ' + str(RV_PK_edges[i, 0]) + ' ' + str(RV_PK_edges[i, 1]) + '\n')
        f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(RV_PK_edges.shape[0]) + '\n')
    
    atmap = np.zeros((nodesXYZ.shape[0]))
    atmap[lvnodes] = lv_PK_time_mat
    atmap[rvnodes] = rv_PK_time_mat
    with open(figPath + meshName  + '_ecg_dtw.ensi.PKtimes', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(0, len(atmap)):
            f.write(str(atmap[i]) + '\n')
            
    # Save the available LV and RV root nodes
    with open(figPath + meshName + '_available_root_nodes.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(lvActivationIndexes)):
            f.write(str(nodesXYZ[lvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[lvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[lvActivationIndexes[i], 2]) + '\n')
        for i in range(0, len(rvActivationIndexes)):
            f.write(str(nodesXYZ[rvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[rvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[rvActivationIndexes[i], 2]) + '\n')




# EVALUATE EIKONAL ECGS USING INFERENCE RESULTS AS BASELINE 2022/05/19 - IDEAL FOR SENSITIVITY TESTING TO PARAMETER CHANGES
def evaluateEikonal(threadsNum_val, meshName, conduction_speeds, resolution_list, figPath, healthy_val,
                load_target, data_type, metric, target_data_path, endocardial_layer, is_clinical, experiment_path_tag,
                min_possible_consistency_count=0, max_possible_consistency_count = 100):
    purkinje_speed = 0.2 # cm/ms # the final purkinje network will use 0.19 but we don't need to use a slightly slower speed to
#allow for curved branches in the networks, because the times are here computed on a jiggsaw-like coarse mesh which will already
#add a little of buffer time
    target_type = data_type+'_'+metric
    
    global has_endocardial_layer
    has_endocardial_layer = endocardial_layer
    # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
    global is_healthy
    is_healthy = healthy_val
    global experiment_output
    experiment_output = 'ecg'
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_bsp
    global nb_leads
    global nb_limb_leads
    nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad   # 8 + lead progression (or 12)
    nb_limb_leads = 2
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    # Only used if is_helathy
    global gf_factor
    # gf_factor = 1.5
    gf_factor = 0.065 # 10/12/2021 - Taggart et al. (2000)
    # gf_factor = 0.067 # 10/12/2021 - Caldwell et al. (2009)
    global gn_factor
    # gn_factor = 0.7
    gn_factor = 0.048 # 10/12/2021 - Taggart et al. (2000)
    # gn_factor = 0.017 # 10/12/2021 - Caldwell et al. (2009)
    print('gn_factor')
    print(gn_factor)
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            # nlhsParam = 2 # t, e
            nlhsParam = 3 # 2022/01/18 # t, e
        else:
            # nlhsParam = 4 # f, t, n, e
            nlhsParam = 5 # 2022/01/18 # f, t, n, e
    else:
        if is_healthy:
            nlhsParam = 1 # t
        else:
            nlhsParam = 3 # f, t, n
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'
    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_edges.csv', delimiter=',') - 1).astype(int)
    # lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
    #                                delimiter=',') - 1).astype(int))  # lv endocardium triangles
    # rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
    #                                delimiter=',') - 1).astype(int))  # rv endocardium triangles
    global lvnodes
    lvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_lvnodes.csv', delimiter=',') - 1).astype(int)               # lv endocardium triangles
    lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
    global rvnodes
    rvtri =(np.loadtxt('metaData/' + meshName + '/' + meshName + '_rvnodes.csv', delimiter=',') - 1).astype(int)                # rv endocardium triangles
    rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
    
    lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, lv_dense_indexes, rv_dense_indexes, nodesCobiveco = generatePurkinjeWithCobiveco(dataPath=dataPath, meshName=meshName,
        figPath=figPath)
    # lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    # rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms
    
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_tetrahedronFibers.csv', delimiter=',')  # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :]  # edge vectors
    
    # Code to plot the root nodes used by Ana in her bidomain simulations for the paper in Frontiers 2019
    rootNodesIndexes_true = None
    rootNodeIndexes_true_file_path = dataPath + meshName + '_rootNodes.csv'
    if os.path.isfile(rootNodeIndexes_true_file_path):
        rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_rootNodes.csv') - 1).astype(int)
        rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    
    global tetrahedrons
    tetrahedrons = (np.loadtxt(dataPath + meshName + '_tri.csv', delimiter=',') - 1).astype(int)
    elems = tetrahedrons
    tetrahedronCenters = np.loadtxt(dataPath + meshName + '_tetrahedronCenters.csv', delimiter=',')
    electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    nb_bsp = electrodePositions.shape[0]
    
    # create heart.ensi.geo file
    print ('Export mesh to ensight format')
    aux_elems = elems+1    # Make indexing Paraview and Matlab friendly
    with open(figPath + meshName+'.ensi.geo', 'w') as f:
        f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
        for i in range(0, nodesXYZ.shape[0]):
            f.write(str(i+1)+'\n')
        for c in [0,1,2]:
            for i in range(0, nodesXYZ.shape[0]):
                f.write(str(nodesXYZ[i,c])+'\n')
        # print('Write tetra4...')
        f.write('tetra4\n  '+str(len(aux_elems))+'\n')
        for i in range(0, len(aux_elems)):
            f.write('  '+str(i+1)+'\n')
        for i in range(0, len(aux_elems)):
            f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\t'+str(aux_elems[i,3])+'\n')
    # create heart.ensi.case file
    with open(figPath+meshName+'.ensi.case', 'w') as f:
        f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
        f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
        f.write('scalar per element: 1	trueNodes_' + target_type + '	'
                + meshName + '_' + target_type + '.ensi.trueNodes\n')
        f.write('scalar per node: 1	PK_times	' + meshName + '_' + target_type + '.ensi.PKtimes\n') # 2022/01/11
        resolution = 'newRV' # TODO: refactor this
  
        for consistency_i in range(min_possible_consistency_count, max_possible_consistency_count, 1):
        # if True:
            if is_clinical:
                previousResultsPath='metaData/Eikonal_Results/Clinical_'+ experiment_path_tag +data_type +'_'+resolution+'/' # TODO Inconsistent with the synthetic naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
                + "_" + str(consistency_i) + '_population.csv')
            else:
                previousResultsPath='metaData/Eikonal_Results/'+data_type+ experiment_path_tag +'_'+resolution+'/' # TODO Inconsistent with the clinical naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_"
                + str(conduction_speeds) + "_" + resolution + "_" + target_type + "_"
                + str(consistency_i) + '_population.csv')
        # previousResultsPath='metaData/Eikonal_Results/Clinical_'+data_type+'_'+resolution+'/'
        
            # fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
            #     + "_" + str(consistency_i) + '_population.csv')
            if os.path.isfile(fileName):
                f.write('scalar per element: 1	kMeans_centroids_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '.ensi.kMeans_centroids\n')
                f.write('scalar per element: 1	cummrootNodes_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '.ensi.cummrootNodes\n')
                f.write('scalar per node: 1	ATMap_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '.ensi.ATMap\n')
                f.write('scalar per node: 1	ATMap_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '_best.ensi.ATMap\n') # 2022/01/18

    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, tetrahedrons.shape[0], 1):
        aux[tetrahedrons[i, 0]].append(i)
        aux[tetrahedrons[i, 1]].append(i)
        aux[tetrahedrons[i, 2]].append(i)
        aux[tetrahedrons[i, 3]].append(i)
    global elements
    elements = [np.array(n) for n in aux]
    aux = None  # Clear Memory

    # Precompute PseudoECG stuff - Calculate the tetrahedrons volumes
    D = nodesXYZ[tetrahedrons[:, 3], :]  # RECYCLED
    A = nodesXYZ[tetrahedrons[:, 0], :] - D  # RECYCLED
    B = nodesXYZ[tetrahedrons[:, 1], :] - D  # RECYCLED
    C = nodesXYZ[tetrahedrons[:, 2], :] - D  # RECYCLED
    D = None  # Clear Memory

    global tVolumes
    tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1),
                                           (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0])  # Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
    tVolumes = tVolumes / np.sum(tVolumes)

    # Calculate the tetrahedron (temporal) voltage gradients
    Mg = np.stack((A, B, C), axis=-1)
    A = None  # Clear Memory
    B = None  # Clear Memory
    C = None  # Clear Memory

    # Calculate the gradients
    global G_pseudo
    G_pseudo = np.zeros(Mg.shape)
    for i in range(Mg.shape[0]):
        G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
    G_pseudo = np.moveaxis(G_pseudo, 1, 2)
    Mg = None  # clear memory

    # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
    r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                               (tetrahedronCenters.shape[0],
                                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1) - electrodePositions

    global d_r
    d_r = np.moveaxis(np.multiply(
        np.moveaxis(r, [0, 1], [-1, -2]),
        np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0, -1)

    # Set endocardial edges aside # 2022/01/18 - Split endocardium into two parts, a fast, and a slower one, namely, a dense and a sparse sub-endocardial Purkinje network
    global isEndocardial
    global isDenseEndocardial # 2022/01/18
    if has_endocardial_layer:
        isEndocardial=np.logical_or(np.all(np.isin(edges, lvnodes), axis=1), np.all(np.isin(edges, rvnodes), axis=1)) # 2022/01/18
        isDenseEndocardial=np.logical_or(np.all(np.isin(edges, lv_dense_indexes), axis=1), np.all(np.isin(edges, rv_dense_indexes), axis=1)) # 2022/01/18
        # print('isEndocardial.shape')
        # print(isEndocardial.shape)
        # print(np.sum(isEndocardial))
        # print(isDenseEndocardial.shape)
        # print(np.sum(isDenseEndocardial))
        # print(np.sum(np.logical_or(isEndocardial, isDenseEndocardial)))
    else:
        isEndocardial = np.zeros((edges.shape[0])).astype(bool)
        isDenseEndocardial = np.zeros((edges.shape[0])).astype(bool) # 2022/01/18
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan,
                         np.int32)  # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None  # Clear Memory

    # Target result
    global reference_lead_is_max # 22/05/03 - Normalise also the limb leads to give them relative importance
    # Load and generate target data
    if load_target:
        target_output = np.genfromtxt(target_data_path, delimiter=',')
        target_output = target_output - (target_output[:, 0:1]+target_output[:, -2:-1])/2 # align at zero # Re-added on 22/05/03 after it was worse without alingment
        reference_lead_max = np.amax(target_output, axis=1) # 22/05/03
        reference_lead_min = np.absolute(np.amin(target_output, axis=1)) # 22/05/03
        reference_lead_is_max_aux = reference_lead_max >= reference_lead_min
        reference_amplitudes = np.zeros(shape=(nb_leads), dtype=np.float64) # 2022/05/04
        reference_amplitudes[reference_lead_is_max_aux] = reference_lead_max[reference_lead_is_max_aux]
        reference_amplitudes[np.logical_not(reference_lead_is_max_aux)] = reference_lead_min[np.logical_not(reference_lead_is_max_aux)]
        target_output[:2, :] = target_output[:2, :] / np.mean(reference_amplitudes[:2]) # 22/05/03 TODO: Uncomment
        target_output[2:nb_leads, :] = target_output[2:nb_leads, :] / np.mean(reference_amplitudes[2:nb_leads]) # 22/05/03 TODO: Uncomment
    else:
        reference_lead_is_max_aux = None # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
        print('TODO: add normalisation rules for synthetic data')
        # TODO: add modifications for new Purkinje strategy # 2022/01/17
        print("TODO: add modifications for new Purkinje strategy # 2022/01/17")
        # lv_rootNodesIndexes_true = rootNodesIndexes_true[np.isin(lvnodes, rootNodesIndexes_true)]
        # lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_rootNodesIndexes_true]).astype(int)
        # rv_rootNodesIndexes_true = rootNodesIndexes_true[np.isin(rvnodes, rootNodesIndexes_true)]
        # rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_rootNodesIndexes_true]).astype(int)
        # print(lv_rootNodesIndexes_true)
        # print(rv_rootNodesIndexes_true)
        # print(rootNodesIndexes_true)
        # print(lvActnode_ids)
        # print(rvActnode_ids)
        # # lv_rootNodesTimes_true = lvHisBundletimes[lvActnode_ids]
        # lv_rootNodesTimes_true = lv_PK_time_mat[lvActnode_ids]
        # # rv_rootNodesTimes_true = rvHisBundletimes[rvActnode_ids]
        # rv_rootNodesTimes_true = rv_PK_time_mat[rvActnode_ids]
        # print(lv_rootNodesTimes_true)
        # print(rv_rootNodesTimes_true)
        # print('revise this section before using this option, 30/11/2021')
        # raise()
        # rootNodesTimes_true = np.concatenate((lv_rootNodesTimes_true, rv_rootNodesTimes_true), axis=0)
        # target_output = eikonal_ecg(np.array([conduction_speeds])/1000., rootNodesIndexes_true, rootNodesTimes_true)[0, :nb_leads, :]
        # lead_size = np.sum(np.logical_not(np.isnan(target_output[0, :])))
        # target_output = target_output[:, :lead_size]
    
    reference_lead_is_max = reference_lead_is_max_aux # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead


    aggregation = pandas.DataFrame([], columns=['consistency_i', 'resolution', 'pred_roots',
        'rootNodes', 'roots_count', 'corr_mat', 'discrepancies'])
    # Iterate consistencies
    coeficients_array = []
    for consistency_i in range(min_possible_consistency_count, max_possible_consistency_count, 1):
        # print('consistency_i ' + str(consistency_i))
        # Count resolutions available
        resolution_count = 0
        aux_resolution_list = []
        for resolution_i in range(len(resolution_list)):
            resolution = resolution_list[resolution_i]
            if is_clinical:
                previousResultsPath='metaData/Eikonal_Results/Clinical_'+ experiment_path_tag +data_type +'_'+resolution+'/' # TODO Inconsistent with the synthetic naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
                + "_" + str(consistency_i) + '_population.csv')
            else:
                previousResultsPath='metaData/Eikonal_Results/'+data_type + experiment_path_tag +'_'+resolution+'/' # TODO Inconsistent with the clinical naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_"
                + str(conduction_speeds) + "_" + resolution + "_" + target_type + "_"
                + str(consistency_i) + '_population.csv')
            
            
            # Check if the file exists
            if os.path.isfile(fileName):
                resolution_count = resolution_count + 1
                aux_resolution_list.append(resolution)
        # print('resolution_count ' + str(resolution_count))
        if resolution_count > 0:
            # Create figure
            
            # Iterate resolutions
            for resolution_i in range(len(aux_resolution_list)):
                resolution = aux_resolution_list[resolution_i]
                if is_clinical:
                    previousResultsPath='metaData/Eikonal_Results/Clinical_'+ experiment_path_tag +data_type +'_'+resolution+'/' # TODO Inconsistent with the synthetic naming
                    fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
                    + "_" + str(consistency_i) + '_population.csv')
                else:
                    previousResultsPath='metaData/Eikonal_Results/'+data_type+ experiment_path_tag +'_'+resolution+'/' # TODO Inconsistent with the clinical naming
                    fileName = (previousResultsPath + meshName + "_" + resolution + "_"
                    + str(conduction_speeds) + "_" + resolution + "_" + target_type + "_"
                    + str(consistency_i) + '_population.csv')
                
                # if not os.path.isfile(fileName):
                #     print('NOT')
                #     print(fileName)
                if os.path.isfile(fileName):
                    print(fileName)
                    # load root nodes with the current resolution
                    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_lv_activationIndexes_' + resolution + 'Res.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
                    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_rv_activationIndexes_' + resolution + 'Res.csv', delimiter=',') - 1).astype(int)
                    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
                    
                    # PRECALCULATE DISTANCES BETWEEN NODES IN THE ENDOCARDIUM TO ALLOW K-MEANS TO BE COMPUTED ON THE RESULTS FROM THE INFERENCE AND ALSO
                    #TO ALLOW COMPUTING THE ACTIVATION TIMES AT THE ROOT NODES # MOVED UP ON THE 29/11/2021
                    # Project canditate root nodes to the endocardial layer: As from 2021 the root nodes are always generated at the endocardium, but this is still a good security measure
                    for i in range(lvActivationIndexes.shape[0]):
                        if lvActivationIndexes[i] not in lvnodes:
                            lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
                    lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
                    for i in range(rvActivationIndexes.shape[0]):
                        if rvActivationIndexes[i] not in rvnodes:
                            rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
                    rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
                    # Time cost from each root node
                    # rootNodeActivationTimes = np.around(np.concatenate((lv_PK_time_mat[lvActnode_ids], rv_PK_time_mat[rvActnode_ids]), axis=0), decimals=4) # 02/12/2021 Tested how many decimals were needed to get the
                    # Save the pseudo-Purkinje networks considered during the inference for visulalisation and plotting purposes - LV
                    lvedges_indexes = np.all(np.isin(edges, lvnodes), axis=1)
                    lv_edges = edges[lvedges_indexes, :]
                    lvPK_edges_indexes = np.zeros(lv_edges.shape[0], dtype=np.bool_)
                    lv_root_to_PKpath_mat = lv_PK_path_mat[lvActnode_ids, :]
                    for i in range(0, lv_edges.shape[0], 1):
                        for j in range(0, lv_root_to_PKpath_mat.shape[0], 1):
                            if np.all(np.isin(lv_edges[i, :], lvnodes[lv_root_to_PKpath_mat[j, :][lv_root_to_PKpath_mat[j, :] != nan_value]])):
                                lvPK_edges_indexes[i] = 1
                                break
                    LV_PK_edges = lv_edges[lvPK_edges_indexes, :]
                    # LV_PK_edges = np.concatenate((LV_PK_edges, lv_edges_his), axis=0)
                    # RV
                    rvedges_indexes = np.all(np.isin(edges, rvnodes), axis=1)
                    rv_edges = edges[rvedges_indexes, :]
                    rvPK_edges_indexes = np.zeros(rv_edges.shape[0], dtype=np.bool_)
                    rv_root_to_PKpath_mat = rv_PK_path_mat[rvActnode_ids, :]
                    # rv_edges_crossing_to_roots = rv_edges_crossing[rvActnode_ids, :]
                    # rv_edges_crossing_to_roots = rv_edges_crossing_to_roots[np.logical_not(np.any(rv_edges_crossing_to_roots == nan_value, axis=1)), :]
                    # rv_edges_crossing_to_roots = rvnodes[rv_edges_crossing_to_roots]
                    for i in range(0, rv_edges.shape[0], 1):
                        for j in range(0, rv_root_to_PKpath_mat.shape[0], 1):
                            if np.all(np.isin(rv_edges[i, :], rvnodes[rv_root_to_PKpath_mat[j, :][rv_root_to_PKpath_mat[j, :] != nan_value]])):
                                rvPK_edges_indexes[i] = 1
                                break
                    RV_PK_edges = rv_edges[rvPK_edges_indexes, :]
                    # RV_PK_edges = np.concatenate((RV_PK_edges, rv_edges_his), axis=0)
                    # RV_PK_edges = np.concatenate((RV_PK_edges, rv_edges_crossing_to_roots), axis=0)
                    
                    # Start computations to produce the ECG Predictions figure
                    compilation_params = np.array([np.concatenate((np.full(nlhsParam+1, 0.09), np.ones(rootNodeActivationIndexes.shape).astype(int)))]) # Compile Numba
                    eikonal_ecg_2(compilation_params, rootNodeActivationIndexes, np.zeros(rootNodeActivationIndexes.shape)) # Compile Numba # 02/12/2021
                    # Load population of particles
                    population_original = np.loadtxt(fileName, delimiter=',')
                    discrepancies_original = np.loadtxt(fileName.replace('population', 'discrepancy'), delimiter=',')
                    
                    print('population_original.shape')
                    print(population_original.shape)
                    # Remove duplicates with identical speeds but different numerical values in them, since these are 'randomly' generated to terminate the inference,
                    population_original = np.around(population_original, decimals=3) # ... even from different but equivalent root node locations September 2021
                    population_original, unique_particles_index = np.unique(population_original, return_index=True, axis=0) # Non-repeated particles
                    discrepancies_original = discrepancies_original[unique_particles_index]
                    best_discrepancy_id = np.argmin(discrepancies_original)
                    reference_parameter_set = population_original[best_discrepancy_id, :]
                    
                    fig_hist, axs_hist = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(30, 12), sharey='row')
                    fig, axs = plt.subplots(nrows=4, ncols=nb_leads, constrained_layout=True, figsize=(40, 24), sharey='row')
                    mult_factor = np.linspace(0.5, 2., threadsNum-1)
                    speed_names = ['Purkinje', 'sheet', 'DenseEndo', 'SparseEndo']
                    for parameter_i in range(4):
                        population = np.zeros((threadsNum, reference_parameter_set.shape[0]+1))
                        population[:, 0] = purkinje_speed
                        population[:, 1:] = np.transpose(np.reshape(np.repeat(reference_parameter_set, threadsNum, axis=0), (reference_parameter_set.shape[0], threadsNum)))
                        if parameter_i > 0:
                            new_speeds = mult_factor*reference_parameter_set[parameter_i-1]
                        else:
                            new_speeds = np.linspace(0.1, 2., threadsNum-1)
                        population[1:, parameter_i] = new_speeds
                        
                        # axs_hist[0, parameter_i].set_title(speed_names[parameter_i])
                        if parameter_i == 0:
                            pk_speed = np.amax(new_speeds) # time is in ms
                        elif parameter_i == 1:
                            pk_speed = 0.2 # time is in ms
                        elif parameter_i == 2:
                            pk_speed = 0.3 # time is in ms
                        elif parameter_i == 3:
                            pk_speed = 0.4 # time is in ms
                        axs_hist[0, parameter_i].set_title(pk_speed)
                        lv_PK_time_mat = lv_PK_distance_mat/pk_speed # time is in ms
                        rv_PK_time_mat = rv_PK_distance_mat/pk_speed # time is in ms
                        
                        axs_hist[1, parameter_i].set_title(speed_names[parameter_i])
                        print('rootNodeActivationTimes - NOT USED IN THIS INFERENCE')
                        not_used_pk_times_aux = np.around(np.concatenate((lv_PK_time_mat[lvActnode_ids], rv_PK_time_mat[rvActnode_ids]), axis=0), decimals=4)
                        not_used_pk_times_aux = not_used_pk_times_aux - np.amin(not_used_pk_times_aux)
                        print(not_used_pk_times_aux)
                        axs_hist[0, parameter_i].hist(not_used_pk_times_aux)
                        rootNodeDistances = np.around(np.concatenate((lv_PK_distance_mat[lvActnode_ids], rv_PK_distance_mat[rvActnode_ids]), axis=0), decimals=4)
                        prediction_list = eikonal_ecg_2(population, rootNodeActivationIndexes, rootNodeDistances)
                        # Compute correlation coefficients to all ECGs in the unique population
                        corr_mat = np.zeros((prediction_list.shape[0]-1, nb_leads)) # TODO - Uncomment 2022/05/09
                        reference_ecg = prediction_list[0, :, :]
                        reference_ecg = reference_ecg[:, np.logical_not(np.isnan(reference_ecg[0, :]))]
                        for j in range(1, prediction_list.shape[0]):
                            not_nan_size = np.sum(np.logical_not(np.isnan(prediction_list[j, 0, :])))
                            for i in range(nb_leads):
                                # prediction_lead = prediction_list[j, i, :]
                                prediction_lead = prediction_list[j, i, :not_nan_size]
                                if prediction_lead.shape[0] >= reference_ecg.shape[1]:
                                    a = prediction_lead
                                    b = reference_ecg[i, :]
                                else:
                                    a = reference_ecg[i, :]
                                    b = prediction_lead
                                b_aux = np.zeros(a.shape)
                                b_aux[:b.shape[0]] = b
                                b_aux[b.shape[0]:] = b[-1]
                                b = b_aux
                                new_corr_value = np.corrcoef(a, b)[0,1]
                                corr_mat[j-1, i] = new_corr_value
                        corr_list = np.mean(corr_mat, axis=1, dtype=np.float64) # cannot use a more sofisticated mean (e.g. geometric mean) because of negative and zero values
                        axs_hist[1, parameter_i].plot(new_speeds, corr_list)
                        
    
                        
                        # # Simulate the best ATMap solution from the inferred population
                        # inference_atmap = eikonal_atm(population[best_id, :], rootNodeActivationIndexes, rootNodeActivationTimes) # 2022/01/18 - Test the plot of the the best CC instead of the kmeans particle.
                        # with open(figPath + meshName + '_' + target_type + '_' + resolution + '_' + str(consistency_i) + '_best.ensi.ATMap', 'w') as f:
                        #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
                        #     for i in range(0, len(inference_atmap)):
                        #         f.write(str(inference_atmap[i]) + '\n')
                        
                        
                        # Calculate figure ms width
                        max_count = np.sum(np.logical_not(np.isnan(target_output[0, :])))
                        for i in range(prediction_list.shape[0]):
                            max_count = max(max_count, np.sum(np.logical_not(np.isnan(prediction_list[i, 0, :]))))
                        
                        # Plot into figure with Simulation and Target and Inference ECGs
                        for i in range(nb_leads):
                            axs_res = axs[parameter_i, i]
                            axs_res.plot(target_output[i, :], 'g-', label='Clinical', linewidth=2)
                            axs_res.plot(target_output[i, :], 'k-', label='Reference', linewidth=2)
                            axs_res.plot(target_output[i, :], 'r-', label='<'+speed_names[parameter_i], linewidth=2)
                            axs_res.plot(target_output[i, :], 'b-', label='>'+speed_names[parameter_i], linewidth=2)

                            for j in range(1, prediction_list.shape[0]):
                                prediction_lead = prediction_list[j, i, :]
                                lead_size = np.sum(np.logical_not(np.isnan(prediction_list[j, i, :])))
                                prediction_lead = prediction_lead[:lead_size]
                                if new_speeds[j-1] <= population[0, parameter_i]:
                                    axs_res.plot(prediction_lead, 'b-', linewidth=.3) # September 2021
                                else:
                                    axs_res.plot(prediction_lead, 'r-', linewidth=.3) # September 2021
                            axs_res.plot(prediction_lead, 'm-', linewidth=2.) # largest value
                            prediction_lead = prediction_list[j, i, :]
                            lead_size = np.sum(np.logical_not(np.isnan(prediction_list[1, i, :])))
                            prediction_lead = prediction_lead[:lead_size]
                            axs_res.plot(prediction_lead, 'c-', linewidth=2.) # smallest value
                            axs_res.plot(target_output[i, :], 'g-', linewidth=2)
                            axs_res.plot(reference_ecg[i, :], 'k-', linewidth=2)
                            # axs_res.plot(target_output[i, :], 'k-', linewidth=1.2)
                            # axs_res.plot(inference_ecg[i, :], 'r-', linewidth=1.2) # 2021/12/15 - TODO Uncomment
                            # axs_res.plot(best_ecg[i, :], 'r-', linewidth=1.2) # 2021/12/15 - TODO Comment
                            # axs_res.plot(best_ecg[i, :], color='purple', linestyle='-', linewidth=1.2) # 2021/12/15 - TODO Uncomment
                            # decorate figure
                            axs_res.xaxis.set_major_locator(MultipleLocator(40))
                            axs_res.xaxis.set_minor_locator(MultipleLocator(20))
                            axs_res.yaxis.set_major_locator(MultipleLocator(1))
                            #axs[0, i].yaxis.set_minor_locator(MultipleLocator(1))
                            axs_res.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                            axs_res.xaxis.grid(True, which='major')
                            axs_res.xaxis.grid(True, which='minor')
                            axs_res.yaxis.grid(True, which='major')
                            #axs[0, i].yaxis.grid(True, which='minor')
                            for tick in axs_res.xaxis.get_major_ticks():
                                tick.label.set_fontsize(14)
                            for tick in axs_res.yaxis.get_major_ticks():
                                tick.label.set_fontsize(14)
                        
                        axs_res.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    
    
                        max_count += 5
                        axs_res.set_xlim(0, max_count)
                    
                    figName = (figPath + meshName + '_speeds_evaluation_'+str(consistency_i)+'.png')
                    plt.savefig(figName, dpi=400)
                    plt.show()
    print('double Done')
    print('Done')


# PLOT THE PREDICTED ECGS FROM THE INFERENCE PROCESS 23/12/2020 - further addapted to clinical data on the 2021/05/24
def compareECGfigure(threadsNum_val, meshName, conduction_speeds, resolution_list, figPath, healthy_val,
                load_target, data_type, metric, target_data_path, endocardial_layer, is_clinical, experiment_path_tag,
                min_possible_consistency_count=0, max_possible_consistency_count = 100):
    purkinje_speed = 0.4 # cm/ms # the final purkinje network will use 0.19 but we don't need to use a slightly slower speed to
#allow for curved branches in the networks, because the times are here computed on a jiggsaw-like coarse mesh which will already
#add a little of buffer time
    target_type = data_type+'_'+metric
    #TODO
    # Fix this function, not working on 23/12/2020
    global has_endocardial_layer
    has_endocardial_layer = endocardial_layer
    # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
    global is_healthy
    is_healthy = healthy_val
    global experiment_output
    experiment_output = 'ecg'
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    global nb_bsp
    global nb_leads
    global nb_limb_leads
    nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad   # 8 + lead progression (or 12)
    nb_limb_leads = 2
    frequency = 1000  # Hz
    freq_cut = 150  # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2)  # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    # Only used if is_helathy
    global gf_factor
    # gf_factor = 1.5
    gf_factor = 0.065 # 10/12/2021 - Taggart et al. (2000)
    # gf_factor = 0.067 # 10/12/2021 - Caldwell et al. (2009)
    global gn_factor
    # gn_factor = 0.7
    gn_factor = 0.048 # 10/12/2021 - Taggart et al. (2000)
    # gn_factor = 0.017 # 10/12/2021 - Caldwell et al. (2009)
    print('gn_factor')
    print(gn_factor)
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            # nlhsParam = 2 # t, e
            nlhsParam = 3 # 2022/01/18 # t, e
        else:
            # nlhsParam = 4 # f, t, n, e
            nlhsParam = 5 # 2022/01/18 # f, t, n, e
    else:
        if is_healthy:
            nlhsParam = 1 # t
        else:
            nlhsParam = 3 # f, t, n
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'
    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv',
                                   delimiter=',') - 1).astype(int))  # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv',
                                   delimiter=',') - 1).astype(int))  # rv endocardium triangles
    global lvnodes
    lvtri = (np.loadtxt('metaData/' + meshName + '/' + meshName + '_lvnodes.csv', delimiter=',') - 1).astype(int)               # lv endocardium triangles
    lvnodes = np.unique(lvtri).astype(int)  # lv endocardium nodes
    global rvnodes
    rvtri =(np.loadtxt('metaData/' + meshName + '/' + meshName + '_rvnodes.csv', delimiter=',') - 1).astype(int)                # rv endocardium triangles
    rvnodes = np.unique(rvtri).astype(int)  # rv endocardium nodes
    
    
    lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, lv_dense_indexes, rv_dense_indexes, nodesCobiveco = generatePurkinjeWithCobiveco(dataPath=dataPath, meshName=meshName,
        figPath=figPath)
    lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms
    
    # #TODO DELETE THIS SECTION OF CODE: # 2022/04/07
    # # # create heart.ensi.geo file
    # # elems = (np.loadtxt(dataPath + meshName + '_tri.csv', delimiter=',') - 1).astype(int)
    # # # elems = tetrahedrons
    # # print ('Export mesh to ensight format')
    # # aux_elems = elems+1    # Make indexing Paraview and Matlab friendly
    # # with open(figPath + meshName+'.ensi.geo', 'w') as f:
    # #     f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
    # #     for i in range(0, nodesXYZ.shape[0]):
    # #         f.write(str(i+1)+'\n')
    # #     for c in [0,1,2]:
    # #         for i in range(0, nodesXYZ.shape[0]):
    # #             f.write(str(nodesXYZ[i,c])+'\n')
    # #     # print('Write tetra4...')
    # #     f.write('tetra4\n  '+str(len(aux_elems))+'\n')
    # #     for i in range(0, len(aux_elems)):
    # #         f.write('  '+str(i+1)+'\n')
    # #     for i in range(0, len(aux_elems)):
    # #         f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\t'+str(aux_elems[i,3])+'\n')
    # # # create heart.ensi.case file
    # # with open(figPath+meshName+'.ensi.case', 'w') as f:
    # #     f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
    # #     f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
    # #     f.write('scalar per element: 1	trueNodes_' + target_type + '	'
    # #             + meshName + '_' + target_type + '.ensi.trueNodes\n')
    # #     f.write('scalar per node: 1	PK_times	' + meshName + '_' + target_type + '.ensi.PKtimes\n') # 2022/01/11
    # #     resolution = 'newRV' # TODO: refactor this
    # # raise
    # #TODO DELETE THIS SECTION OF CODE
    #
    # # Start of new Purkinje code # 2022/01/17
    # lv_his_bundle_nodes = np.loadtxt(dataPath + meshName + '_lvhisbundle_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
    # rv_his_bundle_nodes = np.loadtxt(dataPath + meshName + '_rvhisbundle_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
    # lv_dense_nodes = np.loadtxt(dataPath + meshName + '_lvdense_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # rv_dense_nodes = np.loadtxt(dataPath + meshName + '_rvdense_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_septalwall_nodes = np.loadtxt(dataPath + meshName + '_lvseptalwall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_freewallnavigation_nodes = np.loadtxt(dataPath + meshName + '_lvfreewallnavigationpoints_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_freewall_nodes = np.loadtxt(dataPath + meshName + '_lvfreewallextended_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/04/05
    # # lv_interpapillary_freewall_nodes = np.loadtxt(dataPath + meshName + '_lvfreewall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # TODO 2022/05/05 - Adding redundancy through the LV PK ring
    # # LV nodes that aren't freewall, septal or dense are considered to be paraseptal
    # rv_freewall_nodes = np.loadtxt(dataPath + meshName + '_rvfreewall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # rv_lowerthird_nodes = np.loadtxt(dataPath + meshName + '_rvlowerthird_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/04/06
    # lv_histop_nodes = np.loadtxt(dataPath + meshName + '_lvhistop_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # rv_histop_nodes = np.loadtxt(dataPath + meshName + '_rvhistop_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # lv_hisbundleConnected_nodes = np.loadtxt(dataPath + meshName + '_lvhisbundleConnected_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # rv_hisbundleConnected_nodes = np.loadtxt(dataPath + meshName + '_rvhisbundleConnected_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_apexnavigation_nodes = np.loadtxt(dataPath + meshName + '_lvapexnavigation_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # # Project xyz points to nodes in the endocardial layer
    # if True:
    #     lv_his_bundle_indexes = np.zeros((lv_his_bundle_nodes.shape[0])).astype(int)
    #     for i in range(lv_his_bundle_nodes.shape[0]):
    #         lv_his_bundle_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_his_bundle_indexes, axis=0, return_index=True)[1] # use the unique function without sorting the contents of the array
    #     lv_his_bundle_indexes = lv_his_bundle_indexes[sorted(indexes)]
    #
    #     rv_his_bundle_indexes = np.zeros((rv_his_bundle_nodes.shape[0])).astype(int)
    #     for i in range(rv_his_bundle_nodes.shape[0]):
    #         rv_his_bundle_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_his_bundle_indexes, axis=0, return_index=True)[1]
    #     rv_his_bundle_indexes = rv_his_bundle_indexes[sorted(indexes)]
    #
    #     lv_dense_indexes = np.zeros((lv_dense_nodes.shape[0])).astype(int)
    #     for i in range(lv_dense_nodes.shape[0]):
    #         lv_dense_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_dense_indexes, axis=0, return_index=True)[1]
    #     lv_dense_indexes = lv_dense_indexes[sorted(indexes)]
    #
    #     rv_dense_indexes = np.zeros((rv_dense_nodes.shape[0])).astype(int)
    #     for i in range(rv_dense_nodes.shape[0]):
    #         rv_dense_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_dense_indexes, axis=0, return_index=True)[1]
    #     rv_dense_indexes = rv_dense_indexes[sorted(indexes)]
    #
    #     lv_septalwall_indexes = np.zeros((lv_septalwall_nodes.shape[0])).astype(int)
    #     for i in range(lv_septalwall_nodes.shape[0]):
    #         lv_septalwall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_septalwall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_septalwall_indexes, axis=0, return_index=True)[1]
    #     lv_septalwall_indexes = lv_septalwall_indexes[sorted(indexes)]
    #
    #     lv_freewallnavigation_indexes = np.zeros((lv_freewallnavigation_nodes.shape[0])).astype(int)
    #     for i in range(lv_freewallnavigation_nodes.shape[0]):
    #         lv_freewallnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewallnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_freewallnavigation_indexes, axis=0, return_index=True)[1]
    #     lv_freewallnavigation_indexes = lv_freewallnavigation_indexes[sorted(indexes)]
    #
    #     lv_freewall_indexes = np.zeros((lv_freewall_nodes.shape[0])).astype(int)
    #     for i in range(lv_freewall_nodes.shape[0]):
    #         lv_freewall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_freewall_indexes, axis=0, return_index=True)[1]
    #     lv_freewall_indexes = lv_freewall_indexes[sorted(indexes)]
    #
    #     #TODO
    #     # lv_interpapillary_freewall_indexes = np.zeros((lv_interpapillary_freewall_nodes.shape[0])).astype(int) # TODO 2022/05/05 - Adding redundancy through the LV PK ring
    #     # for i in range(lv_interpapillary_freewall_nodes.shape[0]):
    #     #     lv_interpapillary_freewall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_interpapillary_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     # indexes = np.unique(lv_interpapillary_freewall_indexes, axis=0, return_index=True)[1]
    #     # lv_interpapillary_freewall_indexes = lv_interpapillary_freewall_indexes[sorted(indexes)]
    #
    #     rv_freewall_indexes = np.zeros((rv_freewall_nodes.shape[0])).astype(int)
    #     for i in range(rv_freewall_nodes.shape[0]):
    #         rv_freewall_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_freewall_indexes, axis=0, return_index=True)[1]
    #     rv_freewall_indexes = rv_freewall_indexes[sorted(indexes)]
    #
    #     rv_lowerthird_indexes = np.zeros((rv_lowerthird_nodes.shape[0])).astype(int)
    #     for i in range(rv_lowerthird_nodes.shape[0]):
    #         rv_lowerthird_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_lowerthird_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_lowerthird_indexes, axis=0, return_index=True)[1]
    #     rv_lowerthird_indexes = rv_lowerthird_indexes[sorted(indexes)]
    #
    #     lv_histop_indexes = np.zeros((lv_histop_nodes.shape[0])).astype(int)
    #     for i in range(lv_histop_nodes.shape[0]):
    #         lv_histop_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_histop_indexes, axis=0, return_index=True)[1]
    #     lv_histop_indexes = lv_histop_indexes[sorted(indexes)]
    #
    #     rv_histop_indexes = np.zeros((rv_histop_nodes.shape[0])).astype(int)
    #     for i in range(rv_histop_nodes.shape[0]):
    #         rv_histop_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_histop_indexes, axis=0, return_index=True)[1]
    #     rv_histop_indexes = rv_histop_indexes[sorted(indexes)]
    #
    #     lv_hisbundleConnected_indexes = np.zeros((lv_hisbundleConnected_nodes.shape[0])).astype(int)
    #     for i in range(lv_hisbundleConnected_nodes.shape[0]):
    #         lv_hisbundleConnected_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    #     lv_hisbundleConnected_indexes = lv_hisbundleConnected_indexes[sorted(indexes)]
    #
    #     rv_hisbundleConnected_indexes = np.zeros((rv_hisbundleConnected_nodes.shape[0])).astype(int)
    #     for i in range(rv_hisbundleConnected_nodes.shape[0]):
    #         rv_hisbundleConnected_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    #     rv_hisbundleConnected_indexes = rv_hisbundleConnected_indexes[sorted(indexes)]
    #
    #     lv_apexnavigation_indexes = np.zeros((lv_apexnavigation_nodes.shape[0])).astype(int)
    #     for i in range(lv_apexnavigation_nodes.shape[0]):
    #         lv_apexnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_apexnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_apexnavigation_indexes, axis=0, return_index=True)[1]
    #     lv_apexnavigation_indexes = lv_apexnavigation_indexes[sorted(indexes)]
    #
    # # Set LV endocardial edges aside
    # lvnodesXYZ = nodesXYZ[lvnodes, :]
    # lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    # lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    # lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    # lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
    # lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    # aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    # for i in range(0, len(lvunfoldedEdges), 1):
    #     aux[lvunfoldedEdges[i, 0]].append(i)
    # lvneighbours = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    # # Calculate Djikstra distances from the LV connected (branching) part in the his-Bundle (meshes are in cm)
    # lv_hisbundleConnected_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_hisbundleConnected_indexes]).astype(int)
    # lvHisBundledistance_mat, lvHisBundlepath_mat = djikstra(lv_hisbundleConnected_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # Calculate Djikstra distances from the LV freewall (meshes are in cm)
    # lv_freewallnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_freewallnavigation_indexes]).astype(int)
    # lvFreewalldistance_mat, lvFreewallpath_mat = djikstra(lv_freewallnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # Calculate Djikstra distances from the paraseptalwall (meshes are in cm)
    # lv_apexnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_apexnavigation_indexes]).astype(int)
    # lvParaseptalwalldistance_mat, lvParaseptalwallpath_mat = djikstra(lv_apexnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/04/05
    # # Calculate the offsets to the top of the LV his bundle for the LV his bundle connected nodes
    # lv_histop_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_histop_indexes]).astype(int)
    # lvHisBundledistance_offset = lvHisBundledistance_mat[lv_histop_ids[0], :] # offset to the top of the his-bundle
    # # Calculate within the his-bundle for plotting purposes
    # lvHis_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_his_bundle_indexes]).astype(int)
    # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[lvnodes[lvHis_ids], :] - nodesXYZ[lvnodes[lv_histop_ids], :], 2), axis=1))
    # sorted_indexes = np.argsort(distances_to_top)
    # sorted_lvHis_ids = lvHis_ids[sorted_indexes]
    # lv_edges_his = np.array([np.array([lvnodes[sorted_lvHis_ids[i]], lvnodes[sorted_lvHis_ids[i+1]]]) for i in range(0, sorted_lvHis_ids.shape[0]-1, 1)])
    # # Calculate the offset to the apex navigation point
    # apex_navigation_reference_his_node = np.argmin(lvHisBundledistance_mat[lv_apexnavigation_ids[0], :])
    # lvapexdistance_offset = lvHisBundledistance_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node] + lvHisBundledistance_offset[apex_navigation_reference_his_node] # offset of the apex node itself
    # lvapexpath_offset = lvHisBundlepath_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node, :]
    # # Make the paths on the lateral walls go to the free wall and then down the apex
    # lvFreeWalldistance_offset = (lvFreewalldistance_mat[lv_apexnavigation_ids[0], :] + lvapexdistance_offset) # offset to the pass-through node in the apex + offset of the apex node itself
    # lvFreeWallpath_offset = np.concatenate((lvFreewallpath_mat[lv_apexnavigation_ids[0], :, :], np.tile(lvapexpath_offset, (lvFreewallpath_mat.shape[1], 1))), axis=1)
    # # Make the paths on the paraseptal walls go down the apex
    # lvParaseptalWalldistance_offset = lvapexdistance_offset # offset of the apex node itself
    # lvParaseptalWallpath_offset = np.tile(lvapexpath_offset, (lvParaseptalwallpath_mat.shape[1], 1))
    # # Add the pieces of itinerary to the so called "free wall paths", mostly for plotting purposes
    # lvFreewallpath_mat = np.concatenate((lvFreewallpath_mat, np.tile(lvFreeWallpath_offset, (lvFreewallpath_mat.shape[0], 1, 1))), axis=2)
    # # Add the pieces of itinerary to the so called "paraseptal wall paths", mostly for plotting purposes
    # lvParaseptalwallpath_mat = np.concatenate((lvParaseptalwallpath_mat, np.tile(lvParaseptalWallpath_offset, (lvParaseptalwallpath_mat.shape[0], 1, 1))), axis=2)
    # # For each endocardial node, chose which path it should take: directly to the his bundle, or through the free wall
    # lvHisBundledistance_vec_indexes = np.argmin(lvHisBundledistance_mat, axis=1)
    # lvFreewalldistance_vec_indexes = np.argmin(lvFreewalldistance_mat, axis=1)
    # lvParaseptaldistance_vec_indexes = np.argmin(lvParaseptalwalldistance_mat, axis=1) # if there is only one routing node in the apex, all indexes will be zero
    # # Initialise data structures
    # lv_PK_distance_mat = np.full((lvnodes.shape[0]), np.nan, np.float64)
    # lv_PK_path_mat = np.full((lvnodes.shape[0], max(lvHisBundlepath_mat.shape[2], lvFreewallpath_mat.shape[2])), np.nan, np.int32)
    # # Assign paths to each subgroup of root nodes
    # for endo_node_i in range(lvnodes.shape[0]):
    #     if lvnodes[endo_node_i] in lv_dense_indexes or lvnodes[endo_node_i] in lv_septalwall_indexes: # Septal-wall and dense region
    #         lv_PK_distance_mat[endo_node_i] = lvHisBundledistance_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i]] + lvHisBundledistance_offset[lvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvHisBundlepath_mat.shape[2]] = lvHisBundlepath_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i], :]
    #     elif lvnodes[endo_node_i] in lv_freewall_indexes: # FreeWall
    #         # aux_0 = aux_0 + 1
    #         lv_PK_distance_mat[endo_node_i] = lvFreewalldistance_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i]] + lvFreeWalldistance_offset[lvFreewalldistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvFreewallpath_mat.shape[2]] = lvFreewallpath_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i], :]
    #     else: # Paraseptal
    #         lv_PK_distance_mat[endo_node_i] = lvParaseptalwalldistance_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i]] + lvParaseptalWalldistance_offset #[lvParaseptaldistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvParaseptalwallpath_mat.shape[2]] = lvParaseptalwallpath_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i], :]
    # # Take care of redundant paths
    # # TODO 2022/05/05 - Adding redundancy through the LV PK ring - Postponed!
    # # Time cost from each point in the left his-bundle to every point in the LV endocardium
    # lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    #
    # # Set RV endocardial edges aside
    # rvnodesXYZ = nodesXYZ[rvnodes, :]
    # rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    # rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    # rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    # rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
    # rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    # aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    # for i in range(0, len(rvunfoldedEdges), 1):
    #     aux[rvunfoldedEdges[i, 0]].append(i)
    # rvneighbours = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    # # Calculate Djikstra distances from the RV connected (branching) part in the his-Bundle (meshes are in cm)
    # rv_hisbundleConnected_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_hisbundleConnected_indexes]).astype(int)
    # rvHisBundledistance_mat, rvHisBundlepath_mat = djikstra(rv_hisbundleConnected_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # 2022/01/10
    # # Calculate the offsets to the top of the RV his bundle for the RV his bundle connected nodes
    # rv_histop_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_histop_indexes]).astype(int)
    # rvHisBundledistance_offset = rvHisBundledistance_mat[rv_histop_ids[0], :] # offset to the top of the his-bundle
    # # Calculate within the his-bundle for plotting purposes
    # rvHis_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_his_bundle_indexes]).astype(int)
    # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[rvnodes[rvHis_ids], :] - nodesXYZ[rvnodes[rv_histop_ids], :], 2), axis=1))
    # sorted_indexes = np.argsort(distances_to_top)
    # sorted_rvHis_ids = rvHis_ids[sorted_indexes]
    # rv_edges_his = np.array([np.array([rvnodes[sorted_rvHis_ids[i]], rvnodes[sorted_rvHis_ids[i+1]]]) for i in range(0, sorted_rvHis_ids.shape[0]-1, 1)])
    # # Calculate Crossing Distances RV freewall (meshes are in cm)
    # rvCrossingHisBundledistance_mat = np.sqrt(np.sum(np.power(rvnodesXYZ[:, np.newaxis, :] - rvnodesXYZ[rvHis_ids, :], 2), axis=2))
    # rvCrossingHisBundlepath_mat = np.full((rvnodes.shape[0], rvHis_ids.shape[0], 2), np.nan, np.int32)
    # for i in range(rvnodesXYZ.shape[0]):
    #     for j in range(rvHis_ids.shape[0]):
    #         rvCrossingHisBundlepath_mat[i, j, :] = np.array([i, rvHis_ids[j]])
    # # Calculate the offsets to the top of the RV his bundle for the RV his bundle NOT-connected nodes
    # rv_histopdistance_mat, rv_histoppath_mat = djikstra(rv_histop_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # offets to the top of the his 2022/01/10
    # rvCrossingHisBundledistance_offset = np.squeeze(rv_histopdistance_mat[rvHis_ids, :])
    # # For each endocardial node, chose which path it should take: directly to the his bundle following the endocardial wall or as a false tendon from the his bundle and crossing the cavity
    # rvHisBundledistance_vec_indexes = np.argmin(rvHisBundledistance_mat, axis=1)
    # rvCrossingHisBundledistance_vec_indexes = np.argmin(rvCrossingHisBundledistance_mat, axis=1)
    # rv_edges_crossing = []
    # rv_PK_distance_mat = np.full((rvnodes.shape[0]), np.nan, np.float64)
    # rv_PK_path_mat = np.full((rvnodes.shape[0], max(rvHisBundlepath_mat.shape[2], rvCrossingHisBundlepath_mat.shape[2])), np.nan, np.int32)
    # # rv_freewall_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_freewall_indexes]).astype(int)
    # for endo_node_i in range(rvnodes.shape[0]):
    #     # if rvnodes[endo_node_i] in rv_freewall_indexes:
    #     if rvnodes[endo_node_i] in rv_lowerthird_indexes: # Restrain false tendons to only the lower 1/3 in the RV 2022/04/06
    #         rv_PK_distance_mat[endo_node_i] = rvCrossingHisBundledistance_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i]] + rvCrossingHisBundledistance_offset[rvCrossingHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         rv_PK_path_mat[endo_node_i, :rvCrossingHisBundlepath_mat.shape[2]] = rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :]
    #         rv_edges_crossing.append(rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :])
    #     else:
    #         rv_PK_distance_mat[endo_node_i] = rvHisBundledistance_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i]] + rvHisBundledistance_offset[rvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         rv_PK_path_mat[endo_node_i, :rvHisBundlepath_mat.shape[2]] = rvHisBundlepath_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i], :]
    #         rv_edges_crossing.append(np.array([nan_value, nan_value], dtype=np.int32))
    # # Time cost from each point in the left his-bundle to every point in the RV endocardium (for plotting purposes)
    # rv_edges_crossing = np.asarray(rv_edges_crossing, dtype=np.int32)
    # rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms
    # # Save the times that each point would be activated by the Purkinje fibres
    # atmap = np.zeros((nodesXYZ.shape[0]))
    # atmap[lvnodes] = lv_PK_time_mat
    # atmap[rvnodes] = rv_PK_time_mat
    # with open(figPath + meshName + '_' + target_type + '.ensi.PKtimes', 'w') as f:
    #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #     for i in range(0, len(atmap)):
    #         f.write(str(atmap[i]) + '\n')
    # # End of new Purkinje code # 2022/01/17
    
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_tetrahedronFibers.csv', delimiter=',')  # tetrahedron fiber directions

    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :]  # edge vectors
    
    # Code to plot the root nodes used by Ana in her bidomain simulations for the paper in Frontiers 2019
    rootNodesIndexes_true = None
    rootNodeIndexes_true_file_path = dataPath + meshName + '_rootNodes.csv'
    if os.path.isfile(rootNodeIndexes_true_file_path):
        rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_rootNodes.csv') - 1).astype(int)
        rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    
    global tetrahedrons
    tetrahedrons = (np.loadtxt(dataPath + meshName + '_tri.csv', delimiter=',') - 1).astype(int)
    elems = tetrahedrons
    tetrahedronCenters = np.loadtxt(dataPath + meshName + '_tetrahedronCenters.csv', delimiter=',')
    electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
    nb_bsp = electrodePositions.shape[0]
    
    # create heart.ensi.geo file
    print ('Export mesh to ensight format')
    aux_elems = elems+1    # Make indexing Paraview and Matlab friendly
    with open(figPath + meshName+'.ensi.geo', 'w') as f:
        f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
        for i in range(0, nodesXYZ.shape[0]):
            f.write(str(i+1)+'\n')
        for c in [0,1,2]:
            for i in range(0, nodesXYZ.shape[0]):
                f.write(str(nodesXYZ[i,c])+'\n')
        # print('Write tetra4...')
        f.write('tetra4\n  '+str(len(aux_elems))+'\n')
        for i in range(0, len(aux_elems)):
            f.write('  '+str(i+1)+'\n')
        for i in range(0, len(aux_elems)):
            f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\t'+str(aux_elems[i,3])+'\n')
    # create heart.ensi.case file
    with open(figPath+meshName+'.ensi.case', 'w') as f:
        f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
        f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
        f.write('scalar per element: 1	trueNodes_' + target_type + '	'
                + meshName + '_' + target_type + '.ensi.trueNodes\n')
        f.write('scalar per node: 1	PK_times	' + meshName + '_' + target_type + '.ensi.PKtimes\n') # 2022/01/11
        resolution = 'newRV' # TODO: refactor this
        # for resolution in resolution_list:
        #     f.write('scalar per element: 1	kMeans_centroids_'+target_type+'_'+resolution+'	'
        #         + meshName + "_" + target_type + "_" + resolution + '.ensi.kMeans_centroids\n')
        #     f.write('scalar per element: 1	cummrootNodes_'+target_type+'_' +resolution+'	'
        #         + meshName + "_" + target_type + "_" + resolution + '.ensi.cummrootNodes\n')
        #     f.write('scalar per node: 1	ATMap_'+target_type+'_' +resolution+'	'
        #         + meshName + "_" + target_type + "_" + resolution + '.ensi.ATMap\n')
        # if True:
        #     consistency_i = 2
        for consistency_i in range(min_possible_consistency_count, max_possible_consistency_count, 1):
        # if True:
            if is_clinical:
                previousResultsPath='metaData/Eikonal_Results/Clinical_'+ experiment_path_tag +data_type +'_'+resolution+'/' # TODO Inconsistent with the synthetic naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
                + "_" + str(consistency_i) + '_population.csv')
            else:
                previousResultsPath='metaData/Eikonal_Results/'+data_type+ experiment_path_tag +'_'+resolution+'/' # TODO Inconsistent with the clinical naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_"
                + str(conduction_speeds) + "_" + resolution + "_" + target_type + "_"
                + str(consistency_i) + '_population.csv')
        # previousResultsPath='metaData/Eikonal_Results/Clinical_'+data_type+'_'+resolution+'/'
        
            # fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
            #     + "_" + str(consistency_i) + '_population.csv')
            if os.path.isfile(fileName):
                f.write('scalar per element: 1	kMeans_centroids_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '.ensi.kMeans_centroids\n')
                f.write('scalar per element: 1	cummrootNodes_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '.ensi.cummrootNodes\n')
                f.write('scalar per node: 1	ATMap_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '.ensi.ATMap\n')
                f.write('scalar per node: 1	ATMap_'+target_type+'_'
                            +resolution+'_'+str(consistency_i)+'	' + meshName + "_" + target_type + "_" + resolution
                    + "_" + str(consistency_i) + '_best.ensi.ATMap\n') # 2022/01/18

    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, tetrahedrons.shape[0], 1):
        aux[tetrahedrons[i, 0]].append(i)
        aux[tetrahedrons[i, 1]].append(i)
        aux[tetrahedrons[i, 2]].append(i)
        aux[tetrahedrons[i, 3]].append(i)
    global elements
    elements = [np.array(n) for n in aux]
    aux = None  # Clear Memory

    # Precompute PseudoECG stuff - Calculate the tetrahedrons volumes
    D = nodesXYZ[tetrahedrons[:, 3], :]  # RECYCLED
    A = nodesXYZ[tetrahedrons[:, 0], :] - D  # RECYCLED
    B = nodesXYZ[tetrahedrons[:, 1], :] - D  # RECYCLED
    C = nodesXYZ[tetrahedrons[:, 2], :] - D  # RECYCLED
    D = None  # Clear Memory

    global tVolumes
    tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1),
                                           (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0])  # Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
    tVolumes = tVolumes / np.sum(tVolumes)

    # Calculate the tetrahedron (temporal) voltage gradients
    Mg = np.stack((A, B, C), axis=-1)
    A = None  # Clear Memory
    B = None  # Clear Memory
    C = None  # Clear Memory

    # Calculate the gradients
    global G_pseudo
    G_pseudo = np.zeros(Mg.shape)
    for i in range(Mg.shape[0]):
        G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
    G_pseudo = np.moveaxis(G_pseudo, 1, 2)
    Mg = None  # clear memory

    # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
    r = np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
                               (tetrahedronCenters.shape[0],
                                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1) - electrodePositions

    global d_r
    d_r = np.moveaxis(np.multiply(
        np.moveaxis(r, [0, 1], [-1, -2]),
        np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0, -1)

    # Set endocardial edges aside # 2022/01/18 - Split endocardium into two parts, a fast, and a slower one, namely, a dense and a sparse sub-endocardial Purkinje network
    global isEndocardial
    global isDenseEndocardial # 2022/01/18
    if has_endocardial_layer:
        isEndocardial=np.logical_or(np.all(np.isin(edges, lvnodes), axis=1), np.all(np.isin(edges, rvnodes), axis=1)) # 2022/01/18
        isDenseEndocardial=np.logical_or(np.all(np.isin(edges, lv_dense_indexes), axis=1), np.all(np.isin(edges, rv_dense_indexes), axis=1)) # 2022/01/18
        # print('isEndocardial.shape')
        # print(isEndocardial.shape)
        # print(np.sum(isEndocardial))
        # print(isDenseEndocardial.shape)
        # print(np.sum(isDenseEndocardial))
        # print(np.sum(np.logical_or(isEndocardial, isDenseEndocardial)))
    else:
        isEndocardial = np.zeros((edges.shape[0])).astype(bool)
        isDenseEndocardial = np.zeros((edges.shape[0])).astype(bool) # 2022/01/18
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan,
                         np.int32)  # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None  # Clear Memory

    # Target result
    global reference_lead_is_max # 22/05/03 - Normalise also the limb leads to give them relative importance
    # Load and generate target data
    if load_target:
        target_output = np.genfromtxt(target_data_path, delimiter=',')
        target_output = target_output - (target_output[:, 0:1]+target_output[:, -2:-1])/2 # align at zero # Re-added on 22/05/03 after it was worse without alingment
        reference_lead_max = np.amax(target_output, axis=1) # 22/05/03
        reference_lead_min = np.absolute(np.amin(target_output, axis=1)) # 22/05/03
        reference_lead_is_max_aux = reference_lead_max >= reference_lead_min
        reference_amplitudes = np.zeros(shape=(nb_leads), dtype=np.float64) # 2022/05/04
        reference_amplitudes[reference_lead_is_max_aux] = reference_lead_max[reference_lead_is_max_aux]
        reference_amplitudes[np.logical_not(reference_lead_is_max_aux)] = reference_lead_min[np.logical_not(reference_lead_is_max_aux)]
        target_output[:2, :] = target_output[:2, :] / np.mean(reference_amplitudes[:2]) # 22/05/03 TODO: Uncomment
        target_output[2:nb_leads, :] = target_output[2:nb_leads, :] / np.mean(reference_amplitudes[2:nb_leads]) # 22/05/03 TODO: Uncomment
    else:
        reference_lead_is_max_aux = None # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
        print('TODO: add normalisation rules for synthetic data')
        # TODO: add modifications for new Purkinje strategy # 2022/01/17
        print("TODO: add modifications for new Purkinje strategy # 2022/01/17")
        lv_rootNodesIndexes_true = rootNodesIndexes_true[np.isin(lvnodes, rootNodesIndexes_true)]
        lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_rootNodesIndexes_true]).astype(int)
        rv_rootNodesIndexes_true = rootNodesIndexes_true[np.isin(rvnodes, rootNodesIndexes_true)]
        rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_rootNodesIndexes_true]).astype(int)
        print(lv_rootNodesIndexes_true)
        print(rv_rootNodesIndexes_true)
        print(rootNodesIndexes_true)
        print(lvActnode_ids)
        print(rvActnode_ids)
        # lv_rootNodesTimes_true = lvHisBundletimes[lvActnode_ids]
        lv_rootNodesTimes_true = lv_PK_time_mat[lvActnode_ids]
        # rv_rootNodesTimes_true = rvHisBundletimes[rvActnode_ids]
        rv_rootNodesTimes_true = rv_PK_time_mat[rvActnode_ids]
        print(lv_rootNodesTimes_true)
        print(rv_rootNodesTimes_true)
        print('revise this section before using this option, 30/11/2021')
        raise()
        rootNodesTimes_true = np.concatenate((lv_rootNodesTimes_true, rv_rootNodesTimes_true), axis=0)
        target_output = eikonal_ecg(np.array([conduction_speeds])/1000., rootNodesIndexes_true, rootNodesTimes_true)[0, :nb_leads, :]
        lead_size = np.sum(np.logical_not(np.isnan(target_output[0, :])))
        target_output = target_output[:, :lead_size]
    
    reference_lead_is_max = reference_lead_is_max_aux # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead


    aggregation = pandas.DataFrame([], columns=['consistency_i', 'resolution', 'pred_roots',
        'rootNodes', 'roots_count', 'corr_mat', 'discrepancies'])
    # Iterate consistencies
    coeficients_array = []
    result_summary = []
    corr_mat_summary = []
    particle_summary = []
    for consistency_i in range(min_possible_consistency_count, max_possible_consistency_count, 1):
        # print('consistency_i ' + str(consistency_i))
        # Count resolutions available
        resolution_count = 0
        aux_resolution_list = []
        for resolution_i in range(len(resolution_list)):
            resolution = resolution_list[resolution_i]
            if is_clinical:
                previousResultsPath='metaData/Eikonal_Results/Clinical_'+ experiment_path_tag +data_type +'_'+resolution+'/' # TODO Inconsistent with the synthetic naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
                + "_" + str(consistency_i) + '_population.csv')
            else:
                previousResultsPath='metaData/Eikonal_Results/'+data_type + experiment_path_tag +'_'+resolution+'/' # TODO Inconsistent with the clinical naming
                fileName = (previousResultsPath + meshName + "_" + resolution + "_"
                + str(conduction_speeds) + "_" + resolution + "_" + target_type + "_"
                + str(consistency_i) + '_population.csv')
            
            
            # Check if the file exists
            if os.path.isfile(fileName):
                resolution_count = resolution_count + 1
                aux_resolution_list.append(resolution)
        # print('resolution_count ' + str(resolution_count))
        if resolution_count > 0:
            # Create figure
            fig, axs = plt.subplots(nrows=resolution_count, ncols=nb_leads, constrained_layout=True, figsize=(11, 2.5*resolution_count), sharey='all')
            # fig2, axs2 = plt.subplots(nrows=resolution_count, ncols=2, constrained_layout=True, figsize=(11, 2.5*resolution_count), sharey='all')
            # Iterate resolutions
            for resolution_i in range(len(aux_resolution_list)):
                resolution = aux_resolution_list[resolution_i]
                if is_clinical:
                    previousResultsPath='metaData/Eikonal_Results/Clinical_'+ experiment_path_tag +data_type +'_'+resolution+'/' # TODO Inconsistent with the synthetic naming
                    fileName = (previousResultsPath + meshName + "_" + resolution + "_" + target_type
                    + "_" + str(consistency_i) + '_population.csv')
                else:
                    previousResultsPath='metaData/Eikonal_Results/'+data_type+ experiment_path_tag +'_'+resolution+'/' # TODO Inconsistent with the clinical naming
                    fileName = (previousResultsPath + meshName + "_" + resolution + "_"
                    + str(conduction_speeds) + "_" + resolution + "_" + target_type + "_"
                    + str(consistency_i) + '_population.csv')
                
                # if not os.path.isfile(fileName):
                #     print('NOT')
                #     print(fileName)
                if os.path.isfile(fileName):
                    print(fileName)
                    # load root nodes with the current resolution
                    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_lv_activationIndexes_' + resolution + 'Res.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
                    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_rv_activationIndexes_' + resolution + 'Res.csv', delimiter=',') - 1).astype(int)
                    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
                    
                    # # Save the available LV and RV root nodes
                    # with open(figPath + meshName + '_available_root_nodes.csv', 'w') as f:
                    #     f.write('"Points:0","Points:1","Points:2"\n')
                    #     for i in range(0, len(lvActivationIndexes)):
                    #         f.write(str(nodesXYZ[lvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[lvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[lvActivationIndexes[i], 2]) + '\n')
                    #     for i in range(0, len(rvActivationIndexes)):
                    #         f.write(str(nodesXYZ[rvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[rvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[rvActivationIndexes[i], 2]) + '\n')

                    
                    
                    # PRECALCULATE DISTANCES BETWEEN NODES IN THE ENDOCARDIUM TO ALLOW K-MEANS TO BE COMPUTED ON THE RESULTS FROM THE INFERENCE AND ALSO
                    #TO ALLOW COMPUTING THE ACTIVATION TIMES AT THE ROOT NODES # MOVED UP ON THE 29/11/2021
                    # Project canditate root nodes to the endocardial layer: As from 2021 the root nodes are always generated at the endocardium, but this is still a good security measure
                    for i in range(lvActivationIndexes.shape[0]):
                        if lvActivationIndexes[i] not in lvnodes:
                            lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
                    lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
                    for i in range(rvActivationIndexes.shape[0]):
                        if rvActivationIndexes[i] not in rvnodes:
                            rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
                    rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
                    # Time cost from each root node
                    rootNodeActivationTimes = np.around(np.concatenate((lv_PK_time_mat[lvActnode_ids], rv_PK_time_mat[rvActnode_ids]), axis=0), decimals=4) # 02/12/2021 Tested how many decimals were needed to get the
                    # Save the pseudo-Purkinje networks considered during the inference for visulalisation and plotting purposes - LV
                    lvedges_indexes = np.all(np.isin(edges, lvnodes), axis=1)
                    lv_edges = edges[lvedges_indexes, :]
                    lvPK_edges_indexes = np.zeros(lv_edges.shape[0], dtype=np.bool_)
                    lv_root_to_PKpath_mat = lv_PK_path_mat[lvActnode_ids, :]
                    for i in range(0, lv_edges.shape[0], 1):
                        for j in range(0, lv_root_to_PKpath_mat.shape[0], 1):
                            if np.all(np.isin(lv_edges[i, :], lvnodes[lv_root_to_PKpath_mat[j, :][lv_root_to_PKpath_mat[j, :] != nan_value]])):
                                lvPK_edges_indexes[i] = 1
                                break
                    LV_PK_edges = lv_edges[lvPK_edges_indexes, :]
                    # LV_PK_edges = np.concatenate((LV_PK_edges, lv_edges_his), axis=0)
                    # Save the available LV Purkinje network
                    with open(figPath + meshName + '_available_LV_PKnetwork.vtk', 'w') as f:
                        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
                        for i in range(0, nodesXYZ.shape[0], 1):
                            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
                        f.write('LINES '+str(LV_PK_edges.shape[0])+' '+str(LV_PK_edges.shape[0]*3)+'\n')
                        for i in range(0, LV_PK_edges.shape[0], 1):
                            f.write('2 ' + str(LV_PK_edges[i, 0]) + ' ' + str(LV_PK_edges[i, 1]) + '\n')
                        f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(LV_PK_edges.shape[0]) + '\n')
                    # Save the available LV and RV root nodes
                    with open(figPath + meshName + '_available_root_nodes.csv', 'w') as f:
                        f.write('"Points:0","Points:1","Points:2"\n')
                        for i in range(0, len(lvActivationIndexes)):
                            f.write(str(nodesXYZ[lvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[lvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[lvActivationIndexes[i], 2]) + '\n')
                        for i in range(0, len(rvActivationIndexes)):
                            f.write(str(nodesXYZ[rvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[rvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[rvActivationIndexes[i], 2]) + '\n')
                    # RV
                    rvedges_indexes = np.all(np.isin(edges, rvnodes), axis=1)
                    rv_edges = edges[rvedges_indexes, :]
                    rvPK_edges_indexes = np.zeros(rv_edges.shape[0], dtype=np.bool_)
                    rv_root_to_PKpath_mat = rv_PK_path_mat[rvActnode_ids, :]
                    # rv_edges_crossing_to_roots = rv_edges_crossing[rvActnode_ids, :]
                    # rv_edges_crossing_to_roots = rv_edges_crossing_to_roots[np.logical_not(np.any(rv_edges_crossing_to_roots == nan_value, axis=1)), :]
                    # rv_edges_crossing_to_roots = rvnodes[rv_edges_crossing_to_roots]
                    for i in range(0, rv_edges.shape[0], 1):
                        for j in range(0, rv_root_to_PKpath_mat.shape[0], 1):
                            if np.all(np.isin(rv_edges[i, :], rvnodes[rv_root_to_PKpath_mat[j, :][rv_root_to_PKpath_mat[j, :] != nan_value]])):
                                rvPK_edges_indexes[i] = 1
                                break
                    RV_PK_edges = rv_edges[rvPK_edges_indexes, :]
                    # RV_PK_edges = np.concatenate((RV_PK_edges, rv_edges_his), axis=0)
                    # RV_PK_edges = np.concatenate((RV_PK_edges, rv_edges_crossing_to_roots), axis=0)
                    # # Save the available RV Purkinje network
                    with open(figPath + meshName + '_available_RV_PKnetwork.vtk', 'w') as f:
                        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
                        for i in range(0, nodesXYZ.shape[0], 1):
                            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
                        f.write('LINES '+str(RV_PK_edges.shape[0])+' '+str(RV_PK_edges.shape[0]*3)+'\n')
                        for i in range(0, RV_PK_edges.shape[0], 1):
                            f.write('2 ' + str(RV_PK_edges[i, 0]) + ' ' + str(RV_PK_edges[i, 1]) + '\n')
                        f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(RV_PK_edges.shape[0]) + '\n')
                    
                    # Start computations to produce the ECG Predictions figure
                    compilation_params = np.array([np.concatenate((np.full(nlhsParam, 0.09), np.ones(rootNodeActivationIndexes.shape).astype(int)))]) # Compile Numba
                    eikonal_ecg(compilation_params, rootNodeActivationIndexes, rootNodeActivationTimes) # Compile Numba # 02/12/2021
                    # Load population of particles
                    population_original = np.loadtxt(fileName, delimiter=',')
                    discrepancies_original = np.loadtxt(fileName.replace('population', 'discrepancy'), delimiter=',') # TODO uncomment
                    
                    print('population_original.shape')
                    print(population_original.shape)
                    # Remove duplicates with identical speeds but different numerical values in them, since these are 'randomly' generated to terminate the inference,
                    population_original = np.around(population_original, decimals=3) # ... even from different but equivalent root node locations September 2021
                    population_original, unique_particles_index = np.unique(population_original, return_index=True, axis=0) # Non-repeated particles
                    
                    discrepancies_original = discrepancies_original[unique_particles_index]
                    # best_id = np.argmin(discrepancies_original)
                    # Generate ECGs
                    prediction_list = eikonal_ecg(population_original, rootNodeActivationIndexes, rootNodeActivationTimes)
                    t_time_aux_s = time.time()
                    # Compute correlation coefficients to all ECGs in the unique population
                    corr_mat = np.zeros((prediction_list.shape[0], nb_leads)) # TODO - Uncomment 2022/05/09
                    for j in range(prediction_list.shape[0]):
                        not_nan_size = np.sum(np.logical_not(np.isnan(prediction_list[j, 0, :])))
                        for i in range(nb_leads): # TODO - Uncomment 2022/05/09
                        #TODO: add per lead CC values and check that the function does what is expexted!!! 2022/05/08 - For some reason I get lower CC than expexted!!
                            # prediction_lead = prediction_list[j, i, :]
                            prediction_lead = prediction_list[j, i, :not_nan_size]
                            if prediction_lead.shape[0] >= target_output.shape[1]:
                                a = prediction_lead
                                b = target_output[i, :]
                            else:
                                a = target_output[i, :]
                                b = prediction_lead
                            b_aux = np.zeros(a.shape)
                            b_aux[:b.shape[0]] = b
                            b_aux[b.shape[0]:] = b[-1]
                            b = b_aux
                            new_corr_value = np.corrcoef(a, b)[0,1]
                            corr_mat[j, i] = new_corr_value
                    corr_list = np.mean(corr_mat, axis=1, dtype=np.float64) # cannot use a more sofisticated mean (e.g. geometric mean) because of negative and zero values
                    t_time_final_aux = time.time()-t_time_aux_s
                    print('Time: ' + str(t_time_final_aux/prediction_list.shape[0])) # TODO DELETE
                    
                    # print('corr_list')
                    # print(corr_list.shape)
                    # print(corr_list)
                    best_id = np.argmax(corr_list)
                    corr_mat_summary.append(corr_mat[best_id, :])
                    particle_summary.append(population_original[best_id, :])
                    print('best_id')
                    print(best_id)
                    
                    for i in range(nb_leads):
                        if resolution_count > 1:
                            axs_res = axs[resolution_i, i]
                        else:
                            axs_res = axs[i]
                        axs_res.set_title(str(round(corr_mat[best_id, i], 2)), fontsize=16)

                    print()
                    
                    # print('OLD')
                    print('Average correlation coefficient: '+ str(round(np.mean(corr_list), 2)))
                    print('Best particle correlation coefficient: '+ str(round(corr_list[best_id], 2)))
                    print('Max correlation coefficient: '+ str(round(np.amax(corr_list), 2)))
                    # print('Best Index: '+ str(round(np.argmax(corr_list), 2)))
                    print('Worst correlation coefficient: '+ str(round(np.min(corr_list), 2)))
                    # print('Median correlation coefficient: '+ str(round(np.median(corr_list), 2)))
                    
                   
                   
                   
                    # print(population_original.shape)
                    # print('corr_mat')
                    # print(corr_mat.shape)
                    # print(corr_mat)
                    # print('corr best split')
                    # print(corr_mat[best_id, :])
                    # print(np.mean(corr_mat[best_id, :]))
                    
                    print()
                    
                    
                    
                    
                    # corr_mat = np.zeros((prediction_list.shape[0])) # TODO - Delete 2022/05/09
                    # # print(prediction_list.shape)
                    # best_corr = -1.0
                    # best_pred = None
                    # for j in range(prediction_list.shape[0]):
                    #     not_nan_size = np.sum(np.logical_not(np.isnan(prediction_list[j, 0, :]))) # All leads have the same number of nan-values
                    #     prediction_aux = prediction_list[j, : , :not_nan_size]
                    #     if prediction_aux.shape[1] >= target_output.shape[1]:
                    #         a = prediction_aux
                    #         b = target_output
                    #     else:
                    #         a = target_output
                    #         b = prediction_aux
                    #     b_aux = np.zeros(a.shape)
                    #     b_aux[:, :b.shape[1]] = b
                    #     b_aux[:, b.shape[1]:] = b[:, -1][:, np.newaxis]
                    #     b = b_aux
                    #     if prediction_aux.shape[1] >= target_output.shape[1]:
                    #         prediction_aux_limb = a[:nb_limb_leads, :].flatten()
                    #         prediction_aux_precordial = a[nb_limb_leads:nb_leads, :].flatten()
                    #         target_output_aux_limb = b[:nb_limb_leads, :].flatten()
                    #         target_output_aux_precordial = b[nb_limb_leads:nb_leads, :].flatten()
                    #     else:
                    #         prediction_aux_limb = b[:nb_limb_leads, :].flatten()
                    #         prediction_aux_precordial = b[nb_limb_leads:nb_leads, :].flatten()
                    #         target_output_aux_limb = a[:nb_limb_leads, :].flatten()
                    #         target_output_aux_precordial = a[nb_limb_leads:nb_leads, :].flatten()
                    #
                    #     new_corr_value_limb = np.corrcoef(target_output_aux_limb, prediction_aux_limb)[0,1]
                    #     new_corr_value_precordial = np.corrcoef(target_output_aux_precordial, prediction_aux_precordial)[0,1]
                    #     corr_mat[j] = new_corr_value_limb * nb_limb_leads/nb_leads + new_corr_value_precordial * (nb_leads - nb_limb_leads)/nb_leads
                    #
                    #     if resolution_count > 1:
                    #         axs_res = axs2[resolution_i, :]
                    #     else:
                    #         axs_res = axs2
                    #     if corr_mat[j] >= best_corr:
                    #         best_corr = corr_mat[j]
                    #         best_pred = [prediction_aux[:nb_limb_leads, :].flatten(), prediction_aux[nb_limb_leads:nb_leads, :].flatten()]
                    #         axs_res[0].set_title(str(round(new_corr_value_limb, 4)), fontsize=16)
                    #         axs_res[1].set_title(str(round(new_corr_value_precordial, 4)), fontsize=16)
                    #     axs_res[0].plot(prediction_aux[:nb_limb_leads, :].flatten(), 'b-', linewidth=.05)
                    #     axs_res[1].plot(prediction_aux[nb_limb_leads:nb_leads, :].flatten(), 'b-', linewidth=.05)
                    #
                    # corr_list = corr_mat # np.mean(corr_mat, axis=1, dtype=np.float64) # cannot use a more sofisticated mean (e.g. geometric mean) because of negative and zero values
                    #
                    # axs_res[0].plot(prediction_aux[:nb_limb_leads, :].flatten(), 'b-', label='Pred-'+resolution, linewidth=.05)
                    # axs_res[1].plot(prediction_aux[nb_limb_leads:nb_leads, :].flatten(), 'b-', label='Pred-'+resolution, linewidth=.05)
                    # axs_res[0].plot(target_output[:nb_limb_leads, :].flatten(), 'k-', label='True', linewidth=1.2)
                    # axs_res[1].plot(target_output[nb_limb_leads:nb_leads, :].flatten(), 'k-', label='True', linewidth=1.2)
                    # axs_res[0].plot(best_pred[0], 'r-', label='Best-'+resolution, linewidth=1.2)
                    # axs_res[1].plot(best_pred[1], 'r-', label='Best-'+resolution, linewidth=1.2)
                    # axs_res[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
                    #
                    # print('NEW')
                    # # print('corr_list')
                    # # print(corr_list.shape)
                    # # print(corr_list)
                    # print('Average correlation coefficient: '+ str(round(np.mean(corr_list), 2)))
                    # print('Best correlation coefficient: '+ str(round(np.max(corr_list), 2)))
                    # # print('Best Index: '+ str(round(np.argmax(corr_list), 2)))
                    # print('Worst correlation coefficient: '+ str(round(np.min(corr_list), 2)))
                    # print('Median correlation coefficient: '+ str(round(np.median(corr_list), 2)))
                    # print()
                    # # best_id = np.argmax(corr_list)
                    # # print('best_id')
                    # # print(best_id)
                    # # print(population_original.shape)
                    # # print('corr_mat')
                    # # print(corr_mat.shape)
                    # # print(corr_mat)
                    # # print('corr best split')
                    # # print(corr_mat[best_id])
                    # # print(np.mean(corr_mat[best_id]))
                    #
                    # # print()
                    
                    

                    pass
                    # TODO: DO NOT AVERAGE PARTICLES FOR THE PURKINJE INFERENCE PAPER, JUST TAKE THE BEST ONE # 2022/01/18
                    pass
                    # TODO: this section used to select what particles were good enough or consistent enough to be presented as the result of the inference
                    # TODO: now this is no longer necessary because we use all the particles that reach the final stage (without repetitions) as part of the solution to the inference
                    # # Select the best particles for the average process
                    # # nb_best_particles_to_average = int(population_original.shape[0]/1) # TODO change back preselect only the top 5% particles (discrepancy or correlation coefficient?)
                    # # discrepancies_original = np.loadtxt(fileName.replace('population', 'discrepancy'), delimiter=',') # TODO uncomment
                    # # discrepancies_original = discrepancies_original[unique_particles_index] # TODO uncomment
                    # discrepancies_original = corr_list # TODO uncomment
                    # discrepancies_sort_indexes = np.flip(np.argsort(discrepancies_original)) # TODO uncomment
                    # # print('corr_list[discrepancies_sort_indexes[0]]')
                    # # print(corr_list[discrepancies_sort_indexes[0]])
                    # # print(corr_list[discrepancies_sort_indexes[-1]])
                    # population_original = population_original[discrepancies_sort_indexes, :]
                    # discrepancies_original = discrepancies_original[discrepancies_sort_indexes]
                    # # preselect only the 5% with the best discrepancies
                    # # population = population_original[:nb_best_particles_to_average, :]
                    population = population_original #[:nb_best_particles_to_average, :] # 2022/01/18
                    # discrepancies = discrepancies_original #[:nb_best_particles_to_average] # 2022/01/18
                    # # print(discrepancies.shape)
                    # # print(discrepancies_original.shape)
                    # # print('Average correlation discrepancy: '+ str(round(np.mean(discrepancies_original), 3)))
                    # # print('Best correlation discrepancy: '+ str(round(np.max(discrepancies_original), 3)))
                    # # print('Worst correlation discrepancy: '+ str(round(np.min(discrepancies_original), 3)))
                    # # print('Median correlation discrepancy: '+ str(round(np.median(discrepancies_original), 3)))
                    # best_id = np.argmin(discrepancies)
                    # # best_params = population[best_id, :] # 06/12/2021 - Test the plot of the the best CC instead of the kmeans particle.
                    # # Recalculate ECGs to plot and average
                    # prediction_list = eikonal_ecg(population, rootNodeActivationIndexes, rootNodeActivationTimes)
                    best_ecg_CC = prediction_list[best_id, :, :] # 2022/01/18
                    best_ecg_CC = best_ecg_CC[:, np.logical_not(np.isnan(best_ecg_CC[0, :]))]
                    result_summary.append(best_ecg_CC)
                    # inference_ecg = best_ecg # TODO fix this # 2022/01/18
                    
                    best_discrepancy_id = np.argmin(discrepancies_original)
                    best_ecg_D = prediction_list[best_discrepancy_id, :, :]
                    best_ecg_D = best_ecg_D[:, np.logical_not(np.isnan(best_ecg_D[0, :]))]
                    
                    print('by CC, best CC: ' + str(np.amax(corr_list)) + ', and its D: ' + str(discrepancies_original[np.argmax(corr_list)]))
                    print('by D, best CC: ' + str(corr_list[best_discrepancy_id]) + ', and its best D: ' + str(discrepancies_original[best_discrepancy_id]))
                    
                    # print('best_ecg.shape')
                    # print(best_ecg.shape)
                    # print(population[best_id, :])
                    # # print(np.array([best_params]).shape)
                    # # print(population.shape)
                    # # best_ecg = eikonal_ecg(np.array([best_params]), rootNodeActivationIndexes, rootNodeActivationTimes) # 06/12/2021 - Test the plot of the the best CC instead of the kmeans particle.
                    # # aux_best_ecg = eikonal_ecg(best_params[np.newaxis, :], rootNodeActivationIndexes, rootNodeActivationTimes) # 06/12/2021 - Test the plot of the the best CC instead of the
                    # # kmeans particle.
                    # # print('np.sum(best_ecg[0, :, :] - prediction_list[best_id, :, :])')
                    # # print(np.nansum(np.abs(best_ecg[0, :, :] - best_ecg[0, :, :])))
                    # # print(np.nansum(prediction_list[best_id, :, :] - prediction_list[best_id, :, :]))
                    # # print(np.nansum(np.abs(best_ecg[0, :, :] - prediction_list[best_id, :, :])))
                    # # print(np.nansum(prediction_list[-1, :, :] - prediction_list[best_id, :, :]))
                    # # print(np.nansum(np.abs(aux_best_ecg - best_ecg)))
                    # # print(best_id - 0)
                    # # print(best_id)
                    # # print((best_ecg[0, :, :] - prediction_list[best_id, :, :])*10000)
                    # # print(prediction_list.shape)
                    # # print(best_ecg.shape)
                    # # print(best_params[np.newaxis, :].shape)
                    # # print(population.shape)
                    # # inference_ecg = prediction_list[0, :, :] # TODO remove this
                    # # TODO: no need to recalculate correlations because there was no k-means averaging - 2022/05/11
                    if False: # Do not average particles for this paper # 2022/01/18
                        # THIS IS INSIDE THE "IF FALSE"
                        
                        # Changed on September 2021, speeds and root nodes should be only averaged when coming from similar particles.
                        # Select what particles to average together, this should smooth out indistinguishable differences in particles and improve when the ground truth could not be reached
                        #without drastically changing the predicted ECGs which are already optimal.
                        
                        # Preselect particles based on their number of root nodes on the LV and RV; here we should select
                        # only hearts with a specific number of root nodes to avoid mixing results from too different inference results
                        # September 2021
                        rootNodes = np.round(population[:, nlhsParam:]).astype(int)
                        # LV preselection based on most frequent number of root nodes in the LV
                        lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
                        lv_num_nodes = np.sum(lv_rootNodes_part, axis=1)
                        # Select the particles to average as those that have the same number of root nodes
                        lv_num_nodes_non_rep, lv_num_nodes_counts = np.unique(lv_num_nodes, return_counts=True, axis=0)
                        lv_num_clusters = lv_num_nodes_non_rep[np.argmax(lv_num_nodes_counts)] # most common number of root nodes in the LV
                        population_to_average = population[lv_num_nodes == lv_num_clusters, :] # Indexes of the particles with this number of root nodes in the LV
                        # RV preselection based on most frequent number of root nodes in the RV from the pre-selection in the LV
                        rootNodes_to_average = np.round(population_to_average[:, nlhsParam:]).astype(int)
                        rv_rootNodes_part = rootNodes_to_average[:, len(lvActivationIndexes):] # Refine to only those root node parameter sets which have X root nodes on the LV
                        rv_num_nodes = np.sum(rv_rootNodes_part, axis=1)
                        rv_num_nodes_non_rep, rv_num_nodes_counts = np.unique(rv_num_nodes, return_counts=True, axis=0)
                        rv_num_clusters = rv_num_nodes_non_rep[np.argmax(rv_num_nodes_counts)] # most common number of root nodes in the RV, conditioned to the most freq number in the LV
                        population_to_average = population_to_average[rv_num_nodes == rv_num_clusters, :] # Indexes of the particles with these number of root nodes in the RV
                        
                        # At this point: population_to_average contains the particles with different speed values, and the most frequent numbers of root nodes in both ventricles (conditionally)
                        inferred_speeds = np.median(population_to_average[:, :nlhsParam], axis=0) # compute the median conduction speeds from these similar particles in terms of root nodes
                        # K-means for root nodes: Choose the number of clusters 'k' using the particles that have already been refined September 2021
                        rootNodes = np.round(population_to_average[:, nlhsParam:]).astype(int)
                        roots_count = rootNodes.shape[0]
                        lv_rootNodes_part = rootNodes[:, :len(lvActivationIndexes)]
                        rv_rootNodes_part = rootNodes[:, len(lvActivationIndexes):]
                        
                        # Start of computations for averaging the inferred population - Root nodes - LV
                        lvnodesXYZ = nodesXYZ[lvnodes, :]
                        # Set LV endocardial edges aside
                        lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
                        lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
                        lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
                        lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
                        lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
                        aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
                        for i in range(0, len(lvunfoldedEdges), 1):
                            aux[lvunfoldedEdges[i, 0]].append(i)
                        lvneighbours = [np.array(n) for n in aux]
                        aux = None  # Clear Memory
                        lvdistance_mat, lvpath_mat = djikstra(lvActnode_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours)
                        
                        new_lv_roots = []
                        new_lv_roots_count = 0
                        lv_rootNodes = 0
                        lv_rootNodesTimes_pred = []
                        if lv_num_clusters > 0:
                            # Choose the initial centroides
                            k_lv_rootNodes_list, lvunique_counts = np.unique(lv_rootNodes_part, return_counts=True, axis=0)
                            lv_rootNodes_list_index = np.sum(k_lv_rootNodes_list, axis=1) == lv_num_clusters
                            k_lv_rootNodes_list = k_lv_rootNodes_list[lv_rootNodes_list_index, :]
                            lvunique_counts = lvunique_counts[lv_rootNodes_list_index]
                            lv_centroid_ids = lvActnode_ids[k_lv_rootNodes_list[np.argmax(lvunique_counts), :].astype(bool)].astype(int)
                            # Transform the root nodes predicted data into k-means data
                            lvdata_ids = np.concatenate(([lvActivationIndexes[(lv_rootNodes_part[i, :]).astype(bool)] for i in range(lv_rootNodes_part.shape[0])]), axis=0)
                            lvdata_ids = np.asarray([np.flatnonzero(lvActivationIndexes == node_id)[0] for node_id in lvdata_ids]).astype(int)
                            # K-means algorithm: LV
                            k_means_lv_labels, k_means_lv_centroids = k_means(lvnodesXYZ[lvActnode_ids, :], lvdata_ids, lv_num_clusters, lv_centroid_ids, lvdistance_mat, lvnodesXYZ, max_iter=10)
                            # Any node in the endocardium can be a result
                            new_lv_roots = np.unique(lvnodes[k_means_lv_centroids])
                            # Check if something went wrong
                            if new_lv_roots.shape[0] != lv_num_clusters:
                                print('LV')
                                raise
                            lv_rootNodes = np.sum(lv_rootNodes_part, axis=0)
                            new_lv_roots, new_lv_roots_count = np.unique(new_lv_roots, return_counts=True)
                            # Prepare for ATM simulation
                            lvActnode_ids_pred = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in new_lv_roots]).astype(int)
                            lv_rootNodesTimes_pred = lv_PK_time_mat[lvActnode_ids_pred]
                        
                        # Root nodes - RV
                        rvnodesXYZ = nodesXYZ[rvnodes, :]
                        # Set endocardial edges aside
                        rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
                        rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
                        rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
                        rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
                        rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
                        aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
                        for i in range(0, len(rvunfoldedEdges), 1):
                            aux[rvunfoldedEdges[i, 0]].append(i)
                        rvneighbours = [np.array(n) for n in aux]
                        aux = None  # Clear Memory
                        rvdistance_mat, rvpath_mat = djikstra(rvActnode_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours)
                        
                        new_rv_roots = []
                        new_rv_roots_count = 0
                        rv_rootNodes = 0
                        rv_rootNodesTimes_pred = []
                        if rv_num_clusters > 0:
                            # Choose the initial centroides
                            k_rv_rootNodes_list, rvunique_counts = np.unique(rv_rootNodes_part, return_counts=True, axis=0)
                            rv_rootNodes_list_index = np.sum(k_rv_rootNodes_list, axis=1) == rv_num_clusters
                            k_rv_rootNodes_list = k_rv_rootNodes_list[rv_rootNodes_list_index, :]
                            rvunique_counts = rvunique_counts[rv_rootNodes_list_index]
                            rv_centroid_ids = rvActnode_ids[k_rv_rootNodes_list[np.argmax(rvunique_counts), :].astype(bool)].astype(int)
                            # Transform the root nodes predicted data into k-means data
                            rvdata_ids = np.concatenate(([rvActivationIndexes[(rv_rootNodes_part[i, :]).astype(bool)] for i in range(rv_rootNodes_part.shape[0])]), axis=0)
                            rvdata_ids = np.asarray([np.flatnonzero(rvActivationIndexes == node_id)[0] for node_id in rvdata_ids]).astype(int)
                            # K-means algorithm: RV
                            k_means_rv_labels, k_means_rv_centroids = k_means(rvnodesXYZ[rvActnode_ids, :], rvdata_ids, rv_num_clusters, rv_centroid_ids, rvdistance_mat, rvnodesXYZ, max_iter=10)
                            # Any node in the endocardium can be a result
                            new_rv_roots = np.unique(rvnodes[k_means_rv_centroids])
                            # Check if something went wrong
                            if new_rv_roots.shape[0] != rv_num_clusters:
                                print('RV')
                                print(fileName)
                                print(rv_centroid_ids)
                                print(k_means_rv_centroids)
                                print(rvnodes[k_means_rv_centroids])
                                print(lv_centroid_ids)
                                raise
                            rv_rootNodes = np.sum(rv_rootNodes_part, axis=0)
                            new_rv_roots, new_rv_roots_count = np.unique(new_rv_roots, return_counts=True)
                            # Prepare for ATM simulation
                            rvActnode_ids_pred = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in new_rv_roots]).astype(int)
                            rv_rootNodesTimes_pred = rv_PK_time_mat[rvActnode_ids_pred]
                        
                        # Simulate the ATMap solution to the inference
                        pred_roots = np.concatenate((new_lv_roots, new_rv_roots), axis=0).astype(int)
                        rootNodeActivationTimes_pred = np.around(np.concatenate((lv_rootNodesTimes_pred, rv_rootNodesTimes_pred), axis=0), decimals=4) # 04/12/2021
                        inferred_params = np.concatenate((inferred_speeds, np.ones((pred_roots.shape[0])))) # 06/12/2021 - Test the plot of the the best CC instead of the kmeans particle.
                        inference_atmap = eikonal_atm(inferred_params, pred_roots, rootNodeActivationTimes_pred) # 06/12/2021 - Test the plot of the the best CC instead of the kmeans particle.
                        with open(figPath + meshName + '_' + target_type + '_' + resolution + '_' + str(consistency_i) + '_kmeans.ensi.ATMap', 'w') as f:
                            f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
                            for i in range(0, len(inference_atmap)):
                                f.write(str(inference_atmap[i]) + '\n')
                        
                        # Save K-means CENTROIDS
                        atmap = np.zeros((nodesXYZ.shape[0]))
                        atmap[new_lv_roots] = np.round(100*new_lv_roots_count, 2)
                        atmap[new_rv_roots] = np.round(100*new_rv_roots_count, 2)
                        # Translate the atmap to be an element-wise map
                        elem_atmap = np.zeros((elems.shape[0]))
                        for i in range(elems.shape[0]):
                            elem_atmap[i] = np.sum(atmap[elems[i]])
                        with open(figPath + meshName + "_" + target_type + "_" + resolution
                        + "_" + str(consistency_i) + '.ensi.kMeans_centroids', 'w') as f:
                            f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
                            for i in range(0, len(elem_atmap)):
                                f.write(str(elem_atmap[i])+'\n')
                                
                        
                        # Simulate the ECG solution to the inference
                        inference_ecg = eikonal_ecg(np.array([inferred_speeds]), pred_roots, rootNodeActivationTimes)
                        inference_ecg = inference_ecg[0, :, :]
                        inference_ecg = inference_ecg[:, np.logical_not(np.isnan(inference_ecg[0, :]))]
                        if inference_ecg.shape[1] >= target_output.shape[1]:
                            a = inference_ecg
                            b = target_output
                        else:
                            a = target_output
                            b = inference_ecg
                        b_aux = np.zeros(a.shape)
                        b_aux[:, :b.shape[1]] = b
                        b_aux[:, b.shape[1]:] = b[:, -2:-1]
                        b = b_aux
                        # Plot into figure with Simulation and Target ECGs and compute correlation coef
                        inferred_kmeans_corr_list = []
                        for i in range(nb_leads):
                            inferred_kmeans_corr_list.append(np.corrcoef(a[i, :], b[i, :])[0,1])
                        inferred_corr_coef = round(np.mean(np.asarray(inferred_kmeans_corr_list)), 2)
                        coeficients_array.append(inferred_corr_coef)
                        
                        print('Speeds inferred: ' + str(np.round(1000*inferred_speeds)))
                        print('pred_roots: ' + str(pred_roots))
                        print()
                        print('K-means correlation coefficient: '+ str(inferred_corr_coef))
                        
                        # SAVE ROOT NODE RESULTS INTO PANDAS
                        rootNodes_list = []
                        for i in range(rootNodes.shape[0]):
                            rootNodes_list.append(rootNodeActivationIndexes[rootNodes[i, :].astype(bool)])
                        rootNodes = np.concatenate((rootNodes_list), axis=0)
                        df_tmp = pandas.DataFrame([(consistency_i, resolution, pred_roots,
                            rootNodes, rootNodes.shape[0], corr_mat, discrepancies)], columns=['consistency_i', 'resolution',
                            'pred_roots', 'rootNodes', 'roots_count', 'corr_mat', 'discrepancies'])
                        aggregation = aggregation.append(df_tmp)
                        np.savetxt(figPath + meshName + '_pred_roots_'+ resolution +'_'+str(consistency_i)+'.csv', pred_roots.astype(int)+1, delimiter=',')
                    
                        # THIS IS OUTSIDE THE "IF FALSE"
                    
                    # Save CUMM ROOT NODES TOGETHER
                    atmap = np.zeros((nodesXYZ.shape[0]))
                    if not is_clinical:
                        atmap[rootNodesIndexes_true] = -1000
                    # atmap[lvActivationIndexes] = np.round(100*lv_rootNodes/roots_count, 2)#-6 # substracting the baseline value
                    atmap[lvActivationIndexes] = np.round(100*np.sum(np.round(population[:, nlhsParam:]).astype(int)[:, :len(lvActivationIndexes)], axis=0)/np.round(population[:, nlhsParam:]).astype(int).shape[0], 2)
                    # atmap[rvActivationIndexes] = np.round(100*rv_rootNodes/roots_count, 2)#-6 # substracting the baseline value
                    atmap[rvActivationIndexes] = np.round(100*np.sum(np.round(population[:, nlhsParam:]).astype(int)[:, len(lvActivationIndexes):], axis=0)/np.round(population[:, nlhsParam:]).astype(int).shape[0], 2)
                    # print('a')
                    # print(np.round(population[:, nlhsParam:]).astype(int).shape[0] == population.shape[0])
                    # print(np.sum(np.round(population[:, nlhsParam:]).astype(int)[:, :len(lvActivationIndexes)], axis=0) == np.sum(np.round(population[:, nlhsParam:len(lvActivationIndexes)+nlhsParam]).astype(int), axis=0))
                    # Translate the atmap to be an element-wise map
                    elem_atmap = np.zeros((elems.shape[0]))
                    for i in range(elems.shape[0]):
                        elem_atmap[i] = np.sum(atmap[elems[i]])
                    with open(figPath + meshName + '_' + target_type + '_' + resolution + '_' + str(consistency_i) + '.ensi.cummrootNodes', 'w') as f:
                            f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
                            for i in range(0, len(elem_atmap)):
                                f.write(str(elem_atmap[i])+'\n')
                        
                    
                    # Simulate the best ATMap solution from the inferred population
                    inference_atmap = eikonal_atm(population[best_id, :], rootNodeActivationIndexes, rootNodeActivationTimes) # 2022/01/18 - Test the plot of the the best CC instead of the kmeans particle.
                    # np.savetxt(dataPath + meshName + '_ATM.csv', inference_atmap, delimiter=',')# TODO: delte this line 2022/02/04
                    with open(figPath + meshName + '_' + target_type + '_' + resolution + '_' + str(consistency_i) + '_best.ensi.ATMap', 'w') as f:
                        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
                        for i in range(0, len(inference_atmap)):
                            f.write(str(inference_atmap[i]) + '\n')
                    
                    # Save inferred Purkinje network for visulisation
                    aux_rootNodes = np.round(population[best_id, nlhsParam:]).astype(np.bool_)
                    # print(aux_rootNodes.shape)
                    # print(lvActnode_ids.shape)
                    # print(lvActivationIndexes.shape)
                    # print(rvActnode_ids.shape)
                    # print(rvActivationIndexes.shape)
                    inferred_lvActnode_ids = lvActnode_ids[aux_rootNodes[:len(lvActivationIndexes)]]
                    inferred_rvActnode_ids = rvActnode_ids[aux_rootNodes[len(lvActivationIndexes):]]
                    # print('np.sum(aux_rootNodes)')
                    # print(np.sum(aux_rootNodes))
                    print('nlhsParam ' + str(nlhsParam))
                    # Save the pseudo-Purkinje networks considered during the inference for visulalisation and plotting purposes - LV
                    lvedges_indexes = np.all(np.isin(edges, lvnodes), axis=1)
                    lv_edges = edges[lvedges_indexes, :]
                    inferred_lvPK_edges_indexes = np.zeros(lv_edges.shape[0], dtype=np.bool_)
                    lv_root_to_PKpath_mat = lv_PK_path_mat[inferred_lvActnode_ids, :]
                    for i in range(0, lv_edges.shape[0], 1):
                        for j in range(0, lv_root_to_PKpath_mat.shape[0], 1):
                            if np.all(np.isin(lv_edges[i, :], lvnodes[lv_root_to_PKpath_mat[j, :][lv_root_to_PKpath_mat[j, :] != nan_value]])):
                                inferred_lvPK_edges_indexes[i] = 1
                                break
                    inferred_LV_PK_edges = lv_edges[inferred_lvPK_edges_indexes, :]
                    # inferred_LV_PK_edges = np.concatenate((inferred_LV_PK_edges, lv_edges_his), axis=0)
                    # Save the available LV Purkinje network
                    with open(figPath + meshName + '_' + str(consistency_i) + '_inferred_LV_PKnetwork.vtk', 'w') as f:
                        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
                        for i in range(0, nodesXYZ.shape[0], 1):
                            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
                        f.write('LINES '+str(inferred_LV_PK_edges.shape[0])+' '+str(inferred_LV_PK_edges.shape[0]*3)+'\n')
                        for i in range(0, inferred_LV_PK_edges.shape[0], 1):
                            f.write('2 ' + str(inferred_LV_PK_edges[i, 0]) + ' ' + str(inferred_LV_PK_edges[i, 1]) + '\n')
                        f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(inferred_LV_PK_edges.shape[0]) + '\n')
                    # Save the available LV and RV root nodes
                    inferred_lvActivationIndexes = lvActivationIndexes[aux_rootNodes[:len(lvActivationIndexes)]]
                    # inferred_lvActivationTimes = lv_PK_time_mat[inferred_lvActnode_ids]
                    inferred_rvActivationIndexes = rvActivationIndexes[aux_rootNodes[len(lvActivationIndexes):]]
                    # inferred_rvActivationTimes = rv_PK_time_mat[inferred_rvActnode_ids]
                    inferred_ActivationIndexes = rootNodeActivationIndexes[np.round(population[best_id, nlhsParam:]).astype(np.bool_)]
                    inferred_ActivationTimes = rootNodeActivationTimes[np.round(population[best_id, nlhsParam:]).astype(np.bool_)]
                    # print('aha')
                    # print(lv_PK_time_mat.shape)
                    # print(rv_PK_time_mat.shape)
                    # print(len(aux_rootNodes[:len(lvActivationIndexes)]))
                    # print(len(aux_rootNodes[len(lvActivationIndexes):]))
                    # print(lvActivationIndexes.shape)
                    # print(rvActivationIndexes.shape)
                    # print(lvActnode_ids.shape)
                    # print(rvActnode_ids.shape)
                    # # print(inferred_lvActivationTimes.shape)
                    # # print(inferred_rvActivationTimes.shape)
                    # print(lvnodes.shape)
                    # print(rvnodes.shape)
                    # print(inferred_ActivationIndexes - np.concatenate((inferred_lvActivationIndexes, inferred_rvActivationIndexes)))
                    # print(inferred_ActivationTimes.shape)
                    # print(inferred_ActivationTimes)
                    # print(inferred_ActivationTimes-np.amin(inferred_ActivationTimes))
                    
                    # print(len(inferred_lvActivationIndexes))
                    # print(len(inferred_rvActivationIndexes))
                    with open(figPath + meshName + '_' + str(consistency_i) + '_inferred_root_nodes.csv', 'w') as f:
                        f.write('"x","y","z"\n')
                        for i in range(0, len(inferred_lvActivationIndexes)):
                            f.write(str(nodesXYZ[inferred_lvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[inferred_lvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[inferred_lvActivationIndexes[i], 2]) + '\n')
                        for i in range(0, len(inferred_rvActivationIndexes)):
                            f.write(str(nodesXYZ[inferred_rvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[inferred_rvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[inferred_rvActivationIndexes[i], 2]) + '\n')
                    with open(figPath + meshName + '_' + str(consistency_i) + '_inferred_root_nodes_xyz.csv', 'w') as f:
                        for i in range(0, len(inferred_ActivationIndexes)):
                            f.write(str(nodesXYZ[inferred_ActivationIndexes[i], 0]) + ',' + str(nodesXYZ[inferred_ActivationIndexes[i], 1]) + ',' + str(nodesXYZ[inferred_ActivationIndexes[i], 2]) + '\n')
                        # for i in range(0, len(inferred_rvActivationIndexes)):
                        #     f.write(str(nodesXYZ[inferred_rvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[inferred_rvActivationIndexes[i], 1]) + ',' + str(nodesXYZ[inferred_rvActivationIndexes[i], 2]) + '\n')
                    with open(figPath + meshName + '_' + str(consistency_i) + '_inferred_root_nodes_times.csv', 'w') as f:
                        for i in range(0, len(inferred_ActivationTimes)):
                            f.write(str(inferred_ActivationTimes[i]) + '\n')
                    with open(figPath + meshName + '_' + str(consistency_i) + '_inferred_root_nodes_relative_times.csv', 'w') as f:
                        for i in range(0, len(inferred_ActivationTimes)):
                            f.write(str(inferred_ActivationTimes[i]-np.amin(inferred_ActivationTimes)) + '\n')
                        # for i in range(0, len(inferred_rvActivationTimes)):
                            # f.write(str(inferred_rvActivationTimes[i]) + '\n')
                    with open(figPath + meshName + '_' + str(consistency_i) + '_inferred_root_nodes_cobiveco.csv', 'w') as f:
                        for i in range(0, len(inferred_ActivationIndexes)):
                            f.write(str(nodesCobiveco[inferred_ActivationIndexes[i], 0]) + ',' + str(nodesCobiveco[inferred_ActivationIndexes[i], 1]) + ','
                            + str(nodesCobiveco[inferred_ActivationIndexes[i], 2]) + ',' + str(nodesCobiveco[inferred_ActivationIndexes[i], 3]) + '\n')
                    print('nodesCobiveco.shape')
                    print(nodesCobiveco.shape)
                    print(nodesXYZ.shape)
                    
                    # RV
                    rvedges_indexes = np.all(np.isin(edges, rvnodes), axis=1)
                    rv_edges = edges[rvedges_indexes, :]
                    inferred_rvPK_edges_indexes = np.zeros(rv_edges.shape[0], dtype=np.bool_)
                    rv_root_to_PKpath_mat = rv_PK_path_mat[inferred_rvActnode_ids, :]
                    # rv_edges_crossing_to_roots = rv_edges_crossing[inferred_rvActnode_ids, :]
                    # rv_edges_crossing_to_roots = rv_edges_crossing_to_roots[np.logical_not(np.any(rv_edges_crossing_to_roots == nan_value, axis=1)), :]
                    # rv_edges_crossing_to_roots = rvnodes[rv_edges_crossing_to_roots]
                    for i in range(0, rv_edges.shape[0], 1):
                        for j in range(0, rv_root_to_PKpath_mat.shape[0], 1):
                            if np.all(np.isin(rv_edges[i, :], rvnodes[rv_root_to_PKpath_mat[j, :][rv_root_to_PKpath_mat[j, :] != nan_value]])):
                                inferred_rvPK_edges_indexes[i] = 1
                                break
                    inferred_RV_PK_edges = rv_edges[inferred_rvPK_edges_indexes, :]
                    # inferred_RV_PK_edges = np.concatenate((inferred_RV_PK_edges, rv_edges_his), axis=0)
                    # inferred_RV_PK_edges = np.concatenate((inferred_RV_PK_edges, rv_edges_crossing_to_roots), axis=0)
                    # Save the available RV Purkinje network
                    with open(figPath + meshName + '_' + str(consistency_i) + '_inferred_RV_PKnetwork.vtk', 'w') as f:
                        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
                        for i in range(0, nodesXYZ.shape[0], 1):
                            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
                        f.write('LINES '+str(inferred_RV_PK_edges.shape[0])+' '+str(inferred_RV_PK_edges.shape[0]*3)+'\n')
                        for i in range(0, inferred_RV_PK_edges.shape[0], 1):
                            f.write('2 ' + str(inferred_RV_PK_edges[i, 0]) + ' ' + str(inferred_RV_PK_edges[i, 1]) + '\n')
                        f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(inferred_RV_PK_edges.shape[0]) + '\n')
                    
                    
                    
                    
                    
                    
                    
                    # Calculate figure ms width
                    max_count = np.sum(np.logical_not(np.isnan(target_output[0, :])))
                    for i in range(prediction_list.shape[0]):
                        max_count = max(max_count, np.sum(np.logical_not(np.isnan(prediction_list[i, 0, :]))))
                    
                    # Plot into figure with Simulation and Target and Inference ECGs
                    # fig, axs = plt.subplots(nrows=len(resolution_list), ncols=nb_leads, constrained_layout=True, figsize=(11, 2.5*len(resolution_list)), sharey='all')
                    # corr_list = []
                    # TODO: Part of this code is already repeated above!! 2022/04/28
                    # corr_mat = np.zeros((prediction_list.shape[0], nb_leads))
                    for i in range(nb_leads):
                        if resolution_count > 1:
                            axs_res = axs[resolution_i, i]
                        else:
                            axs_res = axs[i]
                        # axs_res.plot(target_output[i, :], 'k-', label='Target', linewidth=1.2)
                        axs_res.plot(target_output[i, :], 'k-', label='Clinical', linewidth=1.2)
                        # axs_res.plot(target_output[i, :], 'b-', label='Predicted-'+resolution, linewidth=1.)
                        # axs_res.plot(inference_ecg[i, :], 'r-', label='Kmeans-'+resolution, linewidth=1.2) # 2021/12/15 - TODO Uncomment
                        # axs_res.plot(best_ecg_D[i, :], 'r-', label='Best-D', linewidth=1.2) # 2021/12/15 - TODO Comment
                        # axs_res.plot(best_ecg_CC[i, :], 'b-', label='Best-CC', linewidth=1.2) # 2021/12/15 - TODO Comment
                        axs_res.plot(best_ecg_CC[i, :], 'b-', label='Simulated', linewidth=1.2) # 2021/12/15 - TODO Comment
                        # axs_res.plot(best_ecg[i, :], color='purple', linestyle='-', label='Best-'+resolution, linewidth=1.2) # 2021/12/15 - TODO Uncomment
                        # for j in range(prediction_list.shape[0]):
                        #     prediction_lead = prediction_list[j, i, :]
                        #     lead_size = np.sum(np.logical_not(np.isnan(prediction_list[j, i, :])))
                        #     prediction_lead = prediction_lead[:lead_size]
                        #     axs_res.plot(prediction_lead, 'b-', linewidth=.05) # September 2021
                            # TODO: no need to recalculate correlations because there was no k-means averaging - 2022/05/11
                            # if prediction_lead.shape[0] >= target_output.shape[1]:
                            #     a = prediction_lead
                            #     b = target_output[i, :]
                            # else:
                            #     a = target_output[i, :]
                            #     b = prediction_lead
                            # b_aux = np.zeros(a.shape)
                            # b_aux[:b.shape[0]] = b
                            # b_aux[b.shape[0]:] = b[-1]
                            # b = b_aux
                            # new_corr_value = np.corrcoef(a, b)[0,1]
                            # # corr_list.append(new_corr_value)
                            # corr_mat[j, i] = new_corr_value
                        
                        # axs_res.plot(target_output[i, :], 'k-', linewidth=1.2)
                        # axs_res.plot(inference_ecg[i, :], 'r-', linewidth=1.2) # 2021/12/15 - TODO Uncomment
                        # axs_res.plot(best_ecg[i, :], 'r-', linewidth=1.2) # 2021/12/15 - TODO Comment
                        # axs_res.plot(best_ecg[i, :], color='purple', linestyle='-', linewidth=1.2) # 2021/12/15 - TODO Uncomment
                        # decorate figure
                        axs_res.xaxis.set_major_locator(MultipleLocator(40))
                        axs_res.xaxis.set_minor_locator(MultipleLocator(20))
                        axs_res.yaxis.set_major_locator(MultipleLocator(1))
                        #axs[0, i].yaxis.set_minor_locator(MultipleLocator(1))
                        axs_res.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                        axs_res.xaxis.grid(True, which='major')
                        axs_res.xaxis.grid(True, which='minor')
                        axs_res.yaxis.grid(True, which='major')
                        #axs[0, i].yaxis.grid(True, which='minor')
                        for tick in axs_res.xaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        for tick in axs_res.yaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                    
                    axs_res.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

                    # corr_list = np.mean(corr_mat, axis=1, dtype=np.float64) # cannot use a more sofisticated mean because of negative and zero values
                    
                    
                    # print('Average correlation coefficient: '+ str(round(np.mean(corr_list), 2)))
                    # print('Best correlation coefficient: '+ str(round(np.max(corr_list), 2)))
                    # print('Worst correlation coefficient: '+ str(round(np.min(corr_list), 2)))
                    # print('Median correlation coefficient: '+ str(round(np.median(corr_list), 2)))
                    # print()
                    
                    max_count += 5
                    for resolution_i in range(len(resolution_list)):
                        if resolution_count > 1:
                            axs_res = axs[resolution_i, i]
                        else:
                            axs_res = axs[i]
                        axs_res.set_xlim(0, max_count)
                        
                    # axs[0, 0].set_ylabel('standardised voltage', fontsize=22)
                    # axs[1, 0].set_ylabel('standardised voltage', fontsize=22)
                    #     axs[resolution_i, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
                        # axs[1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
                    
                    # if False: # TODO delete this
                    
                    
            if conduction_speeds is None:
                figName = (figPath + meshName + '_clinical_' + target_type + "_" + str(consistency_i) + '_comparison.png')
            else:
                figName = (figPath + meshName + '_' + str(conduction_speeds) + '_' + target_type + "_" + str(consistency_i) + '_comparison.png')
            plt.savefig(figName, dpi=400)
            plt.show()
            # plt.hist(discrepancies_original, bins='auto')  # arguments are passed to np.histogram
            # plt.title("Discrepancies")
            # plt.show()
            # plt.hist(discrepancies, bins='auto')  # arguments are passed to np.histogram
            # plt.title("Best Discrepancies")
            # plt.show()
            
    # for resolution in resolution_list:
    #     res_aggregation = aggregation.loc[(aggregation['resolution'] == resolution)]
    #     rootNodes_list = res_aggregation['rootNodes'].tolist()
    #     pred_roots_list = res_aggregation['pred_roots'].tolist()
    #     roots_count_list = res_aggregation['roots_count'].tolist()
    #     if len(rootNodes_list) > 0 and len(pred_roots_list) > 0 and len(roots_count_list) > 0:
    #         root_count = np.sum(roots_count_list)
    #         rootNodes = np.concatenate((rootNodes_list), axis=0)
    #         pred_roots = np.concatenate((pred_roots_list), axis=0)
    #         rootNodes, rootNodes_count = np.unique(rootNodes, return_counts=True)
    #         pred_roots, pred_roots_count = np.unique(pred_roots, return_counts=True)
    #
    #         # K-means CENTROIDS - ALL
    #         atmap = np.zeros((nodesXYZ.shape[0]))
    #         atmap[pred_roots] = np.round(100*pred_roots_count/len(pred_roots_list), 2)
    #         # Translate the atmap to be an element-wise map
    #         elem_atmap = np.zeros((elems.shape[0]))
    #         for i in range(elems.shape[0]):
    #             elem_atmap[i] = np.sum(atmap[elems[i]])
    #         with open(figPath+meshName+'_'+target_type+'_' +resolution+'.ensi.kMeans_centroids', 'w') as f:
    #             f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
    #             for i in range(0, len(elem_atmap)):
    #                 f.write(str(elem_atmap[i])+'\n')
    #
    #         # CUMM ROOT NODES TOGETHER - ALL
    #         atmap = np.zeros((nodesXYZ.shape[0]))
    #         atmap[rootNodes] = np.round(100*rootNodes_count/root_count, 2)#-6 # substracting the baseline value
    #         # Translate the atmap to be an element-wise map
    #         elem_atmap = np.zeros((elems.shape[0]))
    #         for i in range(elems.shape[0]):
    #             elem_atmap[i] = np.sum(atmap[elems[i]])
    #         with open(figPath+meshName+'_'+target_type+'_'
    #                     +resolution+'.ensi.cummrootNodes', 'w') as f:
    #             f.write('Eikonal Ensight Gold --- Scalar per-element variables file\npart\n\t1\ntetra4\n')
    #             for i in range(0, len(elem_atmap)):
    #                 f.write(str(elem_atmap[i])+'\n')
    #
    #         # INFERRED ACTIVATION MAP
    #         with open(figPath+meshName+'_'+target_type+'_' +resolution+'.ensi.ATMap', 'w') as f:
    #             f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #             for i in range(0, len(inferred_atmap)):
    #                 f.write(str(inferred_atmap[i]) + '\n')
    print('Done')
    
    # Some extra code
    max_count = np.sum(np.logical_not(np.isnan(target_output[0, :])))
    for i in range(len(result_summary)):
        max_count = max(max_count, result_summary[i].shape[0])
        print(result_summary[i].shape[0])
    max_count += 5
    
    colour_list = ['m', 'r', 'b']
    fig_summary, axs_summary = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(6, 1.3), sharey='all')
    for i in range(nb_leads):
        axs_res = axs_summary[i]
        # axs_res.plot(target_output[i, :], 'k-', label='Target', linewidth=1.2)
        axs_res.plot(target_output[i, :], 'k-', label='Clinical', linewidth=2.2)
        for j in range(len(result_summary)):
            result_ecg = result_summary[j]
            axs_res.plot(result_ecg[i, :], colour_list[j]+'-', label='Sim-'+str(j), linewidth=1.5) # 2021/12/15 - TODO Comment
        title_text = leadNames[i]
        # title_text = str(round(corr_mat_summary[0][i], 2))
        # for j in range(1, len(corr_mat_summary)):
        #     title_text = title_text+ '\n'+str(round(corr_mat_summary[j][i], 2)) # Could use latex format to have different colours in text (see
        axs_res.set_title(title_text, fontsize=10)  # https://stackoverflow.com/questions/36264305/matplotlib-multi-colored-title-text-in-practice)
        axs_res.xaxis.set_major_locator(MultipleLocator(40))
        axs_res.xaxis.set_minor_locator(MultipleLocator(20))
        axs_res.yaxis.set_major_locator(MultipleLocator(1))
        axs_res.yaxis.set_minor_locator(MultipleLocator(.5))
        axs_res.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs_res.xaxis.grid(True, which='major')
        axs_res.xaxis.grid(True, which='minor')
        axs_res.yaxis.grid(True, which='major')
        axs_res.yaxis.grid(True, which='minor')
        for tick in axs_res.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in axs_res.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
            
        axs_res.set_xlim(0, max_count)
        axs_res.set_ylim(-2., 2.) # 2022/08/14
        
    axs_res.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    
    figName = (figPath + meshName + '_clinical_ecg_summary.png')
    plt.savefig(figName, dpi=400)
    plt.show()
    with open(figPath + meshName + '_clinical_inferred_particles.csv', 'w') as f:
        for i in range(0, len(particle_summary)):
            particle = particle_summary[i]
            text = str(particle[0])
            for j in range(1, len(particle)):
                text = text + ',' + str(particle[j])
            f.write(text + '\n')
    with open(figPath + meshName + '_clinical_PCC.csv', 'w') as f:
        for i in range(0, len(corr_mat_summary)):
            cc = corr_mat_summary[i]
            text = str(cc[0])
            for j in range(1, len(cc)):
                text = text + ',' + str(cc[j])
            f.write(text + '\n')
    print('double Done')
    # return aggregation, coeficients_array, inference_ecg[0, :], target_output[0, :]

# ------------------------------------------- MAIN SMC-ABC FUNCTIONS ----------------------------------------------------

def run_inference_2021(meshName_val, meshVolume_val, final_path, tmp_path, target_type, metric, threadsNum_val, npart,
                    keep_fraction, rootNodeResolution, conduction_speeds, target_snr_db, healthy_val,
                    load_target, endocardial_layer, target_fileName, is_ECGi_val):
    
    purkinje_speed = 0.4 # cm/ms # the final purkinje network will use 0.19 but we don't need to use a slightly slower speed to
    # print('purkinje_speed: ' + str(purkinje_speed))
#allow for curved branches in the networks, because the times are here computed on a jiggsaw-like coarse mesh which will already
#add a little of buffer time
    global has_endocardial_layer
    has_endocardial_layer = endocardial_layer
    # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
    global is_healthy
    is_healthy = healthy_val
    global meshName
    meshName = meshName_val
    # global meshVolume
    # meshVolume = meshVolume_val
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = [ 'I' , 'II' , 'V1' , 'V2' , 'V3' , 'V4' , 'V5' , 'V6']#, 'lead_prog' ]
    global nb_leads
    global nb_bsp
    frequency = 1000 # Hz
    freq_cut= 150 # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2) # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    # global average_speed_from_Durrer_etal
    # average_speed_from_Durrer_etal = np.array([0.08, 0.0464, 0.03, 0.2]) # from Durrer et al's 80 cm/s 46.4 cm/s 30 cm/s and 200 cm/s
    if has_endocardial_layer:
        # endoRange = np.array([0.1, 0.3]) #np.array([0.08, 0.2]) # value range
        sparseEndoRange = np.array([0.07, 0.15]) # 03/12/2021
        denseEndoRange = np.array([0.1, 0.19]) # 03/12/2021
        # gtRange = np.array([0.01, 0.1]) # September 2021 set to same range for all speeds - originally np.array([0.03, 0.06])  # np.array([0.005, 0.09])  # gl and gn will be set as an initial +10%  -10%  of gt
    # else:
    #     gtRange = np.array([0.005, 0.12])  # gl and gn will be set as an initial +10% -10% of gt
     # Only used if is_helathy
    global gf_factor
    # gf_factor = 1.5
    gf_factor = 0.065 # 10/12/2021 - Taggart et al. (2000)
    # gf_factor = 0.067 # 10/12/2021 - Caldwell et al. (2009)
    global gn_factor
    # gn_factor = 0.7
    gn_factor = 0.048 # 10/12/2021 - Taggart et al. (2000)
    # gn_factor = 0.017 # 10/12/2021 - Caldwell et al. (2009)
    # print(gn_factor)
    if is_healthy:
        gtRange = np.array([0.025, 0.06]) # 07/12/2021
        # gfRange = gtRange * gf_factor
        # gnRange = gtRange * gn_factor
        # print(gfRange)
        # print(gnRange)
    else:
        gfRange = np.array([0.03, 0.09]) # 03/12/2021
        gtRange = np.array([0.02, 0.08]) # 03/12/2021
        gnRange = np.array([0.01, 0.07]) # 03/12/2021
   
    # SMC-ABC configuration
    # desired_Discrepancy = 0.01
    desired_Discrepancy = 0.35 #0.2 # 0.4 # 2022/01/18 # 0.3 # 05/12/2021 # 2022/05/04 # 2022/05/11
    # desired_Discrepancy = 0.5 # 10/12/2021
    # max_MCMC_steps = 100
    max_MCMC_steps = 100 # 05/12/2021
    global nlhsParam
    if has_endocardial_layer:
        if is_healthy:
            nlhsParam = 3 # 2022/01/18
        else:
            nlhsParam = 5 # 2022/01/18
    else:
        nlhsParam = 3
    # Specify the "retain ratio". This is the proportion of samples that would match the current data in the case of N_on = 1 and all particles having the same variable switched on. That is to say,
    # it is an approximate chance of choosing "random updates" over the particle information
    # retain_ratio = 0.5 # Better use 0.5, originally 0.9
    # retain_ratio = 0.7 # 04/12/2021 - reverted back to the original value in Brodie's code
    retain_ratio = 0.5 # 10/12/2021 - reverted back to the original value in Brodie's code
    nRootNodes_range = [3, 9] # Better use [3, 9], originally [2, 14]
    nRootNodes_centre = 6 # Better use 6, originally 7
    nRootNodes_std = 1  # Better use 1, originally 2
    global p_pdf
    # September 2021 - refactoring to make the array as long as the values that it can be and normalise it to add up to 1 probability
    p_pdf = np.empty((nRootNodes_range[1]-nRootNodes_range[0]+1), dtype='float64')
    for N_on in range(nRootNodes_range[0], nRootNodes_range[1]+1):
        p_pdf[N_on-nRootNodes_range[0]] = abs(norm.cdf(N_on-0.5, loc=nRootNodes_centre, scale=nRootNodes_std)
                                        - norm.cdf(N_on+0.5, loc=nRootNodes_centre, scale=nRootNodes_std))
    p_pdf = p_pdf/np.sum(p_pdf)
    global p_cdf
    p_cdf = np.cumsum(p_pdf)
    p_cdf[-1] = 1.1 # I set it to be larger than 1 to account for numerical errors from the round functions
    find_first_larger_than(0.5) # Compile function

    # CALCULATE MISSING RESULTS
    global experiment_output
    experiment_output = target_type
    global is_ECGi
    is_ECGi = is_ECGi_val
    # Paths and tags
    dataPath = 'metaData/' + meshName + '/'
    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_edges.csv', delimiter=',') - 1).astype(int)
    global lvnodes
    lvnodes = np.unique((np.loadtxt(dataPath + meshName + '_lvnodes.csv', delimiter=',') - 1).astype(int)) # lv endocardium triangles
    global rvnodes
    rvnodes = np.unique((np.loadtxt(dataPath + meshName + '_rvnodes.csv', delimiter=',') - 1).astype(int)) # rv endocardium triangles
    # Load potential root node indexes
    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_lv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int) # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_rv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int)
    # Project canditate root nodes to the endocardial layer: As from 2021 the root nodes are always generated at the endocardium, but this is still a good security measure
    for i in range(lvActivationIndexes.shape[0]):
        if lvActivationIndexes[i] not in lvnodes:
            lvActivationIndexes[i] = lvnodes[np.argmin(
                np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    for i in range(rvActivationIndexes.shape[0]):
        if rvActivationIndexes[i] not in rvnodes:
            rvActivationIndexes[i] = rvnodes[np.argmin(
                np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    global rootNodeActivationIndexes
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    # Project canditate root nodes to the endocardial layer: As from 2021 the root nodes are always generated at the endocardium, but this is still a good security measure
    for i in range(lvActivationIndexes.shape[0]):
        if lvActivationIndexes[i] not in lvnodes:
            lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
    for i in range(rvActivationIndexes.shape[0]):
        if rvActivationIndexes[i] not in rvnodes:
            rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
    # # Start of new Purkinje code # 2022/01/17
    # lv_his_bundle_nodes = np.loadtxt(dataPath + meshName + '_lvhisbundle_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
    # rv_his_bundle_nodes = np.loadtxt(dataPath + meshName + '_rvhisbundle_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
    # lv_dense_nodes = np.loadtxt(dataPath + meshName + '_lvdense_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # rv_dense_nodes = np.loadtxt(dataPath + meshName + '_rvdense_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_septalwall_nodes = np.loadtxt(dataPath + meshName + '_lvseptalwall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_freewallnavigation_nodes = np.loadtxt(dataPath + meshName + '_lvfreewallnavigationpoints_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_freewall_nodes = np.loadtxt(dataPath + meshName + '_lvfreewallextended_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/04/05
    # # lv_interpapillary_freewall_nodes = np.loadtxt(dataPath + meshName + '_lvfreewall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/04/05
    # # LV nodes that aren't freewall, septal or dense are considered to be paraseptal
    # rv_freewall_nodes = np.loadtxt(dataPath + meshName + '_rvfreewall_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # rv_lowerthird_nodes = np.loadtxt(dataPath + meshName + '_rvlowerthird_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/04/06
    # lv_histop_nodes = np.loadtxt(dataPath + meshName + '_lvhistop_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # rv_histop_nodes = np.loadtxt(dataPath + meshName + '_rvhistop_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # lv_hisbundleConnected_nodes = np.loadtxt(dataPath + meshName + '_lvhisbundleConnected_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # rv_hisbundleConnected_nodes = np.loadtxt(dataPath + meshName + '_rvhisbundleConnected_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3)) # 2022/01/11
    # lv_apexnavigation_nodes = np.loadtxt(dataPath + meshName + '_lvapexnavigation_xyz.csv', delimiter=',', skiprows=1, usecols=(1,2,3))[np.newaxis, :] # 2022/01/11
    # # Project xyz points to nodes in the endocardial layer
    # if True:
    #     lv_his_bundle_indexes = np.zeros((lv_his_bundle_nodes.shape[0])).astype(int)
    #     for i in range(lv_his_bundle_nodes.shape[0]):
    #         lv_his_bundle_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_his_bundle_indexes, axis=0, return_index=True)[1] # use the unique function without sorting the contents of the array
    #     lv_his_bundle_indexes = lv_his_bundle_indexes[sorted(indexes)]
    #
    #     rv_his_bundle_indexes = np.zeros((rv_his_bundle_nodes.shape[0])).astype(int)
    #     for i in range(rv_his_bundle_nodes.shape[0]):
    #         rv_his_bundle_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_his_bundle_indexes, axis=0, return_index=True)[1]
    #     rv_his_bundle_indexes = rv_his_bundle_indexes[sorted(indexes)]
    #
    #     lv_dense_indexes = np.zeros((lv_dense_nodes.shape[0])).astype(int)
    #     for i in range(lv_dense_nodes.shape[0]):
    #         lv_dense_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_dense_indexes, axis=0, return_index=True)[1]
    #     lv_dense_indexes = lv_dense_indexes[sorted(indexes)]
    #
    #     rv_dense_indexes = np.zeros((rv_dense_nodes.shape[0])).astype(int)
    #     for i in range(rv_dense_nodes.shape[0]):
    #         rv_dense_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_dense_indexes, axis=0, return_index=True)[1]
    #     rv_dense_indexes = rv_dense_indexes[sorted(indexes)]
    #
    #     lv_septalwall_indexes = np.zeros((lv_septalwall_nodes.shape[0])).astype(int)
    #     for i in range(lv_septalwall_nodes.shape[0]):
    #         lv_septalwall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_septalwall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_septalwall_indexes, axis=0, return_index=True)[1]
    #     lv_septalwall_indexes = lv_septalwall_indexes[sorted(indexes)]
    #
    #     lv_freewallnavigation_indexes = np.zeros((lv_freewallnavigation_nodes.shape[0])).astype(int)
    #     for i in range(lv_freewallnavigation_nodes.shape[0]):
    #         lv_freewallnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewallnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_freewallnavigation_indexes, axis=0, return_index=True)[1]
    #     lv_freewallnavigation_indexes = lv_freewallnavigation_indexes[sorted(indexes)]
    #
    #     lv_freewall_indexes = np.zeros((lv_freewall_nodes.shape[0])).astype(int)
    #     for i in range(lv_freewall_nodes.shape[0]):
    #         lv_freewall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_freewall_indexes, axis=0, return_index=True)[1]
    #     lv_freewall_indexes = lv_freewall_indexes[sorted(indexes)]
    #
    #     rv_freewall_indexes = np.zeros((rv_freewall_nodes.shape[0])).astype(int)
    #     for i in range(rv_freewall_nodes.shape[0]):
    #         rv_freewall_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_freewall_indexes, axis=0, return_index=True)[1]
    #     rv_freewall_indexes = rv_freewall_indexes[sorted(indexes)]
    #
    #     rv_lowerthird_indexes = np.zeros((rv_lowerthird_nodes.shape[0])).astype(int)
    #     for i in range(rv_lowerthird_nodes.shape[0]):
    #         rv_lowerthird_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_lowerthird_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_lowerthird_indexes, axis=0, return_index=True)[1]
    #     rv_lowerthird_indexes = rv_lowerthird_indexes[sorted(indexes)]
    #
    #     lv_histop_indexes = np.zeros((lv_histop_nodes.shape[0])).astype(int)
    #     for i in range(lv_histop_nodes.shape[0]):
    #         lv_histop_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_histop_indexes, axis=0, return_index=True)[1]
    #     lv_histop_indexes = lv_histop_indexes[sorted(indexes)]
    #
    #     rv_histop_indexes = np.zeros((rv_histop_nodes.shape[0])).astype(int)
    #     for i in range(rv_histop_nodes.shape[0]):
    #         rv_histop_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_histop_indexes, axis=0, return_index=True)[1]
    #     rv_histop_indexes = rv_histop_indexes[sorted(indexes)]
    #
    #     lv_hisbundleConnected_indexes = np.zeros((lv_hisbundleConnected_nodes.shape[0])).astype(int)
    #     for i in range(lv_hisbundleConnected_nodes.shape[0]):
    #         lv_hisbundleConnected_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    #     lv_hisbundleConnected_indexes = lv_hisbundleConnected_indexes[sorted(indexes)]
    #
    #     rv_hisbundleConnected_indexes = np.zeros((rv_hisbundleConnected_nodes.shape[0])).astype(int)
    #     for i in range(rv_hisbundleConnected_nodes.shape[0]):
    #         rv_hisbundleConnected_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    #     rv_hisbundleConnected_indexes = rv_hisbundleConnected_indexes[sorted(indexes)]
    #
    #     lv_apexnavigation_indexes = np.zeros((lv_apexnavigation_nodes.shape[0])).astype(int)
    #     for i in range(lv_apexnavigation_nodes.shape[0]):
    #         lv_apexnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_apexnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_apexnavigation_indexes, axis=0, return_index=True)[1]
    #     lv_apexnavigation_indexes = lv_apexnavigation_indexes[sorted(indexes)]
    #
    # # Set LV endocardial edges aside
    # lvnodesXYZ = nodesXYZ[lvnodes, :]
    # lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    # lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    # lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    # lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
    # lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    # aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    # for i in range(0, len(lvunfoldedEdges), 1):
    #     aux[lvunfoldedEdges[i, 0]].append(i)
    # lvneighbours = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    # # Calculate Djikstra distances from the LV connected (branching) part in the his-Bundle (meshes are in cm)
    # lv_hisbundleConnected_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_hisbundleConnected_indexes]).astype(int)
    # lvHisBundledistance_mat, lvHisBundlepath_mat = djikstra(lv_hisbundleConnected_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # Calculate Djikstra distances from the LV freewall (meshes are in cm)
    # lv_freewallnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_freewallnavigation_indexes]).astype(int)
    # lvFreewalldistance_mat, lvFreewallpath_mat = djikstra(lv_freewallnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # Calculate Djikstra distances from the paraseptalwall (meshes are in cm)
    # lv_apexnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_apexnavigation_indexes]).astype(int)
    # lvParaseptalwalldistance_mat, lvParaseptalwallpath_mat = djikstra(lv_apexnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/04/05
    # # Calculate the offsets to the top of the LV his bundle for the LV his bundle connected nodes
    # lv_histop_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_histop_indexes]).astype(int)
    # lvHisBundledistance_offset = lvHisBundledistance_mat[lv_histop_ids[0], :] # offset to the top of the his-bundle
    # # Calculate within the his-bundle for plotting purposes
    # lvHis_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_his_bundle_indexes]).astype(int)
    # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[lvnodes[lvHis_ids], :] - nodesXYZ[lvnodes[lv_histop_ids], :], 2), axis=1))
    # sorted_indexes = np.argsort(distances_to_top)
    # sorted_lvHis_ids = lvHis_ids[sorted_indexes]
    # lv_edges_his = np.array([np.array([lvnodes[sorted_lvHis_ids[i]], lvnodes[sorted_lvHis_ids[i+1]]]) for i in range(0, sorted_lvHis_ids.shape[0]-1, 1)])
    # # Calculate the offset to the apex navigation point
    # apex_navigation_reference_his_node = np.argmin(lvHisBundledistance_mat[lv_apexnavigation_ids[0], :])
    # lvapexdistance_offset = lvHisBundledistance_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node] + lvHisBundledistance_offset[apex_navigation_reference_his_node] # offset of the apex node itself
    # lvapexpath_offset = lvHisBundlepath_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node, :]
    # # Make the paths on the lateral walls go to the free wall and then down the apex
    # lvFreeWalldistance_offset = (lvFreewalldistance_mat[lv_apexnavigation_ids[0], :] + lvapexdistance_offset) # offset to the pass-through node in the apex + offset of the apex node itself
    # lvFreeWallpath_offset = np.concatenate((lvFreewallpath_mat[lv_apexnavigation_ids[0], :, :], np.tile(lvapexpath_offset, (lvFreewallpath_mat.shape[1], 1))), axis=1)
    # # Make the paths on the paraseptal walls go down the apex
    # lvParaseptalWalldistance_offset = lvapexdistance_offset # offset of the apex node itself
    # lvParaseptalWallpath_offset = np.tile(lvapexpath_offset, (lvParaseptalwallpath_mat.shape[1], 1))
    # # Add the pieces of itinerary to the so called "free wall paths", mostly for plotting purposes
    # lvFreewallpath_mat = np.concatenate((lvFreewallpath_mat, np.tile(lvFreeWallpath_offset, (lvFreewallpath_mat.shape[0], 1, 1))), axis=2)
    # # Add the pieces of itinerary to the so called "paraseptal wall paths", mostly for plotting purposes
    # lvParaseptalwallpath_mat = np.concatenate((lvParaseptalwallpath_mat, np.tile(lvParaseptalWallpath_offset, (lvParaseptalwallpath_mat.shape[0], 1, 1))), axis=2)
    # # For each endocardial node, chose which path it should take: directly to the his bundle, or through the free wall
    # lvHisBundledistance_vec_indexes = np.argmin(lvHisBundledistance_mat, axis=1)
    # lvFreewalldistance_vec_indexes = np.argmin(lvFreewalldistance_mat, axis=1)
    # lvParaseptaldistance_vec_indexes = np.argmin(lvParaseptalwalldistance_mat, axis=1) # if there is only one routing node in the apex, all indexes will be zero
    # # Initialise data structures
    # lv_PK_distance_mat = np.full((lvnodes.shape[0]), np.nan, np.float64)
    # lv_PK_path_mat = np.full((lvnodes.shape[0], max(lvHisBundlepath_mat.shape[2], lvFreewallpath_mat.shape[2])), np.nan, np.int32)
    # # Assign paths to each subgroup of root nodes
    # for endo_node_i in range(lvnodes.shape[0]):
    #     if lvnodes[endo_node_i] in lv_dense_indexes or lvnodes[endo_node_i] in lv_septalwall_indexes: # Septal-wall and dense region
    #         lv_PK_distance_mat[endo_node_i] = lvHisBundledistance_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i]] + lvHisBundledistance_offset[lvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvHisBundlepath_mat.shape[2]] = lvHisBundlepath_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i], :]
    #     elif lvnodes[endo_node_i] in lv_freewall_indexes: # FreeWall
    #         # aux_0 = aux_0 + 1
    #         lv_PK_distance_mat[endo_node_i] = lvFreewalldistance_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i]] + lvFreeWalldistance_offset[lvFreewalldistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvFreewallpath_mat.shape[2]] = lvFreewallpath_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i], :]
    #     else: # Paraseptal
    #         lv_PK_distance_mat[endo_node_i] = lvParaseptalwalldistance_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i]] + lvParaseptalWalldistance_offset #[lvParaseptaldistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvParaseptalwallpath_mat.shape[2]] = lvParaseptalwallpath_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i], :]
    # # Time cost from each point in the left his-bundle to every point in the LV endocardium
    # lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    #
    # # Set RV endocardial edges aside
    # rvnodesXYZ = nodesXYZ[rvnodes, :]
    # rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    # rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    # rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    # rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
    # rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    # aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    # for i in range(0, len(rvunfoldedEdges), 1):
    #     aux[rvunfoldedEdges[i, 0]].append(i)
    # rvneighbours = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    # # Calculate Djikstra distances from the RV connected (branching) part in the his-Bundle (meshes are in cm)
    # rv_hisbundleConnected_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_hisbundleConnected_indexes]).astype(int)
    # rvHisBundledistance_mat, rvHisBundlepath_mat = djikstra(rv_hisbundleConnected_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # 2022/01/10
    # # Calculate the offsets to the top of the RV his bundle for the RV his bundle connected nodes
    # rv_histop_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_histop_indexes]).astype(int)
    # rvHisBundledistance_offset = rvHisBundledistance_mat[rv_histop_ids[0], :] # offset to the top of the his-bundle
    # # Calculate within the his-bundle for plotting purposes
    # rvHis_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_his_bundle_indexes]).astype(int)
    # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[rvnodes[rvHis_ids], :] - nodesXYZ[rvnodes[rv_histop_ids], :], 2), axis=1))
    # sorted_indexes = np.argsort(distances_to_top)
    # sorted_rvHis_ids = rvHis_ids[sorted_indexes]
    # rv_edges_his = np.array([np.array([rvnodes[sorted_rvHis_ids[i]], rvnodes[sorted_rvHis_ids[i+1]]]) for i in range(0, sorted_rvHis_ids.shape[0]-1, 1)])
    # # Calculate Crossing Distances RV freewall (meshes are in cm)
    # rvCrossingHisBundledistance_mat = np.sqrt(np.sum(np.power(rvnodesXYZ[:, np.newaxis, :] - rvnodesXYZ[rvHis_ids, :], 2), axis=2))
    # rvCrossingHisBundlepath_mat = np.full((rvnodes.shape[0], rvHis_ids.shape[0], 2), np.nan, np.int32)
    # for i in range(rvnodesXYZ.shape[0]):
    #     for j in range(rvHis_ids.shape[0]):
    #         rvCrossingHisBundlepath_mat[i, j, :] = np.array([i, rvHis_ids[j]])
    # # Calculate the offsets to the top of the RV his bundle for the RV his bundle NOT-connected nodes
    # rv_histopdistance_mat, rv_histoppath_mat = djikstra(rv_histop_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # offets to the top of the his 2022/01/10
    # rvCrossingHisBundledistance_offset = np.squeeze(rv_histopdistance_mat[rvHis_ids, :])
    # # For each endocardial node, chose which path it should take: directly to the his bundle following the endocardial wall or as a false tendon from the his bundle and crossing the cavity
    # rvHisBundledistance_vec_indexes = np.argmin(rvHisBundledistance_mat, axis=1)
    # rvCrossingHisBundledistance_vec_indexes = np.argmin(rvCrossingHisBundledistance_mat, axis=1)
    # rv_edges_crossing = []
    # rv_PK_distance_mat = np.full((rvnodes.shape[0]), np.nan, np.float64)
    # rv_PK_path_mat = np.full((rvnodes.shape[0], max(rvHisBundlepath_mat.shape[2], rvCrossingHisBundlepath_mat.shape[2])), np.nan, np.int32)
    # # rv_freewall_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_freewall_indexes]).astype(int)
    # for endo_node_i in range(rvnodes.shape[0]):
    #     # if rvnodes[endo_node_i] in rv_freewall_indexes:
    #     if rvnodes[endo_node_i] in rv_lowerthird_indexes: # Restrain false tendons to only the lower 1/3 in the RV 2022/04/06
    #         rv_PK_distance_mat[endo_node_i] = rvCrossingHisBundledistance_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i]] + rvCrossingHisBundledistance_offset[rvCrossingHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         rv_PK_path_mat[endo_node_i, :rvCrossingHisBundlepath_mat.shape[2]] = rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :]
    #         rv_edges_crossing.append(rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :])
    #     else:
    #         rv_PK_distance_mat[endo_node_i] = rvHisBundledistance_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i]] + rvHisBundledistance_offset[rvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         rv_PK_path_mat[endo_node_i, :rvHisBundlepath_mat.shape[2]] = rvHisBundlepath_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i], :]
    #         rv_edges_crossing.append(np.array([nan_value, nan_value], dtype=np.int32))
    # # Time cost from each point in the left his-bundle to every point in the RV endocardium (for plotting purposes)
    # rv_edges_crossing = np.asarray(rv_edges_crossing, dtype=np.int32)
    # rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms
    
    
    
    
    # if True:
    #     lv_his_bundle_indexes = np.zeros((lv_his_bundle_nodes.shape[0])).astype(int)
    #     for i in range(lv_his_bundle_nodes.shape[0]):
    #         lv_his_bundle_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_his_bundle_indexes, axis=0, return_index=True)[1] # use the unique function without sorting the contents of the array
    #     lv_his_bundle_indexes = lv_his_bundle_indexes[sorted(indexes)]
    #
    #     rv_his_bundle_indexes = np.zeros((rv_his_bundle_nodes.shape[0])).astype(int)
    #     for i in range(rv_his_bundle_nodes.shape[0]):
    #         rv_his_bundle_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_his_bundle_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_his_bundle_indexes, axis=0, return_index=True)[1]
    #     rv_his_bundle_indexes = rv_his_bundle_indexes[sorted(indexes)]
    #
    #     lv_dense_indexes = np.zeros((lv_dense_nodes.shape[0])).astype(int)
    #     for i in range(lv_dense_nodes.shape[0]):
    #         lv_dense_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_dense_indexes, axis=0, return_index=True)[1]
    #     lv_dense_indexes = lv_dense_indexes[sorted(indexes)]
    #
    #     rv_dense_indexes = np.zeros((rv_dense_nodes.shape[0])).astype(int)
    #     for i in range(rv_dense_nodes.shape[0]):
    #         rv_dense_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_dense_indexes, axis=0, return_index=True)[1]
    #     rv_dense_indexes = rv_dense_indexes[sorted(indexes)]
    #
    #     lv_septalwall_indexes = np.zeros((lv_septalwall_nodes.shape[0])).astype(int)
    #     for i in range(lv_septalwall_nodes.shape[0]):
    #         lv_septalwall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_septalwall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_septalwall_indexes, axis=0, return_index=True)[1]
    #     lv_septalwall_indexes = lv_septalwall_indexes[sorted(indexes)]
    #
    #     lv_freewallnavigation_indexes = np.zeros((lv_freewallnavigation_nodes.shape[0])).astype(int)
    #     for i in range(lv_freewallnavigation_nodes.shape[0]):
    #         lv_freewallnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewallnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_freewallnavigation_indexes, axis=0, return_index=True)[1]
    #     lv_freewallnavigation_indexes = lv_freewallnavigation_indexes[sorted(indexes)]
    #
    #     lv_freewall_indexes = np.zeros((lv_freewall_nodes.shape[0])).astype(int)
    #     for i in range(lv_freewall_nodes.shape[0]):
    #         lv_freewall_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_freewall_indexes, axis=0, return_index=True)[1]
    #     lv_freewall_indexes = lv_freewall_indexes[sorted(indexes)]
    #
    #     rv_freewall_indexes = np.zeros((rv_freewall_nodes.shape[0])).astype(int)
    #     for i in range(rv_freewall_nodes.shape[0]):
    #         rv_freewall_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_freewall_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_freewall_indexes, axis=0, return_index=True)[1]
    #     rv_freewall_indexes = rv_freewall_indexes[sorted(indexes)]
    #
    #     lv_histop_indexes = np.zeros((lv_histop_nodes.shape[0])).astype(int)
    #     for i in range(lv_histop_nodes.shape[0]):
    #         lv_histop_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_histop_indexes, axis=0, return_index=True)[1]
    #     lv_histop_indexes = lv_histop_indexes[sorted(indexes)]
    #
    #     rv_histop_indexes = np.zeros((rv_histop_nodes.shape[0])).astype(int)
    #     for i in range(rv_histop_nodes.shape[0]):
    #         rv_histop_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_histop_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_histop_indexes, axis=0, return_index=True)[1]
    #     rv_histop_indexes = rv_histop_indexes[sorted(indexes)]
    #
    #     lv_hisbundleConnected_indexes = np.zeros((lv_hisbundleConnected_nodes.shape[0])).astype(int)
    #     for i in range(lv_hisbundleConnected_nodes.shape[0]):
    #         lv_hisbundleConnected_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    #     lv_hisbundleConnected_indexes = lv_hisbundleConnected_indexes[sorted(indexes)]
    #
    #     rv_hisbundleConnected_indexes = np.zeros((rv_hisbundleConnected_nodes.shape[0])).astype(int)
    #     for i in range(rv_hisbundleConnected_nodes.shape[0]):
    #         rv_hisbundleConnected_indexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - rv_hisbundleConnected_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(rv_hisbundleConnected_indexes, axis=0, return_index=True)[1]
    #     rv_hisbundleConnected_indexes = rv_hisbundleConnected_indexes[sorted(indexes)]
    #
    #     lv_apexnavigation_indexes = np.zeros((lv_apexnavigation_nodes.shape[0])).astype(int)
    #     for i in range(lv_apexnavigation_nodes.shape[0]):
    #         lv_apexnavigation_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_apexnavigation_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_apexnavigation_indexes, axis=0, return_index=True)[1]
    #     lv_apexnavigation_indexes = lv_apexnavigation_indexes[sorted(indexes)]
    #
    # # Set LV endocardial edges aside
    # lvnodesXYZ = nodesXYZ[lvnodes, :]
    # lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    # lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    # lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    # lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
    # lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    # aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    # for i in range(0, len(lvunfoldedEdges), 1):
    #     aux[lvunfoldedEdges[i, 0]].append(i)
    # lvneighbours = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    # # Calculate Djikstra distances from the LV connected (branching) part in the his-Bundle (meshes are in cm)
    # lv_hisbundleConnected_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_hisbundleConnected_indexes]).astype(int)
    # lvHisBundledistance_mat, lvHisBundlepath_mat = djikstra(lv_hisbundleConnected_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # Calculate Djikstra distances from the LV freewall (meshes are in cm)
    # lv_freewallnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_freewallnavigation_indexes]).astype(int)
    # lvFreewalldistance_mat, lvFreewallpath_mat = djikstra(lv_freewallnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/01/11
    # # Calculate Djikstra distances from the paraseptalwall (meshes are in cm)
    # lv_apexnavigation_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_apexnavigation_indexes]).astype(int)
    # lvParaseptalwalldistance_mat, lvParaseptalwallpath_mat = djikstra(lv_apexnavigation_ids, lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=100) # 2022/04/05
    # #TODO: chain this with the end of the hisbundle and set it as the last option, aka paraseptal
    # # Calculate the offsets to the top of the LV his bundle for the LV his bundle connected nodes
    # lv_histop_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_histop_indexes]).astype(int)
    # lvHisBundledistance_offset = lvHisBundledistance_mat[lv_histop_ids[0], :] # offset to the top of the his-bundle
    # # Calculate within the his-bundle for plotting purposes
    # lvHis_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lv_his_bundle_indexes]).astype(int)
    # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[lvnodes[lvHis_ids], :] - nodesXYZ[lvnodes[lv_histop_ids], :], 2), axis=1))
    # sorted_indexes = np.argsort(distances_to_top)
    # sorted_lvHis_ids = lvHis_ids[sorted_indexes]
    # lv_edges_his = np.array([np.array([lvnodes[sorted_lvHis_ids[i]], lvnodes[sorted_lvHis_ids[i+1]]]) for i in range(0, sorted_lvHis_ids.shape[0]-1, 1)])
    # # Calculate the offset to the apex navigation point
    # apex_navigation_reference_his_node = np.argmin(lvHisBundledistance_mat[lv_apexnavigation_ids[0], :])
    # lvapexdistance_offset = (lvHisBundledistance_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node] + lvHisBundledistance_offset[apex_navigation_reference_his_node]) # offset of the apex node itself
    # lvapexpath_offset = lvHisBundlepath_mat[lv_apexnavigation_ids[0], apex_navigation_reference_his_node, :]
    # # Make the paths on the lateral walls go to the free wall and then down the apex
    # lvFreeWalldistance_offset = (lvFreewalldistance_mat[lv_apexnavigation_ids[0], :] + lvapexdistance_offset) # offset to the pass-through node in the apex + offset of the apex node itself
    # lvFreeWallpath_offset = np.concatenate((lvFreewallpath_mat[lv_apexnavigation_ids[0], :, :], np.tile(lvapexpath_offset, (lvFreewallpath_mat.shape[1], 1))), axis=1)
    # # Make the paths on the paraseptal walls go down the apex
    # lvParaseptalWalldistance_offset = lvapexdistance_offset # offset of the apex node itself
    # lvParaseptalWallpath_offset = np.tile(lvapexpath_offset, (lvParaseptalwallpath_mat.shape[1], 1))
    # # Add the pieces of itinerary to the so called "free wall paths", mostly for plotting purposes
    # lvFreewallpath_mat = np.concatenate((lvFreewallpath_mat, np.tile(lvFreeWallpath_offset, (lvFreewallpath_mat.shape[0], 1, 1))), axis=2)
    # # For each endocardial node, chose which path it should take: directly to the his bundle, through the apex or through the free wall and then apex
    # lvHisBundledistance_vec_indexes = np.argmin(lvHisBundledistance_mat, axis=1)
    # lvFreewalldistance_vec_indexes = np.argmin(lvFreewalldistance_mat, axis=1)
    # lvParaseptaldistance_vec_indexes = np.argmin(lvParaseptalwalldistance_mat, axis=1)
    # print('lvParaseptalwalldistance_mat')
    # print(lvParaseptalwalldistance_mat.shape)
    # print(lvParaseptaldistance_vec_indexes.shape)
    # print(lvFreewalldistance_mat.shape)
    # print(lvFreewalldistance_vec_indexes.shape)
    # lv_PK_distance_mat = np.full((lvnodes.shape[0]), np.nan, np.float64)
    # lv_PK_path_mat = np.full((lvnodes.shape[0], max(lvHisBundlepath_mat.shape[2], lvFreewallpath_mat.shape[2])), np.nan, np.int32)
    # for endo_node_i in range(lvnodes.shape[0]):
    #     if lvnodes[endo_node_i] in lv_dense_indexes or lvnodes[endo_node_i] in lv_septalwall_indexes: # Septal-wall and dense region
    #         lv_PK_distance_mat[endo_node_i] = lvHisBundledistance_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i]] + lvHisBundledistance_offset[lvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvHisBundlepath_mat.shape[2]] = lvHisBundlepath_mat[endo_node_i, lvHisBundledistance_vec_indexes[endo_node_i], :]
    #     elif lvnodes[endo_node_i] in lv_freewall_indexes: # FreeWall
    #         lv_PK_distance_mat[endo_node_i] = lvFreewalldistance_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i]] + lvFreeWalldistance_offset[lvFreewalldistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvFreewallpath_mat.shape[2]] = lvFreewallpath_mat[endo_node_i, lvFreewalldistance_vec_indexes[endo_node_i], :]
    #     else: # Paraseptal
    #         lv_PK_distance_mat[endo_node_i] = lvParaseptalwalldistance_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i]] + lvParaseptalWalldistance_offset[lvParaseptaldistance_vec_indexes[
    #         endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         lv_PK_path_mat[endo_node_i, :lvParaseptalwallpath_mat.shape[2]] = lvParaseptalwallpath_mat[endo_node_i, lvParaseptaldistance_vec_indexes[endo_node_i], :]
    # # Time cost from each point in the left his-bundle to every point in the LV endocardium
    # lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    #
    # # Set RV endocardial edges aside
    # rvnodesXYZ = nodesXYZ[rvnodes, :]
    # rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    # rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    # rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    # rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
    # rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    # aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    # for i in range(0, len(rvunfoldedEdges), 1):
    #     aux[rvunfoldedEdges[i, 0]].append(i)
    # rvneighbours = [np.array(n) for n in aux]
    # aux = None  # Clear Memory
    # # Calculate Djikstra distances from the RV connected (branching) part in the his-Bundle (meshes are in cm)
    # rv_hisbundleConnected_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_hisbundleConnected_indexes]).astype(int)
    # rvHisBundledistance_mat, rvHisBundlepath_mat = djikstra(rv_hisbundleConnected_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # 2022/01/10
    # # Calculate the offsets to the top of the RV his bundle for the RV his bundle connected nodes
    # rv_histop_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_histop_indexes]).astype(int)
    # rvHisBundledistance_offset = rvHisBundledistance_mat[rv_histop_ids[0], :] # offset to the top of the his-bundle
    # # Calculate within the his-bundle for plotting purposes
    # rvHis_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_his_bundle_indexes]).astype(int)
    # distances_to_top = np.sqrt(np.sum(np.power(nodesXYZ[rvnodes[rvHis_ids], :] - nodesXYZ[rvnodes[rv_histop_ids], :], 2), axis=1))
    # sorted_indexes = np.argsort(distances_to_top)
    # sorted_rvHis_ids = rvHis_ids[sorted_indexes]
    # rv_edges_his = np.array([np.array([rvnodes[sorted_rvHis_ids[i]], rvnodes[sorted_rvHis_ids[i+1]]]) for i in range(0, sorted_rvHis_ids.shape[0]-1, 1)])
    # # Calculate Crossing Distances RV freewall (meshes are in cm)
    # rvCrossingHisBundledistance_mat = np.sqrt(np.sum(np.power(rvnodesXYZ[:, np.newaxis, :] - rvnodesXYZ[rvHis_ids, :], 2), axis=2))
    # rvCrossingHisBundlepath_mat = np.full((rvnodes.shape[0], rvHis_ids.shape[0], 2), np.nan, np.int32)
    # for i in range(rvnodesXYZ.shape[0]):
    #     for j in range(rvHis_ids.shape[0]):
    #         rvCrossingHisBundlepath_mat[i, j, :] = np.array([i, rvHis_ids[j]])
    # # Calculate the offsets to the top of the RV his bundle for the RV his bundle NOT-connected nodes
    # rv_histopdistance_mat, rv_histoppath_mat = djikstra(rv_histop_ids, rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours) # offets to the top of the his 2022/01/10
    # rvCrossingHisBundledistance_offset = np.squeeze(rv_histopdistance_mat[rvHis_ids, :])
    # # For each endocardial node, chose which path it should take: directly to the his bundle following the endocardial wall or as a false tendon from the his bundle and crossing the cavity
    # rvHisBundledistance_vec_indexes = np.argmin(rvHisBundledistance_mat, axis=1)
    # rvCrossingHisBundledistance_vec_indexes = np.argmin(rvCrossingHisBundledistance_mat, axis=1)
    # rv_edges_crossing = []
    # rv_PK_distance_mat = np.full((rvnodes.shape[0]), np.nan, np.float64)
    # rv_PK_path_mat = np.full((rvnodes.shape[0], max(rvHisBundlepath_mat.shape[2], rvCrossingHisBundlepath_mat.shape[2])), np.nan, np.int32)
    # # rv_freewall_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rv_freewall_indexes]).astype(int)
    # for endo_node_i in range(rvnodes.shape[0]):
    #     if rvnodes[endo_node_i] in rv_freewall_indexes:
    #         rv_PK_distance_mat[endo_node_i] = rvCrossingHisBundledistance_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i]] + rvCrossingHisBundledistance_offset[rvCrossingHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         rv_PK_path_mat[endo_node_i, :rvCrossingHisBundlepath_mat.shape[2]] = rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :]
    #         rv_edges_crossing.append(rvCrossingHisBundlepath_mat[endo_node_i, rvCrossingHisBundledistance_vec_indexes[endo_node_i], :])
    #     else:
    #         rv_PK_distance_mat[endo_node_i] = rvHisBundledistance_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i]] + rvHisBundledistance_offset[rvHisBundledistance_vec_indexes[endo_node_i]] # Purkinje activation times to all points in the endocardium
    #         rv_PK_path_mat[endo_node_i, :rvHisBundlepath_mat.shape[2]] = rvHisBundlepath_mat[endo_node_i, rvHisBundledistance_vec_indexes[endo_node_i], :]
    #         rv_edges_crossing.append(np.array([nan_value, nan_value], dtype=np.int32))
    # # Time cost from each point in the left his-bundle to every point in the RV endocardium (for plotting purposes)
    # rv_edges_crossing = np.asarray(rv_edges_crossing, dtype=np.int32)
    # rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms

    # Time cost from each root node
    
    # Generate pseudo-Purkinje structure
    lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, lv_dense_indexes, rv_dense_indexes, nodesCobiveco = generatePurkinjeWithCobiveco(dataPath=dataPath, meshName=meshName)
    lv_PK_time_mat = lv_PK_distance_mat/purkinje_speed # time is in ms
    rv_PK_time_mat = rv_PK_distance_mat/purkinje_speed # time is in ms
    
    global rootNodeActivationTimes
    rootNodeActivationTimes = np.around(np.concatenate((lv_PK_time_mat[lvActnode_ids], rv_PK_time_mat[rvActnode_ids]), axis=0), decimals=4) # 02/12/2021 Tested how many decimals were needed to get the
    
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions
    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors
    
    nRootLocations=rootNodeActivationIndexes.shape[0]
    nparam = nlhsParam + nRootLocations
    if is_ECGi:
        rootNodesIndexes_true = rootNodeActivationIndexes
    else:
        rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_rootNodes.csv') - 1).astype(int)
        rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    global target_rootNodes
    target_rootNodes = nodesXYZ[rootNodesIndexes_true, :]
    if has_endocardial_layer:
        # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
        if is_healthy:
            # param_boundaries = np.concatenate((np.array([gtRange, endoRange]), np.array([[0, 1] for i in range(nRootLocations)])))
            param_boundaries = np.concatenate((np.array([gtRange, sparseEndoRange, denseEndoRange]), np.array([[0, 1] for i in range(nRootLocations)]))) # 2022/01/18
        else:
        # 23/12/2020: THIS SECTION WAS CHANGED ON THE TO MATCH THE MIA 2021 PAPER
        # param_boundaries = np.concatenate((np.array([gtRange, gtRange, gtRange, endoRange]), np.array([[0, 1] for i in range(nRootLocations)]))) # September 2021 set to same range for all speeds
        #     param_boundaries = np.concatenate((np.array([gfRange, gtRange, gnRange, endoRange]), np.array([[0, 1] for i in range(nRootLocations)]))) # 04/12/2021
            param_boundaries = np.concatenate((np.array([gfRange, gtRange, gnRange, sparseEndoRange, denseEndoRange]), np.array([[0, 1] for i in range(nRootLocations)]))) # 2022/01/18
    else:
        if is_healthy:
            param_boundaries = np.concatenate((np.array([gtRange]), np.array([[0, 1] for i in range(nRootLocations)]))) # 07/12/2021
        else:
            # param_boundaries = np.concatenate((np.array([gtRange, gtRange, gtRange]), np.array([[0, 1] for i in range(nRootLocations)])))
            param_boundaries = np.concatenate((np.array([gfRange, gtRange, gnRange]), np.array([[0, 1] for i in range(nRootLocations)]))) # 04/12/2021
    if experiment_output == 'ecg' or experiment_output == 'bsp':
        global tetrahedrons
        tetrahedrons = (np.loadtxt(dataPath + meshName + '_tri.csv', delimiter=',') - 1).astype(int)
        tetrahedronCenters = np.loadtxt(dataPath + meshName + '_tetrahedronCenters.csv', delimiter=',')
        ecg_electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
        if experiment_output == 'bsp':
            bsp_electrodePositions = np.loadtxt(dataPath + meshName + '_ECGiElectrodePositions.csv', delimiter=',')
            nb_leads = 8 + bsp_electrodePositions.shape[0] # All leads from the ECGi are calculated like the precordial leads
            electrodePositions = np.concatenate((ecg_electrodePositions, bsp_electrodePositions), axis=0)
        elif experiment_output == 'ecg':
            nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad  # 8 + lead progression (or 12)
            electrodePositions = ecg_electrodePositions
        nb_bsp = electrodePositions.shape[0]

        aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
        for i in range(0, tetrahedrons.shape[0], 1):
            aux[tetrahedrons[i, 0]].append(i)
            aux[tetrahedrons[i, 1]].append(i)
            aux[tetrahedrons[i, 2]].append(i)
            aux[tetrahedrons[i, 3]].append(i)
        global elements
        elements = [np.array(n) for n in aux]
        aux = None # Clear Memory

        # Precompute PseudoECG stuff - Calculate the tetrahedrons volumes
        D = nodesXYZ[tetrahedrons[:, 3], :] # RECYCLED
        A = nodesXYZ[tetrahedrons[:, 0], :]-D # RECYCLED
        B = nodesXYZ[tetrahedrons[:, 1], :]-D # RECYCLED
        C = nodesXYZ[tetrahedrons[:, 2], :]-D # RECYCLED
        D = None # Clear Memory

        global tVolumes
        tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1), (np.cross(B, C)[:, :, np.newaxis]))), tetrahedrons.shape[0]) #Tetrahedrons volume, no need to divide by 6
        # since it's being normalised by the sum which includes this 6 scaling factor
        global meshVolume
        meshVolume = np.sum(tVolumes)/6. # used to scale the relevance of signal-length discrepancies in small vs large geometries
        print('volume: '+ str(meshVolume))
        tVolumes = tVolumes/np.sum(tVolumes)

        # Calculate the tetrahedron (temporal) voltage gradients
        Mg = np.stack((A, B, C), axis=-1)
        A = None # Clear Memory
        B = None # Clear Memory
        C = None # Clear Memory

        # Calculate the gradients
        global G_pseudo
        G_pseudo = np.zeros(Mg.shape)
        for i in range(Mg.shape[0]):
            G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
            # If you obtain a Singular Matrix error type, this may be because one of the elements in the mesh is
            # really tinny if you are using a truncated mesh generated with Paraview, the solution is to do a
            # crinkle clip, instead of a regular smooth clip, making sure that the elements are of similar size
            # to each other. The strategy to identify the problem is to search for what element in Mg is giving
            # a singular matrix and see what makes it "special".
        G_pseudo = np.moveaxis(G_pseudo, 1, 2)
        Mg = None # clear memory

        # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
        r=np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
               (tetrahedronCenters.shape[0],
                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1)-electrodePositions

        global d_r
        d_r= np.moveaxis(np.multiply(
            np.moveaxis(r, [0, 1], [-1, -2]),
            np.multiply(np.moveaxis(np.sqrt(np.sum(r**2, axis=2))**(-3), 0, 1), tVolumes)), 0, -1)
    elif experiment_output == 'atm':
        #outputTag = 'ATMap'
        global epiface
        epiface = np.unique((np.loadtxt(dataPath + meshName + '_epiface.csv', delimiter=',') - 1).astype(int)) # epicardium nodes
        global epiface_tri
        epiface_tri = (np.loadtxt(dataPath + meshName + '_epiface.csv', delimiter=',') - 1).astype(int) # epicardium nodes
    else:
        raise
    # Set endocardial edges aside # 2022/01/18 - Split endocardium into two parts, a fast, and a slower one, namely, a dense and a sparse sub-endocardial Purkinje network
    global isEndocardial
    global isDenseEndocardial
    if has_endocardial_layer:
        isEndocardial=np.logical_or(np.all(np.isin(edges, lvnodes), axis=1), np.all(np.isin(edges, rvnodes), axis=1)) # 2022/01/18
        isDenseEndocardial=np.logical_or(np.all(np.isin(edges, lv_dense_indexes), axis=1), np.all(np.isin(edges, rv_dense_indexes), axis=1)) # 2022/01/18
    else:
        isEndocardial = np.zeros((edges.shape[0])).astype(bool)
        isDenseEndocardial = np.zeros((edges.shape[0])).astype(bool)
    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None # Clear Memory
    
    
    # Compute predetermined times of activation of the root nodes
    
    # eikonal_ecg(compilation_params, rootNodeActivationIndexes, rootNodeActivationTimes) # Compile numba function
    
    # Target result
    # global reference_precordial_lead_index # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
    # global reference_precordial_lead_is_max # 22/05/03 - Have some R progression by normalising by the largest positive amplitude lead
    global reference_lead_is_max # 22/05/03 - Normalise also the limb leads to give them relative importance
    # global reference_limb_lead_index # 22/05/03 - Normalise also the limb leads to give them relative importance
    if load_target:
        print(target_fileName)
        if experiment_output == 'atm':
            target_output = np.genfromtxt(target_fileName)[epiface]
        else:
            target_output = np.genfromtxt(target_fileName, delimiter=',')
            # target_output_old = np.copy(target_output)
            # print(np_mean(target_output, axis=1).shape)
            # Stardardise the target ECG signals to have mean == 0 # 2022/04/27 # TODO: Is this a good idea in combination with not aligning the ECGs to start at zero?
            # print(np_std(target_output, axis=1))
            # target_output = target_output - np_mean(target_output, axis=1)[:, np.newaxis] # 22/05/03 TODO: Uncomment
            # print(np_std(target_output, axis=1))
            # target_output = target_output / np_std(target_output, axis=1)[:, np.newaxis] # 22/05/03 TODO: Uncomment
            target_output = target_output - (target_output[:, 0:1]+target_output[:, -2:-1])/2 # align at zero # Re-added on 22/05/03 after it was worse without alingment
            # print(np_std(target_output, axis=1))
            # target_output[:2, :] = target_output[:2, :] / np_std(target_output[:2, :], axis=1)[:, np.newaxis] # 2022/05/03 Keep as before for the limb leads
            # print(np_std(target_output, axis=1))

            # Limb leads
            reference_lead_max = np.amax(target_output, axis=1) # 22/05/03
            # print('reference_lead_max.shape')
            # print(reference_lead_max.shape)
            reference_lead_min = np.absolute(np.amin(target_output, axis=1)) # 22/05/03
            reference_lead_is_max_aux = reference_lead_max >= reference_lead_min
            print('reference_lead_is_max_aux')
            print(reference_lead_is_max_aux)
            reference_amplitudes = np.zeros(shape=(nb_leads), dtype=np.float64) # 2022/05/04
            reference_amplitudes[reference_lead_is_max_aux] = reference_lead_max[reference_lead_is_max_aux]
            reference_amplitudes[np.logical_not(reference_lead_is_max_aux)] = reference_lead_min[np.logical_not(reference_lead_is_max_aux)]
            print('reference_amplitudes')
            print(reference_amplitudes)
            # if reference_limb_lead_is_max_aux:
                # reference_limb_lead_index_aux = np.argmax(np.amax(target_output[:2, :], axis=1)) # 22/05/03 - Have some R progression by normalising by the largest positive amplitude lead
            target_output[:2, :] = target_output[:2, :] / np.mean(reference_amplitudes[:2]) # 22/05/03 TODO: Uncomment
            target_output[2:nb_leads, :] = target_output[2:nb_leads, :] / np.mean(reference_amplitudes[2:nb_leads]) # 22/05/03 TODO: Uncomment
            
            # else:
            #     reference_limb_lead_index_aux = np.argmin(np.amin(target_output[:2, :], axis=1)) # 22/05/03 - Have some R progression by normalising by the largest positive amplitude lead
            #     target_output[:2, :] = target_output[:2, :] / abs(np.amin(target_output[:2, :][reference_limb_lead_index_aux, :])) # 22/05/03
            # Precordial leads
            # reference_precordial_lead_max = np.amax(target_output[2:nb_leads, :], axis=1) # 22/05/03 - Have some R progression by normalising by the largest positive amplitude lead
            # reference_precordial_lead_min = np.amin(target_output[2:nb_leads, :], axis=1) # 22/05/03 - Have some R progression by normalising by the largest positive amplitude lead
            # print(reference_precordial_lead_max)
            # print(reference_precordial_lead_min)
            # reference_precordial_lead_is_max_aux = reference_precordial_lead_max >= np.absolute(reference_precordial_lead_min)
            # print(reference_precordial_lead_is_max_aux)
            # for lead_i in range(nb_leads):
            #     if lead_i < 2:
            #         target_output[2:nb_leads, :] = target_output[2:nb_leads, :] / np.amax(target_output[2:nb_leads, :][reference_precordial_lead_index_aux, :]) # 22/05/03 TODO: Uncomment
            #     else:
            # if reference_precordial_lead_is_max_aux:
            #     reference_precordial_lead_index_aux = np.argmax(np.amax(target_output[2:nb_leads, :], axis=1)) # 22/05/03 - Have some R progression by normalising by the largest positive amplitude lead
            #     target_output[2:nb_leads, :] = target_output[2:nb_leads, :] / np.amax(target_output[2:nb_leads, :][reference_precordial_lead_index_aux, :]) # 22/05/03 TODO: Uncomment
            # else:
            #     reference_precordial_lead_index_aux = np.argmin(np.amin(target_output[2:nb_leads, :], axis=1)) # 22/05/03 - Have some R progression by normalising by the largest positive amplitude lead
            #     target_output[2:nb_leads, :] = target_output[2:nb_leads, :] / abs(np.amin(target_output[2:nb_leads, :][reference_precordial_lead_index_aux, :])) # 22/05/03
            # print(reference_precordial_lead_index_aux)
            # print(np_std(target_output, axis=1))

            # raise
            # print(target_output[:, -1:])
            # # raise
            # # Create figure
            # fig, axs = plt.subplots(nrows=1, ncols=nb_leads, constrained_layout=True, figsize=(22, 8), sharey='all')
            # for i in range(nb_leads):
            #     leadName = leadNames[i]
            #     axs[i].plot(target_output[i, :], 'k-', label='Standard', linewidth=2.)
            #     # axs[i].plot(target_output_old[i, :], 'r--', label='Target', linewidth=1.2)
            #     # decorate figure
            #     axs[i].xaxis.set_major_locator(MultipleLocator(40))
            #     axs[i].xaxis.set_minor_locator(MultipleLocator(20))
            #     axs[i].yaxis.set_major_locator(MultipleLocator(1))
            #     axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     axs[i].xaxis.grid(True, which='major')
            #     axs[i].xaxis.grid(True, which='minor')
            #     axs[i].yaxis.grid(True, which='major')
            #     #axs[0, i].yaxis.grid(True, which='minor')
            #     for tick in axs[i].xaxis.get_major_ticks():
            #         tick.label.set_fontsize(14)
            #     for tick in axs[i].yaxis.get_major_ticks():
            #         tick.label.set_fontsize(14)
            #     # axs[resolution_i, i].set_xlim(0, max_count)
            #     axs[i].set_title('Lead ' + leadName, fontsize=14)
            # axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            # plt.show()
            # print('Done')
            # raise
    else:
        # reference_limb_lead_index_aux = -1 # No leads # 22/05/03 - Have some R progression by normalising by the absolute amplitude lead
        reference_lead_is_max_aux = None # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
        # reference_precordial_lead_index_aux = -1 # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
        # reference_precordial_lead_is_max_aux = None # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
        print('The amplitude normalisation has not been prepared yet for non-clinical data')
        raise
        target_output = eikonal_ecg(np.array([conduction_speeds])/1000, rootNodesIndexes_true)
        if experiment_output == 'atm':
            target_output = target_output[0, epiface]
        else:
            target_output = target_output[0, :, :]
            target_output = target_output[:, np.logical_not(np.isnan(target_output[0, :]))]
    
    if target_snr_db > 0:
        if experiment_output == 'ecg' or experiment_output == 'bsp':
            target_output_aux = np.zeros((target_output.shape[0], target_output.shape[1]+200))
            target_output_aux[:, 100:-100] = target_output
            target_output = target_output_aux
        
            # Add noise white Gaussian noise to the signals using target SNR
            ecg_watts = target_output ** 2
            # Calculate signal power and convert to dB
            sig_avg_watts = np.mean(ecg_watts, axis=1)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            target_output_noised = np.zeros(target_output.shape)
            mean_noise = 0 # white Gaussian noise
            for i in range(target_output_noised.shape[0]):
                noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts[i]), ecg_watts.shape[1])
                # Noise up the original signal
                target_output_noised[i,:] = target_output[i,:] + noise_volts
            # Denoise the noised signal
            freq_cut= 100 # Cut-off frequency of the filter Hz
            w = freq_cut / (frequency / 2) # Normalize the frequency
            b_filt, a_filt = signal.butter(8, w, 'low')
            target_output_denoised = signal.filtfilt(b_filt, a_filt, target_output_noised) # Filter ECG signal
    
            target_output = target_output[:, 100:-100]
            target_output_denoised = target_output_denoised[:, 100:-100]
            target_output_noised = target_output_noised[:, 100:-100]
            
            # fig, axs = plt.subplots(nrows=2, ncols=len(leadNames), constrained_layout=True, figsize=(16, 6), sharey='all')
            #
            # # Calculate figure ms width
            # for i in range(len(leadNames)):
            #     leadName = leadNames[i]
            #     # Print out Pearson's correlation coefficients for each lead
            #     print(leadName + ': ' + str(np.corrcoef(target_output[i, :], target_output_denoised[i, :])[0,1]))
            #     axs[0, i].plot(target_output[i, :], 'k-', label='Clean', linewidth=1.5)
            #     axs[0, i].plot(target_output_denoised[i, :], 'g-', label='Denoised', linewidth=1.5)
            #     axs[0, i].plot(target_output_noised[i, :], 'r-', label='Noised', linewidth=1.5)
            #
            #     axs[1, i].plot(target_output[i, :], 'k-', label='Clean', linewidth=1.5)
            #     axs[1, i].plot(target_output_denoised[i, :], 'g-', label='Denoised', linewidth=1.5)
            #
            #     # decorate figure
            #     axs[0, i].set_xlim(0, target_output.shape[1])
            #     # axs[0, i].xaxis.set_major_locator(MultipleLocator(40))
            #     axs[0, i].yaxis.set_major_locator(MultipleLocator(2))
            #     # axs[0, i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     # axs[0, i].xaxis.grid(True, which='major')
            #     axs[0, i].yaxis.grid(True, which='major')
            #     # for tick in axs[0, i].xaxis.get_major_ticks():
            #     #     tick.label1.set_fontsize(14)
            #     for tick in axs[0, i].yaxis.get_major_ticks():
            #         tick.label1.set_fontsize(14)
            #     axs[0, i].set_title('Lead ' + leadName, fontsize=14)
            #     axs[1, i].set_xlabel('ms', fontsize=14)
            #
            #     axs[1, i].set_xlim(0, target_output.shape[1])
            #     axs[1, i].xaxis.set_major_locator(MultipleLocator(40))
            #     axs[1, i].yaxis.set_major_locator(MultipleLocator(2))
            #     axs[1, i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     axs[1, i].xaxis.grid(True, which='major')
            #     axs[1, i].yaxis.grid(True, which='major')
            #     for tick in axs[1, i].xaxis.get_major_ticks():
            #         tick.label1.set_fontsize(14)
            #     for tick in axs[1, i].yaxis.get_major_ticks():
            #         tick.label1.set_fontsize(14)
            # plt.show()
            #
            # fig, axs = plt.subplots(nrows=2, ncols=len(leadNames), constrained_layout=True, figsize=(16, 6), sharey='all')
            # for i in range(len(leadNames)):
            #     randInt = np.random.randint(0, nb_leads-1)
            #     leadName = 'Rand '+str(randInt)
            #     # Print out Pearson's correlation coefficients for each lead
            #     print(leadName + ': ' + str(np.corrcoef(target_output[randInt, :], target_output_denoised[randInt, :])[0,1]))
            #     axs[0, i].plot(target_output[randInt, :], 'k-', label='Clean', linewidth=1.5)
            #     axs[0, i].plot(target_output_denoised[randInt, :], 'g-', label='Denoised', linewidth=1.5)
            #     axs[0, i].plot(target_output_noised[randInt, :], 'r-', label='Noised', linewidth=1.5)
            #
            #     axs[1, i].plot(target_output[randInt, :], 'k-', label='Clean', linewidth=1.5)
            #     axs[1, i].plot(target_output_denoised[randInt, :], 'g-', label='Denoised', linewidth=1.5)
            #
            #     # decorate figure
            #     axs[0, i].set_xlim(0, target_output.shape[1])
            #     # axs[0, i].xaxis.set_major_locator(MultipleLocator(40))
            #     axs[0, i].yaxis.set_major_locator(MultipleLocator(2))
            #     # axs[0, i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     # axs[0, i].xaxis.grid(True, which='major')
            #     axs[0, i].yaxis.grid(True, which='major')
            #     # for tick in axs[0, i].xaxis.get_major_ticks():
            #     #     tick.label1.set_fontsize(14)
            #     for tick in axs[0, i].yaxis.get_major_ticks():
            #         tick.label1.set_fontsize(14)
            #     axs[0, i].set_title('Lead ' + leadName, fontsize=14)
            #     axs[1, i].set_xlabel('ms', fontsize=14)
            #
            #     axs[1, i].set_xlim(0, target_output.shape[1])
            #     axs[1, i].xaxis.set_major_locator(MultipleLocator(40))
            #     axs[1, i].yaxis.set_major_locator(MultipleLocator(2))
            #     axs[1, i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     axs[1, i].xaxis.grid(True, which='major')
            #     axs[1, i].yaxis.grid(True, which='major')
            #     for tick in axs[1, i].xaxis.get_major_ticks():
            #         tick.label1.set_fontsize(14)
            #     for tick in axs[1, i].yaxis.get_major_ticks():
            #         tick.label1.set_fontsize(14)
            # plt.show()
            print('noise correlation: ' + str(np.mean(np.asarray([np.corrcoef(target_output[i, :], target_output_denoised[i, :])[0,1]
                for i in range(target_output.shape[0])]))))
            target_output = target_output_denoised # always work with the denoised ECG as if obtained in the clinic
        
        else: # Activation map data TODO: check that this is actually a good idea 21/02/2021
            reference_limb_lead_index_aux = -1 # No leads # 22/05/03 - Have some R progression by normalising by the absolute amplitude lead
            reference_limb_lead_is_max_aux = None # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
            reference_precordial_lead_index_aux = -1 # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
            reference_precordial_lead_is_max_aux = None # No leads # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
            # target_output_aux = np.zeros((target_output.shape[0]+200))
            # target_output_aux[100:-100] = target_output
            # target_output = target_output_aux
            # Add noise white Gaussian noise to the signals using target SNR
            ecg_watts = target_output ** 2
            # Calculate signal power and convert to dB
            sig_avg_watts = np.mean(ecg_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0 # white Gaussian noise
            noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), ecg_watts.shape[0])
            # Noise up the original signal
            target_output_noised = target_output + noise_volts
            # Denoise the noised signal
            freq_cut= 100 # Cut-off frequency of the filter Hz
            w = freq_cut / (frequency / 2) # Normalize the frequency
            b_filt, a_filt = signal.butter(8, w, 'low')
            target_output_denoised = signal.filtfilt(b_filt, a_filt, target_output_noised) # Filter ECG signal
            # target_output = target_output[100:-100]
            # target_output_denoised = target_output_denoised[100:-100]
            # target_output_noised = target_output_noised[100:-100]
            
            print('noise correlation: ' + str(np.corrcoef(target_output, target_output_noised)[0,1]))
            print('De-noise correlation: ' + str(np.corrcoef(target_output, target_output_denoised)[0,1]))
            target_output = target_output_noised # filtering the activation map is a terrible idea

    if experiment_output == 'bsp':
        # Define the weight of each electrode based on the redundancy of information in a 5 cm radius
        electrodes_no_limb = electrodePositions[4:, :]  # don't do for limb electrodes, the limb
                                                        # leads (I and II)these will get importance
                                                        # at the end of this code.
        aux_len = []
        lead_weights = np.zeros((nb_leads))
        # min weight
        aux_min_weight = 1000.
        aux_min_distance = None
        aux_min_closest_indexes = None
        aux_min_correlation_list = None
        aux_min_id = None
        # max weight
        aux_max_weight = 0.
        aux_max_distance = None
        aux_max_closest_indexes = None
        aux_max_correlation_list = None
        aux_max_id = None
        for i_electrode in range(electrodes_no_limb.shape[0]):
            aux_distances = np.sqrt(np.sum((electrodes_no_limb - electrodes_no_limb[i_electrode, :])**2, 1))
            closest_indexes = np.logical_and(aux_distances > 0., # discard the same electrode
                              aux_distances < 7.5, where=True # include electrodes in an area of (more or less due to curvature) 15 cm diameter
                              ).nonzero()[0] # without 12-lead ECG leads
            aux_len.append(len(closest_indexes))
            correlation_list = np.zeros(closest_indexes.shape)
            closest_distances = aux_distances[closest_indexes] # without leads I and II
            for i_closest in range(closest_indexes.shape[0]):
                closest_ind = closest_indexes[i_closest]
                # adapt index to number of leads (+2) for leads I and II
                correlation_list[i_closest] = (np.corrcoef(target_output[i_electrode + 2, :], # adapt index to number of leads
                        target_output[closest_ind + 2, :] # adapt index to number of leads (+2) for leads I and II
                        )[0, 1]/closest_distances[i_closest] # normalise by the distance to the electrode
                        )
            # print(correlation_list)
            # min-max normalisation considering the possible min and max from the Pearson's correlation coefficient
            correlation_list = (1 - correlation_list)/2. # Use the CC negated because the more different a lead is, the more weight will get
            new_weight = np.mean(correlation_list)
            # print(correlation_list)
            lead_weights[i_electrode + 2] = new_weight # ignore leads I and II for now
            # print(new_weight)
        
        # Ratio from 12-lead ECG to Body surface potential measurements
        ratio_12ecg_to_bspm = 0.90 # 90%
        # Normalise weights so that they add to 1
        lead_weights[2:] = lead_weights[2:]/np.sum(lead_weights[2:])
        # Set lead I and II as the average of the leads with the most weight
        lead_weights[:2] = np.mean(lead_weights[np.argsort(lead_weights)[-10:]]) # set leads I and II to have a relatively high weight, though not the highest
        lead_weights = lead_weights/np.sum(lead_weights)
        lead_weights = lead_weights * nb_leads # define it as a weighting factor
    
    reference_lead_is_max = reference_lead_is_max_aux # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
    # reference_limb_lead_is_max = reference_limb_lead_is_max_aux # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
    # reference_precordial_lead_index = reference_precordial_lead_index_aux # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
    # reference_precordial_lead_is_max = reference_precordial_lead_is_max_aux # 22/05/03 - Have some R progression by normalising by the largest absolute amplitude lead
    
    # Load and compile Numba - # 22/05/03 - Moved to after loading global variable: reference_precordial_lead_index
    compilation_params = np.array([np.concatenate((np.full(nlhsParam, 0.09), np.ones(rootNodeActivationIndexes.shape).astype(int)))])
    eikonal_ecg(compilation_params, rootNodeActivationIndexes, rootNodeActivationTimes) # 07/12/2021
    compProb(np.array([[0, 1, 0, 1]]), np.array([[0, 1, 0, 1]]), retain_ratio, nRootNodes_range[0]) # Compile numba function
    
    global dtw_lead_weights # prepare global variable used for the dtw computation for bsp data
                            # since 06/02/2021
    if experiment_output == 'bsp':
        dtw_lead_weights = lead_weights.astype(float)
    elif experiment_output == 'ecg':
        dtw_lead_weights = np.ones((nb_leads)).astype(float)
    else:
        dtw_lead_weights = None
        
    # print(dtw_lead_weights)
    
    if experiment_output == 'bsp':
        # print ('Export mesh to ensight format')
        torso_face = (np.loadtxt(dataPath + meshName + '_torsoface.csv', delimiter=',') - 1).astype(int)
        torso_xyz = (np.loadtxt(dataPath + meshName + '_torsoxyz.csv', delimiter=',') - 1).astype(int)
        # aux_elems = torso_face + 1    # Make indexing Paraview and Matlab friendly
        # with open('bspNoiseMidNew_Figures/' + meshName+'.ensi.geo', 'w') as f:
        #     f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(torso_xyz.shape[0])+'\n')
        #     for i in range(0, torso_xyz.shape[0]):
        #         f.write(str(i+1)+'\n')
        #     for c in [0,1,2]:
        #         for i in range(0, torso_xyz.shape[0]):
        #             f.write(str(torso_xyz[i,c])+'\n')
        #     print('Write tria3...')
        #     f.write('tria3\n  '+str(len(aux_elems))+'\n')
        #     for i in range(0, len(aux_elems)):
        #         f.write('  '+str(i+1)+'\n')
        #     for i in range(0, len(aux_elems)):
        #         f.write(str(aux_elems[i,0])+'\t'+str(aux_elems[i,1])+'\t'+str(aux_elems[i,2])+'\n')
        # with open('bspNoiseMidNew_Figures/' + meshName+'.ensi.case', 'w') as f:
        #     f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
        #     f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
        #     f.write('scalar per node: 1	corr'+'	'+meshName+'.ensi.corr'+'\n')

        # Save Spatio-temporal local information gain from each electrode
        correlation_list = lead_weights[2:]
        torso_surface_node_IDs = np.unique(torso_face.flatten(order='C'))
        torso_surface_nodes = torso_xyz[torso_surface_node_IDs, :]
        distances = np.linalg.norm(torso_surface_nodes[:, np.newaxis, :]-electrodePositions[4:, :], ord=2, axis=2)
        closest_electrode = np.argmin(distances, axis=1)
        correlation_map = np.zeros((torso_xyz.shape[0]))
        correlation_map[torso_surface_node_IDs] = correlation_list[closest_electrode]

        np.savetxt('bspNoiseMidNew_Figures/' + meshName + '_torso_electrodes.csv', electrodePositions[4:, :], delimiter=',')
        np.savetxt('bspNoiseMidNew_Figures/' + meshName + '_torso_correlations.csv', correlation_map, delimiter=',')
    
    global t_start
    t_start = time.time()
    print('Starting ' + final_path)
    # print('fix this funciton and also the stopping criteria in ecg and bsp')
    sampler = ABCSMC(nparam=nparam,
                     simulator=eikonal_ecg,
                     target_data=target_output,
                     max_MCMC_steps=max_MCMC_steps,
                     param_boundaries=param_boundaries,
                     nlhsParam=nlhsParam,
                     maxRootNodeJiggleRate=0.1,
                     nRootLocations=nRootLocations,
                     retain_ratio=retain_ratio,
                     nRootNodes_range=nRootNodes_range,
                     resultFile=final_path,
                     experiment_output=experiment_output,
                     experiment_metric=metric,
                     hermite_order=None,
                     hermite_mean=None,
                     hermite_std=None,
                     nRootNodes_centre=nRootNodes_centre,
                     nRootNodes_std=nRootNodes_std,
                     npart=npart,
                     keep_fraction=keep_fraction,
                     conduction_speeds=conduction_speeds,
                     target_rootNodes=rootNodesIndexes_true
                    )
    sampler.sample(desired_Discrepancy)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    else:
        print("The file:"+tmp_path+"\n does not exist")
    
    print('Finished: '+final_path)
    
 
# ------------------------------------------- DISCRETE SMC-ABC FUNCTIONS ----------------------------------------------------
@numba.njit
def find_first_larger_than(item):
    """return the index of the first occurence of item in vec"""
    for i, v in enumerate(p_cdf):
        if v > item: return i
    return -1

@numba.njit
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

# @numba.njit # October 2021 TODO: precompile this function with Numba once I figure out how to update Numba in CSCS - the version of Numba from August 2021 can handle numpy.random.dirichlet https://numba.readthedocs.io/en/stable/release-notes.html
def jiggleDiscreteNonFixed_one(part_binaries, retain_ratio, nRootNodes_range):
    n_parts, n_vars = part_binaries.shape
    alpha = n_parts * (1 - retain_ratio) / (retain_ratio * n_vars - 1)
    on = np.zeros((n_vars), dtype=np.bool_)
    # if np.random.uniform(0, 1) < 0.8:
    if np.random.uniform(0, 1) < retain_ratio: # 05/12/2021
        N_on = int(np.round(np.sum(part_binaries[int(np.random.randint(0, part_binaries.shape[0])), :])))
    else:
        N_on = find_first_larger_than(np.random.uniform(0, 1)) + nRootNodes_range[0] # September 2021
    
    # Use only the probability of the particles with same number of nodes active
    part_binaries_N = part_binaries[np.sum(part_binaries, axis=1)==N_on, :]
    for j in range(N_on):
        open_sites = np.nonzero(np.logical_not(on))[0]
        alpha_aux = alpha + np.sum( part_binaries_N[
            np_all_axis1(part_binaries_N[:, on]), :][:, open_sites], axis=0) # TODO: verify that it works well!! This is a Numba compatible implementation of np.all with an axis argument,
            # see: https://stackoverflow.com/questions/61304720/workaround-for-numpy-np-all-axis-argument-compatibility-with-numba
        w = np.random.dirichlet(alpha_aux) # TODO: Numba new release will support this fucntion, check again in 2022
        r = open_sites[(np.random.multinomial(1, w)).astype(dtype=np.bool_)] # Numpy likes that Generator is used, e.g., np.random.Generator.multinomial(1, w), but to allow for Numba I haven't done it
        on[r] = 1 # True for Numba
    return on


@numba.njit
def compProb(new_binaries, part_binaries, retain_ratio, min_nb_root_nodes):
    # This code can be compiled as non-python code by Numba, which makes it quite fast
    n_parts, n_vars = part_binaries.shape
    alpha = n_parts * (1 - retain_ratio) / (retain_ratio * n_vars - 1)
    p_trial_i = 0.
    N_on = int(np.sum(new_binaries)) # WARNING! Changed on the 23/06/2020, before it was N_on = np.sum(new_binaries)
    p_nRootNodes = (0.8 * np.sum(np.sum(part_binaries, axis=1)==N_on)/part_binaries.shape[0] + 0.2 * p_pdf[int(N_on - 1 - min_nb_root_nodes)])
    # if N_on < 10: # September 2021 changed to always be lower than 10, because otherwise it's computationally intractable - Idea: comparmentise ventricles for computationally tractable more roots
    part_binaries_N = part_binaries[np.sum(part_binaries, axis=1)==N_on, :]
    # Permutations that Numba can understand
    A = np.nonzero(new_binaries)[0]
    k = len(A)
    numba_permutations = [[i for i in range(0)]]
    for i in range(k):
        numba_permutations = [[a] + b for a in A for b in numba_permutations if (a in b)==False]
    for combo in numba_permutations:
        on = np.zeros((n_vars), dtype=np.bool_)
        p_trial_here = p_nRootNodes
        for j in range(len(combo)):
            pb = part_binaries_N[:, on]
            aux_i = np.empty((pb.shape[0]), dtype=np.bool_)
            for part_i in range(pb.shape[0]):
                aux_i[part_i] = np.all(pb[part_i])
            aux_p = part_binaries_N[aux_i, :]
            aux = np.sum(aux_p[:, combo[j]], axis=0)
            aux1 = ((n_vars - j) * alpha + np.sum((aux_p[:, np.logical_not(on)])))
            aux2 = (alpha + aux) / aux1
            p_trial_here *= aux2
            on[combo[j]] = 1
        p_trial_i += p_trial_here
    return p_trial_i


def doNothing(X):
    # TODO: obtain summary metrics for the ECG
    return X


# ------------------------------------------- MAPPING AND GENERATION FUNCTIONS ----------------------------------------------------

def createVentricleApexToBase(endo_node_indexes, mesh_nodesXYZ, cobiveco_mesh_ab_values):
# function longAx = computeLongAxis(sur, directionVec)
# % Computes the heart's long axis as the vector minimizing the dot product
# % with the surface normals of the LV endocardium, i.e. the vector that is
# % "most orthogonal" to the surface normals.
# %
# % longAx = computeLongAxis(sur, basePoints)
# %
# % Inputs:
# %   sur: LV endocardial surface as VTK struct
# %   directionVec: A vector coarsely directed from base towards apex
# %
# % Outputs:
# %   longAx: Unit vector directed from base towards apex
# %
# % Written by Steffen Schuler, Institute of Biomedical Engineering, KIT
#
# TR = vtkToTriangulation(sur);
# normals = TR.faceNormal;
# areaWeights = doublearea(TR.Points, TR.ConnectivityList);
# areaWeights = areaWeights/mean(areaWeights);
#
# % p-norm is chosen to have average properties of 1-norm and 2-norm
# h = (1/sqrt(2)+1)/2;
# p = 1/(log2(sqrt(2)/h));
#
# objFun = @(longAxis) sum(abs(areaWeights.*normals*longAxis').^p) + size(normals,1)*abs(norm(longAxis)-1)^p;
# options = optimset('MaxFunEvals', 1e4);
# longAx = NaN(3);
# objVal = NaN(3,1);
# [longAx(1,:),objVal(1)] = fminsearch(objFun, [1 0 0], options);
# [longAx(2,:),objVal(2)] = fminsearch(objFun, [0 1 0], options);
# [longAx(3,:),objVal(3)] = fminsearch(objFun, [0 0 1], options);
# [~,minInd] = min(objVal);
# longAx = longAx(minInd,:);
# longAx = longAx/norm(longAx);
#
# % make sure longAxis is directed from base towards apex
# d = directionVec(:)' * longAx';
# longAx = sign(d) * longAx;
#
# end
    pass


def plotMeshes3D(mesh1_xyz, mesh2_xyz=None): # Plot the 3D points from 9 perspectives to check the data
    fig = plt.figure(constrained_layout=True, figsize = (15,20))
    for i in range(9):
        ax = fig.add_subplot(331+i, projection='3d')
        ax.scatter(mesh1_xyz[:, 0], mesh1_xyz[:, 1], mesh1_xyz[:, 2], c='b', marker='o', s=.1)
        if mesh2_xyz is not None:
            ax.scatter(mesh2_xyz[:, 0], mesh2_xyz[:, 1], mesh2_xyz[:, 2], c='r', marker='o', s=.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(10, 45*i)
    plt.show()


# For each point in points_to_map_xyz, returns the index of the closest point in the reference cloud of points reference_points_xyz
def mapIndexes(points_to_map_xyz, reference_points_xyz, return_unique_only=False):
    mapped_indexes = np.zeros((points_to_map_xyz.shape[0])).astype(int)
    for i in range(points_to_map_xyz.shape[0]):
        mapped_indexes[i] = np.argmin(np.linalg.norm(reference_points_xyz - points_to_map_xyz[i, :], ord=2, axis=1)).astype(int)
    if return_unique_only: # use the unique function without sorting the contents of the array (meta_indexes)
        unique_meta_indexes = np.unique(mapped_indexes, axis=0, return_index=True)[1] # indexes to the indexes (meta_indexes) that are unique
        mapped_indexes = mapped_indexes[sorted(unique_meta_indexes)]
    return mapped_indexes
    

def obtainVTKField(txt, tag_start, tag_end): # Reads one field as an array in a vtk file-format
    start = txt.find(tag_start)
    end = txt.find(tag_end, start)
    tagged_txt =  txt[txt.find('\n', start):end]
    tagged_txt = tagged_txt.replace('\n', ' ').split(None) # PYTHON documentation: If sep is not specified or is None, a different splitting algorithm is applied: runs of consecutive whitespace are regarded
    # as a single separator, and the result will contain no empty strings at the start or end if the string has leading or trailing whitespace.
    return np.array([float(x) for x in tagged_txt])
    

def getCobiveco(cobiveco_fileName): # Read Cobiveco data
    with open(cobiveco_fileName, 'r') as f:
        cobiveco_txt = f.read()
    # Read Cobiveco fields
    cobiveco_nodesXYZ = obtainVTKField(cobiveco_txt, 'POINTS', 'METADATA')
    cobiveco_nodesXYZ = np.reshape(cobiveco_nodesXYZ, (int(cobiveco_nodesXYZ.shape[0]/3), 3))
    # Apex-to-Base - ab
    ab = obtainVTKField(cobiveco_txt, 'ab ', 'METADATA')
    # Rotation angle - rt
    rt = obtainVTKField(cobiveco_txt, 'rt ', 'METADATA')
    # Transmurality - tm
    tm = obtainVTKField(cobiveco_txt, 'tm ', 'METADATA')
    # Ventricle - tv
    tv = obtainVTKField(cobiveco_txt, 'tv ', 'METADATA')
    return cobiveco_nodesXYZ, np.transpose(np.array([ab, rt, tm, tv], dtype=float)) #TODO: check that transpose is working well


# Returns the mapping for each points in a mesh points_to_map_xyz to the closest point in a mesh target_points_xyz using Cobiveco coordinates on a remeshed version
def generateCobiveco_map(points_to_map_xyz, target_points_xyz, cobiveco_to_map, target_cobiveco): # of the original meshes
    # Map between points_to_map_xyz and its Cobiveco coordinates
    points_to_map_cobiveco_indexes = mapIndexes(points_to_map_xyz, cobiveco_to_map[0], return_unique_only=False)
    cobiveco_points_to_map = cobiveco_to_map[1][points_to_map_cobiveco_indexes, :]
    # Map to REF Cobiveco coordinates
    cobiveco_points_mapping = np.zeros((points_to_map_xyz.shape[0]), dtype=int)
    # LV
    lv_mask_cobiveco_points_to_map = cobiveco_points_to_map[:, 3]==0
    lv_mask_cobiveco_ref = np.nonzero(target_cobiveco[1][:, 3]==0)[0]
    lv_cobiveco_mapping = mapIndexes(cobiveco_points_to_map[lv_mask_cobiveco_points_to_map, :], target_cobiveco[1][lv_mask_cobiveco_ref, :], return_unique_only=False)
    cobiveco_points_mapping[lv_mask_cobiveco_points_to_map] = lv_mask_cobiveco_ref[lv_cobiveco_mapping]
    # RV
    rv_mask_cobiveco_points_to_map = np.logical_not(lv_mask_cobiveco_points_to_map)
    rv_mask_cobiveco_ref = np.nonzero(target_cobiveco[1][:, 3]==1)[0]
    rv_cobiveco_mapping = mapIndexes(cobiveco_points_to_map[rv_mask_cobiveco_points_to_map, :], target_cobiveco[1][rv_mask_cobiveco_ref, :], return_unique_only=False)
    cobiveco_points_mapping[rv_mask_cobiveco_points_to_map] = rv_mask_cobiveco_ref[rv_cobiveco_mapping]
    # Map to REF Cobiveco xyz
    points_mapped_to_ref_cobiveco_xyz = target_cobiveco[0][cobiveco_points_mapping, :]
    # Map to REF XYZ
    points_mapped_xyz_indexes = mapIndexes(points_mapped_to_ref_cobiveco_xyz, target_points_xyz, return_unique_only=False)
    return points_mapped_xyz_indexes


def save_cobiveco(figPath, meshName, nodesXYZ, paraview_elems, cobiveco_mapped):
    if not os.path.isfile(figPath + meshName+'.ensi.geo'):
        with open(figPath + meshName+'.ensi.geo', 'w') as f:
            f.write('Problem name:  '+meshName+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(nodesXYZ.shape[0])+'\n')
            for i in range(0, nodesXYZ.shape[0]):
                f.write(str(i+1)+'\n')
            for c in [0,1,2]:
                for i in range(0, nodesXYZ.shape[0]):
                    f.write(str(nodesXYZ[i,c])+'\n')
            # print('Write tetra4...')
            f.write('tetra4\n  '+str(len(paraview_elems))+'\n')
            for i in range(0, len(paraview_elems)):
                f.write('  '+str(i+1)+'\n')
            for i in range(0, len(paraview_elems)):
                f.write(str(paraview_elems[i,0])+'\t'+str(paraview_elems[i,1])+'\t'+str(paraview_elems[i,2])+'\t'+str(paraview_elems[i,3])+'\n')
    # create heart.ensi.case file
    if os.path.isfile(figPath + meshName + '.ensi.case'):
        with open(figPath + meshName + '.ensi.case', 'a') as f:
            f.write('scalar per node: 1	ab	'+ meshName + '.ensi.ab\n')
            f.write('scalar per node: 1	rt	'+ meshName + '.ensi.rt\n')
            f.write('scalar per node: 1	tm	'+ meshName + '.ensi.tm\n')
            f.write('scalar per node: 1	tv	'+ meshName + '.ensi.tv\n')
    else:
        with open(figPath + meshName + '.ensi.case', 'w') as f:
            f.write('#\n# Eikonal generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\theart\n#\n')
            f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+meshName+'.ensi.geo\nVARIABLE\n')
            f.write('scalar per node: 1	ab	'+ meshName + '.ensi.ab\n')
            f.write('scalar per node: 1	rt	'+ meshName + '.ensi.rt\n')
            f.write('scalar per node: 1	tm	'+ meshName + '.ensi.tm\n')
            f.write('scalar per node: 1	tv	'+ meshName + '.ensi.tv\n')
    
    with open(figPath + meshName +  '.ensi.ab', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(cobiveco_mapped.shape[0]):
            f.write(str(cobiveco_mapped[i, 0]) + '\n')
    with open(figPath + meshName +  '.ensi.rt', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(cobiveco_mapped.shape[0]):
            f.write(str(cobiveco_mapped[i, 1]) + '\n')
    with open(figPath + meshName +  '.ensi.tm', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(cobiveco_mapped.shape[0]):
            f.write(str(cobiveco_mapped[i, 2]) + '\n')
    with open(figPath + meshName +  '.ensi.tv', 'w') as f:
        f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(cobiveco_mapped.shape[0]):
            f.write(str(cobiveco_mapped[i, 3]) + '\n')


# Save cobiveco projected onto original mesh for paraview visualisation
def save_cobiveco_projection(figPath, meshName, nodesXYZ, paraview_elems,  cobiveco_xyz, cobiveco_ab_rt_tm_tv):
    cobiveco_indexes = mapIndexes(nodesXYZ, cobiveco_xyz, return_unique_only=False)
    cobiveco_mapped = cobiveco_ab_rt_tm_tv[cobiveco_indexes, :]
    save_cobiveco(figPath, meshName, nodesXYZ, paraview_elems, cobiveco_mapped)
    

# Closest neighbour using Cobiveco remeshed meshes and their Cobiveco coordinate systems - Mapping all the points didn't work that well
def mappWithCobiveco(meshName_ref='DTI003', meshName_target='DTI4586_2_coarse', points_to_map_filenames=['_lvhisbundle_xyz.csv'], points_are_indexes=[False]):
    # Paths and tags
    ref_dataPath = 'metaData/' + meshName_ref + '/'
    target_dataPath = 'metaData/' + meshName_target + '/'
    # Load meshes
    ref_nodesXYZ = np.loadtxt(ref_dataPath + meshName_ref + '_xyz.csv', delimiter=',')
    target_nodesXYZ = np.loadtxt(target_dataPath + meshName_target + '_xyz.csv', delimiter=',')
    # Read VTK Cobiveco files - REFERENCE
    ref_cobiveco_xyz, ref_cobiveco_ab_rt_tm_tv = getCobiveco(cobiveco_fileName=ref_dataPath + meshName_ref + '_cobiveco.vtk')
    save_cobiveco_projection(ref_dataPath, meshName_ref, ref_nodesXYZ, np.loadtxt(ref_dataPath + meshName_ref + '_tri.csv', delimiter=',').astype(int), ref_cobiveco_xyz, ref_cobiveco_ab_rt_tm_tv)
    target_cobiveco_xyz, target_cobiveco_ab_rt_tm_tv = getCobiveco(cobiveco_fileName=target_dataPath + meshName_target + '_cobiveco.vtk')
    save_cobiveco_projection(target_dataPath, meshName_target, target_nodesXYZ, np.loadtxt(target_dataPath + meshName_target + '_tri.csv', delimiter=',').astype(int), target_cobiveco_xyz, target_cobiveco_ab_rt_tm_tv)
    index_mapping_from_ref_to_target = generateCobiveco_map(points_to_map_xyz=ref_nodesXYZ, target_points_xyz=target_nodesXYZ,
        cobiveco_to_map=[ref_cobiveco_xyz, ref_cobiveco_ab_rt_tm_tv], target_cobiveco=[target_cobiveco_xyz, target_cobiveco_ab_rt_tm_tv])
    # Iterate over files to map to the target mesh
    for file_iter in range(len(points_to_map_filenames)):
        points_to_map_filename = points_to_map_filenames[file_iter]
        # print(points_to_map_filename)
        points_are_index = points_are_indexes[file_iter]
        # Load data to map
        if not points_are_index:
            points_to_map = np.loadtxt(ref_dataPath + meshName_ref + points_to_map_filename, delimiter=',', skiprows=1, usecols=(1,2,3))
            if len(points_to_map.shape) < 2:
                points_to_map = points_to_map[np.newaxis, :]
            points_to_map = np.unique(points_to_map, axis=0)
            indexes_to_map = mapIndexes(points_to_map, ref_nodesXYZ, return_unique_only=False)
        else:
            points_to_map = (np.loadtxt(ref_dataPath + meshName_ref + points_to_map_filename, delimiter=',')- 1).astype(int)
            indexes_to_map = np.asarray(points_to_map, dtype=int)
        points_mapped_indexes = index_mapping_from_ref_to_target[indexes_to_map]
        points_mapped_xyz = target_nodesXYZ[points_mapped_indexes, :]
        if not points_are_index: # Routing xyz point values for constructing the PK network
            with open(target_dataPath + meshName_target + points_to_map_filename, 'w') as f:
                f.write('"","x","y","z"\n')
                for i in range(points_mapped_xyz.shape[0]):
                    f.write(str(points_mapped_indexes[i]) + ',' + str(points_mapped_xyz[i, 0]) + ',' + str(points_mapped_xyz[i, 1]) + ',' + str(points_mapped_xyz[i, 2]) + '\n')
        else: # Possible root node indexes
            with open(target_dataPath + meshName_target + points_to_map_filename, 'w') as f:
                for i in range(points_mapped_indexes.shape[0]):
                    f.write(str(points_mapped_indexes[i]) + '\n')
    print('mapping from ' + meshName_ref + ' to ' + meshName_target + ' COMPLETED!')


# Load Cobiveco and use it's coordinates to define the pseudo-Purkinje tree for the inference
def generatePurkinjeWithCobiveco(dataPath= 'metaData/DTI003/', meshName='DTI003', figPath='Figures_Clinical_PK/'):
    # load root nodes with the current resolution
    # lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_lv_activationIndexes_newRVRes.csv', delimiter=',') - 1).astype(int)  # possible root nodes for the chosen mesh
    # rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_rv_activationIndexes_newRVRes.csv', delimiter=',') - 1).astype(int)
    # for i in range(lvActivationIndexes.shape[0]):
    #     if lvActivationIndexes[i] not in lvnodes:
    #         lvActivationIndexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - nodesXYZ[lvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    # lvActnode_ids = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvActivationIndexes]).astype(int)
    # for i in range(rvActivationIndexes.shape[0]):
    #     if rvActivationIndexes[i] not in rvnodes:
    #         rvActivationIndexes[i] = rvnodes[np.argmin(np.linalg.norm(nodesXYZ[rvnodes, :] - nodesXYZ[rvActivationIndexes[i], :], ord=2, axis=1)).astype(int)]
    # rvActnode_ids = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvActivationIndexes]).astype(int)
    
    
    djikstra_max_path_len = 200
    lv_dense_indexes = np.zeros(lvnodes.shape, dtype=bool)
    rv_dense_indexes = np.zeros(rvnodes.shape, dtype=bool)
    # t_s = time.time()
    # Read VTK Cobiveco files - REFERENCE
    cobiveco_xyz, cobiveco_ab_rt_tm_tv = getCobiveco(cobiveco_fileName=dataPath + meshName + '_cobiveco.vtk')
    # Prepare for Djikstra - Set LV endocardial edges aside
    lvnodesXYZ = nodesXYZ[lvnodes, :]
    lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
    lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    for i in range(0, len(lvunfoldedEdges), 1):
        aux[lvunfoldedEdges[i, 0]].append(i)
    lvneighbours = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    # Set RV endocardial edges aside
    rvnodesXYZ = nodesXYZ[rvnodes, :]
    rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
    rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    for i in range(0, len(rvunfoldedEdges), 1):
        aux[rvunfoldedEdges[i, 0]].append(i)
    rvneighbours = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    # Map between nodesXYZ and its Cobiveco coordinates
    cobiveco_indexes = mapIndexes(nodesXYZ, cobiveco_xyz, return_unique_only=False)
    nodesCobiveco = cobiveco_ab_rt_tm_tv[cobiveco_indexes, :]
    # Define Purkinje tree using Cobiveco-based rules - Initialise data structures
    lv_PK_distance_mat = np.full(lvnodes.shape, nan_value, np.float64)
    lv_PK_path_mat = np.full((lvnodes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    rv_PK_distance_mat = np.full(rvnodes.shape, nan_value, np.float64)
    rv_PK_path_mat = np.full((rvnodes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_visited = np.zeros(lvnodes.shape, dtype=bool)
    rv_visited = np.zeros(rvnodes.shape, dtype=bool)
    # Rule 1) his-av node at coordinates [1., 0.85, 1., :] == [basal, septal, endo, :]
    lv_hisBase_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([1., 0.85, 1., 0.]), ord=2, axis=1))) # [basal, septal, endo, lv]
    rv_hisBase_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([1., 0.85, 1., 1.]), ord=2, axis=1))) # [basal, septal, endo, rv]
    lv_hisBase_distance_mat, lv_hisBase_path_mat = djikstra(np.asarray([lv_hisBase_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=djikstra_max_path_len)
    rv_hisBase_distance_mat, rv_hisBase_path_mat = djikstra(np.asarray([rv_hisBase_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours, max_path_len=djikstra_max_path_len)
    # Rule 2) hisbundle goes down to most apical endocardial point while trying to keep a straight rotation trajectory [0., 0.85, 1., :] == [basal, septal, endo, :]
    lv_hisApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0., 0.85, 1., 0.]), ord=2, axis=1))) # int(np.argmin(nodesCobiveco[lvnodes, 0])) # [basal, septal, endo, lv]
    rv_hisApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0., 0.85, 1., 1.]), ord=2, axis=1))) # int(np.argmin(nodesCobiveco[rvnodes, 0]))
    lv_hisBundle_indexes = lv_hisBase_path_mat[lv_hisApex_index, 0, :] # The nodes in this path are the LV his bundle
    lv_hisBundle_indexes = lv_hisBundle_indexes[lv_hisBundle_indexes != nan_value]
    sorted_indexes = np.argsort(lv_hisBase_distance_mat[lv_hisBundle_indexes, 0]) # Sort nodes by distance to the reference
    lv_hisBundle_indexes = lv_hisBundle_indexes[sorted_indexes] # Sort nodes by distance to the reference
    rv_hisBundle_indexes = rv_hisBase_path_mat[rv_hisApex_index, 0, :] # The nodes in this path are the LV his bundle
    rv_hisBundle_indexes = rv_hisBundle_indexes[rv_hisBundle_indexes != nan_value]
    sorted_indexes = np.argsort(rv_hisBase_distance_mat[rv_hisBundle_indexes, 0]) # Sort nodes by distance to the reference
    rv_hisBundle_indexes = rv_hisBundle_indexes[sorted_indexes] # Sort nodes by distance to the reference
    # lv_hisBundle_offsets = lv_hisBase_distance_mat[lv_hisBundle_indexes, 0]
    # rv_hisBundle_offsets = rv_hisBase_distance_mat[rv_hisBundle_indexes, 0]
    # Rule 3) The apical and Lateral/Freewall in the RV can connect directly to their closest point in the hisbundle that has ab < 0.8
    rv_hisBundle_ab_values = nodesCobiveco[rvnodes[rv_hisBundle_indexes], 0]
    rv_hisBundle_meta_indexes = np.nonzero(rv_hisBundle_ab_values < 0.8)[0]
    rv_ab_values = nodesCobiveco[rvnodes, 0]
    rv_ab_dist = np.abs(rv_ab_values[:, np.newaxis] - rv_hisBundle_ab_values[rv_hisBundle_meta_indexes])
    rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(np.asarray(rv_hisBundle_indexes[rv_hisBundle_meta_indexes], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours, max_path_len=djikstra_max_path_len)
    rv_hisbundle_connections = np.argmin(np.abs(rv_ab_dist), axis=1) # match root nodes to the hisbundles as a rib-cage (same ab values) #np.argmin(rv_hisbundle_distance_mat, axis=1)
    rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(rv_hisbundle_connections.shape[0]):
        offset = rv_hisBase_path_mat[rv_hisBundle_indexes[rv_hisBundle_meta_indexes[rv_hisbundle_connections[i]]], 0, :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
        rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + rv_hisBase_distance_mat[
            rv_hisBundle_indexes[rv_hisBundle_meta_indexes[rv_hisbundle_connections[i]]], 0]
    rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
    rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
    rv_apical_lateral_mask = ((nodesCobiveco[rvnodes, 0] <= 0.2) | ((0.2 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.5))) & np.logical_not(rv_visited)
    rv_PK_distance_mat[rv_apical_lateral_mask] = rv_hisbundle_distance_mat[rv_apical_lateral_mask]
    rv_PK_path_mat[rv_apical_lateral_mask, :] = rv_hisbundle_path_mat[rv_apical_lateral_mask, :]
    rv_visited[rv_apical_lateral_mask] = True
    
    
    
    
    
    
    
    
    # rv_dense_indexes[rv_apical_lateral_mask] = True
    # Rule 4) The apical hisbundle can directly connects to Septal and Apical (and Paraseptal for the RV) root nodes after it crosses the Apex-to-Base 0.4/0.2 threshold LV/RV
    lv_hisMiddle_index = lv_hisBundle_indexes[int(np.argmin(np.abs(nodesCobiveco[lvnodes[lv_hisBundle_indexes], 0] - 0.4)))] # [basal, septal, endo, lv]
    rv_hisMiddle_index = rv_hisBundle_indexes[int(np.argmin(np.abs(nodesCobiveco[rvnodes[rv_hisBundle_indexes], 0] - 0.2)))] # [basal, septal, endo, rv]
    # print('Hisbundle Index checks')
    # print(lv_hisMiddle_index)
    # print(lv_hisBundle_indexes)
    # print(lv_hisApex_index)
    # print(lv_hisBase_distance_mat[lv_hisBundle_indexes, 0])
    # print('RV')
    # print(rv_hisMiddle_index)
    # print(rv_hisBundle_indexes)
    # print(rv_hisApex_index)
    # print(rv_hisBase_distance_mat[rv_hisBundle_indexes, 0])
    # index = find_first(lv_hisMiddle_index, lv_hisBundle_indexes)
    # print('Test')
    # print(index)
    # print(lv_hisBundle_indexes[index])
    # print(lv_hisBundle_indexes[index:])
    lv_hisConnected_indexes = lv_hisBundle_indexes[find_first(lv_hisMiddle_index, lv_hisBundle_indexes):]
    rv_hisConnected_indexes = rv_hisBundle_indexes[find_first(rv_hisMiddle_index, rv_hisBundle_indexes):]
    
    # # Rule 3) lv hisbundle goes through point [0.4, 0.85, 1., 0.] == [basal, septal, endo, lv] and rv hisbundle goes through point [0.2, 0.85, 1., 1.]
    # lv_hisMiddle_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.85, 1., 0.]), ord=2, axis=1))) # [basal, septal, endo, lv]
    # rv_hisMiddle_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.85, 1., 1.]), ord=2, axis=1))) # [basal, septal, endo, rv]
    # Compute paths from middle poitns to the rest since these are the first points to allow connections to the endocardium
    
    
    # lv_hisMiddle_distance_mat, lv_hisMiddle_path_mat = djikstra(np.asarray([lv_hisMiddle_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=djikstra_max_path_len)
    # rv_hisMiddle_distance_mat, rv_hisMiddle_path_mat = djikstra(np.asarray([rv_hisMiddle_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours, max_path_len=djikstra_max_path_len)
    # lv_hisMiddleToBase_distance = lv_hisMiddle_distance_mat[lv_hisBase_index, 0] # LV His Basal offset
    
    
    # lv_hisMiddleToBase_path = lv_hisMiddle_path_mat[lv_hisBase_index, 0, :] # For visualisation only - path offset
    # lv_hisMiddleToBase_path = lv_hisMiddleToBase_path[lv_hisMiddleToBase_path != nan_value] # For visualisation only - path offset
    # sorted_indexes = np.flip(np.argsort(lv_hisMiddle_distance_mat[lv_hisMiddleToBase_path, 0])) # Sort nodes by inverse-distance to the reference
    # lv_hisMiddleToBase_path = lv_hisMiddleToBase_path[sorted_indexes] # Sort nodes by inverse-distance to the reference - Base path offset
    # rv_hisMiddleToBase_distance = rv_hisMiddle_distance_mat[rv_hisBase_index, 0] # RV His Basal offset
    # rv_hisMiddleToBase_path = rv_hisMiddle_path_mat[rv_hisBase_index, 0, :] # For visualisation only - path offset
    # rv_hisMiddleToBase_path = rv_hisMiddleToBase_path[rv_hisMiddleToBase_path != nan_value] # For visualisation only - path offset
    # sorted_indexes = np.flip(np.argsort(rv_hisMiddle_distance_mat[rv_hisMiddleToBase_path, 0])) # Sort nodes by inverse-distance to the reference
    # rv_hisMiddleToBase_path = rv_hisMiddleToBase_path[sorted_indexes] # Sort nodes by inverse-distance to the reference - Base path offset
    
    
    # lv_hisConnected_indexes = lv_hisMiddle_path_mat[lv_hisApex_index, 0, :] # The nodes in this path can connect to LV root nodes
    # lv_hisConnected_indexes = lv_hisConnected_indexes[lv_hisConnected_indexes != nan_value] # The nodes in this path can connect to LV root nodes
    # sorted_indexes = np.argsort(lv_hisMiddle_distance_mat[lv_hisConnected_indexes, 0]) # Sort nodes by distance to the reference
    # lv_hisConnected_indexes = lv_hisConnected_indexes[sorted_indexes] # Sort nodes by distance to the reference
    # lv_hisConnected_offsets = lv_hisMiddle_distance_mat[lv_hisConnected_indexes, 0] + lv_hisMiddleToBase_distance # Offsets to the hisbundle that can be connected
    # lv_hisConnected_path_offset = np.full((lv_hisConnected_indexes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # for i in range(lv_hisConnected_indexes.shape[0]):
    #     index = lv_hisConnected_indexes[i]
    #     offset = lv_hisMiddle_path_mat[index, :]
    #     offset = offset[offset != nan_value] # For visualisation only - path offset
    #     sorted_indexes = np.argsort(lv_hisMiddle_distance_mat[offset, 0]) # Sort nodes by distance to the reference
    #     offset = offset[sorted_indexes] # Sort nodes by distance to the reference - Base path offset
    #     offset = np.concatenate((lv_hisMiddleToBase_path, offset), axis=0)
    #     lv_hisConnected_path_offset[i, :offset.shape[0]] = offset
    # rv_hisConnected_indexes = rv_hisMiddle_path_mat[rv_hisApex_index, 0, :] # The nodes in this path can connect to RV root nodes
    # rv_hisConnected_indexes = rv_hisConnected_indexes[rv_hisConnected_indexes != nan_value] # The nodes in this path can connect to RV root nodes
    # sorted_indexes = np.argsort(rv_hisMiddle_distance_mat[rv_hisConnected_indexes, 0]) # Sort nodes by distance to the reference
    # rv_hisConnected_indexes = rv_hisConnected_indexes[sorted_indexes] # Sort nodes by distance to the reference
    # rv_hisConnected_offsets = rv_hisMiddle_distance_mat[rv_hisConnected_indexes, 0] + rv_hisMiddleToBase_distance # Offsets to the hisbundle that can be connected
    # rv_hisConnected_path_offset = np.full((rv_hisConnected_indexes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # for i in range(rv_hisConnected_indexes.shape[0]):
    #     index = rv_hisConnected_indexes[i]
    #     offset = rv_hisMiddle_path_mat[index, :]
    #     offset = offset[offset != nan_value] # For visualisation only - path offset
    #     sorted_indexes = np.argsort(rv_hisMiddle_distance_mat[offset, 0]) # Sort nodes by distance to the reference
    #     offset = offset[sorted_indexes] # Sort nodes by distance to the reference - Base path offset
    #     offset = np.concatenate((rv_hisMiddleToBase_path, offset), axis=0)
    #     rv_hisConnected_path_offset[i, :offset.shape[0]] = offset
    
    # Rule 5) Root nodes in the Apical regions of the heart connect to their closest Apical hisbundle node
    lv_hisbundle_distance_mat, lv_hisbundle_path_mat = djikstra(np.asarray(lv_hisConnected_indexes, dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=djikstra_max_path_len)
    rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(np.asarray(rv_hisConnected_indexes, dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours, max_path_len=djikstra_max_path_len)
    lv_hisbundle_connections = np.argmin(lv_hisbundle_distance_mat, axis=1)
    rv_hisbundle_connections = np.argmin(rv_hisbundle_distance_mat, axis=1)
    lv_hisbundle_path_mat_aux = np.full((lv_hisbundle_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_hisbundle_distance_mat_aux = np.full((lv_hisbundle_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(lv_hisbundle_connections.shape[0]):
        offset = lv_hisBase_path_mat[lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0, :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = lv_hisbundle_path_mat[i, lv_hisbundle_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        lv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
        lv_hisbundle_distance_mat_aux[i] = lv_hisbundle_distance_mat[i, lv_hisbundle_connections[i]] + lv_hisBase_distance_mat[lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0]
    lv_hisbundle_path_mat = lv_hisbundle_path_mat_aux
    lv_hisbundle_distance_mat = lv_hisbundle_distance_mat_aux
    rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(rv_hisbundle_connections.shape[0]):
        offset = rv_hisBase_path_mat[rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0, :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
        rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + rv_hisBase_distance_mat[rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0]
    rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
    rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
    # Rule 5) Apical|Septal|Paraseptal regions of the heart are defined as AB < 0.4/0.2 in the LV/RV | [0.7 < RT < 1.] | [0. < RT < 0.2] & [0.5 < RT < 0.7], respectively
    lv_apical_septal_mask = ((nodesCobiveco[lvnodes, 0] <= 0.4) | ((0.7 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 1.))) & np.logical_not(lv_visited)
    rv_apical_septal_paraseptal_mask = (((nodesCobiveco[rvnodes, 0] <= 0.2) | ((0.7 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 1.))) |
        (((0.0 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.2)) | ((0.5 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.7)))) & np.logical_not(rv_visited)
    lv_PK_distance_mat[lv_apical_septal_mask] = lv_hisbundle_distance_mat[lv_apical_septal_mask]
    lv_PK_path_mat[lv_apical_septal_mask, :] = lv_hisbundle_path_mat[lv_apical_septal_mask, :]
    lv_visited[lv_apical_septal_mask] = True
    rv_PK_distance_mat[rv_apical_septal_paraseptal_mask] = rv_hisbundle_distance_mat[rv_apical_septal_paraseptal_mask]
    rv_PK_path_mat[rv_apical_septal_paraseptal_mask, :] = rv_hisbundle_path_mat[rv_apical_septal_paraseptal_mask, :]
    rv_visited[rv_apical_septal_paraseptal_mask] = True
    # lv_dense_indexes[lv_apical_septal_mask] = True
    # rv_dense_indexes[rv_apical_septal_paraseptal_mask] = True
    
    
    
    
    
    # Rule 6) Paraseptal regions of the heart are connected from apex to base through either [0.4/0.2, 0.1, 1., :] or  [0.4/0.2, 0.6, 1., :] LV/RV
    lv_ant_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.6, 1., 0.]), ord=2, axis=1))) # [mid, paraseptal, endo, lv]
    lv_post_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.1, 1., 0.]), ord=2, axis=1))) # [mid, paraseptal, endo, lv]
    # rv_ant_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.6, 1., 1.]), ord=2, axis=1))) # [mid, paraseptal, endo, rv]
    # rv_post_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.1, 1., 1.]), ord=2, axis=1))) # [mid, paraseptal, endo, rv]
    if not lv_visited[lv_ant_paraseptalApex_index]:
        lv_PK_distance_mat[lv_ant_paraseptalApex_index] = lv_hisbundle_distance_mat[lv_ant_paraseptalApex_index]
        lv_PK_path_mat[lv_ant_paraseptalApex_index, :] = lv_hisbundle_path_mat[lv_ant_paraseptalApex_index, :]
        lv_visited[lv_ant_paraseptalApex_index] = True
    if not lv_visited[lv_post_paraseptalApex_index]:
        lv_PK_distance_mat[lv_post_paraseptalApex_index] = lv_hisbundle_distance_mat[lv_post_paraseptalApex_index]
        lv_PK_path_mat[lv_post_paraseptalApex_index, :] = lv_hisbundle_path_mat[lv_post_paraseptalApex_index, :]
        lv_visited[lv_post_paraseptalApex_index] = True
    # if not rv_visited[rv_ant_paraseptalApex_index]:
    #     rv_PK_distance_mat[rv_ant_paraseptalApex_index] = rv_hisbundle_distance_mat[rv_ant_paraseptalApex_index]
    #     rv_PK_path_mat[rv_ant_paraseptalApex_index, :] = rv_hisbundle_path_mat[rv_ant_paraseptalApex_index, :]
    #     rv_visited[rv_ant_paraseptalApex_index] = True
    # if not rv_visited[rv_post_paraseptalApex_index]:
    #     rv_PK_distance_mat[rv_post_paraseptalApex_index] = rv_hisbundle_distance_mat[rv_post_paraseptalApex_index]
    #     rv_PK_path_mat[rv_post_paraseptalApex_index, :] = rv_hisbundle_path_mat[rv_post_paraseptalApex_index, :]
    #     rv_visited[rv_post_paraseptalApex_index] = True
    lv_paraseptalApex_offsets = np.array([lv_PK_distance_mat[lv_ant_paraseptalApex_index], lv_PK_distance_mat[lv_post_paraseptalApex_index]], dtype=float)
    # rv_paraseptalApex_offsets = np.array([rv_PK_distance_mat[rv_ant_paraseptalApex_index], rv_PK_distance_mat[rv_post_paraseptalApex_index]], dtype=float)
    lv_paraseptal_distance_mat, lv_paraseptal_path_mat = djikstra(np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC,
        lvneighbours, max_path_len=djikstra_max_path_len)
    # rv_paraseptal_distance_mat, rv_paraseptal_path_mat = djikstra(np.asarray([rv_ant_paraseptalApex_index, rv_post_paraseptalApex_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC,
    #     rvneighbours, max_path_len=djikstra_max_path_len)
    lv_paraseptal_connections = np.argmin(lv_paraseptal_distance_mat, axis=1)
    # rv_paraseptal_connections = np.argmin(rv_paraseptal_distance_mat, axis=1)
    lv_paraseptal_path_mat_aux = np.full((lv_paraseptal_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_paraseptal_distance_mat_aux = np.full((lv_paraseptal_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(lv_paraseptal_connections.shape[0]):
        offset = lv_PK_path_mat[np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int)[lv_paraseptal_connections[i]], :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = lv_paraseptal_path_mat[i, lv_paraseptal_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        lv_paraseptal_path_mat_aux[i, :path.shape[0]] = path
        lv_paraseptal_distance_mat_aux[i] = lv_paraseptal_distance_mat[i, lv_paraseptal_connections[i]] + lv_paraseptalApex_offsets[lv_paraseptal_connections[i]]
    lv_paraseptal_path_mat = lv_paraseptal_path_mat_aux
    lv_paraseptal_distance_mat = lv_paraseptal_distance_mat_aux
    # rv_paraseptal_path_mat_aux = np.full((rv_paraseptal_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # rv_paraseptal_distance_mat_aux = np.full((rv_paraseptal_distance_mat.shape[0]), nan_value, dtype=np.float64)
    # for i in range(rv_paraseptal_connections.shape[0]):
    #     offset = rv_PK_path_mat[np.asarray([rv_ant_paraseptalApex_index, rv_post_paraseptalApex_index], dtype=int)[rv_paraseptal_connections[i]], :]
    #     offset = offset[offset != nan_value] # For visualisation only - path offset
    #     path = rv_paraseptal_path_mat[i, rv_paraseptal_connections[i], :]
    #     path = path[path != nan_value] # For visualisation only - path offset
    #     path = np.concatenate((offset, path), axis=0)
    #     rv_paraseptal_path_mat_aux[i, :path.shape[0]] = path
    #     rv_paraseptal_distance_mat_aux[i] = rv_paraseptal_distance_mat[i, rv_paraseptal_connections[i]] + rv_paraseptalApex_offsets[rv_paraseptal_connections[i]]
    # rv_paraseptal_path_mat = rv_paraseptal_path_mat_aux
    # rv_paraseptal_distance_mat = rv_paraseptal_distance_mat_aux
    # Rule 7) Paraseptal regions of the heart are defined as [0. < rotation-angle (RT) < 0.2] & [0.5 < RT < 0.7], these are connected to their closest paraseptal routing point (anterior or posterior)
    lv_paraseptal_mask = (((0.0 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 0.2)) | ((0.5 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 0.7))) & np.logical_not(lv_visited)
    # rv_paraseptal_mask = (((0.0 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.2)) | ((0.5 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.7))) & np.logical_not(rv_visited)
    lv_PK_distance_mat[lv_paraseptal_mask] = lv_paraseptal_distance_mat[lv_paraseptal_mask]
    lv_PK_path_mat[lv_paraseptal_mask, :] = lv_paraseptal_path_mat[lv_paraseptal_mask, :]
    lv_visited[lv_paraseptal_mask] = True
    # rv_PK_distance_mat[rv_paraseptal_mask] = rv_paraseptal_distance_mat[rv_paraseptal_mask]
    # rv_PK_path_mat[rv_paraseptal_mask, :] = rv_paraseptal_path_mat[rv_paraseptal_mask, :]
    # rv_visited[rv_paraseptal_mask] = True
    # Rule 8) Freewall regions of the heart are connected from apex to base through [0.4, 0.35, 1., :] in the LV
    lv_freewallApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.35, 1., 0.]), ord=2, axis=1))) # [mid, freewall, endo, lv]
    # rv_freewallApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.35, 1., 1.]), ord=2, axis=1))) # [mid, freewall, endo, rv]
    if not lv_visited[lv_freewallApex_index]:
        lv_PK_distance_mat[lv_freewallApex_index] = lv_hisbundle_distance_mat[lv_freewallApex_index]
        lv_PK_path_mat[lv_freewallApex_index, :] = lv_hisbundle_path_mat[lv_freewallApex_index, :]
        lv_visited[lv_freewallApex_index] = True
    # if not rv_visited[rv_freewallApex_index]:
    #     rv_PK_distance_mat[rv_freewallApex_index] = rv_hisbundle_distance_mat[rv_freewallApex_index]
    #     rv_PK_path_mat[rv_freewallApex_index, :] = rv_hisbundle_path_mat[rv_freewallApex_index, :]
    #     rv_visited[rv_freewallApex_index] = True
    lv_freewallApex_offset = lv_PK_distance_mat[lv_freewallApex_index]
    lv_freewallApex_path_offset = lv_PK_path_mat[lv_freewallApex_index, :]
    lv_freewallApex_path_offset = lv_freewallApex_path_offset[lv_freewallApex_path_offset != nan_value]
    # rv_freewallApex_offset = rv_PK_distance_mat[rv_freewallApex_index]
    # rv_freewallApex_path_offset = rv_PK_path_mat[rv_freewallApex_index, :]
    # rv_freewallApex_path_offset = rv_freewallApex_path_offset[rv_freewallApex_path_offset != nan_value]
    lv_freewall_distance_mat, lv_freewall_path_mat = djikstra(np.asarray([lv_freewallApex_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC,
        lvneighbours, max_path_len=djikstra_max_path_len)
    # rv_freewall_distance_mat, rv_freewall_path_mat = djikstra(np.asarray([rv_freewallApex_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC,
    #     rvneighbours, max_path_len=djikstra_max_path_len)
    lv_freewall_path_mat_aux = np.full((lv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_freewall_distance_mat_aux = np.full((lv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(lv_freewall_distance_mat.shape[0]):
        path = lv_freewall_path_mat[i, 0, :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((lv_freewallApex_path_offset, path), axis=0)
        lv_freewall_path_mat_aux[i, :path.shape[0]] = path
        lv_freewall_distance_mat_aux[i] = lv_freewall_distance_mat[i, 0] + lv_freewallApex_offset
    lv_freewall_path_mat = lv_freewall_path_mat_aux
    lv_freewall_distance_mat = lv_freewall_distance_mat_aux
    # rv_freewall_path_mat_aux = np.full((rv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # rv_freewall_distance_mat_aux = np.full((rv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    # for i in range(rv_freewall_distance_mat.shape[0]):
    #     path = rv_freewall_path_mat[i, 0, :]
    #     path = path[path != nan_value] # For visualisation only - path offset
    #     path = np.concatenate((rv_freewallApex_path_offset, path), axis=0)
    #     rv_freewall_path_mat_aux[i, :path.shape[0]] = path
    #     rv_freewall_distance_mat_aux[i] = rv_freewall_distance_mat[i, 0] + rv_freewallApex_offset
    # rv_freewall_path_mat = rv_freewall_path_mat_aux
    # rv_freewall_distance_mat = rv_freewall_distance_mat_aux
    # Rule 9) Latera/Freewall in the RV connects directly to any part of the hisbundle
    # rv_freewall_distance_mat, rv_freewall_path_mat = djikstra(np.asarray(rv_hisMiddleToBase_path, dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC,
    #     rvneighbours, max_path_len=djikstra_max_path_len)
    # rv_freewall_offsets = rv_freewall_distance_mat[rv_hisBase_index, :]
    # rv_freewall_path_offsets = rv_freewall_path_mat[rv_hisBase_index, :, :]
    # rv_freewall_connections = np.argmin(rv_freewall_distance_mat, axis=1)
    # rv_freewall_path_mat_aux = np.full((rv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # rv_freewall_distance_mat_aux = np.full((rv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    # for i in range(rv_freewall_connections.shape[0]):
    #     offset = rv_freewall_path_offsets[rv_freewall_connections[i], :]
    #     offset = offset[offset != nan_value] # For visualisation only - path offset
    #     path = rv_freewall_path_mat[i, rv_freewall_connections[i], :]
    #     path = path[path != nan_value] # For visualisation only - path offset
    #     path = np.concatenate((offset, path), axis=0)
    #     rv_freewall_path_mat_aux[i, :path.shape[0]] = path
    #     rv_freewall_distance_mat_aux[i] = rv_freewall_distance_mat[i, rv_freewall_connections[i]] + rv_freewall_offsets[rv_freewall_connections[i]]
    # rv_freewall_path_mat = rv_freewall_path_mat_aux
    # rv_freewall_distance_mat = rv_freewall_distance_mat_aux
    # Rule 10) Freewall/Lateral regions of the heart are defined as [0.2 < rotation-angle (RT) < 0.5], these are connected to the lateral routing point
    lv_freewall_mask = ((0.2 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 0.5)) & np.logical_not(lv_visited)
    lv_PK_distance_mat[lv_freewall_mask] = lv_freewall_distance_mat[lv_freewall_mask]
    lv_PK_path_mat[lv_freewall_mask, :] = lv_freewall_path_mat[lv_freewall_mask, :]
    lv_visited[lv_freewall_mask] = True
    # rv_PK_distance_mat[rv_freewall_mask] = rv_freewall_distance_mat[rv_freewall_mask]
    # rv_PK_path_mat[rv_freewall_mask, :] = rv_freewall_path_mat[rv_freewall_mask, :]
    # rv_visited[rv_freewall_mask] = True
    # Rule 11) Dense fast endocardial layer
    lv_apical_mask = nodesCobiveco[lvnodes, 0] <= 0.4
    lv_dense_indexes[lv_apical_mask] = True
    rv_apical_freewall_mask = ((nodesCobiveco[rvnodes, 0] <= 0.4) | ((0.2 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.5)))
    rv_dense_indexes[rv_apical_freewall_mask] = True
    
    
    
    
    
    lv_PK_time_mat = lv_PK_distance_mat/0.2 # time is in ms
    rv_PK_time_mat = rv_PK_distance_mat/0.2 # time is in ms
    
    # # Save the pseudo-Purkinje networks considered during the inference for visulalisation and plotting purposes - LV
    # lvedges_indexes = np.all(np.isin(edges, lvnodes), axis=1)
    # lv_edges = edges[lvedges_indexes, :]
    # lvPK_edges_indexes = np.zeros(lv_edges.shape[0], dtype=np.bool_)
    # lv_root_to_PKpath_mat = lv_PK_path_mat[lvActnode_ids, :]
    # for i in range(0, lv_edges.shape[0], 1):
    #     for j in range(0, lv_root_to_PKpath_mat.shape[0], 1):
    #         path = lvnodes[lv_root_to_PKpath_mat[j, :][lv_root_to_PKpath_mat[j, :] != nan_value]]
    #         for k in range(0, path.shape[0]-1, 1):
    #             new_edge = path[k:k+2]
    #             if np.all(np.isin(lv_edges[i, :], new_edge)):
    #                 lvPK_edges_indexes[i] = 1
    #                 break
    # LV_PK_edges = lv_edges[lvPK_edges_indexes, :]
    # # Save the available LV Purkinje network
    # with open(figPath + meshName + '_available_LV_PKnetwork.vtk', 'w') as f:
    #     f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
    #     for i in range(0, nodesXYZ.shape[0], 1):
    #         f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
    #     f.write('LINES '+str(LV_PK_edges.shape[0])+' '+str(LV_PK_edges.shape[0]*3)+'\n')
    #     for i in range(0, LV_PK_edges.shape[0], 1):
    #         f.write('2 ' + str(LV_PK_edges[i, 0]) + ' ' + str(LV_PK_edges[i, 1]) + '\n')
    #     f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(LV_PK_edges.shape[0]) + '\n')
    # # RV
    # rvedges_indexes = np.all(np.isin(edges, rvnodes), axis=1)
    # rv_edges = edges[rvedges_indexes, :]
    # rvPK_edges_indexes = np.zeros(rv_edges.shape[0], dtype=np.bool_)
    # rv_root_to_PKpath_mat = rv_PK_path_mat[rvActnode_ids, :]
    # for i in range(0, rv_edges.shape[0], 1):
    #     for j in range(0, rv_root_to_PKpath_mat.shape[0], 1):
    #         path = rvnodes[rv_root_to_PKpath_mat[j, :][rv_root_to_PKpath_mat[j, :] != nan_value]]
    #         for k in range(0, path.shape[0]-1, 1):
    #             new_edge = path[k:k+2]
    #             if np.all(np.isin(rv_edges[i, :], new_edge)):
    #                 rvPK_edges_indexes[i] = 1
    #                 break
    #
    # RV_PK_edges = rv_edges[rvPK_edges_indexes, :]
    # # Save the available RV Purkinje network
    # with open(figPath + meshName + '_available_RV_PKnetwork.vtk', 'w') as f:
    #     f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS '+str(nodesXYZ.shape[0])+' double\n')
    #     for i in range(0, nodesXYZ.shape[0], 1):
    #         f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) +'\n')
    #     f.write('LINES '+str(RV_PK_edges.shape[0])+' '+str(RV_PK_edges.shape[0]*3)+'\n')
    #     for i in range(0, RV_PK_edges.shape[0], 1):
    #         f.write('2 ' + str(RV_PK_edges[i, 0]) + ' ' + str(RV_PK_edges[i, 1]) + '\n')
    #     f.write('POINT_DATA '+str(nodesXYZ.shape[0])+'\n\nCELL_DATA '+ str(RV_PK_edges.shape[0]) + '\n')
    #
    #
    # atmap = np.zeros((nodesXYZ.shape[0]))
    # atmap[lvnodes] = lv_PK_time_mat
    # atmap[rvnodes] = rv_PK_time_mat
    # with open(figPath + meshName  + '_ecg_dtw.ensi.PKtimes', 'w') as f:
    #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
    #     for i in range(0, len(atmap)):
    #         f.write(str(atmap[i]) + '\n')
    
    
    # raise
    
    
    
    
    
    
    # lv_dense_indexes = np.zeros((lv_dense_nodes.shape[0])).astype(int)
    #     for i in range(lv_dense_nodes.shape[0]):
    #         lv_dense_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_dense_indexes, axis=0, return_index=True)[1]
    #     lv_dense_indexes = lv_dense_indexes[sorted(indexes)]
    
    
    
    
    
    # rv_freewall_path_mat_aux = np.full((rv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # rv_freewall_distance_mat_aux = np.full((rv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    # for i in range(rv_freewall_distance_mat.shape[0]):
    #     path = rv_freewall_path_mat[i, 0, :]
    #     path = path[path != nan_value] # For visualisation only - path offset
    #     path = np.concatenate((rv_freewallApex_path_offset, path), axis=0)
    #     rv_freewall_path_mat_aux[i, :path.shape[0]] = path
    #     rv_freewall_distance_mat_aux[i] = rv_freewall_distance_mat[i, 0] + rv_freewallApex_offset
    # rv_freewall_path_mat = rv_freewall_path_mat_aux
    # rv_freewall_distance_mat = rv_freewall_distance_mat_aux
    
    
    # print('lv_visited')
    # print(np.sum(lv_visited))
    # print(lv_visited.shape)
    #
    # print('rv_visited')
    # print(np.sum(rv_visited))
    # print(rv_visited.shape)
    #
    # t_e = time.time()-t_s
    # print('Time: '+ str(t_e))
    
    lv_dense_indexes = lvnodes[lv_dense_indexes]
    rv_dense_indexes = rvnodes[rv_dense_indexes]
    
    return lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, lv_dense_indexes, rv_dense_indexes, nodesCobiveco
    

def backup_generatePurkinjeWithCobiveco(dataPath= 'metaData/DTI003/', meshName='DTI003', figPath='Figures_Clinical_PK/'):
    djikstra_max_path_len = 150
    lv_dense_indexes = np.zeros(lvnodes.shape, dtype=bool)
    rv_dense_indexes = np.zeros(rvnodes.shape, dtype=bool)
    t_s = time.time()
    # Read VTK Cobiveco files - REFERENCE
    cobiveco_xyz, cobiveco_ab_rt_tm_tv = getCobiveco(cobiveco_fileName=dataPath + meshName + '_cobiveco.vtk')
    # Prepare for Djikstra - Set LV endocardial edges aside
    lvnodesXYZ = nodesXYZ[lvnodes, :]
    lvedges = edges[np.all(np.isin(edges, lvnodes), axis=1), :]
    lvedges[:, 0] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 0]]).astype(int)
    lvedges[:, 1] = np.asarray([np.flatnonzero(lvnodes == node_id)[0] for node_id in lvedges[:, 1]]).astype(int)
    lvedgeVEC = lvnodesXYZ[lvedges[:, 0], :] - lvnodesXYZ[lvedges[:, 1], :]  # edge vectors
    lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, lvnodesXYZ.shape[0], 1)]
    for i in range(0, len(lvunfoldedEdges), 1):
        aux[lvunfoldedEdges[i, 0]].append(i)
    lvneighbours = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    # Set RV endocardial edges aside
    rvnodesXYZ = nodesXYZ[rvnodes, :]
    rvedges = edges[np.all(np.isin(edges, rvnodes), axis=1), :]
    rvedges[:, 0] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 0]]).astype(int)
    rvedges[:, 1] = np.asarray([np.flatnonzero(rvnodes == node_id)[0] for node_id in rvedges[:, 1]]).astype(int)
    rvedgeVEC = rvnodesXYZ[rvedges[:, 0], :] - rvnodesXYZ[rvedges[:, 1], :]  # edge vectors
    rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    aux = [[] for i in range(0, rvnodesXYZ.shape[0], 1)]
    for i in range(0, len(rvunfoldedEdges), 1):
        aux[rvunfoldedEdges[i, 0]].append(i)
    rvneighbours = [np.array(n) for n in aux]
    aux = None  # Clear Memory
    # Map between nodesXYZ and its Cobiveco coordinates
    cobiveco_indexes = mapIndexes(nodesXYZ, cobiveco_xyz, return_unique_only=False)
    nodesCobiveco = cobiveco_ab_rt_tm_tv[cobiveco_indexes, :]
    # Define Purkinje tree using Cobiveco-based rules - Initialise data structures
    lv_PK_distance_mat = np.full(lvnodes.shape, nan_value, np.float64)
    lv_PK_path_mat = np.full((lvnodes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    rv_PK_distance_mat = np.full(rvnodes.shape, nan_value, np.float64)
    rv_PK_path_mat = np.full((rvnodes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_visited = np.zeros(lvnodes.shape, dtype=bool)
    rv_visited = np.zeros(rvnodes.shape, dtype=bool)
    # Rule 1) his-av node at coordinates [1., 0.85, 1., :] == [basal, septal, endo, :]
    lv_hisBase_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([1., 0.85, 1., 0.]), ord=2, axis=1))) # [basal, septal, endo, lv]
    rv_hisBase_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([1., 0.85, 1., 1.]), ord=2, axis=1))) # [basal, septal, endo, rv]
    lv_hisBase_distance_mat, lv_hisBase_path_mat = djikstra(np.asarray([lv_hisBase_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=djikstra_max_path_len)
    rv_hisBase_distance_mat, rv_hisBase_path_mat = djikstra(np.asarray([rv_hisBase_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours, max_path_len=djikstra_max_path_len)
    # Rule 2) hisbundle goes down to most apical endocardial point while trying to keep a straight rotation trajectory [0., 0.85, 1., :] == [basal, septal, endo, :]
    lv_hisApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0., 0.85, 1., 0.]), ord=2, axis=1))) # int(np.argmin(nodesCobiveco[lvnodes, 0])) # [basal, septal, endo, lv]
    rv_hisApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0., 0.85, 1., 1.]), ord=2, axis=1))) # int(np.argmin(nodesCobiveco[rvnodes, 0]))
    lv_hisBundle_indexes = lv_hisBase_path_mat[lv_hisApex_index, 0, :] # The nodes in this path are the LV his bundle
    lv_hisBundle_indexes = lv_hisBundle_indexes[lv_hisBundle_indexes != nan_value]
    rv_hisBundle_indexes = rv_hisBase_path_mat[rv_hisApex_index, 0, :] # The nodes in this path are the LV his bundle
    rv_hisBundle_indexes = rv_hisBundle_indexes[rv_hisBundle_indexes != nan_value]
    lv_hisBundle_offsets = lv_hisBase_distance_mat[lv_hisBundle_indexes]
    rv_hisBundle_offsets = rv_hisBase_distance_mat[rv_hisBundle_indexes]
    # Rule 3) The RV hisbundle can directly connect to any root node on the RV Lateral/freewall
    sorted_indexes = np.argsort(rv_hisBase_distance_mat[rv_hisBundle_indexes, 0]) # Sort nodes by distance to the reference
    rv_hisBundle_indexes = rv_hisBundle_indexes[sorted_indexes] # Sort nodes by distance to the reference
    rv_hisConnected_path_offset = np.full((rv_hisConnected_indexes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    for i in range(rv_hisConnected_indexes.shape[0]):
        index = rv_hisConnected_indexes[i]
        offset = rv_hisMiddle_path_mat[index, :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        sorted_indexes = np.argsort(rv_hisMiddle_distance_mat[offset, 0]) # Sort nodes by distance to the reference
        offset = offset[sorted_indexes] # Sort nodes by distance to the reference - Base path offset
        offset = np.concatenate((rv_hisMiddleToBase_path, offset), axis=0)
        rv_hisConnected_path_offset[i, :offset.shape[0]] = offset
    # Rule 3) The apical hisbundle can directly connect to root nodes after it crosses the Apex-to-Base 0.4/0.2 threshold LV/RV
    lv_hisMiddle_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, 0] - np.array([0.4]), ord=2, axis=1))) # [basal, septal, endo, lv]
    rv_hisMiddle_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, 0] - np.array([0.2]), ord=2, axis=1))) # [basal, septal, endo, rv]
    # # Rule 3) lv hisbundle goes through point [0.4, 0.85, 1., 0.] == [basal, septal, endo, lv] and rv hisbundle goes through point [0.2, 0.85, 1., 1.]
    # lv_hisMiddle_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.85, 1., 0.]), ord=2, axis=1))) # [basal, septal, endo, lv]
    # rv_hisMiddle_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.85, 1., 1.]), ord=2, axis=1))) # [basal, septal, endo, rv]
    # Compute paths from middle poitns to the rest since these are the first points to allow connections to the endocardium
    lv_hisMiddle_distance_mat, lv_hisMiddle_path_mat = djikstra(np.asarray([lv_hisMiddle_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=djikstra_max_path_len)
    rv_hisMiddle_distance_mat, rv_hisMiddle_path_mat = djikstra(np.asarray([rv_hisMiddle_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours, max_path_len=djikstra_max_path_len)
    lv_hisMiddleToBase_distance = lv_hisMiddle_distance_mat[lv_hisBase_index, 0] # LV His Basal offset
    lv_hisMiddleToBase_path = lv_hisMiddle_path_mat[lv_hisBase_index, 0, :] # For visualisation only - path offset
    lv_hisMiddleToBase_path = lv_hisMiddleToBase_path[lv_hisMiddleToBase_path != nan_value] # For visualisation only - path offset
    sorted_indexes = np.flip(np.argsort(lv_hisMiddle_distance_mat[lv_hisMiddleToBase_path, 0])) # Sort nodes by inverse-distance to the reference
    lv_hisMiddleToBase_path = lv_hisMiddleToBase_path[sorted_indexes] # Sort nodes by inverse-distance to the reference - Base path offset
    rv_hisMiddleToBase_distance = rv_hisMiddle_distance_mat[rv_hisBase_index, 0] # RV His Basal offset
    rv_hisMiddleToBase_path = rv_hisMiddle_path_mat[rv_hisBase_index, 0, :] # For visualisation only - path offset
    rv_hisMiddleToBase_path = rv_hisMiddleToBase_path[rv_hisMiddleToBase_path != nan_value] # For visualisation only - path offset
    sorted_indexes = np.flip(np.argsort(rv_hisMiddle_distance_mat[rv_hisMiddleToBase_path, 0])) # Sort nodes by inverse-distance to the reference
    rv_hisMiddleToBase_path = rv_hisMiddleToBase_path[sorted_indexes] # Sort nodes by inverse-distance to the reference - Base path offset
    lv_hisConnected_indexes = lv_hisMiddle_path_mat[lv_hisApex_index, 0, :] # The nodes in this path can connect to LV root nodes
    lv_hisConnected_indexes = lv_hisConnected_indexes[lv_hisConnected_indexes != nan_value] # The nodes in this path can connect to LV root nodes
    sorted_indexes = np.argsort(lv_hisMiddle_distance_mat[lv_hisConnected_indexes, 0]) # Sort nodes by distance to the reference
    lv_hisConnected_indexes = lv_hisConnected_indexes[sorted_indexes] # Sort nodes by distance to the reference
    lv_hisConnected_offsets = lv_hisMiddle_distance_mat[lv_hisConnected_indexes, 0] + lv_hisMiddleToBase_distance # Offsets to the hisbundle that can be connected
    lv_hisConnected_path_offset = np.full((lv_hisConnected_indexes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    for i in range(lv_hisConnected_indexes.shape[0]):
        index = lv_hisConnected_indexes[i]
        offset = lv_hisMiddle_path_mat[index, :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        sorted_indexes = np.argsort(lv_hisMiddle_distance_mat[offset, 0]) # Sort nodes by distance to the reference
        offset = offset[sorted_indexes] # Sort nodes by distance to the reference - Base path offset
        offset = np.concatenate((lv_hisMiddleToBase_path, offset), axis=0)
        lv_hisConnected_path_offset[i, :offset.shape[0]] = offset
    rv_hisConnected_indexes = rv_hisMiddle_path_mat[rv_hisApex_index, 0, :] # The nodes in this path can connect to RV root nodes
    rv_hisConnected_indexes = rv_hisConnected_indexes[rv_hisConnected_indexes != nan_value] # The nodes in this path can connect to RV root nodes
    sorted_indexes = np.argsort(rv_hisMiddle_distance_mat[rv_hisConnected_indexes, 0]) # Sort nodes by distance to the reference
    rv_hisConnected_indexes = rv_hisConnected_indexes[sorted_indexes] # Sort nodes by distance to the reference
    rv_hisConnected_offsets = rv_hisMiddle_distance_mat[rv_hisConnected_indexes, 0] + rv_hisMiddleToBase_distance # Offsets to the hisbundle that can be connected
    rv_hisConnected_path_offset = np.full((rv_hisConnected_indexes.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    for i in range(rv_hisConnected_indexes.shape[0]):
        index = rv_hisConnected_indexes[i]
        offset = rv_hisMiddle_path_mat[index, :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        sorted_indexes = np.argsort(rv_hisMiddle_distance_mat[offset, 0]) # Sort nodes by distance to the reference
        offset = offset[sorted_indexes] # Sort nodes by distance to the reference - Base path offset
        offset = np.concatenate((rv_hisMiddleToBase_path, offset), axis=0)
        rv_hisConnected_path_offset[i, :offset.shape[0]] = offset
    # Rule 4) Root nodes in the Apical regions of the heart connect to their closest Apical hisbundle node
    lv_hisbundle_distance_mat, lv_hisbundle_path_mat = djikstra(np.asarray(lv_hisConnected_indexes, dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC, lvneighbours, max_path_len=djikstra_max_path_len)
    rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(np.asarray(rv_hisConnected_indexes, dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC, rvneighbours, max_path_len=djikstra_max_path_len)
    lv_hisbundle_connections = np.argmin(lv_hisbundle_distance_mat, axis=1)
    rv_hisbundle_connections = np.argmin(rv_hisbundle_distance_mat, axis=1)
    lv_hisbundle_path_mat_aux = np.full((lv_hisbundle_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_hisbundle_distance_mat_aux = np.full((lv_hisbundle_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(lv_hisbundle_connections.shape[0]):
        offset = lv_hisConnected_path_offset[lv_hisbundle_connections[i]]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = lv_hisbundle_path_mat[i, lv_hisbundle_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        lv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
        lv_hisbundle_distance_mat_aux[i] = lv_hisbundle_distance_mat[i, lv_hisbundle_connections[i]] + lv_hisConnected_offsets[lv_hisbundle_connections[i]]
    lv_hisbundle_path_mat = lv_hisbundle_path_mat_aux
    lv_hisbundle_distance_mat = lv_hisbundle_distance_mat_aux
    rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(rv_hisbundle_connections.shape[0]):
        offset = rv_hisConnected_path_offset[rv_hisbundle_connections[i]]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
        rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + rv_hisConnected_offsets[rv_hisbundle_connections[i]]
    rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
    rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
    # Rule 5) Apical regions of the heart are defined as apex-to-base (AB) < 0.4/0.2 in the LV/RV and Septal regions as [0.7 < rotation-angle (RT) < 1.] these are connected directly to the hisbundle
    lv_apical_mask = (nodesCobiveco[lvnodes, 0] <= 0.4) | (0.7 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 1.) & np.logical_not(lv_visited)
    rv_apical_mask = (nodesCobiveco[rvnodes, 0] <= 0.2) | (0.7 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 1.) & np.logical_not(rv_visited)
    lv_PK_distance_mat[lv_apical_mask] = lv_hisbundle_distance_mat[lv_apical_mask]
    lv_PK_path_mat[lv_apical_mask, :] = lv_hisbundle_path_mat[lv_apical_mask, :]
    lv_visited[lv_apical_mask] = True
    rv_PK_distance_mat[rv_apical_mask] = rv_hisbundle_distance_mat[rv_apical_mask]
    rv_PK_path_mat[rv_apical_mask, :] = rv_hisbundle_path_mat[rv_apical_mask, :]
    rv_visited[rv_apical_mask] = True
    lv_dense_indexes[lv_apical_mask] = True
    rv_dense_indexes[rv_apical_mask] = True
    # Rule 6) Paraseptal regions of the heart are connected from apex to base through either [0.4/0.2, 0.1, 1., :] or  [0.4/0.2, 0.6, 1., :] LV/RV
    lv_ant_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.6, 1., 0.]), ord=2, axis=1))) # [mid, paraseptal, endo, lv]
    lv_post_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.1, 1., 0.]), ord=2, axis=1))) # [mid, paraseptal, endo, lv]
    rv_ant_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.6, 1., 1.]), ord=2, axis=1))) # [mid, paraseptal, endo, rv]
    rv_post_paraseptalApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.1, 1., 1.]), ord=2, axis=1))) # [mid, paraseptal, endo, rv]
    if not lv_visited[lv_ant_paraseptalApex_index]:
        lv_PK_distance_mat[lv_ant_paraseptalApex_index] = lv_hisbundle_distance_mat[lv_ant_paraseptalApex_index]
        lv_PK_path_mat[lv_ant_paraseptalApex_index, :] = lv_hisbundle_path_mat[lv_ant_paraseptalApex_index, :]
        lv_visited[lv_ant_paraseptalApex_index] = True
    if not lv_visited[lv_post_paraseptalApex_index]:
        lv_PK_distance_mat[lv_post_paraseptalApex_index] = lv_hisbundle_distance_mat[lv_post_paraseptalApex_index]
        lv_PK_path_mat[lv_post_paraseptalApex_index, :] = lv_hisbundle_path_mat[lv_post_paraseptalApex_index, :]
        lv_visited[lv_post_paraseptalApex_index] = True
    if not rv_visited[rv_ant_paraseptalApex_index]:
        rv_PK_distance_mat[rv_ant_paraseptalApex_index] = rv_hisbundle_distance_mat[rv_ant_paraseptalApex_index]
        rv_PK_path_mat[rv_ant_paraseptalApex_index, :] = rv_hisbundle_path_mat[rv_ant_paraseptalApex_index, :]
        rv_visited[rv_ant_paraseptalApex_index] = True
    if not rv_visited[rv_post_paraseptalApex_index]:
        rv_PK_distance_mat[rv_post_paraseptalApex_index] = rv_hisbundle_distance_mat[rv_post_paraseptalApex_index]
        rv_PK_path_mat[rv_post_paraseptalApex_index, :] = rv_hisbundle_path_mat[rv_post_paraseptalApex_index, :]
        rv_visited[rv_post_paraseptalApex_index] = True
    lv_paraseptalApex_offsets = np.array([lv_PK_distance_mat[lv_ant_paraseptalApex_index], lv_PK_distance_mat[lv_post_paraseptalApex_index]], dtype=float)
    rv_paraseptalApex_offsets = np.array([rv_PK_distance_mat[rv_ant_paraseptalApex_index], rv_PK_distance_mat[rv_post_paraseptalApex_index]], dtype=float)
    lv_paraseptal_distance_mat, lv_paraseptal_path_mat = djikstra(np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC,
        lvneighbours, max_path_len=djikstra_max_path_len)
    rv_paraseptal_distance_mat, rv_paraseptal_path_mat = djikstra(np.asarray([rv_ant_paraseptalApex_index, rv_post_paraseptalApex_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC,
        rvneighbours, max_path_len=djikstra_max_path_len)
    lv_paraseptal_connections = np.argmin(lv_paraseptal_distance_mat, axis=1)
    rv_paraseptal_connections = np.argmin(rv_paraseptal_distance_mat, axis=1)
    lv_paraseptal_path_mat_aux = np.full((lv_paraseptal_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_paraseptal_distance_mat_aux = np.full((lv_paraseptal_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(lv_paraseptal_connections.shape[0]):
        offset = lv_PK_path_mat[np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int)[lv_paraseptal_connections[i]], :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = lv_paraseptal_path_mat[i, lv_paraseptal_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        lv_paraseptal_path_mat_aux[i, :path.shape[0]] = path
        lv_paraseptal_distance_mat_aux[i] = lv_paraseptal_distance_mat[i, lv_paraseptal_connections[i]] + lv_paraseptalApex_offsets[lv_paraseptal_connections[i]]
    lv_paraseptal_path_mat = lv_paraseptal_path_mat_aux
    lv_paraseptal_distance_mat = lv_paraseptal_distance_mat_aux
    rv_paraseptal_path_mat_aux = np.full((rv_paraseptal_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    rv_paraseptal_distance_mat_aux = np.full((rv_paraseptal_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(rv_paraseptal_connections.shape[0]):
        offset = rv_PK_path_mat[np.asarray([rv_ant_paraseptalApex_index, rv_post_paraseptalApex_index], dtype=int)[rv_paraseptal_connections[i]], :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = rv_paraseptal_path_mat[i, rv_paraseptal_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        rv_paraseptal_path_mat_aux[i, :path.shape[0]] = path
        rv_paraseptal_distance_mat_aux[i] = rv_paraseptal_distance_mat[i, rv_paraseptal_connections[i]] + rv_paraseptalApex_offsets[rv_paraseptal_connections[i]]
    rv_paraseptal_path_mat = rv_paraseptal_path_mat_aux
    rv_paraseptal_distance_mat = rv_paraseptal_distance_mat_aux
    # Rule 7) Paraseptal regions of the heart are defined as [0. < rotation-angle (RT) < 0.2] & [0.5 < RT < 0.7], these are connected to their closest paraseptal routing point (anterior or posterior)
    lv_paraseptal_mask = (((0.0 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 0.2)) | ((0.5 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 0.7))) & np.logical_not(lv_visited)
    rv_paraseptal_mask = (((0.0 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.2)) | ((0.5 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.7))) & np.logical_not(rv_visited)
    lv_PK_distance_mat[lv_paraseptal_mask] = lv_paraseptal_distance_mat[lv_paraseptal_mask]
    lv_PK_path_mat[lv_paraseptal_mask, :] = lv_paraseptal_path_mat[lv_paraseptal_mask, :]
    lv_visited[lv_paraseptal_mask] = True
    rv_PK_distance_mat[rv_paraseptal_mask] = rv_paraseptal_distance_mat[rv_paraseptal_mask]
    rv_PK_path_mat[rv_paraseptal_mask, :] = rv_paraseptal_path_mat[rv_paraseptal_mask, :]
    rv_visited[rv_paraseptal_mask] = True
    # Rule 8) Freewall regions of the heart are connected from apex to base through [0.4, 0.35, 1., :] in the LV
    lv_freewallApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[lvnodes, :] - np.array([0.4, 0.35, 1., 0.]), ord=2, axis=1))) # [mid, freewall, endo, lv]
    # rv_freewallApex_index = int(np.argmin(np.linalg.norm(nodesCobiveco[rvnodes, :] - np.array([0.2, 0.35, 1., 1.]), ord=2, axis=1))) # [mid, freewall, endo, rv]
    if not lv_visited[lv_freewallApex_index]:
        lv_PK_distance_mat[lv_freewallApex_index] = lv_hisbundle_distance_mat[lv_freewallApex_index]
        lv_PK_path_mat[lv_freewallApex_index, :] = lv_hisbundle_path_mat[lv_freewallApex_index, :]
        lv_visited[lv_freewallApex_index] = True
    # if not rv_visited[rv_freewallApex_index]:
    #     rv_PK_distance_mat[rv_freewallApex_index] = rv_hisbundle_distance_mat[rv_freewallApex_index]
    #     rv_PK_path_mat[rv_freewallApex_index, :] = rv_hisbundle_path_mat[rv_freewallApex_index, :]
    #     rv_visited[rv_freewallApex_index] = True
    lv_freewallApex_offset = lv_PK_distance_mat[lv_freewallApex_index]
    lv_freewallApex_path_offset = lv_PK_path_mat[lv_freewallApex_index, :]
    lv_freewallApex_path_offset = lv_freewallApex_path_offset[lv_freewallApex_path_offset != nan_value]
    # rv_freewallApex_offset = rv_PK_distance_mat[rv_freewallApex_index]
    # rv_freewallApex_path_offset = rv_PK_path_mat[rv_freewallApex_index, :]
    # rv_freewallApex_path_offset = rv_freewallApex_path_offset[rv_freewallApex_path_offset != nan_value]
    lv_freewall_distance_mat, lv_freewall_path_mat = djikstra(np.asarray([lv_freewallApex_index], dtype=int), lvnodesXYZ, lvunfoldedEdges, lvedgeVEC,
        lvneighbours, max_path_len=djikstra_max_path_len)
    # rv_freewall_distance_mat, rv_freewall_path_mat = djikstra(np.asarray([rv_freewallApex_index], dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC,
    #     rvneighbours, max_path_len=djikstra_max_path_len)
    lv_freewall_path_mat_aux = np.full((lv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    lv_freewall_distance_mat_aux = np.full((lv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(lv_freewall_distance_mat.shape[0]):
        path = lv_freewall_path_mat[i, 0, :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((lv_freewallApex_path_offset, path), axis=0)
        lv_freewall_path_mat_aux[i, :path.shape[0]] = path
        lv_freewall_distance_mat_aux[i] = lv_freewall_distance_mat[i, 0] + lv_freewallApex_offset
    lv_freewall_path_mat = lv_freewall_path_mat_aux
    lv_freewall_distance_mat = lv_freewall_distance_mat_aux
    # rv_freewall_path_mat_aux = np.full((rv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # rv_freewall_distance_mat_aux = np.full((rv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    # for i in range(rv_freewall_distance_mat.shape[0]):
    #     path = rv_freewall_path_mat[i, 0, :]
    #     path = path[path != nan_value] # For visualisation only - path offset
    #     path = np.concatenate((rv_freewallApex_path_offset, path), axis=0)
    #     rv_freewall_path_mat_aux[i, :path.shape[0]] = path
    #     rv_freewall_distance_mat_aux[i] = rv_freewall_distance_mat[i, 0] + rv_freewallApex_offset
    # rv_freewall_path_mat = rv_freewall_path_mat_aux
    # rv_freewall_distance_mat = rv_freewall_distance_mat_aux
    # Rule 9) Latera/Freewall in the RV connects directly to any part of the hisbundle
    rv_freewall_distance_mat, rv_freewall_path_mat = djikstra(np.asarray(rv_hisMiddleToBase_path, dtype=int), rvnodesXYZ, rvunfoldedEdges, rvedgeVEC,
        rvneighbours, max_path_len=djikstra_max_path_len)
    rv_freewall_offsets = rv_freewall_distance_mat[rv_hisBase_index, :]
    rv_freewall_path_offsets = rv_freewall_path_mat[rv_hisBase_index, :, :]
    rv_freewall_connections = np.argmin(rv_freewall_distance_mat, axis=1)
    rv_freewall_path_mat_aux = np.full((rv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    rv_freewall_distance_mat_aux = np.full((rv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    for i in range(rv_freewall_connections.shape[0]):
        offset = rv_freewall_path_offsets[rv_freewall_connections[i], :]
        offset = offset[offset != nan_value] # For visualisation only - path offset
        path = rv_freewall_path_mat[i, rv_freewall_connections[i], :]
        path = path[path != nan_value] # For visualisation only - path offset
        path = np.concatenate((offset, path), axis=0)
        rv_freewall_path_mat_aux[i, :path.shape[0]] = path
        rv_freewall_distance_mat_aux[i] = rv_freewall_distance_mat[i, rv_freewall_connections[i]] + rv_freewall_offsets[rv_freewall_connections[i]]
    rv_freewall_path_mat = rv_freewall_path_mat_aux
    rv_freewall_distance_mat = rv_freewall_distance_mat_aux
    # Rule 10) Freewall/Lateral regions of the heart are defined as [0.2 < rotation-angle (RT) < 0.5], these are connected to the lateral routing point
    lv_freewall_mask = (0.2 <= nodesCobiveco[lvnodes, 1]) & (nodesCobiveco[lvnodes, 1] <= 0.5) & np.logical_not(lv_visited)
    rv_freewall_mask = (0.2 <= nodesCobiveco[rvnodes, 1]) & (nodesCobiveco[rvnodes, 1] <= 0.5) & np.logical_not(rv_visited)
    lv_PK_distance_mat[lv_freewall_mask] = lv_freewall_distance_mat[lv_freewall_mask]
    lv_PK_path_mat[lv_freewall_mask, :] = lv_freewall_path_mat[lv_freewall_mask, :]
    lv_visited[lv_freewall_mask] = True
    rv_PK_distance_mat[rv_freewall_mask] = rv_freewall_distance_mat[rv_freewall_mask]
    rv_PK_path_mat[rv_freewall_mask, :] = rv_freewall_path_mat[rv_freewall_mask, :]
    rv_visited[rv_freewall_mask] = True
    rv_dense_indexes[rv_freewall_mask] = True
    
    
    
    
    # lv_dense_indexes = np.zeros((lv_dense_nodes.shape[0])).astype(int)
    #     for i in range(lv_dense_nodes.shape[0]):
    #         lv_dense_indexes[i] = lvnodes[np.argmin(np.linalg.norm(nodesXYZ[lvnodes, :] - lv_dense_nodes[i, :], ord=2, axis=1)).astype(int)]
    #     indexes = np.unique(lv_dense_indexes, axis=0, return_index=True)[1]
    #     lv_dense_indexes = lv_dense_indexes[sorted(indexes)]
    
    
    
    
    
    # rv_freewall_path_mat_aux = np.full((rv_freewall_path_mat.shape[0], djikstra_max_path_len), nan_value, dtype=np.int32)
    # rv_freewall_distance_mat_aux = np.full((rv_freewall_distance_mat.shape[0]), nan_value, dtype=np.float64)
    # for i in range(rv_freewall_distance_mat.shape[0]):
    #     path = rv_freewall_path_mat[i, 0, :]
    #     path = path[path != nan_value] # For visualisation only - path offset
    #     path = np.concatenate((rv_freewallApex_path_offset, path), axis=0)
    #     rv_freewall_path_mat_aux[i, :path.shape[0]] = path
    #     rv_freewall_distance_mat_aux[i] = rv_freewall_distance_mat[i, 0] + rv_freewallApex_offset
    # rv_freewall_path_mat = rv_freewall_path_mat_aux
    # rv_freewall_distance_mat = rv_freewall_distance_mat_aux
    
    
    print('lv_visited')
    print(np.sum(lv_visited))
    print(lv_visited.shape)
    
    print('rv_visited')
    print(np.sum(rv_visited))
    print(rv_visited.shape)
    
    t_e = time.time()-t_s
    print('Time: '+ str(t_e))
    
    lv_dense_indexes = lvnodes[lv_dense_indexes]
    rv_dense_indexes = rvnodes[rv_dense_indexes]
    
    return lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, lv_dense_indexes, rv_dense_indexes, nodesCobiveco

    
# TODO: this function is very inneficient, but it's Ok because it's only called once per inference
def generateRootNodes(n, nRootLocs, nRootNodes_centre, nRootNodes_std, nRootNodes_range):
    rootNodes = np.zeros((n, nRootLocs))
    for i in range(n):
        N_on = 0 # TODO change to constraint the number of root nodes within the ranges
        while N_on < nRootNodes_range[0] or N_on > nRootNodes_range[1]:
            N_on = int(round(np.random.normal(loc=nRootNodes_centre, scale=nRootNodes_std)))
        rootNodes[i, np.random.permutation(nRootLocs)[:N_on-1]] = 1
        rootNodes[i, i%nRootLocs] = 1 # Ensure that all root nodes are represented at least once
    return rootNodes


# ------------------------------------------- DISTANCE FUNCTIONS ----------------------------------------------------
def dtw_ecg(predicted, target_ecg, max_slope=1.5, w_max=10.):
    """25/02/2021: I have realised that my trianglogram method is equivalent to the vanila paralelogram in practice but less
    computationally efficient. I initially thought that using a parallelogram implied that all warping was to be undone
    towards the end of the signal, like comparing people reading the same text on the same amount of time but with a variability
    on the speed for each part of the sentance. However, that was not the case. The parallelogram serves the same purpose
    as the trianlogram when the constraint of equal ending is put in place, namely, only (N-1, M-1) is evaluated. This
    constraint forces both signals to represent the same information rather than one signal being only half of the other.
    Therefore, by using the trianglogram plus the restriction, the only thing I am achieving is to do more calculations
    strictly necessary. However, the original implementation of the parallelogram allows an amount of warping proportional
    to the length difference between the two signals, which could lead to non-physiological warping. Here instead, the
    maximum amount of warping is defined in w_max and max_slope; this feature is key because the discrepancy calculation
    needs to be equivalent for all signals throughout the method regardless of their length."""
    
    """Dynamic Time Warping distance specific for comparing electrocardiogram signals.
    It implements a trianglogram constraint (inspired from Itakura parallelogram (Itakura, 1975)).
    It also implements weight penalty with a linear increasing cost away from the true diagonal (i.e. i=j).
    Moreover, it implements a step-pattern with slope-P value = 0.5 from (Sakoe and Chiba, 1978).
    Finally, the algorithm penalises the difference between the lenght of the two signals and adds it to the DTW distance.
    Options
    -------
    max_slope : float Maximum slope of the trianglogram.
    w_max : float weigth coeficient to the distance from the diagonal.
    small_c :  float weight coeficient to the difference in lenght between the signals being compared.
    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).
    .. [2] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).
    """
    # Don't compute repetitions - Numpy Unique has an issue with NaN values: https://github.com/numpy/numpy/issues/2111
    small_c = 0.05 * 171 / meshVolume # Scaling factor to account for differences in mesh size # 2022/05/04
    p = np.copy(predicted[:, 0, :])
    p[np.isnan(p)]=np.inf
    p, particle_index, unique_index = np.unique(p, return_index=True, return_inverse=True, axis=0)
    p=None
    predicted = predicted[particle_index, :, :]
    
    # This code has parallel pympy sections as well as numba parallel sections
    nParts = predicted.shape[0]
    nLeads = predicted.shape[1]
    res = pymp.shared.array(nParts, dtype='float64')
    # if True: # TODO DELETE THIS
    #     for conf_i in range(0, nParts): # TODO DELETE THIS
    # warps = pymp.shared.array(nParts, dtype='float64') # TODO DELETE THIS 2022/05/17
    with pymp.Parallel(min(nParts, threadsNum)) as p1:
        for conf_i in p1.range(0, nParts):
            # warps_aux = 0.  # TODO DELETE THIS 2022/05/17
            mask = np.logical_not(np.isnan(predicted[conf_i, :, :]))
            pred_ecg = np.squeeze(predicted[conf_i:conf_i+1, :, mask[0, :]])  # using slicing index does not change the shape of the object,
                                                                              # however, mixing slicing with broadcasting does change it, which then
                                                                              # requires moving the axis with np.moveaxis
            # Lengths of each sequence to be compared
            n_timestamps_1 = pred_ecg.shape[1]
            n_timestamps_2 = target_ecg.shape[1]
            
            # Computes the region (in-window area using a trianglogram)
            # WARNING: this is a shorter version of the code for generating the region which does not account for special cases, the full version is the fuction "trianglorgram" from myfunctions.py
            max_slope_ = max_slope
            min_slope_ = 1 / max_slope_
            scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2)
            max_slope_ *= scale_max
            scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1)
            min_slope_ *= scale_min
            centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
            lower_bound = min_slope_ * np.arange(n_timestamps_1)
            lower_bound = np.round(lower_bound, 2)
            lower_bound = np.floor(lower_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path
            upper_bound = max_slope_ * np.arange(n_timestamps_1) + 1
            upper_bound = np.round(upper_bound, 2)
            upper_bound = np.ceil(upper_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path
            region_original = np.asarray([lower_bound, upper_bound]).astype('int64')
            region_original = np.clip(region_original[:, :n_timestamps_1], 0, n_timestamps_2) # Project region on the feasible set
            
            part_dtw = 0.
            # Compute the DTW for each lead and particle separately so that leads can be wraped differently from each other
            for lead_i in range(nLeads):
                region = np.copy(region_original)
                x = pred_ecg[lead_i, :]
                y = target_ecg[lead_i, :]
                
                # Computes cost matrix from dtw input
                dist_ = lambda x, y : (x - y) ** 2 # The squared term will penalise differences exponentially, which is desirable especially when relative amplitudes are important

                # Computs the cost matrix considering the window (0 inside, np.inf outside)
                cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
                m = np.amax(cost_mat.shape)
                for i in numba.prange(n_timestamps_1):
                    for j in numba.prange(region[0, i], region[1, i]):
                        # cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.) # This new weight considers that wraping in time is cheaper the later it's done #* (w_max/(1+math.exp(-g*(abs(i-j)-m/2)))+1.) # + abs(i-j)*small_c # Weighted version of the DTW algorithm
                        # TODO: BELLOW MODIFIED ON THE 2022/05/16 - to resemble more what it used to be, larger pennalty for warping
                        cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.) + abs(i-j)*small_c*10 # 07/12/2021 When going from synthetic data into clinical we observe that the
                        # singals cannot be as similar anymore due to the difference in source models between real patients and Eikonal, -
                        # additional penalty for warping:  +  abs( i-j)*small_c*10
                        # TODO: BELLOW MODIFIED ON THE 2022/05/16 - to resemble more what it used to be, larger pennalty for warping
                        # cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.) + abs(i-j)*0.5 * 171 / meshVolume # 2022/05/16 larger penalty for warping

                # Computes the accumulated cost matrix
                acc_cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
                acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
                    cost_mat[0, 0: region[1, 0]]
                )
                acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
                    cost_mat[0: region[1, 0], 0]
                )
                region_ = np.copy(region)
                region_[0] = np.maximum(region_[0], 1)
                for i in range(1, n_timestamps_1):
                    for j in range(region_[0, i], region_[1, i]):
                        # Implementation of a Slope-constraint as a step-pattern:
                        # This constraint will enforce that the algorithm can only take up to 2 consecutive steps along the time wraping directions.
                        # I decided to make it symetric because in (Sakoe and Chiba, 1978) they state that symetric means that DTW(A, B) == DTW(B, A), although I am not convinced why it's not the case in the asymetric implementation.
                        # Besides, the asymetric case has a bias towards the diagonal which I thought could be desirable in our case, that said, having DTW(A, B) == DTW(B, A) may prove even more important, especially in further
                        # applications of this algorithm for ECG comparison.
                        # This implementation is further explained in (Sakoe and Chiba, 1978) and correspondes to the one with P = 0.5, (P = n/m, where P is a rule being inforced, n is the number of steps in the diagonal
                        # direction and m is the steps in the time wraping direction).
                        acc_cost_mat[i, j] = min(
                            acc_cost_mat[i - 1, j-3] + 2*cost_mat[i, j-2] + cost_mat[i, j-1] + cost_mat[i, j],
                            acc_cost_mat[i - 1, j-2] + 2*cost_mat[i, j-1] + cost_mat[i, j],
                            acc_cost_mat[i - 1, j - 1] + 2*cost_mat[i, j],
                            acc_cost_mat[i - 2, j-1] + 2*cost_mat[i-1, j] + cost_mat[i, j],
                            acc_cost_mat[i - 3, j-1] + 2*cost_mat[i-2, j] + cost_mat[i-1, j] + cost_mat[i, j]
                        )
                dtw_dist = acc_cost_mat[-1, -1]/(n_timestamps_1 + n_timestamps_2) # Normalisation M+N according to (Sakoe and Chiba, 1978)
                # signals have the same lenght
                dtw_dist = dtw_dist / np.amax(np.abs(y)) * np.amax(np.abs(target_ecg)) # 2022/05/04 - Normalise by lead amplitude to weight all leads similarly
                part_dtw += math.sqrt(dtw_dist) #I would rather have leads not compensating for each other # 2022/05/04 Add normalisation by max abs amplitude in each lead
                # part_dtw += dtw_dist #I would rather have leads not compensating for each other # 2022/05/16 - Fixed mistake! I was doing the opposite of the comment # TODO check urgent NOW!
            res[conf_i] = part_dtw / nLeads + small_c * (n_timestamps_1-n_timestamps_2)**2/min(n_timestamps_1,n_timestamps_2) # 2022/05/16 # old - I think that it's too strong
            # res[conf_i] = part_dtw / nLeads + small_c * abs(n_timestamps_1-n_timestamps_2)/(n_timestamps_1 + n_timestamps_2) # 2022/05/16 # TODO check urgent NOW!
            # warps[conf_i] = warps_aux  # TODO DELETE THIS 2022/05/17
    # if plot_warps:  # TODO DELETE THIS 2022/05/17
    #     return res[unique_index]  # TODO DELETE THIS 2022/05/17
    # else:  # TODO DELETE THIS 2022/05/17
    return res[unique_index]


def computeDiscrepancyForATMaps(predicted, target): #TODO Change this urgently 2021 February
    # if is_ECGi:#TODO Change this urgently 2021 February
    res = np.sqrt(np.mean((predicted[:, epiface] - target)**2, axis=1))
    # else:#TODO Change this urgently 2021 February
    #     res = np.sqrt(np.mean((predicted[:, epiface] - target[epiface])**2, axis=1))#TODO Change this urgently 2021 February
    return res


def computeATMerror(prediction_list, target): #TODO Change this urgently 2021 February
    # if is_ECGi: #TODO Change this urgently 2021 February
    error = np.sqrt(np.mean((prediction_list[:, epiface] - target[np.newaxis, :])**2, axis=1))
    # else:
    #     error = np.sqrt(np.mean((prediction_list - target[np.newaxis, :])**2, axis=1)) #TODO Change this urgently 2021 February
    return error


# ------------------------------------------- UTILS FUNCTIONS ----------------------------------------------------
@numba.njit()
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


# ------------------------------------------- CORE SMC-ABC FUNCTIONS ----------------------------------------------------

class ABCSMC(object):

    def __init__(self, nparam, simulator, target_data,
#                  scale_param,
#                  prior,
                 max_MCMC_steps,
                 param_boundaries, nlhsParam,
                 maxRootNodeJiggleRate,
                 retain_ratio, nRootNodes_range,
                 resultFile, experiment_output,
                 experiment_metric, hermite_mean, hermite_std, hermite_order,
                 nRootNodes_centre, nRootNodes_std, nRootLocations, npart, keep_fraction,
                 conduction_speeds, target_rootNodes):

        self.experiment_output = experiment_output
        self.experiment_metric = experiment_metric
        self.data = target_data
        self.nRootNodes_range = nRootNodes_range
        self.nRootNodes_centre = nRootNodes_centre
        self.nRootNodes_std = nRootNodes_std
        self.nRootLocations = nRootLocations
        self.npart = npart
        self.keep_fraction = keep_fraction
        self.max_MCMC_steps = max_MCMC_steps
        self.simulator = simulator
        self.nparam = nparam
        self.nlhsParam = nlhsParam
        self.param_boundaries = param_boundaries
        self.conduction_speeds = conduction_speeds
        self.target_rootNodes = target_rootNodes
        self.nRootLocations = nRootLocations
        part_theta = self.iniParticles(self.npart)#prior, transformed_ranges)
#         part_theta[:, scale_param] = np.exp(part_theta[:, scale_param])
        self.part_theta = part_theta
        part_output = self.simulator(self.part_theta, rootNodeActivationIndexes, rootNodeActivationTimes)
        # Calculate sample covariance matrix from the initial particles
        # Use diagonals of this to extract only variances (no covariances)
        # Store the inverse, so that it only need be calculated once
        #    self.invC_S = np.linalg.inv(np.diag(np.diag(np.cov(self.part_output))))
        
        if self.experiment_output == 'ecg' or experiment_output == 'bsp':
            if self.experiment_metric == 'dtw':
                self.f_discrepancy = dtw_ecg
                self.f_summary = doNothing
                self.f_compProb = compProb
                self.f_discrepancy_plot = dtw_trianglorgram
            elif self.experiment_metric == 'hermite':
                # Needs to be included for considering Hermitte coefs
                # self.f_summary = computeHermiteNormalised
                # self.f_discrepancy = computeDiscrepancyForECGasHermite
                self.hermite_mean = hermite_mean
                self.hermite_std = hermite_std
                self.hermite_order = hermite_order
        elif self.experiment_output == 'atm':
            self.f_discrepancy = computeDiscrepancyForATMaps
            self.f_summary = doNothing
            self.f_compProb = compProb
        self.part_discrepancy = self.f_discrepancy(part_output, self.data)
        self.maxRootNodeJiggleRate = maxRootNodeJiggleRate
        self.retain_ratio = retain_ratio
        self.resultFile=resultFile
        
    
    def iniParticles(self, npart):
        # 2/12/2021 :
        # TODO: run the current version, record the time it takes and then change the initialisation so that:
        #   - criterion='correlation' : means that it will minimise the maximum correlation coefficient. - DONE
        #   - looks like both, the Transversal and Endocardial speed are using the first factor in the LHS (e.g. lhs_theta[:, 0]), change so that they use different ones. - DONE
        #   - use random sampling for the normal and fibre directions but within the boundaries of the allowed values. - DONE
        # Do LHS only for Transversal and Endocardial speed and adjust the others
        # lhs_theta = lhs(2, samples=npart*2, criterion='maximin') # Compute more than the required amount and then pick randomly the correct amount
        # lhs_theta = lhs(2, samples=npart, criterion='corr') # 04/12/2021 - correlation option still not available?
        lhs_theta = lhs(3, samples=npart, criterion='corr') # 2022/01/18
        # speeds = np.zeros((npart*2, self.nlhsParam)) # Compute more than the required amount and then pick randomly the correct amount
        speeds = np.zeros((npart, self.nlhsParam)) # 04/12/2021
        if is_healthy:
            sheet_ind = 0
            endo_ind = 1
        else:
            sheet_ind = 1
            endo_ind = 3
        # Myocardial conduction speeds are defined with respect to the transversal/sheet directed speed
        speeds[:, sheet_ind] = lhs_theta[:, 0] * (self.param_boundaries[sheet_ind, 1] - self.param_boundaries[sheet_ind, 0]) + self.param_boundaries[sheet_ind, 0] # sheet speed
        # for i in range(npart*2):
        if not is_healthy: # 07/12/2021
            random_fibre = np.random.random_sample(npart) # 04/12/2021
            random_normal = np.random.random_sample(npart) # 04/12/2021
            for i in range(npart): # 04/12/2021
                speeds[i, 0] = random_fibre[i] * (self.param_boundaries[0, 1] - speeds[i, sheet_ind]) + speeds[i, sheet_ind]  # fibre speed
                speeds[i, 2] = random_normal[i] * (speeds[i, sheet_ind] - self.param_boundaries[2, 0]) + self.param_boundaries[2, 0]  # sheet-normal speed
        if has_endocardial_layer:
            # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
            # if is_healthy:
            #     speeds[:, 0] = lhs_theta[:, 0] * (self.param_boundaries[0, 1] - self.param_boundaries[0, 0]) + self.param_boundaries[0, 0] # sheet speed
            #     speeds[:, 1] = lhs_theta[:, 1] * (self.param_boundaries[1, 1] - self.param_boundaries[1, 0]) + self.param_boundaries[1, 0] # endocardial speed
            # else:
            # speeds[:, 3] = lhs_theta[:, 0] * (self.param_boundaries[3, 1] - self.param_boundaries[3, 0]) + self.param_boundaries[3, 0] # endocardial speed
            # speeds[:, endo_ind] = lhs_theta[:, 1] * (self.param_boundaries[endo_ind, 1] - self.param_boundaries[endo_ind, 0]) + self.param_boundaries[endo_ind, 0] # 2/12/2021 endocardial speed
            speeds[:, endo_ind] = lhs_theta[:, 1] * (self.param_boundaries[endo_ind, 1] - self.param_boundaries[endo_ind, 0]) + self.param_boundaries[endo_ind, 0] # 2022/01/18 endocardial speed
            speeds[:, endo_ind+1] = lhs_theta[:, 2] * (self.param_boundaries[endo_ind+1, 1] - self.param_boundaries[endo_ind+1, 0]) + self.param_boundaries[endo_ind+1, 0] # 2022/01/18 endocardial speed
        # else:
        #     speeds[:, 1] = lhs_theta[:, 0] * (self.param_boundaries[1, 1] - self.param_boundaries[1, 0]) + self.param_boundaries[1, 0] # sheet speed
        #     speeds[:, 0] = speeds[:, 1] * gf_factor # fibre speed
        #     speeds[:, 2] = speeds[:, 1] * gn_factor # sheet-normal speed
        # by leaving 3 decimals we keep a precision of 1 cm/s, knowing that values will be around 100-20 cm/s
        speeds = np.round(speeds, decimals=3) # precision of 1 cm/s
        # speeds = np.true_divide(np.rint(1000*speeds * 2**(-1)), 2**(-1))/1000.0 # np.round(speeds, decimals=4) # 2022/05/16 Changed to precision of 2 cm/s
        # Regulate max and min speed physiological parameter boundaries
        physiological_speed_indexes = np.logical_and(speeds <= self.param_boundaries[:self.nlhsParam, 1], speeds >= self.param_boundaries[:self.nlhsParam, 0])
        physiological_speed_indexes = np.all(physiological_speed_indexes, axis=1) # only accept if all speeds are within human physiological ranges
        if is_healthy: # 07/12/2021
            temp_accepted_indexes = physiological_speed_indexes # 07/12/2021
        else: # 07/12/2021
            # Check for Non healthy fibre-oriented speeds - namely, make sure that gt <= gl and gn <= gt
            healthy_speed_indexes = np.logical_and(speeds[:, 0] >= speeds[:, 1], speeds[:, 1] >= speeds[:, 2])
            temp_accepted_indexes = np.logical_and(physiological_speed_indexes, healthy_speed_indexes)
        # temp_rejected_indexes = np.logical_not(temp_accepted_indexes)
        # nb_to_replace = np.sum(temp_rejected_indexes)
        nb_accepted = np.sum(temp_accepted_indexes)
        # Compute more than the required amount and then pick randomly the correct amount
        speeds = speeds[temp_accepted_indexes]
        selection = np.random.randint(low=0, high=nb_accepted, size=npart) # Chose the accepted speeds
        speeds = speeds[selection]
        # selection = np.random.randint(low=0, high=nb_accepted, size=nb_to_replace) # Chose the replacement speeds
        # speeds[temp_rejected_indexes] = speeds[temp_accepted_indexes][selection]
        rootNodes = generateRootNodes(npart, self.nRootLocations, self.nRootNodes_centre, self.nRootNodes_std, self.nRootNodes_range)
        theta = np.concatenate((speeds, rootNodes), axis=1)
        return theta
        

    def MVN_move(self, Ctheta, cuttoffD, nMoves, jiggleIndex, nb_particles_to_jiggle):
        accepted = np.full(shape=(nb_particles_to_jiggle), fill_value=0, dtype=np.bool_) # 2022/05/04 - fixed
        prop_output_aux = None
        for i in range(nMoves):
            copied_speeds = self.part_theta[jiggleIndex:, :self.nlhsParam] # September 2021 - this is equivalent to copiedTheta, not sure why this naming
            copied_root_nodes = self.part_theta[jiggleIndex:, self.nlhsParam:]
            current_root_nodes = self.part_theta[:jiggleIndex, self.nlhsParam:]
            copied_root_nodes_unique, unique_indexes = np.unique(copied_root_nodes, return_inverse=True, axis=0)
            copied_probs = pymp.shared.array((copied_root_nodes_unique.shape[0]), dtype=np.float64)
            with pymp.Parallel(min(threadsNum, copied_root_nodes_unique.shape[0])) as p1:
                for j in p1.range(copied_root_nodes_unique.shape[0]):
                    copied_probs[j] = self.f_compProb(copied_root_nodes_unique[j, :], current_root_nodes, self.retain_ratio, self.nRootNodes_range[0])
            copied_probs = copied_probs[unique_indexes]
            
            # Jiggle discrete parameters
            prop_root_nodes = np.empty(copied_root_nodes.shape, dtype=np.float64)
            for j in range(copied_root_nodes.shape[0]):
                prop_root_nodes[j] = jiggleDiscreteNonFixed_one(current_root_nodes, self.retain_ratio, self.nRootNodes_range)
                
            prop_root_nodes_unique, unique_indexes = np.unique(prop_root_nodes, return_inverse=True, axis=0) # Evaluate only unique ones and then copy back to all
            prop_probs = pymp.shared.array((prop_root_nodes_unique.shape[0]), dtype=np.float64)
            with pymp.Parallel(min(threadsNum, prop_root_nodes_unique.shape[0])) as p1:
                for j in p1.range(prop_root_nodes_unique.shape[0]):
                    prop_probs[j] = self.f_compProb(prop_root_nodes_unique[j, :], current_root_nodes, self.retain_ratio, self.nRootNodes_range[0])

            prop_probs = prop_probs[unique_indexes] # Copy back to all from unique ones
            # print(np.min(prop_probs))
            # if np.any(prop_probs == 0):
            #     print('hello')
            #     print(prop_root_nodes[np.argmin(prop_probs), :])
            
            non_select_roots = np.random.rand(nb_particles_to_jiggle) > copied_probs/prop_probs
            prop_root_nodes[non_select_roots, :] = copied_root_nodes[non_select_roots, :]
            prop_theta = np.concatenate((copied_speeds, prop_root_nodes), axis=1) # Compile new root nodes with copied speeds
            prop_output = self.simulator(prop_theta, rootNodeActivationIndexes, rootNodeActivationTimes) # Simulate new root nodes
            prop_output_aux = prop_output # TODO: DELETE
            prop_discrepancy = self.f_discrepancy(prop_output, self.data) # Discrepancy of the new root nodes
            accepted = np.logical_or(prop_discrepancy < cuttoffD, accepted)
            self.part_theta[jiggleIndex:][prop_discrepancy < cuttoffD] = prop_theta[prop_discrepancy < cuttoffD] # Keep the particles that have lower discrepancy than the cuttoff
            self.part_discrepancy[jiggleIndex:][prop_discrepancy < cuttoffD] = prop_discrepancy[prop_discrepancy < cuttoffD]
            
            # Jiggle continuous parameters - after jiggling the discrete ones (i.e. conduction speeds and root nodes, respectively)
            copied_root_nodes = self.part_theta[jiggleIndex:, self.nlhsParam:] # It's very important to check again the root nodes to get the ones that have changed
            prop_speeds = (copied_speeds + np.random.multivariate_normal(
                np.zeros((self.nlhsParam,)), Ctheta, size=nb_particles_to_jiggle))
            prop_speeds = np.round(prop_speeds, decimals=3) # decimals=4) precision of 1 mm/s # 2022/05/09 - changed to 1 cm/s # 2022/05/16 - changed to 2 cm/s
            # prop_speeds = np.true_divide(np.rint(1000*prop_speeds * 2**(-1)), 2**(-1))/1000.0 # 2022/05/16 - changed to 5 cm/s - data is in cm/ms, pass to cm/s, then operate, then back to cm/ms
            # Regulate max and min speed physiological parameter boundaries
            physiological_speed_indexes = np.logical_and(prop_speeds <= self.param_boundaries[:self.nlhsParam, 1], prop_speeds >= self.param_boundaries[:self.nlhsParam, 0])
            physiological_speed_indexes = np.all(physiological_speed_indexes, axis=1) # only accept if all speeds are within human physiological ranges
            # Check for Non healthy fibre-oriented speeds - namely, make sure that gt <= gl and gn <= gt
            if is_healthy: # 07/12/2021 - Only considers fast endocardial and transmural
                temp_accepted_indexes = physiological_speed_indexes  # 07/12/2021
            else: # 07/12/2021
                healthy_speed_indexes = np.logical_and(prop_speeds[:, 0] >= prop_speeds[:, 1], prop_speeds[:, 1] >= prop_speeds[:, 2]) # Check orthotropic speeds
                temp_accepted_indexes = np.logical_and(physiological_speed_indexes, healthy_speed_indexes)
            prop_theta = np.concatenate((prop_speeds, copied_root_nodes), axis=1)
            temp_accepted_prop_theta = prop_theta[temp_accepted_indexes, :]
            reduced_prop_output = self.simulator(temp_accepted_prop_theta, rootNodeActivationIndexes, rootNodeActivationTimes)
            output_shape = list(reduced_prop_output.shape)
            output_shape[0] = temp_accepted_indexes.shape[0]
            output_shape = tuple(output_shape)
            prop_output = np.empty(output_shape, dtype=np.float64)
            prop_output[temp_accepted_indexes] = reduced_prop_output
            reduced_prop_discrepancy = self.f_discrepancy(reduced_prop_output, self.data)
            discrepancy_accepted_indexes = reduced_prop_discrepancy < cuttoffD
            temp_accepted_indexes[temp_accepted_indexes] = discrepancy_accepted_indexes
            accepted = np.logical_or(temp_accepted_indexes, accepted)
            self.part_theta[jiggleIndex:][temp_accepted_indexes] = prop_theta[temp_accepted_indexes]
            self.part_discrepancy[jiggleIndex:][temp_accepted_indexes] = reduced_prop_discrepancy[discrepancy_accepted_indexes]
        return accepted, prop_output_aux # TODO: DELETE

    
    def sample(self, targetDiscrepancy):
        # cum_est_accept_rate = 0. # 2022/05/04 TODO: DELETE
        # cum_real_accept_rate = 0. # 2022/05/04 TODO: DELETE
        # cum_count = 0. # 2022/05/04 TODO: DELETE
        looping = True
        self.visualise(particle_indexes=[int(np.argmin(self.part_discrepancy)),int(np.argmax(self.part_discrepancy))])
        visualise_count_ini = 1000 # zero will print at every iteration
        visualise_count = visualise_count_ini
        worst_keep = int(np.round(self.npart*self.keep_fraction))
        ant_part_theta = None
        ant_discrepancy = None
        ant_cutoffdiscrepancy = None
        iteration_count = 0
        worst_keep_ini = int(np.round(self.npart*self.keep_fraction))
        worst_keep = worst_keep_ini
        while looping:
            # cum_count = cum_count+1. # 2022/05/04 TODO: DELETE
            index = np.argsort(self.part_discrepancy)
            self.part_discrepancy = self.part_discrepancy[index]
            self.part_theta = self.part_theta[index, :]
            # Select the new cuttoff discrepancy
            cuttoffDiscrepancy = self.part_discrepancy[worst_keep]
            # Select which particles are going to be copied into the ones that don't make the cut this round
            nb_particles_to_jiggle = self.npart-(worst_keep)
            selection = np.random.randint(low=0, high=worst_keep, size=nb_particles_to_jiggle) # Chose the replacement particles
            self.part_theta[worst_keep:] = self.part_theta[selection]
            self.part_discrepancy[worst_keep:] = self.part_discrepancy[selection]
            # Optimal factor ... don't look at me! Ask Brodie, he is the expert
            Ctheta = 2.38**2/self.nlhsParam * np.cov(self.part_theta[:, :self.nlhsParam].T)
            # Jiggle just once to compute how many jiggles may be needed
            prop_output_aux = None
            est_accept, prop_output_aux = self.MVN_move(Ctheta, cuttoffDiscrepancy, 1, worst_keep, nb_particles_to_jiggle) # 2022/05/04
            est_accept_rate = np.mean(est_accept) # 2022/05/04 This value was being consistently overestimated!!
            # some extra operations have been added to ensure that there are no divisions by zero
            # This computes the number of MCMC jiggles are required in this iteration of the main loop
            MCMC_steps = min(math.ceil(math.log(0.05)/math.log(1-min(max(est_accept_rate, 1e-8), 1-1e-8))), self.max_MCMC_steps)
            # Run the remaining MCMC steps to compleete the amount just calculated - there is no need to keep the output of accepted nodes
            accepted, prop_output_aux = self.MVN_move(Ctheta, cuttoffDiscrepancy, MCMC_steps-1, worst_keep, nb_particles_to_jiggle)  # 2022/05/04 TODO: DELETE
            if prop_output_aux is not None:
                prop_output_bad = prop_output_aux[np.logical_not(accepted)]
            else:
                prop_output_bad = None
            nUnique = len(np.unique(self.part_theta, axis=0))
            bestInd = np.argmin(self.part_discrepancy)
            worstInd = np.argmax(self.part_discrepancy) # 2022/05/04
            # cum_est_accept_rate = cum_est_accept_rate + est_accept_rate # 2022/05/04 TODO: DELETE
            # cum_real_accept_rate = cum_real_accept_rate + np.mean(accepted) # 2022/05/04 TODO: DELETE
            
            # Stoping criteria
            if experiment_output == 'atm':
                unique_lim_nb = int(np.round(self.npart*0.5))
            else:
                # unique_lim_nb = int(np.round(self.npart*0.5))
                unique_lim_nb = int(np.round(self.npart*0.5)) # 08/12/2021 it was 10% # 2022/05/08 - Made it 50% again
                # unique_lim_nb = int(np.round(self.npart*0.5)) # 10/12/2021
                # unique_lim_nb = 1 # 05/12/2021
            # if nUnique < unique_lim_nb or cuttoffDiscrepancy < targetDiscrepancy:
            if nUnique < unique_lim_nb or cuttoffDiscrepancy < targetDiscrepancy: # 2022/05/11 - Simplify strategy.
            # #or (np.amax(self.part_discrepancy) - np.amin(self.part_discrepancy) <  0.03 and np.amin(self.part_discrepancy) < 0.4 and nUnique
            # < int(np.round(self.npart*self.keep_fraction))): # 08/12/2021 # 2022/05/08 TODO Check if this was a good idea - Trying to make it end sooner, without being worse
                looping = 0
                if not (os.path.isfile(self.resultFile) and os.path.isfile(self.resultFile.replace('population', 'discrepancy')) and os.path.isfile(self.resultFile.replace('population', 'prediction'))):
                    np.savetxt(self.resultFile, self.part_theta, delimiter=',')
                    # CAUTION! SINCE DISCREPANCY IS ONLY A LIST, THIS WON'T ADD ANY ',' AND WILL IGNORE THE DELIMITER
                    np.savetxt(self.resultFile.replace('population', 'discrepancy'), self.part_discrepancy, delimiter=',')
                    prop_data = self.simulator(np.array([self.part_theta[bestInd, :]]), rootNodeActivationIndexes, rootNodeActivationTimes)
                    if self.experiment_output == 'ecg' or experiment_output == 'bsp':
                        prop_data = prop_data[0, :, :]
                    elif self.experiment_output == 'atm':
                        prop_data = prop_data[0, :]
                    np.savetxt(self.resultFile.replace('population', 'prediction'), prop_data, delimiter=',')
                    visualise_count = 0
            
            
            # print(accepted.shape)# TODO: DELETE
            # print(prop_output_aux.shape) # TODO: DELETE

            if visualise_count < 1:
                #TODO Change this urgently 2021 February # TODO: DELETE
                print('\n'+ meshName + ' Iteration: '+str(iteration_count)+' ; After '+str(MCMC_steps) + ' MCMC_steps and '+str(round((time.time()-t_start)/3600.0, 2))+' hours'
                      # + '\nest_accept_rate ' + str(est_accept_rate)
                      + '\naccepted rate % ' + str(round(np.mean(accepted)*100))
                      + '\nnUnique % ' + str(round(nUnique/self.npart*100))
                      + '\nbest discrepancy ' + str(self.part_discrepancy[bestInd])
                      + '\ncuttoffDiscrepancy ' + str(cuttoffDiscrepancy)
                      # + '\nworst discrepancy ' + str(self.part_discrepancy[worstInd])
                      # + '\ncum_est_accept_rate ' + str(cum_est_accept_rate/cum_count)
                      # + '\ncum_accepted rate ' + str(cum_real_accept_rate/cum_count)
                     )# TODO: DELETE
                # randId = random.randint(0, self.npart-1) # 08/12/2021
                self.visualise(particle_indexes=[bestInd, worstInd], prop_output_bad=prop_output_bad) # 2022/05/04
                visualise_count = visualise_count_ini
                # break # TODO: DELETE 2022/05/17
                # Print the paramters exploration progress
                # Root nodes
                if False:
                    rootNodesIndexes = []
                    for i in range(self.part_theta.shape[0]):
                        x = self.part_theta[i, nlhsParam:]
                        y = np.empty_like(x)
                        rootNodesParam = np.round_(x, 0, y)
                        y = None
                        rootNodesIndexes.append(rootNodeActivationIndexes[rootNodesParam==1])
                    rootNodesIndexes = np.concatenate(rootNodesIndexes, axis=0)
                    # Root nodes of the best discrepancy
                    x = self.part_theta[bestInd, nlhsParam:]
                    y = np.empty_like(x)
                    rootNodesParam = np.round_(x, 0, y)
                    y = None
                    best_roots = nodesXYZ[rootNodeActivationIndexes[rootNodesParam==1], :]
                    
                    endo_node_ids = np.concatenate((lvnodes, rvnodes))
                    fig = plt.figure(constrained_layout=True, figsize = (15,20))
                    fig.suptitle(self.resultFile +' nUnique % ' + str(nUnique/self.npart*100), fontsize=24)
                    ax = fig.add_subplot(331, projection='3d')
                    pred_roots = nodesXYZ[rootNodesIndexes, :]
                    target_roots = nodesXYZ[self.target_rootNodes, :]
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    
                    ax = fig.add_subplot(332, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(10, 0)
                    
                    ax = fig.add_subplot(333, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(10, 45)
                    
                    ax = fig.add_subplot(334, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(10, 90)
                    
                    ax = fig.add_subplot(335, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(10, 135)
                    
                    ax = fig.add_subplot(336, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(10, 180)
                    
                    ax = fig.add_subplot(337, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(10, 225)
                    
                    ax = fig.add_subplot(338, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(10, 270)
                    
                    ax = fig.add_subplot(339, projection='3d')
                    ax.scatter(nodesXYZ[endo_node_ids, 0], nodesXYZ[endo_node_ids, 1], nodesXYZ[endo_node_ids, 2], c='b', marker='o', s=.1)
                    ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 2], c='r', marker='o', s=10)
                    ax.scatter(best_roots[:, 0], best_roots[:, 1], best_roots[:, 2], c='g', marker='o', s=100)
                    if not is_ECGi:
                        ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 2], c='k', marker='x', s=50)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.view_init(elev=10, azim=315) # vertical_axis='z' - doesnt accept this argument - 2022/05/11
                    
                    plt.show()
                    
                    # Conduction speeds
                    # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
                    if is_healthy and has_endocardial_layer:
                        speed_name_list = ['Sheet speed', 'Sparse endocardial speed', 'Dense endocardial speed'] # 2022/01/18
                    elif has_endocardial_layer:
                        speed_name_list = ['Fibre speed', 'Sheet speed', 'Normal speed', 'Sparse endocardial speed', 'Dense endocardial speed'] # 2022/01/18
                    else:
                        speed_name_list = ['Fibre speed', 'Sheet speed', 'Normal speed']
                    fig, axs = plt.subplots(2, len(speed_name_list), constrained_layout=True, figsize = (20,10))
                    fig.suptitle(meshName +' nUnique % ' + str(nUnique/self.npart*100), fontsize=24)
                    # Prepare previous results
                    if not ant_part_theta is None:
                        ant_good_particles = ant_discrepancy < cuttoffDiscrepancy
                        ant_good_theta = ant_part_theta[ant_good_particles, :]*1000
                        ant_bad_theta = ant_part_theta[np.logical_not(ant_good_particles), :]*1000
                    # Iterate over speeds
                    for speed_iter in range(len(speed_name_list)):
                        # Plot new results
                        axs[1, speed_iter].plot(self.part_theta[:worst_keep, speed_iter]*1000, self.part_discrepancy[:worst_keep], 'bo')
                        axs[1, speed_iter].plot(self.part_theta[worst_keep:, speed_iter]*1000, self.part_discrepancy[worst_keep:], 'go')
                        axs[1, speed_iter].plot(self.part_theta[bestInd, speed_iter]*1000, self.part_discrepancy[bestInd], 'ro')
                        axs[1, speed_iter].set_title('New ' + speed_name_list[speed_iter], fontsize=16)
                        # TODO: October 2021 - This entire section needs lots of refactoring, the If clauses make no sense and the naming of variables is inconsistent.
                        if not is_ECGi and not (ant_part_theta is None):
                            true_speed_value = self.conduction_speeds[speed_iter]
                            old_median = np.median(ant_part_theta[:, speed_iter]*1000)
                            new_median = np.median(self.part_theta[:, speed_iter]*1000)
                            axs[1, speed_iter].axvline(x=true_speed_value, c='cyan')
                            if abs(true_speed_value - new_median) <= abs(true_speed_value - old_median):
                                axs[1, speed_iter].axvline(x=new_median, c='green')
                            else:
                                axs[1, speed_iter].axvline(x=new_median, c='red')
                                
                        # Plot previous results for comparison
                        if not ant_part_theta is None:
                            axs[0, speed_iter].plot(ant_good_theta[:, speed_iter], ant_discrepancy[ant_good_particles], 'go')
                            axs[0, speed_iter].plot(ant_bad_theta[:, speed_iter], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                            axs[0, speed_iter].set_title('Ant ' + speed_name_list[speed_iter], fontsize=16)
                            if not is_ECGi:
                                axs[0, speed_iter].axvline(x=self.conduction_speeds[speed_iter], c='cyan')
                            axs[0, speed_iter].axvline(x=np.median(ant_part_theta[:, speed_iter]*1000), c='magenta')
                            axs[0, speed_iter].axhline(y=cuttoffDiscrepancy, color='blue')
                        axs[1, speed_iter].set_xlabel('cm/s', fontsize=16)
                        # axs[0, 1].plot(ant_good_theta[:, 1], ant_discrepancy[ant_good_particles], 'go')
                        # axs[0, 1].plot(ant_bad_theta[:, 1], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                        # axs[0, 1].set_title('Ant Sheet speed', fontsize=16)
                        # if not is_ECGi:
                        #     axs[0, 1].axvline(x=self.conduction_speeds[1], c='grey')
                        #     axs[0, 1].axvline(x=np.median(ant_part_theta[:, 1]*1000), c='purple')
                        # axs[0, 1].axhline(y=cuttoffDiscrepancy, color='blue')
                        # axs[0, 2].plot(ant_good_theta[:, 2], ant_discrepancy[ant_good_particles], 'go')
                        # axs[0, 2].plot(ant_bad_theta[:, 2], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                        # axs[0, 2].set_title('Ant Normal speed', fontsize=16)
                        # if not is_ECGi:
                        #     axs[0, 2].axvline(x=self.conduction_speeds[2], c='grey')
                        #     axs[0, 2].axvline(x=np.median(ant_part_theta[:, 2]*1000), c='purple')
                        # if has_endocardial_layer:
                        #     axs[0, 2].axhline(y=cuttoffDiscrepancy, color='blue')
                        #     axs[0, 3].plot(ant_good_theta[:, 3], ant_discrepancy[ant_good_particles], 'go')
                        #     axs[0, 3].plot(ant_bad_theta[:, 3], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                        #     axs[0, 3].set_title('Ant Endocardial speed', fontsize=16)
                        #     if not is_ECGi:
                        #         axs[0, 3].axvline(x=self.conduction_speeds[3], c='grey')
                        #         axs[0, 3].axvline(x=np.median(ant_part_theta[:, 3]*1000), c='purple')
                        #     axs[0, 3].axhline(y=cuttoffDiscrepancy, color='blue')
                    
                    # Plot new results
                    # Iterate over speeds
                    #     for speed_iter in range(len(speed_name_list)):
                    # speed_iter = 0
                    # axs[1, 0].plot(self.part_theta[:worst_keep, 0]*1000, self.part_discrepancy[:worst_keep], 'bo')
                    # axs[1, 0].plot(self.part_theta[worst_keep:, 0]*1000, self.part_discrepancy[worst_keep:], 'go')
                    # axs[1, 0].plot(self.part_theta[randId, 0]*1000, self.part_discrepancy[randId], 'ro')
                    # axs[1, 0].set_title('New Fibre speed', fontsize=16)
                    # if not is_ECGi:
                    #     true_speed_value = self.conduction_speeds[speed_iter]
                    #     old_median = np.median(self.ant_part_theta[:, speed_iter]*1000)
                    #     new_median = np.median(self.part_theta[:, speed_iter]*1000)
                    #     axs[1, 0].axvline(x=self.conduction_speeds[speed_iter], c='grey')
                    #     if abs(true_speed_value - new_median) <= abs(true_speed_value - old_median):
                    #         axs[1, 0].axvline(x=new_median, c='red')
                    #     else:
                    #         axs[1, 0].axvline(x=new_median, c='green')
                    # #axs[1, 0].axhline(y=cuttoffDiscrepancy, color='r')
                    # axs[1, 1].plot(self.part_theta[:worst_keep, 1]*1000, self.part_discrepancy[:worst_keep], 'bo')
                    # axs[1, 1].plot(self.part_theta[worst_keep:, 1]*1000, self.part_discrepancy[worst_keep:], 'go')
                    # axs[1, 1].plot(self.part_theta[randId, 1]*1000, self.part_discrepancy[randId], 'ro')
                    # axs[1, 1].set_title('New Sheet speed', fontsize=16)
                    # if not is_ECGi:
                    #     axs[1, 1].axvline(x=self.conduction_speeds[1], c='grey')
                    #     axs[1, 1].axvline(x=np.median(self.part_theta[:, 1]*1000), c='red')
                    # #axs[1, 1].axhline(y=cuttoffDiscrepancy, color='r')
                    # axs[1, 2].plot(self.part_theta[:worst_keep, 2]*1000, self.part_discrepancy[:worst_keep], 'bo')
                    # axs[1, 2].plot(self.part_theta[worst_keep:, 2]*1000, self.part_discrepancy[worst_keep:], 'go')
                    # axs[1, 2].plot(self.part_theta[randId, 2]*1000, self.part_discrepancy[randId], 'ro')
                    # axs[1, 2].set_title('New Normal speed', fontsize=16)
                    # if not is_ECGi:
                    #     axs[1, 2].axvline(x=np.median(self.part_theta[:, 2]*1000), c='red')
                    #     axs[1, 2].axvline(x=self.conduction_speeds[2], c='grey')
                    # #axs[1, 2].axhline(y=cuttoffDiscrepancy, color='r')
                    # if has_endocardial_layer:
                    #     axs[1, 3].plot(self.part_theta[:worst_keep, 3]*1000, self.part_discrepancy[:worst_keep], 'bo')
                    #     axs[1, 3].plot(self.part_theta[worst_keep:, 3]*1000, self.part_discrepancy[worst_keep:], 'go')
                    #     axs[1, 3].plot(self.part_theta[randId, 3]*1000, self.part_discrepancy[randId], 'ro')
                    #     axs[1, 3].set_title('New Endocardial speed', fontsize=16)
                    #     if not is_ECGi:
                    #         axs[1, 3].axvline(x=self.conduction_speeds[3], c='grey')
                    #         axs[1, 3].axvline(x=np.median(self.part_theta[:, 3]*1000), c='red')
                    axs[0, 0].set_ylabel('discrepancy', fontsize=16)
                    axs[1, 0].set_ylabel('discrepancy', fontsize=16)
                    plt.show()
                
                # if is_healthy:
                #     fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize = (10,10))
                #     fig.suptitle(meshName +' nUnique % ' + str(nUnique/self.npart*100), fontsize=24)
                #     axs[1, 0].plot(self.part_theta[:worst_keep, 0]*1000, self.part_discrepancy[:worst_keep], 'bo')
                #     axs[1, 0].plot(self.part_theta[worst_keep:, 0]*1000, self.part_discrepancy[worst_keep:], 'go')
                #     axs[1, 0].plot(self.part_theta[randId, 0]*1000, self.part_discrepancy[randId], 'ro')
                #     axs[1, 0].set_title('New Sheet speed', fontsize=16)
                #     if not is_ECGi:
                #         axs[1, 1].axvline(x=self.conduction_speeds[1], c='grey')
                #     #axs[1, 1].axhline(y=cuttoffDiscrepancy, color='r')
                #     axs[1, 1].plot(self.part_theta[:worst_keep, 1]*1000, self.part_discrepancy[:worst_keep], 'bo')
                #     axs[1, 1].plot(self.part_theta[worst_keep:, 1]*1000, self.part_discrepancy[worst_keep:], 'go')
                #     axs[1, 1].plot(self.part_theta[randId, 1]*1000, self.part_discrepancy[randId], 'ro')
                #     axs[1, 1].set_title('New Endocardial speed', fontsize=16)
                #     if not is_ECGi:
                #         axs[1, 3].axvline(x=self.conduction_speeds[3], c='grey')
                #     if not ant_part_theta is None:
                #         ant_good_particles = ant_discrepancy < cuttoffDiscrepancy
                #         ant_good_theta = ant_part_theta[ant_good_particles, :]*1000
                #         ant_bad_theta = ant_part_theta[np.logical_not(ant_good_particles), :]*1000
                #         axs[0, 0].plot(ant_good_theta[:, 0], ant_discrepancy[ant_good_particles], 'go')
                #         axs[0, 0].plot(ant_bad_theta[:, 0], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                #         axs[0, 0].set_title('Ant Sheet speed', fontsize=16)
                #         axs[0, 0].axhline(y=cuttoffDiscrepancy, color='blue')
                #         if not is_ECGi:
                #             axs[0, 0].axvline(x=self.conduction_speeds[0], c='grey')
                #         axs[0, 1].plot(ant_good_theta[:, 1], ant_discrepancy[ant_good_particles], 'go')
                #         axs[0, 1].plot(ant_bad_theta[:, 1], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                #         axs[0, 1].set_title('Ant Endocardial speed', fontsize=16)
                #         if not is_ECGi:
                #             axs[0, 1].axvline(x=self.conduction_speeds[1], c='grey')
                #         axs[0, 1].axhline(y=cuttoffDiscrepancy, color='blue')
                #     for axs_i in range(2):
                #         axs[1, axs_i].set_xlabel('cm/s', fontsize=16)
                #     axs[0, 0].set_ylabel('discrepancy', fontsize=16)
                #     axs[1, 0].set_ylabel('discrepancy', fontsize=16)
                #     plt.show()
                # else:
                #     speed_name_list = ['Fibre speed', 'Sheet speed', 'Normal speed']
                #     if has_endocardial_layer:
                #         speed_name_list.append('Endocardial speed')
                #     # Conduction speeds
                #     fig, axs = plt.subplots(2, len(speed_name_list), constrained_layout=True, figsize = (20,10))
                #     fig.suptitle(meshName +' nUnique % ' + str(nUnique/self.npart*100), fontsize=24)
                #     # Prepare previous results
                #     if not ant_part_theta is None:
                #         ant_good_particles = ant_discrepancy < cuttoffDiscrepancy
                #         ant_good_theta = ant_part_theta[ant_good_particles, :]*1000
                #         ant_bad_theta = ant_part_theta[np.logical_not(ant_good_particles), :]*1000
                #     # Iterate over speeds
                #     for speed_iter in range(len(speed_name_list)):
                #         # Plot new results
                #         axs[1, speed_iter].plot(self.part_theta[:worst_keep, speed_iter]*1000, self.part_discrepancy[:worst_keep], 'bo')
                #         axs[1, speed_iter].plot(self.part_theta[worst_keep:, speed_iter]*1000, self.part_discrepancy[worst_keep:], 'go')
                #         axs[1, speed_iter].plot(self.part_theta[randId, speed_iter]*1000, self.part_discrepancy[randId], 'ro')
                #         axs[1, speed_iter].set_title('New ' + speed_name_list[speed_iter], fontsize=16)
                #         if not is_ECGi and not (ant_part_theta is None):
                #             true_speed_value = self.conduction_speeds[speed_iter]
                #             old_median = np.median(ant_part_theta[:, speed_iter]*1000)
                #             new_median = np.median(self.part_theta[:, speed_iter]*1000)
                #             axs[1, speed_iter].axvline(x=self.conduction_speeds[speed_iter], c='grey')
                #             if abs(true_speed_value - new_median) <= abs(true_speed_value - old_median):
                #                 axs[1, speed_iter].axvline(x=new_median, c='green')
                #             else:
                #                 axs[1, speed_iter].axvline(x=new_median, c='red')
                #
                #         # Plot previous results for comparison
                #         if not ant_part_theta is None:
                #             axs[0, speed_iter].plot(ant_good_theta[:, speed_iter], ant_discrepancy[ant_good_particles], 'go')
                #             axs[0, speed_iter].plot(ant_bad_theta[:, speed_iter], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                #             axs[0, speed_iter].set_title('Ant ' + speed_name_list[speed_iter], fontsize=16)
                #             if not is_ECGi:
                #                 axs[0, speed_iter].axvline(x=self.conduction_speeds[speed_iter], c='grey')
                #             axs[0, speed_iter].axvline(x=np.median(ant_part_theta[:, speed_iter]*1000), c='purple')
                #             axs[0, speed_iter].axhline(y=cuttoffDiscrepancy, color='blue')
                #         axs[1, speed_iter].set_xlabel('cm/s', fontsize=16)
                #         # axs[0, 1].plot(ant_good_theta[:, 1], ant_discrepancy[ant_good_particles], 'go')
                #         # axs[0, 1].plot(ant_bad_theta[:, 1], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                #         # axs[0, 1].set_title('Ant Sheet speed', fontsize=16)
                #         # if not is_ECGi:
                #         #     axs[0, 1].axvline(x=self.conduction_speeds[1], c='grey')
                #         #     axs[0, 1].axvline(x=np.median(ant_part_theta[:, 1]*1000), c='purple')
                #         # axs[0, 1].axhline(y=cuttoffDiscrepancy, color='blue')
                #         # axs[0, 2].plot(ant_good_theta[:, 2], ant_discrepancy[ant_good_particles], 'go')
                #         # axs[0, 2].plot(ant_bad_theta[:, 2], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                #         # axs[0, 2].set_title('Ant Normal speed', fontsize=16)
                #         # if not is_ECGi:
                #         #     axs[0, 2].axvline(x=self.conduction_speeds[2], c='grey')
                #         #     axs[0, 2].axvline(x=np.median(ant_part_theta[:, 2]*1000), c='purple')
                #         # if has_endocardial_layer:
                #         #     axs[0, 2].axhline(y=cuttoffDiscrepancy, color='blue')
                #         #     axs[0, 3].plot(ant_good_theta[:, 3], ant_discrepancy[ant_good_particles], 'go')
                #         #     axs[0, 3].plot(ant_bad_theta[:, 3], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                #         #     axs[0, 3].set_title('Ant Endocardial speed', fontsize=16)
                #         #     if not is_ECGi:
                #         #         axs[0, 3].axvline(x=self.conduction_speeds[3], c='grey')
                #         #         axs[0, 3].axvline(x=np.median(ant_part_theta[:, 3]*1000), c='purple')
                #         #     axs[0, 3].axhline(y=cuttoffDiscrepancy, color='blue')
                #
                #     # Plot new results
                #     # Iterate over speeds
                #     #     for speed_iter in range(len(speed_name_list)):
                #     # speed_iter = 0
                #     # axs[1, 0].plot(self.part_theta[:worst_keep, 0]*1000, self.part_discrepancy[:worst_keep], 'bo')
                #     # axs[1, 0].plot(self.part_theta[worst_keep:, 0]*1000, self.part_discrepancy[worst_keep:], 'go')
                #     # axs[1, 0].plot(self.part_theta[randId, 0]*1000, self.part_discrepancy[randId], 'ro')
                #     # axs[1, 0].set_title('New Fibre speed', fontsize=16)
                #     # if not is_ECGi:
                #     #     true_speed_value = self.conduction_speeds[speed_iter]
                #     #     old_median = np.median(self.ant_part_theta[:, speed_iter]*1000)
                #     #     new_median = np.median(self.part_theta[:, speed_iter]*1000)
                #     #     axs[1, 0].axvline(x=self.conduction_speeds[speed_iter], c='grey')
                #     #     if abs(true_speed_value - new_median) <= abs(true_speed_value - old_median):
                #     #         axs[1, 0].axvline(x=new_median, c='red')
                #     #     else:
                #     #         axs[1, 0].axvline(x=new_median, c='green')
                #     # #axs[1, 0].axhline(y=cuttoffDiscrepancy, color='r')
                #     # axs[1, 1].plot(self.part_theta[:worst_keep, 1]*1000, self.part_discrepancy[:worst_keep], 'bo')
                #     # axs[1, 1].plot(self.part_theta[worst_keep:, 1]*1000, self.part_discrepancy[worst_keep:], 'go')
                #     # axs[1, 1].plot(self.part_theta[randId, 1]*1000, self.part_discrepancy[randId], 'ro')
                #     # axs[1, 1].set_title('New Sheet speed', fontsize=16)
                #     # if not is_ECGi:
                #     #     axs[1, 1].axvline(x=self.conduction_speeds[1], c='grey')
                #     #     axs[1, 1].axvline(x=np.median(self.part_theta[:, 1]*1000), c='red')
                #     # #axs[1, 1].axhline(y=cuttoffDiscrepancy, color='r')
                #     # axs[1, 2].plot(self.part_theta[:worst_keep, 2]*1000, self.part_discrepancy[:worst_keep], 'bo')
                #     # axs[1, 2].plot(self.part_theta[worst_keep:, 2]*1000, self.part_discrepancy[worst_keep:], 'go')
                #     # axs[1, 2].plot(self.part_theta[randId, 2]*1000, self.part_discrepancy[randId], 'ro')
                #     # axs[1, 2].set_title('New Normal speed', fontsize=16)
                #     # if not is_ECGi:
                #     #     axs[1, 2].axvline(x=np.median(self.part_theta[:, 2]*1000), c='red')
                #     #     axs[1, 2].axvline(x=self.conduction_speeds[2], c='grey')
                #     # #axs[1, 2].axhline(y=cuttoffDiscrepancy, color='r')
                #     # if has_endocardial_layer:
                #     #     axs[1, 3].plot(self.part_theta[:worst_keep, 3]*1000, self.part_discrepancy[:worst_keep], 'bo')
                #     #     axs[1, 3].plot(self.part_theta[worst_keep:, 3]*1000, self.part_discrepancy[worst_keep:], 'go')
                #     #     axs[1, 3].plot(self.part_theta[randId, 3]*1000, self.part_discrepancy[randId], 'ro')
                #     #     axs[1, 3].set_title('New Endocardial speed', fontsize=16)
                #     #     if not is_ECGi:
                #     #         axs[1, 3].axvline(x=self.conduction_speeds[3], c='grey')
                #     #         axs[1, 3].axvline(x=np.median(self.part_theta[:, 3]*1000), c='red')
                #     axs[0, 0].set_ylabel('discrepancy', fontsize=16)
                #     axs[1, 0].set_ylabel('discrepancy', fontsize=16)
                #     plt.show()
                
            else:
                visualise_count = visualise_count -1
                
            ant_sort = np.argsort(self.part_discrepancy)
            ant_part_theta = self.part_theta[ant_sort, :]
            ant_discrepancy = self.part_discrepancy[ant_sort]
            iteration_count = iteration_count +1
        return self.part_theta
    
        
    def visualise(self, particle_indexes=None, prop_output_bad=None): # TODO: DELETE
        if particle_indexes is not None:
            population_unique, unique_particles_index, unique_particles_reverse_index = np.unique(self.part_theta, return_index=True, return_inverse=True, axis=0) # Non-repeated particles
            # prop_data = self.simulator(population_unique, rootNodeActivationIndexes, rootNodeActivationTimes)  # TODO: UNCOMMENT 2022/05/17
            prop_data = self.simulator(self.part_theta[particle_indexes, :], rootNodeActivationIndexes, rootNodeActivationTimes)# TODO: DELETE 2022/05/17
            
            if self.experiment_output == 'ecg' or self.experiment_output == 'bsp':
                # prop_data = prop_data[unique_particles_reverse_index, :, :] # added on the 2022/05/11 # TODO: UNCOMMENT 2022/05/17
                # Compute the correlation coefficient of the ecg with best discrepancy - 2022/05/11
                # corr_mat = np.zeros((nb_leads))
                # not_nan_size = np.sum(np.logical_not(np.isnan(prop_data[particle_indexes[0], 0, :]))) # TODO: UNCOMMENT 2022/05/17
                not_nan_size = np.sum(np.logical_not(np.isnan(prop_data[0, 0, :]))) # TODO: DELETE 2022/05/17
                lead_i_corr_value_cum = 0.
                fig, axs = plt.subplots(2, nb_leads, constrained_layout=True, figsize = (40,12))
                for i in range(nb_leads):
                    # prediction_lead = prop_data[particle_indexes[0], i, :not_nan_size] # TODO: UNCOMMENT 2022/05/17
                    prediction_lead = prop_data[0, i, :not_nan_size] # TODO: DELETE 2022/05/17
                    if prediction_lead.shape[0] >= self.data.shape[1]:
                        a = prediction_lead
                        b = self.data[i, :]
                    else:
                        a = self.data[i, :]
                        b = prediction_lead
                    b_aux = np.zeros(a.shape)
                    b_aux[:b.shape[0]] = b
                    b_aux[b.shape[0]:] = b[-1]
                    b = b_aux
                    lead_i_corr_value = np.corrcoef(a, b)[0,1]
                    lead_i_corr_value_cum = lead_i_corr_value_cum + lead_i_corr_value
                
                    leadName = leadNames[i]
                    # axs[0, i].plot(self.data[i, :], 'r-', label='Not-pred', linewidth=1.5)  # TODO: UNCOMMENT 2022/05/17
                    # axs[0, i].plot(self.data[i, :], 'b-', label='pred', linewidth=1.5)  # TODO: UNCOMMENT 2022/05/17
                    axs[1, i].plot(prop_data[0, i, :], 'b-', label='best', linewidth=1.5)# TODO: DELETE 2022/05/17
                    axs[1, i].plot(prop_data[1, i, :], 'r-', label='worst', linewidth=1.5)# TODO: DELETE 2022/05/17
                    # axs[1, i].plot(prop_data[particle_indexes[0], i, :], 'b-', label='best', linewidth=1.5) # TODO: UNCOMMENT 2022/05/17
                    # axs[1, i].plot(prop_data[particle_indexes[1], i, :], 'r-', label='worst', linewidth=1.5) # TODO: UNCOMMENT 2022/05/17
                    # if prop_output_bad is not None:  # TODO: UNCOMMENT 2022/05/17
                    #     for j in range(1, prop_output_bad.shape[0]): # TODO: UNCOMMENT 2022/05/17
                    #         axs[0, i].plot(prop_output_bad[j, i, :], 'r-', linewidth=0.5) # TODO: UNCOMMENT 2022/05/17
                    # for j in range(0, prop_data.shape[0]): # TODO: UNCOMMENT 2022/05/17
                    #     axs[0, i].plot(prop_data[j, i, :], 'b-', linewidth=0.08) # TODO: UNCOMMENT 2022/05/17
                    # axs[0, i].plot(self.data[i, :], 'k-', label='true', linewidth=1.5) # TODO: UNCOMMENT 2022/05/17
                    axs[1, i].plot(self.data[i, :], 'k-', label='true', linewidth=1.5)
                    axs[0, i].set_title(leadName + '   ' + str(round(lead_i_corr_value, 2)), fontsize=20)
                    # aux = prop_data[particle_indexes[0], i, :] # TODO: UNCOMMENT 2022/05/17
                    aux = prop_data[0, i, :] # TODO: DELETE 2022/05/17
                    aux = aux[~np.isnan(aux)]
                    dtw_cost, penalty_cost = self.f_discrepancy_plot(aux, self.data[i, :], target_max_amplitude=np.amax(np.abs(self.data)))
                    # aux = prop_data[particle_indexes[1], i, :] # 2022/05/03
                    # aux = aux[~np.isnan(aux)] # 2022/05/03
                    # worst_dtw_cost, worst_penalty_cost = self.f_discrepancy_plot(aux, self.data[i, :], target_max_amplitude=np.amax(np.abs(self.data))) # 2022/05/03
                    # axs[1, i].set_title(str(round(dtw_cost, 2))+'   -   '+str(round(penalty_cost, 2)) +'   /   '+ str(round(worst_dtw_cost, 2))+'   -   '+str(round(worst_penalty_cost, 2)),fontsize=16)
                    axs[1, i].set_title('D: ' + str(round(dtw_cost + penalty_cost, 2))+', CC: '+str(round(lead_i_corr_value, 2)), fontsize=18)
                    # axs[1, i].set_title('D: ' + str(round(dtw_cost, 2))+', P: ' + str(round(penalty_cost, 2))+', CC: '+str(round(lead_i_corr_value, 2)), fontsize=16)
                    # axs[1, i].set_title('D: ' + str(round(dtw_cost, 2))+', P: ' + str(round(penalty_cost, 2))+', WD: ' + str(round(worst_dtw_cost, 2))+', WP: ' + str(round(worst_penalty_cost, 2)), fontsize=16)
                    axs[0, i].set_ylim(bottom=min(np.nanmin(prop_data), np.nanmin((self.data))), top=max(np.nanmax(prop_data), np.nanmax(self.data))) # 2022/05/03
                    # axs[1, i].set_ylim(bottom=min(np.nanmin(prop_data[particle_indexes[0], :, :]), np.nanmin((self.data))), top=max(np.nanmax(prop_data[particle_indexes[0], :, :]),
                    # np.nanmax(self.data))) # 2022/05/03  # TODO: UNCOMMENT 2022/05/17
                    axs[1, i].set_ylim(bottom=min(np.nanmin(prop_data[0, :, :]), np.nanmin((self.data))), top=max(np.nanmax(prop_data[0, :, :]), np.nanmax(self.data))) # TODO: DELETE 2022/05/17
                # axs[0, i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20) # TODO: UNCOMMENT 2022/05/17
                axs[1, i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
                fig.suptitle(meshName + '  Best D.: ' + str(round(self.part_discrepancy[particle_indexes[0]], 2)) + '   Best CC: ' + str(round(lead_i_corr_value_cum / nb_leads, 2)), fontsize=24)
                plt.show()
            else:
                print(meshName +' Activation Time Map Error: ' + str(computeATMerror(prop_data, self.data)[0]))
        return None

# ------------------------------------------- EIKONAL FUNCTIONS ----------------------------------------------------

# Function to insert element
def insertSorted(aList, newV):
    ini_index = 0
    end_index = len(aList)
    index = int((end_index-ini_index)/2)
    for i in range(0, len(aList), 1):
        if newV[1] < aList[index][1]:
            if end_index-ini_index <= 1 or index + 1 == end_index:
                index = index + 1
                break
            else:
                ini_index = index + 1
                index = int(index + (end_index-ini_index)/2 + 1)
        elif newV[1] > aList[index][1]:
            if end_index-ini_index <= 1 or index == ini_index:
                index = index # Place before the current position
                break
            else:
                end_index = index
                index = int(index - (end_index-ini_index)/2)
        else:
            index = ini_index
            break
    aList.insert(index, newV)


@numba.njit()
def eikonal_one_ecg_part1(params, possibleRootNodesIndexes, possibleRootNodesTimes):
    if nlhsParam == 4 or nlhsParam == 2: # TODO delete this # 2022/01/19
        denseEndoSpeed = params[nlhsParam-1] # 2022/01/18
        sparseEndoSpeed = params[nlhsParam-1] # 2022/01/18
    else:
        denseEndoSpeed = params[nlhsParam-1] # 2022/01/18
        sparseEndoSpeed = params[nlhsParam-2] # 2022/01/18
    # print('endoSpeed')
    # print(endoSpeed)
    x = params[nlhsParam:]
    y=np.empty_like(x)
    rootNodesParam = np.round_(x, 0, y)
    y=None
    rootNodesIndexes = possibleRootNodesIndexes[rootNodesParam==1]
    rootNodesTimes = possibleRootNodesTimes[rootNodesParam==1] # 02/12/2021
    
    # Compute the cost of all endocardial edges
    navigationCosts = np.empty((edges.shape[0]))
    #print('2')
    for index in range(edges.shape[0]):
        # Cost for the propagation in the endocardium
        if isEndocardial[index]:
            if isDenseEndocardial[index]:  # Distinguish between two densities of Purkinje network in the endocardium # 2022/01/18
                navigationCosts[index] = math.sqrt(np.dot(edgeVEC[index, :], edgeVEC[index, :])) / denseEndoSpeed # 2022/01/18
            else:
                navigationCosts[index] = math.sqrt(np.dot(edgeVEC[index, :], edgeVEC[index, :])) / sparseEndoSpeed # 2022/01/18
    
    # Current Speed Configuration
    g = np.zeros((3, 3), np.float64) # g matrix
    # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
    if is_healthy: # 07/12/2021
        # np.fill_diagonal(g, [(gf_factor*params[0])**2, params[0]**2, (gn_factor*params[0])**2], wrap=False)# Needs to square each value # 07/12/2021
        np.fill_diagonal(g, [(gf_factor)**2, params[0]**2, (gn_factor)**2], wrap=False)# Needs to square each value # 10/12/2021
    else: # 07/12/2021
        np.fill_diagonal(g, [params[0]**2, params[1]**2, params[2]**2], wrap=False)# Needs to square each value # 07/12/2021
    #print('3')
    # Compute Eikonal edge navigation costs
    for index in range(edges.shape[0]):
        if not isEndocardial[index]:
            # Cost equation for the Eikonal model + Fibrosis at the end
            aux1 = np.dot(g, tetraFibers[index, :, :].T)
            aux2 = np.dot(tetraFibers[index, :, :], aux1)
            aux3 = np.linalg.inv(aux2)
            aux4 = np.dot(edgeVEC[index, :], aux3)
            aux5 = np.dot(aux4, edgeVEC[index:index+1, :].T)
            navigationCosts[index] = np.sqrt(aux5)[0]
    #print('4')
    # Build adjacentcy costs
    adjacentCost = numba.typed.List()
    for i in range(0, nodesXYZ.shape[0], 1):
        not_nan_neighbours = neighbours[i][neighbours[i]!=nan_value]
        adjacentCost.append(np.concatenate((unfoldedEdges[not_nan_neighbours][:, 1:2], np.expand_dims(navigationCosts[not_nan_neighbours%edges.shape[0]], -1)), axis=1))
    return adjacentCost, rootNodesIndexes, rootNodesTimes # 02/12/2021


@numba.njit()
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@numba.njit()
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@numba.njit()
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


# To reproduce previous results with simultaneous activation at the root nodes, use rootNodesTimes = np.ones(...)
def eikonal_ecg(population, rootNodesIndexes, rootNodesTimes):
    if population.shape[1] <= nlhsParam:
        population = np.concatenate((population, np.ones((population.shape[0], rootNodesIndexes.shape[0]))), axis=1)

    population_unique, unique_indexes = np.unique(population, return_inverse=True, axis=0)
    
    # ==========================================
    # Caution! Not working for BSP on 13/01/2021
    # if True:
    #     max_len = 256
    #     prediction_list = np.full((population_unique.shape[0], nb_leads, max_len), np.nan, dtype=np.float64)
    #     prediction_list = np.full((population_unique.shape[0], nodesXYZ.shape[0]), np.nan, dtype=np.float64)
    #     for conf_i in range(prediction_list.shape[0]):
    # gives error about dimension mismatch 2 to 3 or something like that
    # ==========================================
    if experiment_output == 'ecg' or experiment_output == 'bsp':
        max_len = 256
        prediction_list = pymp.shared.array((population_unique.shape[0], nb_leads, max_len), dtype=np.float64)
        prediction_list[:, :, :] = np.nan
    else:
        prediction_list = pymp.shared.array((population_unique.shape[0], nodesXYZ.shape[0]), dtype=np.float64)
        prediction_list[:, :] = np.nan
    # if True: # TODO DELETE THIS
    #     for conf_i in range(prediction_list.shape[0]): # TODO DELETE THIS
    with pymp.Parallel(min(threadsNum, prediction_list.shape[0])) as p1:
        for conf_i in p1.range(prediction_list.shape[0]):
            ## Initialise variables
            activationTimes = np.zeros((nodesXYZ.shape[0],), np.float64)
            visitedNodes = np.zeros((nodesXYZ.shape[0],), dtype=np.bool_)
            # cummCost = 1. # WARNING!! ROOT NODES HAVE A TIME OF 1 ms # Commented on the 24/11/2021
            # Root nodes will be activated at time =   rootNodesTimes
            # cummCost = 0. # We use an offset of zero because now we will have offsets that come from the Purkinje network delays 24/11/2021
            tempTimes = np.zeros((nodesXYZ.shape[0],), np.float64) + 1000
            params = population_unique[conf_i, :]
            adjacentCost, eiknoal_rootNodes, eiknoal_rootActivationTimes = eikonal_one_ecg_part1(params, rootNodesIndexes, rootNodesTimes)
            time_sorting = np.argsort(eiknoal_rootActivationTimes) # 02/12/2021
            # print(eiknoal_rootNodes)
            # print(eiknoal_rootActivationTimes)
            eiknoal_rootNodes = eiknoal_rootNodes[time_sorting] # 02/12/2021
            # print(eiknoal_rootNodes)
            eiknoal_rootActivationTimes = eiknoal_rootActivationTimes[time_sorting] # 02/12/2021
            # print(eiknoal_rootActivationTimes)
            eiknoal_rootActivationTimes = eiknoal_rootActivationTimes - eiknoal_rootActivationTimes[0] + 1. # 02/12/2021 remove the offset from the root nodes - give it a 1 mm of zero activity
            cummCost = eiknoal_rootActivationTimes[0] # 02/12/2021
            # print(cummCost)
            initial_root_nodes_indexes = eiknoal_rootActivationTimes <= cummCost
            # print(initial_root_nodes_indexes)
            # print(eiknoal_rootActivationTimes[initial_root_nodes_indexes])
            # print('hola')
            ## Run the code for the root nodes
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10870] # TODO: 2022 Remove this please - used to test the hypothesis of having delayed root nodes in October 2021
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 13767] # TODO: 2022 Remove this please
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 11512] # TODO: 2022 Remove this please
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10988] # TODO: 2022 Remove this please
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10944] # TODO: 2022 Remove this please
            # visitedNodes[eiknoal_rootNodes] = True # Not simultaneous activation any more; Commented on 24/11/2021
            visitedNodes[eiknoal_rootNodes[initial_root_nodes_indexes]] = True # 02/12/2021 Not simultaneous activation any more
            activationTimes[eiknoal_rootNodes[initial_root_nodes_indexes]] = eiknoal_rootActivationTimes[initial_root_nodes_indexes] # 02/12/2021 Not simultaneous activation any more
            nextNodes = (np.vstack([adjacentCost[eiknoal_rootNodes[rootNode_i]]
                            + np.array([0, eiknoal_rootActivationTimes[rootNode_i]]) for rootNode_i in range(eiknoal_rootNodes.shape[0])])).tolist() # 02/12/2021 Not simultaneous activation any more
            # print(nextNodes)
            # print()
            # print(nextNodes2)
            # visitedNodes2 = np.copy(visitedNodes)
            # tempTimes2 = np.copy(tempTimes)
            # activationTimes2 = np.copy(activationTimes)
            # cummCost2 = cummCost

            # raise()
            
            # on 24/11/2021
            # nextNodes.append(np.array([11512, 15])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([10988, 10])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([10944, 10])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([13767, 25])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([10870, 5])) # TODO: 2022 Remove this please
            
            # VAIG PER AKI!!!! TODO: 01/12/2021
            # SOLUCIO!: TODO: look at the selected root nodes and only consider a root node the one that activates first, then take that time as an offset to correct the QRS onset time
            # TODO: then take the remaining root nodes and push them into the sorted list as if they were just neighbours of the first root node.
            
            
            
            activeNode_i = eiknoal_rootNodes[0]
            sortSecond = lambda x : x[1]
            nextNodes.sort(key=sortSecond, reverse=True)

            while visitedNodes[activeNode_i]:
                nextEdge = nextNodes.pop()
                activeNode_i = int(nextEdge[0])
            cummCost = nextEdge[1]
            if nextNodes: # Check if the list is empty, which can happen while everything being Ok
                tempTimes[(np.array(nextNodes)[:, 0]).astype(np.int32)] = np.array(nextNodes)[:, 1]

            ## Run the whole algorithm
            for i in range(0, nodesXYZ.shape[0]-np.sum(visitedNodes), 1):
                visitedNodes[activeNode_i] = True
                activationTimes[activeNode_i] = cummCost # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
                adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist() # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
                # If I use the actual costs, I only have to do it the first time and then it will just propagate, I will have to use decimals though, so no more uint type arrays.
                for adjacent_i in range(0, len(adjacents), 1):
                    if (not visitedNodes[int(adjacents[adjacent_i][0])]
                    and (tempTimes[int(adjacents[adjacent_i][0])] >
                    adjacents[adjacent_i][1])):
                        insertSorted(nextNodes, adjacents[adjacent_i])
                        tempTimes[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
                while visitedNodes[activeNode_i] and len(nextNodes) > 0:
                    nextEdge = nextNodes.pop()
                    activeNode_i = int(nextEdge[0])
                cummCost = nextEdge[1]

            # Clean Memory
            adjacentCost = None # Clear Mem
            visitedNodes = None # Clear Mem
            tempTimes = None # Clear Mem
            nextNodes = None # Clear Mem
            tempVisited = None # Clear Mem
            navigationCosts = None # Clear Mem

            # activationTimes3 = np.round(activationTimes).astype(np.int32)
            activationTimes = np.round(activationTimes).astype(np.int32)
            # activationTimes = activationTimes2
            # cummCost = cummCost2
            
            # nextNodes = nextNodes2
            # visitedNodes = visitedNodes2
            # tempTimes = tempTimes2
            
            
            
            
            
            # rootNodesTimes = np.around(rootNodesTimes, decimals=4) # 02/12/2021 TODO: use only one decimal
            #
            # ## Initialise variables
            # activationTimes = np.zeros((nodesXYZ.shape[0],), np.float64)
            # visitedNodes = np.zeros((nodesXYZ.shape[0],), dtype=np.bool_)
            # # cummCost = 1. # WARNING!! ROOT NODES HAVE A TIME OF 1 ms # Commented on the 24/11/2021
            # # Root nodes will be activated at time =   rootNodesTimes
            # # cummCost = 0. # We use an offset of zero because now we will have offsets that come from the Purkinje network delays 24/11/2021
            # tempTimes = np.zeros((nodesXYZ.shape[0],), np.float64) + 1000
            # params = population_unique[conf_i, :]
            # adjacentCost, eiknoal_rootNodes, eiknoal_rootActivationTimes = eikonal_one_ecg_part1(params, rootNodesIndexes, rootNodesTimes)
            # time_sorting = np.argsort(eiknoal_rootActivationTimes) # 02/12/2021
            # print(eiknoal_rootNodes)
            # print(eiknoal_rootActivationTimes)
            # eiknoal_rootNodes = eiknoal_rootNodes[time_sorting] # 02/12/2021
            # print(eiknoal_rootNodes)
            # eiknoal_rootActivationTimes = eiknoal_rootActivationTimes[time_sorting] # 02/12/2021
            # print(eiknoal_rootActivationTimes)
            # cummCost = eiknoal_rootActivationTimes[0] # 02/12/2021
            # print(cummCost)
            # initial_root_nodes_indexes = eiknoal_rootActivationTimes <= cummCost
            # print(initial_root_nodes_indexes)
            # print(eiknoal_rootActivationTimes[initial_root_nodes_indexes])
            # print('hola2')
            # ## Run the code for the root nodes
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10870] # TODO: 2022 Remove this please - used to test the hypothesis of having delayed root nodes in October 2021
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 13767] # TODO: 2022 Remove this please
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 11512] # TODO: 2022 Remove this please
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10988] # TODO: 2022 Remove this please
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10944] # TODO: 2022 Remove this please
            # # visitedNodes[eiknoal_rootNodes] = True # Not simultaneous activation any more; Commented on 24/11/2021
            # visitedNodes[eiknoal_rootNodes[initial_root_nodes_indexes]] = True # 02/12/2021 Not simultaneous activation any more
            # activationTimes[eiknoal_rootNodes[initial_root_nodes_indexes]] = eiknoal_rootActivationTimes[initial_root_nodes_indexes] # 02/12/2021 Not simultaneous activation any more
            # nextNodes = (np.vstack([adjacentCost[eiknoal_rootNodes[rootNode_i]]
            #                 + np.array([0, eiknoal_rootActivationTimes[rootNode_i]]) for rootNode_i in range(eiknoal_rootNodes.shape[0])])).tolist() # 02/12/2021 Not simultaneous activation any more
            #
            #
            #
            #
            #
            #
            #
            #
            # activeNode_i = eiknoal_rootNodes[0]
            # sortSecond = lambda x : x[1]
            # nextNodes.sort(key=sortSecond, reverse=True)
            #
            # while visitedNodes[activeNode_i]:
            #     nextEdge = nextNodes.pop()
            #     activeNode_i = int(nextEdge[0])
            # cummCost = nextEdge[1]
            # if nextNodes: # Check if the list is empty, which can happen while everything being Ok
            #     tempTimes[(np.array(nextNodes)[:, 0]).astype(np.int32)] = np.array(nextNodes)[:, 1]
            #
            # # aux_halt_ref_node = 8828
            # # print(aux_halt_ref_node)
            # # print(visitedNodes[aux_halt_ref_node])
            # ## Run the whole algorithm
            # for i in range(0, nodesXYZ.shape[0]-np.sum(visitedNodes), 1):
            #     visitedNodes[activeNode_i] = True
            #     activationTimes[activeNode_i] = cummCost # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
            #     adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist() # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
            #     # If I use the actual costs, I only have to do it the first time and then it will just propagate, I will have to use decimals though, so no more uint type arrays.
            #     for adjacent_i in range(0, len(adjacents), 1):
            #         if (not visitedNodes[int(adjacents[adjacent_i][0])]
            #         and (tempTimes[int(adjacents[adjacent_i][0])] >
            #         adjacents[adjacent_i][1])):
            #             insertSorted(nextNodes, adjacents[adjacent_i])
            #             tempTimes[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
            #     while visitedNodes[activeNode_i] and len(nextNodes) > 0:
            #         nextEdge = nextNodes.pop()
            #         activeNode_i = int(nextEdge[0])
            #     cummCost = nextEdge[1]
            #
            # # print(np.sum(visitedNodes))
            # # print(visitedNodes.shape)
            # # print(np.sum(np.ones(visitedNodes.shape)))
            # # print()
            # # aux_indexes = np.nonzero(visitedNodes == 0)[0]
            # # print(aux_indexes)
            # # print(visitedNodes[aux_indexes])
            # # print(len(adjacentCost))
            # # print(len(neighbours))
            # # print(nodesXYZ.shape)
            # # print(edgeVEC.shape)
            # # # print('s')
            # # # for aux_index in aux_indexes:
            # # #     print(adjacentCost[aux_index])
            # # #     print()
            # # print('m')
            # # aux_ref_node = aux_indexes[0]
            # # print(aux_ref_node)
            # # aux_nei = neighbours[aux_ref_node][neighbours[aux_ref_node]!=nan_value]
            # # print(aux_nei)
            # # aux_edge = unfoldedEdges[aux_nei]
            # # print(aux_edge)
            # # aux_prev_node = aux_edge[:, 1]
            # # print(visitedNodes[aux_prev_node])
            # # print('h')
            # # aux_ref_node = aux_prev_node[0]
            # # print(aux_ref_node)
            # # aux_nei = neighbours[aux_ref_node][neighbours[aux_ref_node]!=nan_value]
            # # print(aux_nei)
            # # aux_edge = unfoldedEdges[aux_nei]
            # # print(aux_edge)
            # # aux_prev_node = aux_edge[:, 1]
            # # print(visitedNodes[aux_prev_node])
            #
            # # Clean Memory
            # adjacentCost = None # Clear Mem
            # visitedNodes = None # Clear Mem
            # tempTimes = None # Clear Mem
            # nextNodes = None # Clear Mem
            # tempVisited = None # Clear Mem
            # navigationCosts = None # Clear Mem
            #
            # activationTimes = np.round(activationTimes).astype(np.int32)
            #
            # atmap = activationTimes - activationTimes3
            # with open('/users/jcamps/repos/Eikonal/Figures_Clinical_PK/DTI003_diff.ensi.ATMap', 'w') as f:
            #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            #     for i in range(0, len(atmap)):
            #         f.write(str(atmap[i]) + '\n')
            #
            # with open('/users/jcamps/repos/Eikonal/Figures_Clinical_PK/DTI003_1.ensi.ATMap', 'w') as f:
            #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            #     for i in range(0, len(activationTimes3)):
            #         f.write(str(activationTimes3[i]) + '\n')
            #
            # with open('/users/jcamps/repos/Eikonal/Figures_Clinical_PK/DTI003_2.ensi.ATMap', 'w') as f:
            #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            #     for i in range(0, len(activationTimes)):
            #         f.write(str(activationTimes[i]) + '\n')
            #
            # raise()
            # print(1)
            # print(params)

            if experiment_output == 'ecg' or experiment_output == 'bsp':
                # Start ECG section ---------------
                nb_timesteps = min(max_len, np.max(activationTimes) + 1) # 1000 Hz is one evaluation every 1 ms
                ECG_aux = np.full((nb_leads, nb_timesteps), np.nan, dtype=np.float64)

                # Calculate voltage per timestep
                Vm = np.zeros((nb_timesteps, nodesXYZ.shape[0])) #+ innactivated_Vm
                for t in range(0, nb_timesteps, 1): # 1000 Hz is one evaluation every 1 ms
                    Vm[t:, activationTimes == t] = 1

                # BSP is  ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                BSP = np.zeros((nb_bsp, nb_timesteps), dtype=np.float64)
                eleContrib = np.zeros((nb_bsp, tetrahedrons.shape[0]), dtype=np.float64)
                for timestep_i in range(1, nb_timesteps, 1):
                    activeNodes = np.nonzero(activationTimes==timestep_i)[0].astype(np.int32)
                    if not len(activeNodes) == 0:
                        activeEle = np.unique(
                            np.concatenate([elements[nodeID] for nodeID in activeNodes]))

                        bMVm = (Vm[timestep_i, tetrahedrons[activeEle, 0:3]]
                            - Vm[timestep_i, tetrahedrons[activeEle, 3]][:, np.newaxis])
                        bd_Vm = np.squeeze(
                        np.matmul(G_pseudo[activeEle, :, :], bMVm[:, :, np.newaxis]), axis=2)
                        eleContrib[:, activeEle] = np.sum(d_r[:, activeEle, :]*bd_Vm, axis=2)
                        BSP[:, timestep_i] = np.sum(eleContrib, axis=1)
                    else:
                        BSP[:, timestep_i] = BSP[:, timestep_i-1]

                # Clear Memory
                activationResults = None
                Vm = None
                eleContrib = None

                # Make 12-lead ECG
                ECG_aux[0, :] = (BSP[0, :] - BSP[1, :])
                ECG_aux[1, :] = (BSP[2, :] - BSP[1, :])
                BSPecg = BSP - np_mean(BSP[0:3, :], axis=0) # 2022/02/07 Fixed willson terminal calculation - was ignoring the third electrode
                BSP = None # Clear Memory
                ECG_aux[2:nb_leads, :] = BSPecg[4:nb_bsp, :]
                ECG_aux = signal.filtfilt(b_filtfilt, a_filtfilt, ECG_aux) # Filter ECG signal
                
                # 2022/05/03 Check if having the R progression (only positive deflection) in the precordials helps the inference (using the same lead as the ground truth)
                # ECG_aux = ECG_aux - ECG_aux[:, 0:1] # align at zero # 2022/05/03 Maybe aligning at zero first and then normalising instead of stardardising
                ECG_aux = ECG_aux - (ECG_aux[:, 0:1]+ECG_aux[:, -2:-1])/2 # align at zero # 2022/05/03
                # ECG_aux[:2, :] = ECG_aux[:2, :] - np_mean(ECG_aux[:2, :], axis=1)[:, np.newaxis] # 2022/05/03 Keep as before for the limb leads - No need after aligning at zero
                # ECG_aux[:2, :] = ECG_aux[:2, :] / np_std(ECG_aux[:2, :], axis=1)[:, np.newaxis] # 2022/05/03 Keep as before for the limb leads
                # if reference_limb_lead_is_max:
                #     ECG_aux[:2, :] = ECG_aux[:2, :] / np.amax(ECG_aux[:2, :][reference_limb_lead_index, :])+0.01 # 2022/05/03 Normalize using the reference lead - avoid division by zero
                # else:
                #     ECG_aux[:2, :] = ECG_aux[:2, :] / abs(np.amin(ECG_aux[:2, :][reference_limb_lead_index, :]))+0.01 # 2022/05/03 Normalize using the reference lead
                # if reference_precordial_lead_is_max:
                #     ECG_aux[2:nb_leads, :] = ECG_aux[2:nb_leads, :] / np.amax(ECG_aux[2:nb_leads, :][reference_precordial_lead_index, :])+0.01 # 2022/05/03 Normalize using the reference lead
                # else:
                #     ECG_aux[2:nb_leads, :] = ECG_aux[2:nb_leads, :] / abs(np.amin(ECG_aux[2:nb_leads, :][reference_precordial_lead_index, :]))+0.01 # 2022/05/03 Normalize using the reference lead
                # ECG_aux = ECG_aux - np_mean(ECG_aux, axis=1)[:, np.newaxis] # 2022/05/03 TODO: Uncomment
                # ECG_aux = ECG_aux / np_std(ECG_aux, axis=1)[:, np.newaxis] # 2022/05/03 TODO: Uncomment
                # ECG_aux = ECG_aux - ECG_aux[:, 0:1] # align at zero # Re-added on 22/05/03 after it was worse without alingment # 2022/05/03 TODO: Uncomment
                
                reference_amplitudes = np.empty(shape=(nb_leads), dtype=np.float64) # 2022/05/04
                reference_amplitudes[reference_lead_is_max] = np.amax(ECG_aux, axis=1)[reference_lead_is_max]
                reference_amplitudes[np.logical_not(reference_lead_is_max)] = np.absolute(np.amin(ECG_aux, axis=1))[np.logical_not(reference_lead_is_max)]
                ECG_aux[:2, :] = ECG_aux[:2, :] / np.mean(reference_amplitudes[:2]) # 22/05/03
                ECG_aux[2:nb_leads, :] = ECG_aux[2:nb_leads, :] / np.mean(reference_amplitudes[2:nb_leads]) # 22/05/03 TODO: Uncomment
            
                
                
                prediction_list[conf_i, :, :ECG_aux.shape[1]] = ECG_aux
            else:
                prediction_list[conf_i, :] = activationTimes
    
    #print(np.sum(np.isnan(prediction_list)))
    return prediction_list[unique_indexes]


# Inclusion of root node distances and on the fly calculation of root node times  # TODO 2022/05/19
def eikonal_ecg_2(population, rootNodesIndexes, rootNodesDistances):  # TODO 2022/05/19
    if population.shape[1] <= nlhsParam:
        population = np.concatenate((population, np.ones((population.shape[0], rootNodesIndexes.shape[0]))), axis=1)

    population_unique, unique_indexes = np.unique(population, return_inverse=True, axis=0)
    pk_speeds = population_unique[:, 0]
    population_unique = population_unique[:, 1:]
    # ==========================================
    # Caution! Not working for BSP on 13/01/2021
    # if True:
    #     max_len = 256
    #     prediction_list = np.full((population_unique.shape[0], nb_leads, max_len), np.nan, dtype=np.float64)
    #     prediction_list = np.full((population_unique.shape[0], nodesXYZ.shape[0]), np.nan, dtype=np.float64)
    #     for conf_i in range(prediction_list.shape[0]):
    # gives error about dimension mismatch 2 to 3 or something like that
    # ==========================================
    if experiment_output == 'ecg' or experiment_output == 'bsp':
        max_len = 256
        prediction_list = pymp.shared.array((population_unique.shape[0], nb_leads, max_len), dtype=np.float64)
        prediction_list[:, :, :] = np.nan
    else:
        prediction_list = pymp.shared.array((population_unique.shape[0], nodesXYZ.shape[0]), dtype=np.float64)
        prediction_list[:, :] = np.nan
    # if True: # TODO DELETE THIS
    #     for conf_i in range(prediction_list.shape[0]): # TODO DELETE THIS
    with pymp.Parallel(min(threadsNum, prediction_list.shape[0])) as p1:
        for conf_i in p1.range(prediction_list.shape[0]):
            ## Initialise variables
            activationTimes = np.zeros((nodesXYZ.shape[0],), np.float64)
            visitedNodes = np.zeros((nodesXYZ.shape[0],), dtype=np.bool_)
            # cummCost = 1. # WARNING!! ROOT NODES HAVE A TIME OF 1 ms # Commented on the 24/11/2021
            # Root nodes will be activated at time =   rootNodesTimes
            # cummCost = 0. # We use an offset of zero because now we will have offsets that come from the Purkinje network delays 24/11/2021
            tempTimes = np.zeros((nodesXYZ.shape[0],), np.float64) + 1000
            params = population_unique[conf_i, :]
            adjacentCost, eiknoal_rootNodes, eiknoal_rootActivationTimes = eikonal_one_ecg_part1(params, rootNodesIndexes, rootNodesDistances/pk_speeds[conf_i]) # TODO 2022/05/19
            time_sorting = np.argsort(eiknoal_rootActivationTimes) # 02/12/2021
            # print(eiknoal_rootNodes)
            # print(eiknoal_rootActivationTimes)
            eiknoal_rootNodes = eiknoal_rootNodes[time_sorting] # 02/12/2021
            # print(eiknoal_rootNodes)
            eiknoal_rootActivationTimes = eiknoal_rootActivationTimes[time_sorting] # 02/12/2021
            # print(eiknoal_rootActivationTimes)
            eiknoal_rootActivationTimes = eiknoal_rootActivationTimes - eiknoal_rootActivationTimes[0] + 1. # 02/12/2021 remove the offset from the root nodes - give it a 1 mm of zero activity
            cummCost = eiknoal_rootActivationTimes[0] # 02/12/2021
            # print(cummCost)
            initial_root_nodes_indexes = eiknoal_rootActivationTimes <= cummCost
            # print(initial_root_nodes_indexes)
            # print(eiknoal_rootActivationTimes[initial_root_nodes_indexes])
            # print('hola')
            ## Run the code for the root nodes
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10870] # TODO: 2022 Remove this please - used to test the hypothesis of having delayed root nodes in October 2021
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 13767] # TODO: 2022 Remove this please
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 11512] # TODO: 2022 Remove this please
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10988] # TODO: 2022 Remove this please
            # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10944] # TODO: 2022 Remove this please
            # visitedNodes[eiknoal_rootNodes] = True # Not simultaneous activation any more; Commented on 24/11/2021
            visitedNodes[eiknoal_rootNodes[initial_root_nodes_indexes]] = True # 02/12/2021 Not simultaneous activation any more
            activationTimes[eiknoal_rootNodes[initial_root_nodes_indexes]] = eiknoal_rootActivationTimes[initial_root_nodes_indexes] # 02/12/2021 Not simultaneous activation any more
            nextNodes = (np.vstack([adjacentCost[eiknoal_rootNodes[rootNode_i]]
                            + np.array([0, eiknoal_rootActivationTimes[rootNode_i]]) for rootNode_i in range(eiknoal_rootNodes.shape[0])])).tolist() # 02/12/2021 Not simultaneous activation any more
            # print(nextNodes)
            # print()
            # print(nextNodes2)
            # visitedNodes2 = np.copy(visitedNodes)
            # tempTimes2 = np.copy(tempTimes)
            # activationTimes2 = np.copy(activationTimes)
            # cummCost2 = cummCost

            # raise()
            
            # on 24/11/2021
            # nextNodes.append(np.array([11512, 15])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([10988, 10])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([10944, 10])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([13767, 25])) # TODO: 2022 Remove this please
            # nextNodes.append(np.array([10870, 5])) # TODO: 2022 Remove this please
            
            # VAIG PER AKI!!!! TODO: 01/12/2021
            # SOLUCIO!: TODO: look at the selected root nodes and only consider a root node the one that activates first, then take that time as an offset to correct the QRS onset time
            # TODO: then take the remaining root nodes and push them into the sorted list as if they were just neighbours of the first root node.
            
            
            
            activeNode_i = eiknoal_rootNodes[0]
            sortSecond = lambda x : x[1]
            nextNodes.sort(key=sortSecond, reverse=True)

            while visitedNodes[activeNode_i]:
                nextEdge = nextNodes.pop()
                activeNode_i = int(nextEdge[0])
            cummCost = nextEdge[1]
            if nextNodes: # Check if the list is empty, which can happen while everything being Ok
                tempTimes[(np.array(nextNodes)[:, 0]).astype(np.int32)] = np.array(nextNodes)[:, 1]

            ## Run the whole algorithm
            for i in range(0, nodesXYZ.shape[0]-np.sum(visitedNodes), 1):
                visitedNodes[activeNode_i] = True
                activationTimes[activeNode_i] = cummCost # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
                adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist() # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
                # If I use the actual costs, I only have to do it the first time and then it will just propagate, I will have to use decimals though, so no more uint type arrays.
                for adjacent_i in range(0, len(adjacents), 1):
                    if (not visitedNodes[int(adjacents[adjacent_i][0])]
                    and (tempTimes[int(adjacents[adjacent_i][0])] >
                    adjacents[adjacent_i][1])):
                        insertSorted(nextNodes, adjacents[adjacent_i])
                        tempTimes[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
                while visitedNodes[activeNode_i] and len(nextNodes) > 0:
                    nextEdge = nextNodes.pop()
                    activeNode_i = int(nextEdge[0])
                cummCost = nextEdge[1]

            # Clean Memory
            adjacentCost = None # Clear Mem
            visitedNodes = None # Clear Mem
            tempTimes = None # Clear Mem
            nextNodes = None # Clear Mem
            tempVisited = None # Clear Mem
            navigationCosts = None # Clear Mem

            # activationTimes3 = np.round(activationTimes).astype(np.int32)
            activationTimes = np.round(activationTimes).astype(np.int32)
            # activationTimes = activationTimes2
            # cummCost = cummCost2
            
            # nextNodes = nextNodes2
            # visitedNodes = visitedNodes2
            # tempTimes = tempTimes2
            
            
            
            
            
            # rootNodesTimes = np.around(rootNodesTimes, decimals=4) # 02/12/2021 TODO: use only one decimal
            #
            # ## Initialise variables
            # activationTimes = np.zeros((nodesXYZ.shape[0],), np.float64)
            # visitedNodes = np.zeros((nodesXYZ.shape[0],), dtype=np.bool_)
            # # cummCost = 1. # WARNING!! ROOT NODES HAVE A TIME OF 1 ms # Commented on the 24/11/2021
            # # Root nodes will be activated at time =   rootNodesTimes
            # # cummCost = 0. # We use an offset of zero because now we will have offsets that come from the Purkinje network delays 24/11/2021
            # tempTimes = np.zeros((nodesXYZ.shape[0],), np.float64) + 1000
            # params = population_unique[conf_i, :]
            # adjacentCost, eiknoal_rootNodes, eiknoal_rootActivationTimes = eikonal_one_ecg_part1(params, rootNodesIndexes, rootNodesTimes)
            # time_sorting = np.argsort(eiknoal_rootActivationTimes) # 02/12/2021
            # print(eiknoal_rootNodes)
            # print(eiknoal_rootActivationTimes)
            # eiknoal_rootNodes = eiknoal_rootNodes[time_sorting] # 02/12/2021
            # print(eiknoal_rootNodes)
            # eiknoal_rootActivationTimes = eiknoal_rootActivationTimes[time_sorting] # 02/12/2021
            # print(eiknoal_rootActivationTimes)
            # cummCost = eiknoal_rootActivationTimes[0] # 02/12/2021
            # print(cummCost)
            # initial_root_nodes_indexes = eiknoal_rootActivationTimes <= cummCost
            # print(initial_root_nodes_indexes)
            # print(eiknoal_rootActivationTimes[initial_root_nodes_indexes])
            # print('hola2')
            # ## Run the code for the root nodes
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10870] # TODO: 2022 Remove this please - used to test the hypothesis of having delayed root nodes in October 2021
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 13767] # TODO: 2022 Remove this please
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 11512] # TODO: 2022 Remove this please
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10988] # TODO: 2022 Remove this please
            # # eiknoal_rootNodes = eiknoal_rootNodes[eiknoal_rootNodes != 10944] # TODO: 2022 Remove this please
            # # visitedNodes[eiknoal_rootNodes] = True # Not simultaneous activation any more; Commented on 24/11/2021
            # visitedNodes[eiknoal_rootNodes[initial_root_nodes_indexes]] = True # 02/12/2021 Not simultaneous activation any more
            # activationTimes[eiknoal_rootNodes[initial_root_nodes_indexes]] = eiknoal_rootActivationTimes[initial_root_nodes_indexes] # 02/12/2021 Not simultaneous activation any more
            # nextNodes = (np.vstack([adjacentCost[eiknoal_rootNodes[rootNode_i]]
            #                 + np.array([0, eiknoal_rootActivationTimes[rootNode_i]]) for rootNode_i in range(eiknoal_rootNodes.shape[0])])).tolist() # 02/12/2021 Not simultaneous activation any more
            #
            #
            #
            #
            #
            #
            #
            #
            # activeNode_i = eiknoal_rootNodes[0]
            # sortSecond = lambda x : x[1]
            # nextNodes.sort(key=sortSecond, reverse=True)
            #
            # while visitedNodes[activeNode_i]:
            #     nextEdge = nextNodes.pop()
            #     activeNode_i = int(nextEdge[0])
            # cummCost = nextEdge[1]
            # if nextNodes: # Check if the list is empty, which can happen while everything being Ok
            #     tempTimes[(np.array(nextNodes)[:, 0]).astype(np.int32)] = np.array(nextNodes)[:, 1]
            #
            # # aux_halt_ref_node = 8828
            # # print(aux_halt_ref_node)
            # # print(visitedNodes[aux_halt_ref_node])
            # ## Run the whole algorithm
            # for i in range(0, nodesXYZ.shape[0]-np.sum(visitedNodes), 1):
            #     visitedNodes[activeNode_i] = True
            #     activationTimes[activeNode_i] = cummCost # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
            #     adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist() # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
            #     # If I use the actual costs, I only have to do it the first time and then it will just propagate, I will have to use decimals though, so no more uint type arrays.
            #     for adjacent_i in range(0, len(adjacents), 1):
            #         if (not visitedNodes[int(adjacents[adjacent_i][0])]
            #         and (tempTimes[int(adjacents[adjacent_i][0])] >
            #         adjacents[adjacent_i][1])):
            #             insertSorted(nextNodes, adjacents[adjacent_i])
            #             tempTimes[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
            #     while visitedNodes[activeNode_i] and len(nextNodes) > 0:
            #         nextEdge = nextNodes.pop()
            #         activeNode_i = int(nextEdge[0])
            #     cummCost = nextEdge[1]
            #
            # # print(np.sum(visitedNodes))
            # # print(visitedNodes.shape)
            # # print(np.sum(np.ones(visitedNodes.shape)))
            # # print()
            # # aux_indexes = np.nonzero(visitedNodes == 0)[0]
            # # print(aux_indexes)
            # # print(visitedNodes[aux_indexes])
            # # print(len(adjacentCost))
            # # print(len(neighbours))
            # # print(nodesXYZ.shape)
            # # print(edgeVEC.shape)
            # # # print('s')
            # # # for aux_index in aux_indexes:
            # # #     print(adjacentCost[aux_index])
            # # #     print()
            # # print('m')
            # # aux_ref_node = aux_indexes[0]
            # # print(aux_ref_node)
            # # aux_nei = neighbours[aux_ref_node][neighbours[aux_ref_node]!=nan_value]
            # # print(aux_nei)
            # # aux_edge = unfoldedEdges[aux_nei]
            # # print(aux_edge)
            # # aux_prev_node = aux_edge[:, 1]
            # # print(visitedNodes[aux_prev_node])
            # # print('h')
            # # aux_ref_node = aux_prev_node[0]
            # # print(aux_ref_node)
            # # aux_nei = neighbours[aux_ref_node][neighbours[aux_ref_node]!=nan_value]
            # # print(aux_nei)
            # # aux_edge = unfoldedEdges[aux_nei]
            # # print(aux_edge)
            # # aux_prev_node = aux_edge[:, 1]
            # # print(visitedNodes[aux_prev_node])
            #
            # # Clean Memory
            # adjacentCost = None # Clear Mem
            # visitedNodes = None # Clear Mem
            # tempTimes = None # Clear Mem
            # nextNodes = None # Clear Mem
            # tempVisited = None # Clear Mem
            # navigationCosts = None # Clear Mem
            #
            # activationTimes = np.round(activationTimes).astype(np.int32)
            #
            # atmap = activationTimes - activationTimes3
            # with open('/users/jcamps/repos/Eikonal/Figures_Clinical_PK/DTI003_diff.ensi.ATMap', 'w') as f:
            #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            #     for i in range(0, len(atmap)):
            #         f.write(str(atmap[i]) + '\n')
            #
            # with open('/users/jcamps/repos/Eikonal/Figures_Clinical_PK/DTI003_1.ensi.ATMap', 'w') as f:
            #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            #     for i in range(0, len(activationTimes3)):
            #         f.write(str(activationTimes3[i]) + '\n')
            #
            # with open('/users/jcamps/repos/Eikonal/Figures_Clinical_PK/DTI003_2.ensi.ATMap', 'w') as f:
            #     f.write('Eikonal Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            #     for i in range(0, len(activationTimes)):
            #         f.write(str(activationTimes[i]) + '\n')
            #
            # raise()
            # print(1)
            # print(params)

            if experiment_output == 'ecg' or experiment_output == 'bsp':
                # Start ECG section ---------------
                nb_timesteps = min(max_len, np.max(activationTimes) + 1) # 1000 Hz is one evaluation every 1 ms
                ECG_aux = np.full((nb_leads, nb_timesteps), np.nan, dtype=np.float64)

                # Calculate voltage per timestep
                Vm = np.zeros((nb_timesteps, nodesXYZ.shape[0])) #+ innactivated_Vm
                for t in range(0, nb_timesteps, 1): # 1000 Hz is one evaluation every 1 ms
                    Vm[t:, activationTimes == t] = 1

                # BSP is  ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                BSP = np.zeros((nb_bsp, nb_timesteps), dtype=np.float64)
                eleContrib = np.zeros((nb_bsp, tetrahedrons.shape[0]), dtype=np.float64)
                for timestep_i in range(1, nb_timesteps, 1):
                    activeNodes = np.nonzero(activationTimes==timestep_i)[0].astype(np.int32)
                    if not len(activeNodes) == 0:
                        activeEle = np.unique(
                            np.concatenate([elements[nodeID] for nodeID in activeNodes]))

                        bMVm = (Vm[timestep_i, tetrahedrons[activeEle, 0:3]]
                            - Vm[timestep_i, tetrahedrons[activeEle, 3]][:, np.newaxis])
                        bd_Vm = np.squeeze(
                        np.matmul(G_pseudo[activeEle, :, :], bMVm[:, :, np.newaxis]), axis=2)
                        eleContrib[:, activeEle] = np.sum(d_r[:, activeEle, :]*bd_Vm, axis=2)
                        BSP[:, timestep_i] = np.sum(eleContrib, axis=1)
                    else:
                        BSP[:, timestep_i] = BSP[:, timestep_i-1]

                # Clear Memory
                activationResults = None
                Vm = None
                eleContrib = None

                # Make 12-lead ECG
                ECG_aux[0, :] = (BSP[0, :] - BSP[1, :])
                ECG_aux[1, :] = (BSP[2, :] - BSP[1, :])
                BSPecg = BSP - np_mean(BSP[0:3, :], axis=0) # 2022/02/07 Fixed willson terminal calculation - was ignoring the third electrode
                BSP = None # Clear Memory
                ECG_aux[2:nb_leads, :] = BSPecg[4:nb_bsp, :]
                ECG_aux = signal.filtfilt(b_filtfilt, a_filtfilt, ECG_aux) # Filter ECG signal
                
                # 2022/05/03 Check if having the R progression (only positive deflection) in the precordials helps the inference (using the same lead as the ground truth)
                # ECG_aux = ECG_aux - ECG_aux[:, 0:1] # align at zero # 2022/05/03 Maybe aligning at zero first and then normalising instead of stardardising
                ECG_aux = ECG_aux - (ECG_aux[:, 0:1]+ECG_aux[:, -2:-1])/2 # align at zero # 2022/05/03
                # ECG_aux[:2, :] = ECG_aux[:2, :] - np_mean(ECG_aux[:2, :], axis=1)[:, np.newaxis] # 2022/05/03 Keep as before for the limb leads - No need after aligning at zero
                # ECG_aux[:2, :] = ECG_aux[:2, :] / np_std(ECG_aux[:2, :], axis=1)[:, np.newaxis] # 2022/05/03 Keep as before for the limb leads
                # if reference_limb_lead_is_max:
                #     ECG_aux[:2, :] = ECG_aux[:2, :] / np.amax(ECG_aux[:2, :][reference_limb_lead_index, :])+0.01 # 2022/05/03 Normalize using the reference lead - avoid division by zero
                # else:
                #     ECG_aux[:2, :] = ECG_aux[:2, :] / abs(np.amin(ECG_aux[:2, :][reference_limb_lead_index, :]))+0.01 # 2022/05/03 Normalize using the reference lead
                # if reference_precordial_lead_is_max:
                #     ECG_aux[2:nb_leads, :] = ECG_aux[2:nb_leads, :] / np.amax(ECG_aux[2:nb_leads, :][reference_precordial_lead_index, :])+0.01 # 2022/05/03 Normalize using the reference lead
                # else:
                #     ECG_aux[2:nb_leads, :] = ECG_aux[2:nb_leads, :] / abs(np.amin(ECG_aux[2:nb_leads, :][reference_precordial_lead_index, :]))+0.01 # 2022/05/03 Normalize using the reference lead
                # ECG_aux = ECG_aux - np_mean(ECG_aux, axis=1)[:, np.newaxis] # 2022/05/03 TODO: Uncomment
                # ECG_aux = ECG_aux / np_std(ECG_aux, axis=1)[:, np.newaxis] # 2022/05/03 TODO: Uncomment
                # ECG_aux = ECG_aux - ECG_aux[:, 0:1] # align at zero # Re-added on 22/05/03 after it was worse without alingment # 2022/05/03 TODO: Uncomment
                
                reference_amplitudes = np.empty(shape=(nb_leads), dtype=np.float64) # 2022/05/04
                reference_amplitudes[reference_lead_is_max] = np.amax(ECG_aux, axis=1)[reference_lead_is_max]
                reference_amplitudes[np.logical_not(reference_lead_is_max)] = np.absolute(np.amin(ECG_aux, axis=1))[np.logical_not(reference_lead_is_max)]
                ECG_aux[:2, :] = ECG_aux[:2, :] / np.mean(reference_amplitudes[:2]) # 22/05/03
                ECG_aux[2:nb_leads, :] = ECG_aux[2:nb_leads, :] / np.mean(reference_amplitudes[2:nb_leads]) # 22/05/03 TODO: Uncomment
            
                
                
                prediction_list[conf_i, :, :ECG_aux.shape[1]] = ECG_aux
            else:
                prediction_list[conf_i, :] = activationTimes
    
    #print(np.sum(np.isnan(prediction_list)))
    return prediction_list[unique_indexes]


# Compute just one activation time map
def eikonal_atm(params, rootNodesIndexes, rootNodesTimes):
    ## Initialise variables
    activationTimes = np.zeros((nodesXYZ.shape[0],), np.float64)
    visitedNodes = np.zeros((nodesXYZ.shape[0],), dtype=np.bool_)
    # cummCost = 1. # WARNING!! ROOT NODES HAVE A TIME OF 1 ms
    tempTimes = np.zeros((nodesXYZ.shape[0],), np.float64) + 1000
    adjacentCost, eiknoal_rootNodes, eiknoal_rootActivationTimes = eikonal_one_ecg_part1(params, rootNodesIndexes, rootNodesTimes)
    time_sorting = np.argsort(eiknoal_rootActivationTimes) # 02/12/2021
    eiknoal_rootNodes = eiknoal_rootNodes[time_sorting] # 02/12/2021
    eiknoal_rootActivationTimes = eiknoal_rootActivationTimes[time_sorting] # 02/12/2021
    eiknoal_rootActivationTimes = eiknoal_rootActivationTimes - eiknoal_rootActivationTimes[0] # 02/12/2021 remove the offset from the root nodes
    cummCost = eiknoal_rootActivationTimes[0] # 02/12/2021
    initial_root_nodes_indexes = eiknoal_rootActivationTimes <= cummCost
    ## Run the code for the root nodes
    visitedNodes[eiknoal_rootNodes[initial_root_nodes_indexes]] = True # 02/12/2021 Not simultaneous activation any more
    activationTimes[eiknoal_rootNodes[initial_root_nodes_indexes]] = eiknoal_rootActivationTimes[initial_root_nodes_indexes] # 02/12/2021 Not simultaneous activation any more
    nextNodes = (np.vstack([adjacentCost[eiknoal_rootNodes[rootNode_i]]
                    + np.array([0, eiknoal_rootActivationTimes[rootNode_i]]) for rootNode_i in range(eiknoal_rootNodes.shape[0])])).tolist() # 02/12/2021 Not simultaneous activation any more

    activeNode_i = eiknoal_rootNodes[0]
    sortSecond = lambda x : x[1]
    nextNodes.sort(key=sortSecond, reverse=True)

    while visitedNodes[activeNode_i]:
        nextEdge = nextNodes.pop()
        activeNode_i = int(nextEdge[0])
    cummCost = nextEdge[1]
    if nextNodes: # Check if the list is empty, which can happen while everything being Ok
        tempTimes[(np.array(nextNodes)[:, 0]).astype(np.int32)] = np.array(nextNodes)[:, 1]

    ## Run the whole algorithm
    for i in range(0, nodesXYZ.shape[0]-np.sum(visitedNodes), 1):
        visitedNodes[activeNode_i] = True
        activationTimes[activeNode_i] = cummCost # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
        adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist() # 28/02/2021 Instead of using cumCost, I could use the actual time cost for each node
        # If I use the actual costs, I only have to do it the first time and then it will just propagate, I will have to use decimals though, so no more uint type arrays.
        for adjacent_i in range(0, len(adjacents), 1):
            if (not visitedNodes[int(adjacents[adjacent_i][0])]
            and (tempTimes[int(adjacents[adjacent_i][0])] >
            adjacents[adjacent_i][1])):
                insertSorted(nextNodes, adjacents[adjacent_i])
                tempTimes[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
        while visitedNodes[activeNode_i] and len(nextNodes) > 0:
            nextEdge = nextNodes.pop()
            activeNode_i = int(nextEdge[0])
        cummCost = nextEdge[1]
    # Clean Memory
    adjacentCost = None # Clear Mem
    visitedNodes = None # Clear Mem
    tempTimes = None # Clear Mem
    nextNodes = None # Clear Mem
    tempVisited = None # Clear Mem
    navigationCosts = None # Clear Mem
    # activationTimes = np.round(activationTimes).astype(np.int32)




    

    # ## Run the code for the root nodes
    # visitedNodes[eiknoal_rootNodes] = True
    # activationTimes[eiknoal_rootNodes] = cummCost
    # nextNodes = (np.vstack([adjacentCost[rootNode] + np.array([0, cummCost]) for rootNode in eiknoal_rootNodes])).tolist()
    
    # activeNode_i = eiknoal_rootNodes[0]
    # sortSecond = lambda x : x[1]
    # nextNodes.sort(key=sortSecond, reverse=True)

    # while visitedNodes[activeNode_i]:
    #     nextEdge = nextNodes.pop()
    #     activeNode_i = int(nextEdge[0])
    # cummCost = nextEdge[1]
    # if nextNodes: # Check if the list is empty, which can happen while everything being Ok
    #     tempTimes[(np.array(nextNodes)[:, 0]).astype(np.int32)] = np.array(nextNodes)[:, 1]

    # ## Run the whole algorithm
    # for i in range(0, len(activationTimes)-(len(eiknoal_rootNodes)), 1):
    #     visitedNodes[activeNode_i] = True
    #     activationTimes[activeNode_i] = cummCost
    #     adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist()
    #     for adjacent_i in range(0, len(adjacents), 1):
    #         if (not visitedNodes[int(adjacents[adjacent_i][0])]
    #         and (tempTimes[int(adjacents[adjacent_i][0])] >
    #         adjacents[adjacent_i][1])):
    #             insertSorted(nextNodes, adjacents[adjacent_i])
    #             tempTimes[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
    #     while visitedNodes[activeNode_i] and len(nextNodes) > 0:
    #         nextEdge = nextNodes.pop()
    #         activeNode_i = int(nextEdge[0])
    #     cummCost = nextEdge[1]
    #
    # # Clean Memory
    # adjacentCost = None # Clear Mem
    # visitedNodes = None # Clear Mem
    # tempTimes = None # Clear Mem
    # nextNodes = None # Clear Mem
    # tempVisited = None # Clear Mem
    # navigationCosts = None # Clear Mem
    return np.round(activationTimes).astype(np.int32)

# Use the 12 lead ECG + Body surface potentials, or at least the limb leads, this way I can use the unipolar calculation as in the precordial leads for any body surface potential from the ECGi electrodes
def pseudoBSP(ATMap):
    ATMap = ATMap.astype(int) # prevent non-integer activation maps, they need to be in ms 03/02/2021
    nbEle = tetrahedrons.shape[0]
    nbEleSplit = int(math.ceil(nbEle / 4))
    nb_timesteps = int(np.max(ATMap) + 1)  # 1000 Hz is one evaluation every 1 ms

    # Calculate voltage per timestep
    Vm = np.zeros((nb_timesteps, nodesXYZ.shape[0]))  # + innactivated_Vm
    for t in range(0, nb_timesteps, 1):  # 1000 Hz is one evaluation every 1 ms
        Vm[t:, ATMap == t] = 1  # activated_Vm

    ECG = np.zeros((nb_leads, nb_timesteps), dtype='float64')

    BSP = np.zeros((nb_bsp, nb_timesteps), dtype='float64')
    eleContrib = np.zeros((nb_bsp, tetrahedrons.shape[0]))
    for timestep_i in range(1, nb_timesteps, 1):
        activeNodes = (np.nonzero(ATMap == timestep_i)[0]).astype(int)
        if not len(activeNodes) == 0:
            activeEle = np.unique(
                np.concatenate([elements[nodeID] for nodeID in activeNodes]))
            bMVm = (Vm[timestep_i, tetrahedrons[activeEle, 0:3]]
                    - Vm[timestep_i, tetrahedrons[activeEle, 3]][:, np.newaxis])
            bd_Vm = np.squeeze(
                np.matmul(G_pseudo[activeEle, :, :], bMVm[:, :, np.newaxis]), axis=2)
            eleContrib[:, activeEle] = np.sum(d_r[:, activeEle, :] * bd_Vm, axis=2)
            BSP[:, timestep_i] = np.sum(eleContrib, axis=1)
        else:
            BSP[:, timestep_i] = BSP[:, timestep_i - 1]
    eleContrib = None
    Vm = None  # Clear Memory

    # BSP is  ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    ECG[0, :] = (BSP[0, :] - BSP[1, :])
    ECG[1, :] = (BSP[2, :] - BSP[1, :])
    BSPecg = BSP - np.mean(BSP[0:2, :], axis=0)
    BSP = None  # Clear Memory
    ECG[2:nb_leads, :] = BSPecg[4:nb_bsp, :]

    ECG = signal.filtfilt(b_filtfilt, a_filtfilt, ECG)

    ECG = (ECG - np.mean(ECG, axis=1)[:, np.newaxis])
    ECG = (ECG / np.std(ECG, axis=1)[:, np.newaxis])
    ECG = ECG - ECG[:, 0][:, np.newaxis]  # align at zero

    return ECG
    

def non_standard_pseudoBSP(ATMap):
    ATMap = ATMap.astype(int) # prevent non-integer activation maps, they need to be in ms 03/02/2021
    nbEle = tetrahedrons.shape[0]
    nbEleSplit = int(math.ceil(nbEle / 4))
    nb_timesteps = int(np.max(ATMap) + 1)  # 1000 Hz is one evaluation every 1 ms

    # Calculate voltage per timestep
    Vm = np.zeros((nb_timesteps, nodesXYZ.shape[0]))  # + innactivated_Vm
    for t in range(0, nb_timesteps, 1):  # 1000 Hz is one evaluation every 1 ms
        Vm[t:, ATMap == t] = 1  # activated_Vm

    ECG = np.zeros((nb_leads, nb_timesteps), dtype='float64')

    BSP = np.zeros((nb_bsp, nb_timesteps), dtype='float64')
    eleContrib = np.zeros((nb_bsp, tetrahedrons.shape[0]))
    for timestep_i in range(1, nb_timesteps, 1):
        activeNodes = (np.nonzero(ATMap == timestep_i)[0]).astype(int)
        if not len(activeNodes) == 0:
            activeEle = np.unique(
                np.concatenate([elements[nodeID] for nodeID in activeNodes]))
            bMVm = (Vm[timestep_i, tetrahedrons[activeEle, 0:3]]
                    - Vm[timestep_i, tetrahedrons[activeEle, 3]][:, np.newaxis])
            bd_Vm = np.squeeze(
                np.matmul(G_pseudo[activeEle, :, :], bMVm[:, :, np.newaxis]), axis=2)
            eleContrib[:, activeEle] = np.sum(d_r[:, activeEle, :] * bd_Vm, axis=2)
            BSP[:, timestep_i] = np.sum(eleContrib, axis=1)
        else:
            BSP[:, timestep_i] = BSP[:, timestep_i - 1]
    eleContrib = None
    Vm = None  # Clear Memory

    # BSP is  ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    ECG[0, :] = (BSP[0, :] - BSP[1, :])
    ECG[1, :] = (BSP[2, :] - BSP[1, :])
    BSPecg = BSP - np.mean(BSP[0:2, :], axis=0)
    BSP = None  # Clear Memory
    ECG[2:nb_leads, :] = BSPecg[4:nb_bsp, :]

    ECG = signal.filtfilt(b_filtfilt, a_filtfilt, ECG) # remove coarse artefacts

    ECG = ECG - ECG[:, 0][:, np.newaxis]  # align at zero
    ECG = (ECG / np.amax(np.absolute(ECG))) # normalise to make maximum = 1 collectively for all leads

    return ECG


# ------------------------------------------- RESULTS ANALYSIS FUNCTIONS ----------------------------------------------------

def djikstra(source_id_list, djikstra_nodesXYZ, djikstra_unfoldedEdges, djikstra_edgeVEC, djikstra_neighbours, max_path_len=100):
    distances = np.zeros((djikstra_nodesXYZ.shape[0], source_id_list.shape[0])).astype(float)
    paths = np.full((djikstra_nodesXYZ.shape[0], source_id_list.shape[0], max_path_len), np.nan, np.int32)
    for source_id_index in range(source_id_list.shape[0]):
        source_id = source_id_list[source_id_index]
        distances_per_source = np.zeros((djikstra_nodesXYZ.shape[0]))
        previous_node_indexes_temp = np.zeros((djikstra_nodesXYZ.shape[0])).astype(int)  # 2022/01/10 This object starts with a wrong solution and it makes it better and better until in the end the
        # solution is the correct one. Idea by me and Jenny :-)
        visitedNodes = np.zeros((djikstra_nodesXYZ.shape[0])).astype(bool)

        # Compute the cost of all endocardial edges
        navigationCosts = np.zeros((int(djikstra_unfoldedEdges.shape[0]/2)))
        for index in range(0, navigationCosts.shape[0]):
            # Cost for the propagation in the endocardium
            navigationCosts[index] = math.sqrt(np.dot(djikstra_edgeVEC[index, :], djikstra_edgeVEC[index, :]))

        # Build adjacentcy costs
        adjacentCost = [np.concatenate((djikstra_unfoldedEdges[djikstra_neighbours[i]][:, 1][:,np.newaxis], navigationCosts[djikstra_neighbours[i]%navigationCosts.shape[0]][:, np.newaxis]), axis=1) for i in range(0, djikstra_nodesXYZ.shape[0], 1)]
        
        cummCost = 0. # Distance from a node to itself is zero
        tempDists = np.zeros((djikstra_nodesXYZ.shape[0],), float) + 1000

        ## Run the code for the root nodes
        visitedNodes[source_id] = True
        distances_per_source[source_id] = cummCost
        nextNodes = (adjacentCost[source_id] + np.array([0, cummCost])).tolist()
        activeNode_i = source_id
        sortSecond = lambda x : x[1]
        nextNodes.sort(key=sortSecond, reverse=True)
        previous_node_indexes_temp[activeNode_i] = activeNode_i  # 2022/01/10
        for nextEdge_aux in nextNodes: # 2022/01/10
            previous_node_indexes_temp[int(nextEdge_aux[0])] = activeNode_i  # 2022/01/10
        while visitedNodes[activeNode_i] and len(nextNodes) > 0:
            nextEdge = nextNodes.pop()
            activeNode_i = int(nextEdge[0])
        cummCost = nextEdge[1]
        if nextNodes: # Check if the list is empty, which can happen while everything being Ok
            tempDists[(np.array(nextNodes)[:, 0]).astype(int)] = np.array(nextNodes)[:, 1]
            
        ## Run the whole algorithm
        for i in range(distances_per_source.shape[0]):
            visitedNodes[activeNode_i] = True
            distances_per_source[activeNode_i] = cummCost
            adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist()
            for adjacent_i in range(0, len(adjacents), 1):
                if (not visitedNodes[int(adjacents[adjacent_i][0])] and (tempDists[int(adjacents[adjacent_i][0])] > adjacents[adjacent_i][1])):
                    insertSorted(nextNodes, adjacents[adjacent_i])
                    tempDists[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
                    previous_node_indexes_temp[int(adjacents[adjacent_i][0])] = activeNode_i  # 2022/01/10
            while visitedNodes[activeNode_i] and len(nextNodes) > 0:
                nextEdge = nextNodes.pop()
                activeNode_i = int(nextEdge[0])
            cummCost = nextEdge[1]
        
        distances[:, source_id_index] = distances_per_source
        for djikstra_node_id in range(0, djikstra_nodesXYZ.shape[0], 1):  # 2022/01/10
            path_per_source = np.full((djikstra_nodesXYZ.shape[0]), np.nan, np.int32)  # 2022/01/10
            path_node_id = djikstra_node_id  # 2022/01/10
            path_node_id_iter = 0  # 2022/01/10
            path_per_source[path_node_id_iter] = path_node_id  # 2022/01/14
            path_node_id_iter = path_node_id_iter + 1  # 2022/01/14
            while path_node_id != source_id:  # 2022/01/10
                path_node_id = previous_node_indexes_temp[path_node_id]  # 2022/01/10
                path_per_source[path_node_id_iter] = path_node_id  # 2022/01/10
                path_node_id_iter = path_node_id_iter + 1  # 2022/01/10
            # If the path is longer than the current size of the matrix, make the matrix a little bigger and continue
            if path_node_id_iter + 1 > max_path_len:  # 2022/01/11
                paths_aux = np.full((djikstra_nodesXYZ.shape[0], source_id_list.shape[0], path_node_id_iter + 10), np.nan, np.int32)  # 2022/01/11
                paths_aux[:, :, :max_path_len] = paths  # 2022/01/11
                paths = paths_aux  # 2022/01/11
                max_path_len = path_node_id_iter + 10  # 2022/01/11
            paths[djikstra_node_id, source_id_index, :] = path_per_source[:max_path_len]  # 2022/01/10
    return distances, paths  # 2022/01/10


def k_means(data, data_ids, k, centroid_ids, distance_mat, endo_nodes, max_iter=100):

    # labels : array containing labels for data points, randomly initialized
    labels = np.random.randint(low=0, high=k, size=data_ids.shape[0])

    # k-means algorithm
    for i in range(max_iter):
        # distances : between datapoints and centroids
        distances = np.array([distance_mat[c_id, data_ids] for c_id in centroid_ids])
        # new_labels : computed by finding centroid with minimal distance
        new_labels = np.argmin(distances, axis=0)
        
        if np.all(labels == new_labels):
            # labels unchanged
            labels = new_labels
#             print('Labels unchanged ! Terminating k-means.')
            break
        else:
            # labels changed
            # difference : percentage of changed labels
            # difference = np.mean(labels != new_labels)
#             print('%4f%% labels changed' % (difference * 100))
            labels = new_labels
            for c in range(k):
                # computing centroids by taking the mean over associated data points
                new_centroid = np.mean(data[data_ids, :][labels == c, :], axis=0)
                
                # Project centroids back to the endocardium
                ind = np.argmin(np.linalg.norm(endo_nodes - new_centroid, ord=2, axis=1)).astype(int)
#                 print('diff ' + str(np.round(np.linalg.norm(endo_nodes[ind, :] - new_centroid, ord=2, axis=0)*10, 2)) + ' mm')
                centroid_ids[c] = ind
    return labels, centroid_ids


def trianglorgram(n_timestamps_1, n_timestamps_2, max_slope=2.):
    """Compute the dtw trianglorgram.
    Parameters
    ----------
    n_timestamps_1 : int
        The size of the first time series.
    n_timestamps_2 : int (optional, default None)
        The size of the second time series. If None, set to `n_timestamps_1`.
    max_slope : float (default = 2.)
        Maximum slope for the parallelogram. Must be >= 1.
    Returns
    -------
    region : array, shape = (2, n_timestamps_1)
        Constraint region. The first row consists of the starting indices
        (included) and the second row consists of the ending indices (excluded)
        of the valid rows for each column.
    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).
    Examples
    --------
    >>> from pyts.metrics import itakura_parallelogram
    >>> print(itakura_parallelogram(5))
    [[0 1 1 2 4]
     [1 3 4 4 5]]
    """
    
    # Compute the slopes of the parallelogram bounds
    max_slope_ = max_slope
    min_slope_ = 1 / max_slope_
    scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2) # ORIGINAL
    max_slope_ *= scale_max
    # We take out this line because we want to consider values around the new diagonal, rather than the true diagonal
#     max_slope_ = max(1., max_slope_) # ORIGINAL, this would include the true diagonal if not included already

    scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1) # ORIGINAL
    min_slope_ *= scale_min
    # We take out this line because we want to consider values around the new diagonal, rather than the true diagonal
#     min_slope_ = min(1., min_slope_) # ORIGINAL, this would include the true diagonal if not included already
    
    # Little fix for max_slope = 1
    if max_slope == 1:
        # Now we create the piecewise linear functions defining the parallelogram
        centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
        lower_bound = np.empty((2, n_timestamps_1))
        lower_bound[0] = min_slope_ * np.arange(n_timestamps_1)
        lower_bound[1] = max_slope_ * centered_scale + n_timestamps_2 - 1

        # take the max of the lower linear funcs
        lower_bound = np.round(lower_bound, 2)
        lower_bound = np.ceil(np.max(lower_bound, axis=0))

        upper_bound = np.empty((2, n_timestamps_1))
        upper_bound[0] = max_slope_ * np.arange(n_timestamps_1) + 1
        upper_bound[1] = min_slope_ * centered_scale + n_timestamps_2

        # take the min of the upper linear funcs
        upper_bound = np.round(upper_bound, 2)
        upper_bound = np.floor(np.min(upper_bound, axis=0))
        
        # This part makes that sometimes dtw(ecg1, ecg2) != dtw(ecg2, ecg1) for ecgs from DTI003, thus should be revised in the future
        if n_timestamps_2 > n_timestamps_1:
            upper_bound[:-1] = lower_bound[1:]
        else:
            upper_bound = lower_bound + 1
    else:
        # Now we create the piecewise linear functions defining the parallelogram
        centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
        lower_bound = min_slope_ * np.arange(n_timestamps_1)

        # take the max of the lower linear funcs
        lower_bound = np.round(lower_bound, 2)
#         lower_bound = np.ceil(lower_bound) # ORIGINAL
        lower_bound = np.floor(lower_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path

        upper_bound = max_slope_ * np.arange(n_timestamps_1) + 1

        # take the min of the upper linear funcs
        upper_bound = np.round(upper_bound, 2)
#         upper_bound = np.floor(upper_bound) # ORIGINAL
        upper_bound = np.ceil(upper_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path

    region = np.asarray([lower_bound, upper_bound]).astype('int64')
    region = np.clip(region[:, :n_timestamps_1], 0, n_timestamps_2) # Project region on the feasible set
    return region


def dtw_trianglorgram(x, y, max_slope=1.5, w_max=10., target_max_amplitude=1.):
    """25/02/2021: I have realised that this method is equivalent to the vanila paralelogram in practice but less
    computationally efficient. I initially thought that using a parallelogram implied that all warping was to be undone
    towards the end of the signal, like comparing people reading the same text on the same amount of time but with a variability
    on the speed for each part of the sentance. However, that was not the case. The parallelogram serves the same purpose
    as the trianlogram when the constraint of equal ending is put in place, namely, only (N-1, M-1) is evaluated. This
    constraint forces both signals to represent the same information rather than one signal being only half of the other.
    Therefore, by using the trianglogram plus the restriction, the only thing I am achieving is to do more calculations
    strictly necessary. However, the original implementation of the parallelogram allows an amount of warping proportional
    to the length difference between the two signals, which could lead to non-physiological warping. Here instead, the
    maximum amount of warping is defined in w_max and max_slope; this feature is key because the discrepancy calculation
    needs to be equivalent for all signals throughout the method regardless of their length. TODO: change method to be the parallelogram to make it more efficient."""
    """Dynamic Time Warping distance specific for comparing electrocardiogram signals.
    It implements a trianglogram constraint (inspired from Itakura parallelogram).
    It also implements weight penalty with a linear increasing cost away from the true diagonal (i.e. i=j).
    Moreover, it implements a step-pattern with slope-P value = 0.5 from (Sakoe and Chiba, 1978).
    Finally, the algorithm penalises the difference between the lenght of the two signals and adds it to the DTW distance.
    Options
    -------
    max_slope : float Maximum slope of the trianglogram.
    w_max : float weigth coeficient to the distance from the diagonal.
    small_c :  float weight coeficient to the difference in lenght between the signals being compared.
    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).
    .. [1] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).
    """

    n_timestamps_1 = x.shape[0]
    n_timestamps_2 = y.shape[0]

    small_c = 0.05 * 171 / meshVolume # Scaling factor to account for differences in mesh size # 2022/05/04

    #print('from code in dtw_trianglogram(...). Sizes:')
    #print(n_timestamps_1)
    #print(n_timestamps_2)
    
    # Computes the region (in-window area using a trianglogram)
    region = trianglorgram(n_timestamps_1, n_timestamps_2, max_slope)
    
    # Computes cost matrix from dtw input
    dist_ = lambda x, y : (x - y) ** 2

    region = check_array(region, dtype='int64')
    region_shape = region.shape
    if region_shape != (2, x.size):
        raise ValueError(
            "The shape of 'region' must be equal to (2, n_timestamps_1) "
            "(got {0}).".format(region_shape)
        )

    # Computs the cost matrix considering the window (0 inside, np.inf outside)
    cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
    m = np.amax(cost_mat.shape)
    for i in numba.prange(n_timestamps_1):
        for j in numba.prange(region[0, i], region[1, i]):
            cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.) # This new weight considers that wraping in time is cheaper the later it's done #* (w_max/(1+math.exp(-g*(abs(i-j)-m/2)))+1.) # + abs(i-j)*small_c # Weighted version of the DTW algorithm

    cost_mat = check_array(cost_mat, ensure_min_samples=2,
                           ensure_min_features=2, ensure_2d=True,
                           force_all_finite=False, dtype='float64')
    
    # Computes the accumulated cost matrix
    acc_cost_mat = np.ones((n_timestamps_1, n_timestamps_2)) * np.inf
    acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
        cost_mat[0, 0: region[1, 0]]
    )
    acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
        cost_mat[0: region[1, 0], 0]
    )
    region_ = np.copy(region)

    region_[0] = np.maximum(region_[0], 1)
    ant_acc_min_i = -1
    acc_count = 0
    for i in range(1, n_timestamps_1):
        for j in range(region_[0, i], region_[1, i]):
            # Implementation of a Slope-constraint as a step-pattern:
            # This constraint will enforce that the algorithm can only take up to 2 consecutive steps along the time wraping directions.
            # I decided to make it symetric because in (Sakoe and Chiba, 1978) they state that symetric means that DTW(A, B) == DTW(B, A), although I am not convinced why it's not the case in the asymetric implementation.
            # Besides, the asymetric case has a bias towards the diagonal which I thought could be desirable in our case, that said, having DTW(A, B) == DTW(B, A) may prove even more important, especially in further
            # applications of this algorithm for ECG comparison.
            # This implementation is further explained in (Sakoe and Chiba, 1978) and correspondes to the one with P = 0.5, (P = n/m, where P is a rule being inforced, n is the number of steps in the diagonal
            # direction and m is the steps in the time wraping direction).
            acc_cost_mat[i, j] = min(
                acc_cost_mat[i - 1, j-3] + 2*cost_mat[i, j-2] + cost_mat[i, j-1] + cost_mat[i, j],
                acc_cost_mat[i - 1, j-2] + 2*cost_mat[i, j-1] + cost_mat[i, j],
                acc_cost_mat[i - 1, j - 1] + 2*cost_mat[i, j],
                acc_cost_mat[i - 2, j-1] + 2*cost_mat[i-1, j] + cost_mat[i, j],
                acc_cost_mat[i - 3, j-1] + 2*cost_mat[i-2, j] + cost_mat[i-1, j] + cost_mat[i, j]
            )
    
    dtw_dist = acc_cost_mat[-1, -1]/(n_timestamps_1 + n_timestamps_2) # Normalisation M+N according to (Sakoe and Chiba, 1978)
    
    dtw_dist = dtw_dist / np.amax(np.abs(y)) * target_max_amplitude # 2022/05/04 - Normalise by lead amplitude to weight all leads similarly
    dtw_dist = math.sqrt(dtw_dist)
    
    # Penalty for ECG-width differences
    ecg_width_cost = small_c * (n_timestamps_1-n_timestamps_2)**2 / min(n_timestamps_1,n_timestamps_2)
    
#     return (dtw_dist+ecg_width_cost, dtw_dist, cost_mat/(n_timestamps_1+n_timestamps_2), acc_cost_mat/(n_timestamps_1+n_timestamps_2), path)
    return dtw_dist, ecg_width_cost


# ------------------------------------------- MAIN FUNCTIONS ----------------------------------------------------

def main_synthetic(args): # preserved as it was in 2021/05/22 for reproducibility of the the MIA paper
    import multiprocessing
    
    run_iter = int(args[0])
    print(run_iter)
    
    target_type = args[1]
    if target_type == 'bsp':
        target_type = 'ecg'
        rootNodeResolution = "newLow"
        target_snr_db = 20
        # file_path_tag = target_type+'Noise_new_code_' + rootNodeResolution # September 2021
        file_path_tag = target_type+'Fixed_2021_' + rootNodeResolution
    elif target_type == 'ecg':
        target_type = 'ecg'
        rootNodeResolution = "newRV"
        # target_snr_db = 0 # September 2021
        target_snr_db = 20 # September 2021
        # file_path_tag = target_type+'Noise_new_code_' + rootNodeResolution # September 2021
        file_path_tag = target_type+'Fixed_2021_' + rootNodeResolution
    else:
        raise
    metric = "dtw"
    
    print(file_path_tag)
    load_target = False  # Needs to be false because these are Eikonal-generated
    
    consistency_count = 3
    meshName_list = ["DTI003", "DTI001", "DTI024", "DTI004"]
    conduction_speeds_list = [
                [50, 32, 29, 150],
                [50, 32, 29, 120],
                [50, 32, 29, 179],
                [88, 49, 45, 179],
                [88, 49, 45, 120]
    ]
    
    healthy = False  # When true will ommit estimating the fibre and sheet-normal speeds and define them as proportional to the sheet speed
    endocardial_layer = True  # Set a fast isotropic endocardial layer
    npart = 512  # 384
    if target_type == 'bsp':
        threadsNum = 16
    else:
        threadsNum = multiprocessing.cpu_count()
    keep_fraction =  (npart - threadsNum) / npart
    
    run_count = 0
    conduction_speeds = None
    consistency_iter = None
    for consistency in range(consistency_count):
        for conduction_speeds_aux in conduction_speeds_list:
            for meshName_val_aux in meshName_list:
                if run_count == run_iter:
                    conduction_speeds = conduction_speeds_aux
                    consistency_iter = consistency
                    meshName_val = meshName_val_aux
                run_count = run_count + 1
    if conduction_speeds is not None:
        # Volumes of the meshes
        if meshName_val == "DTI001":
            meshVolume_val = 74
        elif meshName_val == "DTI024":
            meshVolume_val = 76
        elif meshName_val == "DTI004":
            meshVolume_val = 107
        elif meshName_val == "DTI003":
            meshVolume_val = 171
        elif meshName_val == "DTI032":
            meshVolume_val = 139
        
        finalResultsPath = 'metaData/Eikonal_Results/'+file_path_tag+'/'
        tmpResultsPath = 'metaData/Results/'+file_path_tag+'/'
        
        population_fileName = (meshName_val + '_' + rootNodeResolution + '_' + str(conduction_speeds) + '_'
                    + rootNodeResolution + '_' + target_type + '_' + metric + '_' +
                     str(consistency_iter) + '_population.csv')
        if (not os.path.isfile(finalResultsPath + population_fileName)
            and (not os.path.isfile(tmpResultsPath + population_fileName))):
            target_fileName = None
            if load_target:
                if target_type == 'atm':
                    data_type_tag = 'ATMap'
                else:
                    data_type_tag = target_type
                if 50 in conduction_speeds:
                    speeds_tag = str(conduction_speeds[3]) + '_1x'
                else:
                    speeds_tag = str(conduction_speeds[3]) + '_2x'
                target_fileName = ('metaData/ATMaps/' + meshName_val + '_coarse_true_'+data_type_tag+'_'
                                            + speeds_tag + '.csv')
            if (not load_target) or os.path.isfile(target_fileName):
                f = open(tmpResultsPath + population_fileName, "w")
                f.write("Experiment in progress")
                f.close()
                print(population_fileName)
                run_inference_2021(
                    meshName_val=meshName_val,
                    meshVolume_val=meshVolume_val,
                    conduction_speeds=conduction_speeds,
                    final_path=finalResultsPath + population_fileName,
                    tmp_path=tmpResultsPath + population_fileName,
                    threadsNum_val=threadsNum,
                    target_type=target_type,
                    metric=metric,
                    npart=npart,
                    keep_fraction=keep_fraction,
                    rootNodeResolution=rootNodeResolution,
                    target_snr_db=target_snr_db,
                    healthy_val=healthy,
                    load_target=load_target,
                    endocardial_layer=endocardial_layer,
                    target_fileName=target_fileName,
                    is_ECGi_val=False
                )
            else:
                print('No target file: ' +target_fileName)
        else:
            print('Done by someone else')
    else:
        print('Nothing to be done')
        
        
def main_clinical(args): # added on the 2021/05/21 to run inference on clinical data
    import multiprocessing
    
    run_iter = int(args[0])
    print(run_iter)
    
    target_type = args[1]
    if target_type == 'ecg':
        rootNodeResolution = "newRV"
        target_snr_db = 0
        # file_path_tag = 'Clinical_Fixed_2021_' + target_type + '_' + rootNodeResolution
        file_path_tag = 'Clinical_PK_2021_' + target_type + '_' + rootNodeResolution
    else:
        raise
    metric = "dtw"
    
    print(file_path_tag)
    load_target = True  # Needs to be false because these are Eikonal-generated
    
    # consistency_count = 22
    consistency_count = 3 # 100 # 07/12/2021 # 2022/05/04
    # meshName_list = ["DTI003", "DTI032", "DTI024", "DTI004", "DTI124_1_coarse", "DTI4586_1_coarse", "DTI4586_2_coarse"]
    # meshName_list = ["DTI003"]#, "DTI032", "DTI024", "DTI004", "DTI124_1_coarse", "DTI4586_1_coarse", "DTI4586_2_coarse"]
    # meshName_list = ["DTI004", "DTI003", "DTI124_1_coarse", "DTI4586_2_coarse"]
    meshName_list = ["DTI024", "DTI032"]
    # meshName_list = ["DTI004"]
    # healthy = False  # When true will ommit estimating the fibre and sheet-normal speeds and define them as proportional to the sheet speed
    # healthy = 59 < run_iter < 70 # 08/12/2021
    # healthy = 80 < run_iter < 100 # 08/12/2021
    healthy = run_iter < 300 # 2022/04/27 # TODO DELETE THIS
    endocardial_layer = True  # Set a fast isotropic endocardial layer
    npart = 576  # 2022/05/03
    # npart = 512  # 384 October 2021
    # npart = 2048  # 384 October 2021 # 10/12/2021
    # if 90 <= run_iter < 150:
    #     npart = 576  # 2022/05/03
    #     # npart = 288  # 2022/05/16
    # # if 100 <= run_iter < 110:
    # #     npart = 1152  # 2022/05/03
    # # if 20 < run_iter < 30:
    # #     npart = 288  # 2022/04/27 # TODO DELETE THIS
    # if 20 <= run_iter < 30:
    #     npart = 2304  # 2022/04/27 # TODO DELETE THIS
    # npart = 512  # 2022/01/18
    print('npart ' + str(npart))
    threadsNum = multiprocessing.cpu_count()
    keep_fraction =  (npart - threadsNum) / npart
    print('keep_fraction: ' + str(keep_fraction))
    run_count = 6 #24
    conduction_speeds = None
    consistency_iter = None
    for consistency in range(consistency_count):
        consistency = consistency+6
        for meshName_val_aux in meshName_list:
            if run_count == run_iter:
                consistency_iter = consistency
                meshName_val = meshName_val_aux
                break
            run_count = run_count + 1
        if consistency_iter is not None:
            break
    # if conduction_speeds is not None:
    # Volumes of the meshes
    # if meshName_val == "DTI001":
    #     meshVolume_val = 74
    # elif meshName_val == "DTI024":
    #     meshVolume_val = 76
    # elif meshName_val == "DTI004":
    #     meshVolume_val = 107
    # elif meshName_val == "DTI003":
    #     meshVolume_val = 171
    # elif meshName_val == "DTI032":
    #     meshVolume_val = 139
    # else:
    if consistency_iter is not None:
        meshVolume_val = 100
        
        finalResultsPath = 'metaData/Eikonal_Results/'+file_path_tag+'/'
        tmpResultsPath = 'metaData/Results/'+file_path_tag+'/'
        
        
        population_fileName = (meshName_val + '_' + rootNodeResolution + '_'
            + target_type + '_' + metric + '_' + str(consistency_iter) + '_population.csv')
        print(finalResultsPath + population_fileName)
        if (not os.path.isfile(finalResultsPath + population_fileName)
           and (not os.path.isfile(tmpResultsPath + population_fileName)) and consistency_iter is not None):
            target_fileName = None
            if load_target:
                target_fileName = 'metaData/Clinical/' + meshName_val + '_clinical_'+target_type+'.csv'
                target_fileName = target_fileName.replace('_coarse', '') # TODO FIX this so that it does not take out the coarse word
            # if either the target file to be loaded exists or it will be later simulatied, then continue
            if (not load_target) or os.path.isfile(target_fileName):
                f = open(tmpResultsPath + population_fileName, "w")
                f.write("Experiment in progress")
                f.close()
                print(population_fileName)
                run_inference_2021(
                    meshName_val=meshName_val,
                    meshVolume_val=meshVolume_val,
                    conduction_speeds=conduction_speeds,
                    final_path=finalResultsPath + population_fileName,
                    tmp_path=tmpResultsPath + population_fileName,
                    threadsNum_val=threadsNum,
                    target_type=target_type,
                    metric=metric,
                    npart=npart,
                    keep_fraction=keep_fraction,
                    rootNodeResolution=rootNodeResolution,
                    target_snr_db=target_snr_db,
                    healthy_val=healthy,
                    load_target=load_target,
                    endocardial_layer=endocardial_layer,
                    target_fileName=target_fileName,
                    is_ECGi_val=True
                )
            else:
                print('No target file: ' +target_fileName)
        else:
            print('Done by someone else')
    else:
        print('consistency_iter is None')

if __name__ == "__main__":
    import sys
    main_clinical(sys.argv[1:])
