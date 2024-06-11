import os, sys, tarfile
from pathlib import Path
import netCDF4 as nc
from numpy import *
import matplotlib.pyplot as plt
import glob

import time
import pylab as pl
from IPython import display
from matplotlib import colors
import pandas as pd
import re
import sys

## load an ALT file
# 1) unpack
# 2) read and organize
# 3) tidy up

vbase = 'opd_drops'
#ALT_FILE = 'dharma_alt_020666.tgz'
#INDIR  = '/data/home/floriantornow/dharma_test/'
#OUTDIR = '/data/home/floriantornow/dharma_test/tmp/'

def obtain_time(path,input_filename,type='alt_0'):
    
    file = open(path + '/dharma.log', "r")
    #print(str(int(input_filename.stem.split(type)[1])))#.strip("0"))
    for line in file:
        #col_oi = line.split('|')[0]
        col_oi = line.split('|')
        if len(col_oi) > 5:
        #    print(int(col_oi[1]))
            if int(input_filename.stem.split(type)[1]) == int(col_oi[1]):
                print(line, end='\n')
                time_2d = float(line.split('|')[2])
    return time_2d

def read_2d(FILE,vbase='opd_drops'):
    
    OUTDIR = os.path.dirname(os.path.abspath(FILE)) + '/tmp/'
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    tar = tarfile.open(FILE)
    tar.extractall(OUTDIR)
    tar.close()
    
    ## gather info from first file
    ALT_FILE_TMP = OUTDIR + Path(FILE).stem + '_0000.cdf'
    ds = nc.Dataset(ALT_FILE_TMP)
    pmap = ds.pmap
    nbox = ds.nboxes
    nx = ds.nx
    ny = ds.ny
    ds.close()
    
    var_big = zeros((nx,ny),dtype=float)
    
    ## read and organize
    for j in range(len(pmap)):
        vname = vbase + '_' + str("%04d" % (j+1,))
        
        ib = pmap[j]
        ALT_FILE_TMP = OUTDIR + Path(FILE).stem + '_' + str("%04d" % (ib,)) + '.cdf'
        ds = nc.Dataset(ALT_FILE_TMP)
        #print(ds)
        varF  = ds.variables[vname]
        bnds = varF.bnds
        var = varF[:]
        ngrow = 0
        
        ib = bnds[0]
        jb = bnds[1]
        kb = bnds[2]
        ie = bnds[3]
        je = bnds[4]
        ke = bnds[5]
        mx = ie - ib + 1
        my = je - jb + 1
        mz = ke - kb + 1
        var = reshape(var,(mx,my)) #,order='F')
        var_big[(jb-1):(je),(ib-1):(ie)] = var[0:(mx),0:(my)]
        ds.close()
    
        ## tidy up
    files = glob.glob(OUTDIR + '/*cdf')
    for f in files:
        os.remove(f)
    os.rmdir(OUTDIR)
    
    var_big = var_big.reshape(1,shape(var_big)[0],shape(var_big)[1])
    
    return var_big

def coarsen_2d(var_big,RES_ORG,RES_CLA):
    
    ## coarsen field
    nx_coa = round(var_big.shape[1]*RES_ORG/RES_CLA)
    
    var_coarse = zeros((nx_coa,nx_coa),dtype=float)
    for i in range(var_coarse.shape[0]):
        for j in range(var_coarse.shape[1]):
            ia = i*int(RES_CLA/RES_ORG)
            ie = (i+1)*int(RES_CLA/RES_ORG) - 1
            ja = j*int(RES_CLA/RES_ORG)
            je = (j+1)*int(RES_CLA/RES_ORG) - 1
            var_coarse[i,j] = mean(var_big[ia:ie,ja:je])
    
    return var_coarse

def watershed_check_m(RAD_DAT,CL_OI,TMPP,THRES):
    #CL_OI = pd.DataFrame([[x_pos,y_pos,1]],columns=['x','y','cluster'])
    #TMP = pd.DataFrame([[1,2,0.4]],columns=['x','y','rad'])
    #print('TMP')
    #print(TMPP)
    #print('CL_OI')
    #print(CL_OI)
    X_DIF = abs(CL_OI['x'] - TMPP['x'])
    Y_DIF = abs(CL_OI['y'] - TMPP['y'])
    
    DIST = sqrt(X_DIF**2 + Y_DIF**2)
    #print('Distance')
    #print(DIST)
    
    if DIST > 0:
        ##...
        x_vec = linspace(int(CL_OI['x']),int(TMPP['x']),num=int(ceil(DIST)))
        y_vec = linspace(int(CL_OI['y']),int(TMPP['y']),num=int(ceil(DIST)))
        
        rad_vec = 0*x_vec
        for dd in range(shape(x_vec)[0]):
            rad_vec[dd] = RAD_DAT['alb'][nearest_neigh(RAD_DAT,x_vec[dd],y_vec[dd])]
    else:
        rad_vec = float(TMPP['rad'])
    
    return sum(rad_vec < THRES) > 0

def nearest_neigh(RAD_DAT,x,y):
    return argmin((RAD_DAT['x'] - x)**2 + (RAD_DAT['y'] - y)**2)    

def id_watershed(IMG,THRES_CLOUD,THRES_CONNECT,plotting=False):
    
    ## cluster the 2D field 
    #IMG = var_coarse
    M = IMG.shape[0]
    N = IMG.shape[1]
    y_mat = reshape(repeat(arange(0,M),N),(M,N))
    x_mat = reshape(tile(arange(0,N),M),(M,N))
    
    RAD_VEC = reshape(IMG,(1,IMG.shape[0]*IMG.shape[1]))
    RAD_VEC = sort(RAD_VEC.ravel())
    RAD_VEC = RAD_VEC[::-1]
    
    RAD_DAT = pd.DataFrame(IMG.ravel(),columns=['alb'])
    RAD_DAT['x'] = x_mat.ravel()
    RAD_DAT['y'] = y_mat.ravel()
    L_RAD = len(RAD_DAT)
    print(RAD_DAT)
    
    cl_counter = 1
    POINT_COL = pd.DataFrame()
    print('...associating each cloudy pixel...')
    for ii in range(L_RAD):
        RAD_OI = RAD_VEC[ii]
        if RAD_OI < THRES_CLOUD:
            break
        
        ## find position of next-brighter pixel
        x_pos = int(x_mat[where(IMG==RAD_OI)])
        y_pos = int(y_mat[where(IMG==RAD_OI)])
        
        TMP = pd.DataFrame([[x_pos,y_pos,RAD_OI,ii+1]],columns=['x','y','rad','ind'])
        if len(POINT_COL)==0:
            TMP['cluster'] = cl_counter   
        else:
            # compute cluster centers
            POINT_COL_CP = POINT_COL
            POINT_COL_CP = POINT_COL_CP.set_index('cluster')
            cluster_center = POINT_COL.groupby('cluster').mean()
            
            # check coordinate of interest against each center
            CL_FLAG_COL = pd.DataFrame()
            for cc in range(len(cluster_center)):
                C_OI = cluster_center.loc[cc+1]
                CL_FLAG = watershed_check_m(RAD_DAT,C_OI,TMP.loc[0],THRES=THRES_CONNECT)
                CL_FLAG_COL = pd.concat([CL_FLAG_COL,pd.DataFrame([[CL_FLAG,cc+1]],columns=['connect','cluster'])])
            
            # check coordinate of interest against existing points
            POI_FLAG_COL = pd.DataFrame()
            if(len(POINT_COL)>0):
                for cc in range(len(POINT_COL)):
                    P_OI = POINT_COL.loc[cc+1]
                    POI_FLAG = watershed_check_m(RAD_DAT,P_OI,TMP.loc[0],THRES=THRES_CONNECT)
                    POI_FLAG_COL = pd.concat([POI_FLAG_COL,pd.DataFrame([[POI_FLAG,cc+1]],columns=['connect','ind'])])
            
            POI_FLAG_COL_SUM = 0
            if len(POI_FLAG_COL):
                POI_FLAG_COL_SUM = sum(POI_FLAG_COL['connect'])
            
            if sum(CL_FLAG_COL['connect'])==len(CL_FLAG_COL) and POI_FLAG_COL_SUM==len(POI_FLAG_COL):
                cl_counter += 1
                cl_assign = cl_counter
                print('Cluster center' + str(cl_assign))
            else:          
                if sum(CL_FLAG_COL['connect']) < len(CL_FLAG_COL):
                    CL_FLAG_COL = CL_FLAG_COL.set_index('cluster')
                    cl_assign = CL_FLAG_COL.loc[CL_FLAG_COL['connect']==False].index[0]
                if POI_FLAG_COL_SUM < len(POI_FLAG_COL):
                    POI_FLAG_COL = POI_FLAG_COL.set_index('ind')
                    cl_assign = POINT_COL.loc[POI_FLAG_COL.loc[POI_FLAG_COL['connect']==False].index[0]]['cluster']
            TMP['cluster'] = cl_assign
    
        TMP = TMP.set_index('ind')
        POINT_COL = pd.concat([POINT_COL,TMP])
        
        ## show updates
        if plotting == True and POINT_COL['cluster'].max() > 2:
            cmap = plt.cm.rainbow
            norm = colors.BoundaryNorm(arange(1, POINT_COL['cluster'].max(), 1),cmap.N)
            plt.imshow(IMG,interpolation='none',cmap='gray')
            plt.xlim(0,N-1)
            plt.ylim(0,M-1)
            plt.scatter(POINT_COL['x'],POINT_COL['y'],c=POINT_COL['cluster'],cmap=cmap, norm=norm, s=100, edgecolor='none')
            
            display.clear_output(wait=True)
            display.display(plt.gcf())


    print('...checking interconnection between clusters...')
    for cc in range(len(cluster_center)):
        for pp in range(len(cluster_center)):
        
            DUAL_COL = pd.DataFrame()
            print(str(cc) + '_' + str(pp))
            POINT_COL_cc = POINT_COL.loc[POINT_COL['cluster']==cc+1]
            POINT_COL_pp = POINT_COL.loc[POINT_COL['cluster']==pp+1]
            
            if cc!=pp:
                for ccc in POINT_COL_cc.index:
                    for ppp in POINT_COL_pp.index:
                        DUAL_FLAG = watershed_check_m(RAD_DAT,POINT_COL_cc.loc[ccc],POINT_COL_pp.loc[ppp],THRES=THRES_CONNECT)
                        DUAL_COL = pd.concat([DUAL_COL,pd.DataFrame([[DUAL_FLAG,pp+1]],columns=['connect','cluster_p'])])
            else:
                continue
        
            if len(DUAL_COL) > 0:
                if sum(DUAL_COL['connect'] == False) > 0:
                    print('joining clusters ' + str(cc+1) + ' and ' + str(pp+1))
                    POINT_COL.loc[POINT_COL['cluster']==max([cc+1,pp+1]),'cluster'] = min([cc+1,pp+1])
                
            cmap = plt.cm.rainbow
            norm = colors.BoundaryNorm(arange(1, POINT_COL['cluster'].max(), 1), cmap.N)
        
            plt.imshow(IMG,interpolation='none',cmap='gray')
            plt.xlim(0,N-1)
            plt.ylim(0,M-1)
            plt.scatter(POINT_COL['x'],POINT_COL['y'],c=POINT_COL['cluster'],cmap=cmap, norm=norm, s=100, edgecolor='none')
        
            display.clear_output(wait=True)
            display.display(plt.gcf())

    ## review cluster numbers
    CL_HIST = POINT_COL['cluster'].value_counts()
    CL_NUM = len(CL_HIST)
    CL_VEC = sort(CL_HIST.index)
    
    counter = 1
    for cc in range(CL_NUM):
        print(str(cc+1) + '_' + str(counter))
        POINT_COL.loc[POINT_COL['cluster']==CL_VEC[cc],'cluster'] = counter
        counter += 1
        
    plt.imshow(IMG,interpolation='none',cmap='gray')
    plt.xlim(0,N-1)
    plt.ylim(0,M-1)
    plt.scatter(POINT_COL['x'],POINT_COL['y'],c=POINT_COL['cluster'],cmap=cmap, norm=norm, s=100, edgecolor='none')

    return POINT_COL
        
def threed_loader(FILE,vbase='w'):
    OUTDIR = os.path.dirname(os.path.abspath(FILE)) + '/tmp/'
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    tar = tarfile.open(FILE)
    tar.extractall(OUTDIR)
    tar.close()
    
    ## gather info from first file
    PLT_FILE_TMP = OUTDIR + Path(FILE).stem + '_0000.cdf'
    ds = nc.Dataset(PLT_FILE_TMP)
    #for dim in ds.dimensions.items():
    #    print(dim)
    #print(ds)
    pmap = ds.pmap
    nbox = ds.nboxes
    nx = ds.nx
    ny = ds.ny
    nz = ds.nz
    ds.close()
    var_big = zeros((nz,ny,nx),dtype=float)
    
    ## read and organize
    for j in range(len(pmap)):
        vname = vbase + '_' + str("%04d" % (j+1,))
        
        ib = pmap[j]
        PLT_FILE_TMP = OUTDIR + Path(FILE).stem + '_' + str("%04d" % (ib,)) + '.cdf'
        #print(PLT_FILE_TMP)
        ds = nc.Dataset(PLT_FILE_TMP)
        varF  = ds.variables[vname]
        #print(varF.dimensions)
        bnds = varF.bnds
        var = varF[:]
        ngrow = varF.ngrow
        ishift = varF.ishift
        jshift = varF.jshift
        kshift = varF.kshift
        
        ib = bnds[0]
        jb = bnds[1]
        kb = bnds[2]
        ie = bnds[3]
        je = bnds[4]
        ke = bnds[5]
        mx = ie - ib + 1
        my = je - jb + 1
        mz = ke - kb + 1
        var = reshape(var,(mz,my,mx)) 
        var_big[(kb-1+ngrow):(ke-ngrow-kshift),(jb-1+ngrow):(je-ngrow-jshift),(ib-1+ngrow):(ie-ngrow-ishift)] = var[ngrow:(mz-ngrow-kshift),ngrow:(my-ngrow-jshift),ngrow:(mx-ngrow-ishift)]
        ds.close()
    
    ## tidy up
    files = glob.glob(OUTDIR + '/*cdf')
    for f in files:
        os.remove(f)
    
    return var_big
