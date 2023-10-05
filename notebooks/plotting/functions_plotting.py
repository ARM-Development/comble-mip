import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import glob
import os
import csv

## for questions, please contact 
## Florian Tornow: ft2544@columbia.edu
## Tim Juliano: tjuliano@ucar.edu
## Ann Fridlind: ann.fridlind@nasa.gov

def sat_pres(x):
    
    ## compute saturation vapor pressure
    ## __input__
    ## x...temperature in degC
    
    ## Bolton (1980), return in hPa
    return 6.112*np.exp(17.67*x/(243.5 + x))
    
def load_ceres(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'viirs_2020-03-13_satdat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    ## exclude greater temperoral offsets
    data = data.loc[abs(data['tdiff']) <= t_filter]
    
    ## exclude bispectral retrievals under high SZA
    data['time'] = (data['time.rel'] + t_off)*3600.
    ## placeholder for long- and shortwave flux nomenclature
    #data['zi'] = data['swflx']
    #data['zi.25'] = data['swflx.25']
    #data['zi.75'] = data['swflx.75']
    
    #data['cod'] = data['cod.me']
    data.index = data['time']
     
    data['class'] = data['sat']
    return data


def load_calipso(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'caliop_2020-03-13_satdat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    ## exclude greater temperoral offsets
    data = data.loc[abs(data['tdiff']) <= t_filter]
        
    data['time'] = (data['time.rel'] + t_off)*3600.
    data['zi'] = data['cth']*1000
    data['zi.25'] = data['cth.25']*1000
    data['zi.75'] = data['cth.75']*1000
    
    data['iwp'] = data['iwp']/1000
    data['iwp.25'] = data['iwp.25']/1000
    data['iwp.75'] = data['iwp.75']/1000
    
    data['od'] = data['cod']
    data['od.25'] = data['cod.25']
    data['od.75'] = data['cod.75']
    #data['cod'] = data['cod.me']
    data.index = data['time']
     
    data['class'] = data['sat']
    return data

def load_iwpgong(case='20200313',t_filter = 1.,sza_filter = 80.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'gongiwp_2020-03-13_satdat2.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    ## exclude greater temperoral offsets
    data = data.loc[abs(data['tdiff']) <= t_filter]
        
    data['time'] = (data['time.rel'] + t_off)*3600.
    
    data['iwp'] = data['iwp']/1000
    data['iwp.25'] = data['iwp']/1000
    data['iwp.75'] = data['iwp']/1000
    data.index = data['time']
     
    data['class'] = data['sat']
    return data

def load_sentinel(case='20200313',t_filter = 1.,sza_filter = 80.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'sentinel_2020-03-13_satdat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    ## exclude greater temperoral offsets
    data = data.loc[abs(data['tdiff']) <= t_filter]
    
    ## exclude bispectral retrievals under high SZA
    data.loc[data['sza'] > sza_filter,['cod','cod.25','cod.75']] = np.nan 
    
    data['time'] = (data['time.rel'] + t_off)*3600.
    data['zi'] = data['cth']
    data['zi.25'] = data['cth.25']
    data['zi.75'] = data['cth.75']
    
    data['od'] = data['cod']
    data['od.25'] = data['cod.25']
    data['od.75'] = data['cod.75']
    data.index = data['time']
     
    data['class'] = data['sat']
    return data

def load_viirs(case='20200313',t_filter = 1.,sza_filter = 80.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'viirs_2020-03-13_satdat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    ## exclude greater temperoral offsets
    data = data.loc[abs(data['tdiff']) <= t_filter]
    
    ## exclude bispectral retrievals under high SZA
    data.loc[data['sza'] > sza_filter,['cod','cod.25','cod.75']] = np.nan 
    
    data['time'] = (data['time.rel'] + t_off)*3600.
    data['zi'] = data['cth']
    data['zi.25'] = data['cth.25']
    data['zi.75'] = data['cth.75']
    data['ctt'] = data['ctt'] - 273.15
    data['ctt.25'] = data['ctt.25'] - 273.15
    data['ctt.75'] = data['ctt.75'] - 273.15
    
    #data['cod'] = data['cod.me']
    data['od'] = data['cod']
    data['od.25'] = data['cod.25']
    data['od.75'] = data['cod.75']
    data.index = data['time']
     
    data['class'] = data['sat']
    return data

def load_modis(case='20200313',t_filter = 1.,sza_filter = 80.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'modis_2020-03-13_satdat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    ## exclude greater temperoral offsets
    data = data.loc[abs(data['tdiff']) <= t_filter]
    
    ## exclude bispectral retrievals under high SZA
    data.loc[data['sza'] > sza_filter,['cod','cod.25','cod.75']] = np.nan 
    
    data['time'] = (data['time.rel'] + t_off)*3600.
    data['zi'] = data['cth']
    data['zi.25'] = data['cth.25']
    data['zi.75'] = data['cth.75']
    data.index = data['time']
    #data['cod'] = data['cod.me']
    data['od'] = data['cod']
    data['od.25'] = data['cod.25']
    data['od.75'] = data['cod.75']
     
    data['class'] = data['sat']
    return data

def load_maclwp(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'maclwp_2020-03-13_satdat3.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    data = data.loc[abs(data['tdiff']) <= t_filter]
    data['time'] = (data['time.rel'] + t_off)*3600.
    data.index = data['time']
    #time = data['time'][:]
    
    #pd.DataFrame({"class": ['MAC-LWP']* len(time), "time":time}, index=time)
    
    #print(data.time)
    
    #data_mac = pd.DataFrame({"class": ['MAC-LWP'] * len(time), "time":time}, index=time)
    #data_mac['lwp_bu'] = data['lwp'][:]/1000.
    
    data_mac = data
    data_mac['lwp_bu'] = data['lwp'][:]/1000.
    data_mac['lwp_bu.25'] = data['lwp.25'][:]/1000.
    data_mac['lwp_bu.75'] = data['lwp.75'][:]/1000.
    data_mac['class'] = data_mac['sat']
    return data_mac

def load_kazrkollias(case='20200313',PATH='../../data_files/',aux_dat=pd.DataFrame()):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'kazr-kollias_2020-03-13_dat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    data['time'] = (data['time.rel'] + t_off)*3600.
    data.index = data['time']
    data['zi'] = data['cth']
    data['zi.25'] = data['cth.25']
    data['zi.75'] = data['cth.75']
    data['lwp_bu'] = data['lwp'][:]/1000.
    data['lwp_bu.25'] = data['lwp.25'][:]/1000.
    data['lwp_bu.75'] = data['lwp.75'][:]/1000.
    data['class'] = data['source']
    
    if aux_dat.shape[0] > 0:
        print('KAZR (Kollias): here using auxiliary field to estimate cloud-top temperature')
        aux_dat['zdiff'] = np.abs(aux_dat['zf'] - np.float(data['zi']))
        aux_dat['zdiff.25'] = np.abs(aux_dat['zf'] - np.float(data['zi.25']))
        aux_dat['zdiff.75'] = np.abs(aux_dat['zf'] - np.float(data['zi.75']))
        data['ctt'] = np.mean(aux_dat.loc[aux_dat['zdiff'] < 10,'ta']) - 273.15
        data['ctt.25'] = np.mean(aux_dat.loc[aux_dat['zdiff.25'] < 10,'ta']) - 273.15
        data['ctt.75'] = np.mean(aux_dat.loc[aux_dat['zdiff.75'] < 10,'ta']) - 273.15
    
    return data

def load_kazrclough(case='20200313',PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'kazr-clough_2020-03-13_dat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    data['time'] = (data['time.rel'] + t_off)*3600.
    data.index = data['time']
    data['lwp_bu'] = data['lwp'][:]/1000.
    data['lwp_bu.25'] = data['lwp.25'][:]/1000.
    data['lwp_bu.75'] = data['lwp.75'][:]/1000.
    data['class'] = data['source']
    
    return data


def load_radflux(case='20200313',PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'radflux_2020-03-13_dat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    data['time'] = (data['time.rel'] + t_off)*3600.
    data.index = data['time']
    data['class'] = data['source']
    
    return data

def load_aeri(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    if case == '20200313':
        file = 'aeri_2020-03-13_dat.csv'
        time_near = 18.
    
    data = pd.read_csv(PATH + file)
    data = data.loc[abs(data['time.abs']/3600 - time_near) <= t_filter]
    
    data['time'] = time_near*3600
    data['ta'] = data['temp'] + 273.15
    data['qv'] = data['qv']/1000
    data.index = data['time.abs']/3600
    data['zf'] = data['altitude']*1000
    data['ua'] = float('nan') 
    data['va'] = float('nan') 
    data.loc[data.zf > 5800,'qv'] = float('nan')
    data.loc[data.zf > 5800,'theta'] = float('nan')
    data.loc[data.zf > 5800,'ta'] = float('nan')
    data['class'] = 'AERI'
    
    return data
    
def load_rs(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load radiosonde obs
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    direc = pathlib.Path(PATH)
    R_cp = 0.286 
    eps = 0.622

    if case == '20200313':
        time_near = 18.
    
    NCFILES = list(direc.rglob('anx*cdf'))
    
    var_vec_1d = ['alt','pres','u_wind','v_wind','tdry','dp','rh']
    var_vec_1d_trans = ['zf','pf','ua','va','temp','tdew','rh']
    
    rs_col = pd.DataFrame()
    for fn in NCFILES:
        if fn.stem.split(".")[2] == case:
            print(fn)
            
            group = 'Radiosonde:'+fn.stem.split(".")[3]
            ds = nc.Dataset(fn)
            
            time = ds.variables['time'][:]
            p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
            for vv in var_vec_1d:
                vv_trans = var_vec_1d_trans[var_vec_1d.index(vv)]
                p_df[vv_trans] = ds.variables[vv][:]
                
            ds.close
            rs_col = pd.concat([rs_col,p_df])
    
    ## convert to model output (qv, theta)
    rs_col['theta'] = (273.15 + rs_col['temp'])*(1000./rs_col['pf'])**(R_cp)
    rs_col['qv'] = sat_pres(rs_col['tdew']) / (rs_col['pf'] - ((1 - eps)*sat_pres(rs_col['tdew'])))
    
    ## optional temporal filter 
    rs_col = rs_col[abs(rs_col['time']/3600 - time_near) < t_filter]

    ## for plotting purposes
    rs_col['time'] = time_near*3600.
    rs_col['ta'] = rs_col['temp'] + 273.15

    return rs_col


def load_era5(case='20200313',PATH='../../data_files/'):
    
    ## load ERA5 data along trajectory
    ## __input__
    ## case........string of COMBLE date
    ## PATH........directory
    
    if case =='20200313':
        fn = PATH + 'theta_temp_rh_sh_uvw_sst_along_trajectory_era5ml_28h_end_2020-03-13-18.nc'
        t_offset = 18.

    print(fn)
    ds = nc.Dataset(fn)
    
    ## extract 1D and 2D fields
    var_vec_1d = ['SST','LatHtFlx','SenHtFlx']
    var_vec_1d_trans = ['ts','hfls','hfss']
    fact_1d = [1.,-1.,-1.]
    
    var_vec_2d = ['U','V','W','Theta','SH','GEOS_HT']
    var_vec_2d_trans = ['ua','va','w','theta','qv','zf']
    group = 'ERA5'
    
    time = (ds.variables['Time'][:] + t_offset)*3600
    pf = ds.variables['Pressure'][:] 
    
    p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
    for vv in var_vec_1d:
        vv_trans = var_vec_1d_trans[var_vec_1d.index(vv)]
        p_df[vv_trans] = ds.variables[vv][:]*fact_1d[var_vec_1d.index(vv)]
       
    df_col2 = pd.DataFrame()
    for ii in range(len(pf)):
        p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time, "pf": pf[ii]}, index=time/3600)      
        for vv in var_vec_2d:
            vv_trans = var_vec_2d_trans[var_vec_2d.index(vv)]
            p_df2[vv_trans] = ds.variables[vv][:,:][:,ii]
        df_col2 = pd.concat([df_col2,p_df2])
        
    ds.close()

    return p_df,df_col2        


def load_real_wrf(PATH='../../data_files/'):
    
    ## load realistic WRF-LES output along trajectory
    ## __input__
    ## PATH........directory
    
    fn = PATH + 'REAL_WRF_LES_COMBLE-I.nc'

    print(fn)
    ds = nc.Dataset(fn)
    time = ds.variables['time'][:]
    zf   = ds.variables['zf'][:]
    
    ## extract 1D and 2D fields
    var_vec_1d = ['rmol','zi','ziol']
    var_vec_1d_trans = ['rmol','zi','ziol']
    fact_1d = [1.,1.,1.]
    
    var_vec_2d = ['ua','va','theta']
    var_vec_2d_trans = ['ua','va','theta']
    group = 'REAL-WRF'
    
    p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
    for vv in var_vec_1d:
        vv_trans = var_vec_1d_trans[var_vec_1d.index(vv)]
        p_df[vv_trans] = ds.variables[vv][:]*fact_1d[var_vec_1d.index(vv)]
    
    df_col2 = pd.DataFrame()
    for ii in range(len(zf)):
        p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time, "zf": zf[ii]}, index=time/3600)      
        for vv in var_vec_2d:
            vv_trans = var_vec_2d_trans[var_vec_2d.index(vv)]
            p_df2[vv_trans] = ds.variables[vv][:,:][:,ii]
        df_col2 = pd.concat([df_col2,p_df2])
        
    ds.close()

    return p_df,df_col2 


def load_sims(path,var_vec_1d,var_vec_2d,t_shift = 0,keyword='',make_gray = 0):
    
    ## load ERA5 data along trajectory
    ## __input__
    ## path..........directory (scanning all subdirectories)
    ## var_vec_1d....variables with time dependence
    ## var_vec_2d....variables with time and height dependence
    ## t_shift.......time shift prior to ice edge
    ## keyword.......search for subset of sims within path
    
    direc = pathlib.Path(path)
    NCFILES = list(direc.rglob("*nc"))
    NCFILES_STR = [str(p) for p in pathlib.Path(path).rglob('*.nc')]
    
    ## variables that only have time as dimension
    print('Loading variables: f(time)')
    df_col = pd.DataFrame()
    count = 0
    for fn in NCFILES:
        if keyword in NCFILES_STR[count]:
            print(fn)
            ds = nc.Dataset(fn)
            #print(ds)
            time = ds.variables['time'][:]
            #cwp  = ds.variables['cwp'][:]
            #rwp  = ds.variables['rwp'][:]

            label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
            #label_items = label_items[0:(len(label_items)-1)]
            group = "/".join(label_items)

            #p_df = pd.DataFrame({"class": [group]* len(time), "time":time, "cwp": cwp, "rwp": rwp},index=time/3600)
            p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
            for vv in var_vec_1d:
                if vv in ds.variables:
                    p_df[vv] = ds.variables[vv][:]
                else:
                    print(vv + ' not found in ' + str(fn))
                    p_df[vv] = np.NAN

            ds.close()
            df_col = pd.concat([df_col,p_df])
            
        count+=1
        
    ## variables that have time and height as dimensions
    print('Loading variables: f(time,height)')
    df_col2 = pd.DataFrame()
    count = 0
    for fn in NCFILES:
        if keyword in NCFILES_STR[count]:
            print(fn)
            ds = nc.Dataset(fn)
            time = ds.variables['time'][:]
            zf   = ds.variables['zf'][:]
            qv   = ds.variables['qv'][:,:]
            zf_ndim = zf.ndim

            label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
            #label_items = label_items[0:(len(label_items)-1)]
            group = "/".join(label_items)

            for ii in range(len(zf)):
                if(zf_ndim > 1) & (ii==0):
                    zf = zf[1,:]
                p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time, "zf": zf[ii]}, index=time/3600)      
                for vv in var_vec_2d:
                    #if(ii==0): print(vv)
                    if vv in ds.variables:
                        if(zf_ndim>1) & (vv=='pa'):
                            p_df2[vv] = ds.variables[vv][:][ii]
                        else:
                            p_df2[vv] = ds.variables[vv][:,:][:,ii]
                    else:
                        if(ii==0): print(vv + ' not found in ' + str(fn))
                        p_df2[vv] = np.NAN
                df_col2 = pd.concat([df_col2,p_df2])
            
        count+=1
            
    ## a simple cloud-top temperature
    df_col['ctt'] = np.nan
    for cc in np.unique(df_col['class']):
        df_sub  = df_col.loc[df_col['class']==cc]
        df_sub2 = df_col2.loc[df_col2['class']==cc]
        #print(df_sub)
        if 'ta' in df_col2.columns and 'zi' in df_col.columns:
            for tt in df_sub['time']:
                zi_step = df_sub.loc[df_sub['time'] == tt,'zi']
                ta_step = df_sub2.loc[df_sub2['time'] == tt,['zf','ta']]
                ta_step['zf_diff'] = np.abs(ta_step['zf'] - zi_step)
                df_col.loc[(df_col['class']==cc) & (df_col['time']==tt),'ctt'] = min(ta_step.loc[ta_step.zf_diff == ta_step.zf_diff.min(),'ta'], default=np.NAN) - 273.15
    
    df_col['time']  = df_col['time'] + t_shift*3600.
    df_col2['time'] = df_col2['time'] + t_shift*3600.
    
    if(make_gray == 1):        
        df_col['colflag']  = 'gray'
        df_col2['colflag'] = 'gray'
    else:
        df_col['colflag']  = 'col'
        df_col2['colflag'] = 'col'        
    
    return df_col,df_col2


def plot_1d(df_col,var_vec,t0=-2.,t1=18.,longnames=[]):
    
    ## plot variables with time dependence
    ## __input__
    ## df_col.....data frame containing simulations, reanalysis, and/or observations
    ## var_vec....variables with time dependence
    ## t0.........starting plot time (h relative to ice edge)
    ## t1.........end plot time (h relative to ice edge)
    
    t0 = t0*3600. # convert h to s
    t1 = t1*3600.
    
    ## 1D plots
    plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442"]
    plot_symbol = ['+','x','s','o','D','1','2','3']
    
    counter = 0
    counter_symbol = 0
    counter_plot = 0
        
    fig, axs = plt.subplots(len(var_vec),1,figsize=(5,1 + 2*len(var_vec)))
    for label, df in df_col.groupby('class'):
        df = df[(df.time>=t0) & (df.time<=t1)]
        if (label=='MAC-LWP') | (label=='KAZR (Kollias)')| (label=='KAZR (Clough)'):
            df['lwp'] = df['lwp_bu']
            df['lwp.25'] = df['lwp_bu.25']
            df['lwp.75'] = df['lwp_bu.75']
        for ii in range(len(var_vec)):
            if len(var_vec) == 1:
                obj = axs
            else:
                obj = axs[ii]
            if (label=='MAC-LWP') | (label=='MODIS') | (label=='VIIRS') | (label=='CERES') | (label=='SENTINEL') | (label=='KAZR (Kollias)')| (label=='KAZR (Clough)')| (label=='CALIOP')| (label=='ATMS')| (label=='RADFLUX'):
                obj.scatter(df.time/3600,df[var_vec[ii]],label=label,c='k',marker=plot_symbol[counter_symbol])
                #print(label)
                #print(df[var_vec[ii]])
                if (label=='MAC-LWP') | (label=='VIIRS') | (label=='MODIS') | (label=='CERES')| (label=='SENTINEL') | (label=='KAZR (Kollias)')| (label=='KAZR (Clough)')| (label=='CALIOP')| (label=='RADFLUX'):
                    if np.count_nonzero(np.isnan(df[var_vec[ii]])) < len(df[var_vec[ii]]):
                        error_1 = np.abs(df[var_vec[ii]] - df[var_vec[ii]+'.25'])
                        error_2 = np.abs(df[var_vec[ii]+'.75'] - df[var_vec[ii]])
                        error = [error_1,error_2]
                        obj.errorbar(df.time/3600,df[var_vec[ii]],yerr=error,label=label,c='k',fmt=plot_symbol[counter_symbol])
                if ii==len(var_vec)-1:
                    counter_symbol +=1
            else:                
                ## eliminate doubles and only plot non-gray
                if(len(df['colflag'].unique()) > 1):
                    df = df[df['colflag'] == 'col']
                if(df['colflag'].unique() == 'gray'):
                    obj.plot(df.time/3600,df[var_vec[ii]],label=label,c='gray',zorder=1,linewidth=3,alpha=0.7)
                else:
                    obj.plot(df.time/3600,df[var_vec[ii]],label=label,c=plot_colors[counter_plot],zorder=2)
            obj.grid(alpha=0.2)
            if (len(longnames)>0) & (counter==0):
                obj.text(.01, .99, longnames[ii], ha='left', va='top', transform=obj.transAxes)
        counter +=1
        if not df['colflag'].unique() == 'gray':  counter_plot +=1
        if (label=='MAC-LWP') | (label=='MODIS') | (label=='VIIRS') | (label=='CERES') | (label=='SENTINEL') | (label=='KAZR (Kollias)')| (label=='KAZR (Clough)')| (label=='CALIOP')| (label=='ATMS')| (label=='RADFLUX'): counter_plot -=1    
    i_count = 0

    if len(var_vec) > 1:
        for ax in axs.flat:
            ax.set(xlabel='Time (h)', ylabel=var_vec[i_count])
            #ax.set_xlim([np.min(df_col.time)/3600 - 0.5, np.max(df_col.time)/3600 + 0.5])
            ax.set_xlim(t0/3600. - 0.5, t1/3600. + 0.5)
            i_count += 1
        
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
    else:
        axs.set(xlabel='Time (h)', ylabel=var_vec[i_count])
            
    # Add a legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc='upper center',ncol=2)

    fig.tight_layout()
    
    w1 = 1/abs(1.01 - len(var_vec))
    w2 = 1/abs(18.01 - len(var_vec))
    ww1 = w1/(w1 + w2)
    ww2 = w2/(w1 + w2)
    top_offset = -0.2*ww1 + 0.12*ww2
        
    fig.subplots_adjust(top=0.85 + top_offset) #base + top_offset)
    
    plt.show()


def plot_2d(df_col2,var_vec,times,z_max = 6000.):
    
    ## plot variables with time and height dependence
    ## __input__
    ## df_col2....data frame containing simulations, reanalysis, and/or observations
    ## var_vec....variables with time dependence
    ## times......list with hours of interest
    ## z_max......maximum altitude for plotting (meters)
    
    plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442",'black','gray']
    
    if 'ws' in var_vec:
        if  'ua' in df_col2.columns and 'va' in df_col2.columns:
            print('Computing wind speed')
            df_col2['ws'] = np.sqrt(df_col2['ua']**2 + df_col2['va']**2)
        else:
            print('Please include ua and va!')
            
    if 'wd' in var_vec:
        if  'ua' in df_col2.columns and 'va' in df_col2.columns:
            print('Computing wind direction')
            df_col2['ws'] = np.sqrt(df_col2['ua']**2 + df_col2['va']**2)
            df_col2['wd'] = np.arctan2(df_col2['ua']/df_col2['ws'],df_col2['va']/df_col2['ws'])*180/np.pi + 180.
            
            ## for better plotting, set values < 2 NaN
            df_col2.loc[df_col2['wd'] < 2,'wd'] = np.NAN
            
            ## bring over on one side
            df_col2.loc[df_col2['wd'] < 0,'wd'] = df_col2.loc[df_col2['wd'] < 0,'wd'] + 360
        else:
            print('Please include ua and va!')

    ## apply altitude filter
    df_col2 = df_col2[df_col2['zf'] < (z_max + 500)]
    
    counter = 0
    counter_plot = 0
    fig, axs = plt.subplots(len(var_vec),len(times),figsize=(2*len(times),2 + 2*len(var_vec)))
    for tt in range(len(times)):
        df_sub = df_col2[round(df_col2.time) == times[tt]*3600.]      
        counter_plot = 0
        for label, df in df_col2.groupby('class'):
            df = df[round(df.time) == times[tt]*3600.]
            #print(len(df))
            for ii in range(len(var_vec)):                
                if len(var_vec) == 1 & len(times) == 1:
                    obj = axs
                elif len(var_vec) == 1:
                    axs = axs.flatten()
                    obj = axs[tt]
                elif len(times) == 1:
                    obj = axs[ii]
                else:
                    obj = axs[ii,tt]
                ## eliminate doubles and only plot non-gray
                if(len(df['colflag'].unique()) > 1):
                    df = df[df['colflag'] == 'col']
                if(df['colflag'].unique() == 'gray'):
                    obj.plot(df[var_vec[ii]],df.zf,label=label,c='gray',zorder=1,linewidth=3,alpha=0.7)
                else:
                    obj.plot(df[var_vec[ii]],df.zf,label=label,c=plot_colors[counter_plot],zorder=2)
                obj.grid(alpha=0.2)
                obj.set_ylim([0, z_max])
                if ii==0:
                    obj.set_title(str(times[tt])+'h')
                if tt==0:
                    obj.set(ylabel='Altitude (km)', xlabel=var_vec[ii])
                else:
                    obj.set(xlabel=var_vec[ii])
                    plt.setp(obj.get_yticklabels(), visible=False)
                counter +=1
            if not df['colflag'].unique() == 'gray': counter_plot +=1
                
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 1.1),ncol=2)
    
    fig.tight_layout()
    
    w1 = 1/abs(1.01 - len(var_vec))
    w2 = 1/abs(40.01 - len(var_vec))
    ww1 = w1/(w1 + w2)
    ww2 = w2/(w1 + w2)
    top_offset = -0.2*ww1 + 0.19*ww2
        
    fig.subplots_adjust(top=0.9 + top_offset)
    
    plt.figure(figsize=(10,6))
    plt.show()

