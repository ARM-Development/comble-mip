import netCDF4 as nc
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import glob
import os
import csv
import scipy
import xarray as xr

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
    
    data['cth'] = data['cth']*1000
    data['cth.25'] = data['cth.25']*1000
    data['cth.75'] = data['cth.75']*1000
    
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
    
    ## exclude values obtained during high-cloud influence
    data.loc[(data['time.rel'] + t_off) < 3,['ctt','cth']] = np.nan 
    
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
    
    ## exclude values obtained during high-cloud influence
    data.loc[(data['time.rel'] + t_off) < 3,['ctt','cth']] = np.nan 
    
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

def load_kazrkollias(case='20200313',t_filter = 1.,PATH='../../data_files/',aux_dat=pd.DataFrame()):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'kazr-kollias_2020-03-13_dat2.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    p_df = pd.DataFrame({"class": ['Bulk'], "time":[t_off*3600]}, index=[t_off])
    ## here equating inversion height and cloud-top height
    p_df['zi.25'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cth']>0),'cth'],0.25)
    p_df['zi']    = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cth']>0),'cth'],0.50)
    p_df['zi.75'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cth']>0),'cth'],0.75)
    p_df['cth.25'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cth']>0),'cth'],0.25)
    p_df['cth']    = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cth']>0),'cth'],0.50)
    p_df['cth.75'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cth']>0),'cth'],0.75)
    p_df['lwp_bu.25'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['lwp']>=0),'lwp'],0.25)/1000
    p_df['lwp_bu']    = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['lwp']>=0),'lwp'],0.50)/1000
    p_df['lwp_bu.75'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['lwp']>=0),'lwp'],0.75)/1000
    p_df['class'] = 'KAZR (Kollias)'
    
    if aux_dat.shape[0] > 0:
        print('KAZR (Kollias): here using auxiliary field to estimate cloud-top temperature')
        aux_dat['zdiff'] = np.abs(aux_dat['zf'] - np.float(p_df['zi']))
        aux_dat['zdiff.25'] = np.abs(aux_dat['zf'] - np.float(p_df['zi.25']))
        aux_dat['zdiff.75'] = np.abs(aux_dat['zf'] - np.float(p_df['zi.75']))
        p_df['ctt'] = np.mean(aux_dat.loc[aux_dat['zdiff'] < 10,'ta']) - 273.15
        p_df['ctt.25'] = np.max(aux_dat.loc[aux_dat['zdiff.25'] < 10,'ta']) - 273.15
        p_df['ctt.75'] = np.min(aux_dat.loc[aux_dat['zdiff.75'] < 10,'ta']) - 273.15
    
    return p_df

def load_kazrclough(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'kazr-clough_2020-03-13_dat2.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    p_df = pd.DataFrame({"class": ['Bulk'], "time":[t_off*3600]}, index=[t_off])
    p_df['lwp_bu.25'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['lwp']>=0),'lwp'],0.25)/1000
    p_df['lwp_bu']    = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['lwp']>=0),'lwp'],0.50)/1000
    p_df['lwp_bu.75'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['lwp']>=0),'lwp'],0.75)/1000
    p_df['class'] = 'KAZR (Clough)'
    
    return p_df


def load_radflux(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'radflux_2020-03-13_dat2.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    
    p_df = pd.DataFrame({"class": ['Bulk'], "time":[t_off*3600]}, index=[t_off])
    #p_df['od.25'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cod']>=0),'cod'],0.25)/1000
    #p_df['od']    = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cod']>=0),'cod'],0.50)/1000
    #p_df['od.75'] = np.quantile(data.loc[(abs(data['trel']) <= t_filter) & (data['cod']>=0),'cod'],0.75)/1000
    
    p_df['od']    = np.mean(data.loc[(abs(data['trel']) <= t_filter) & (data['cod']>=0),'cod'])
    p_df['class'] = 'RADFLUX'
    
    return p_df

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

def load_carraflux(case='20200313',PATH='../../data_files/'):

    ## load CARRA surface fluxes
    if case == '20200313':
        fn = 'CARRA_SHF_LHF_along_trajectory_end_2020-03-13-18.nc'
        time_offset = 18.
    
    ds = nc.Dataset(PATH + fn)
    time = ds.variables['Time'][:] + time_offset
    lhf = ds.variables['LHF'][:]
    shf = ds.variables['SHF'][:]
    
    p_df = pd.DataFrame({"class": ['CARRA']*len(time), "time":time*3600}, index=time)
    ## stick to sign convention of LES
    p_df['hfls']    = -lhf
    p_df['hfss']    = -shf
    
    return p_df
    
def load_flux(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load ECOR and Bulk surface turbulent fluxes, obtained near Andenenes
    if case == '20200313':
        fn = 'bulk_aerodynamic_fluxes_031320.nc'
        time_near = 18.
    
    ds = nc.Dataset(PATH + fn)
    time_bulk = ds.variables['MINUTE_OF_DAY_BULK'][:]/60
    time_ecor = ds.variables['MINUTE_OF_DAY_ECOR'][:]/60
    lhf_bulk = ds.variables['bulk_lhf'][:].data
    shf_bulk = ds.variables['bulk_shf'][:].data
    lhf_ecor = ds.variables['ecor_lhf'][:].data
    shf_ecor = ds.variables['ecor_shf'][:].data
    
    p_df = pd.DataFrame({"class": ['Bulk'], "time":[time_near*3600]}, index=[time_near])
    p_df['hfls']    = lhf_bulk[(abs(time_bulk - time_near) <= t_filter) & (lhf_bulk > 0)].mean()
    p_df['hfls.25'] = np.quantile(lhf_bulk[(abs(time_bulk - time_near) <= t_filter) & (lhf_bulk > 0)],0.25)
    p_df['hfls.75'] = np.quantile(lhf_bulk[(abs(time_bulk - time_near) <= t_filter) & (lhf_bulk > 0)],0.75)
    p_df['hfss']    = shf_bulk[(abs(time_bulk - time_near) <= t_filter) & (shf_bulk > 0)].mean()
    p_df['hfss.25'] = np.quantile(shf_bulk[(abs(time_bulk - time_near) <= t_filter) & (shf_bulk > 0)],0.25)
    p_df['hfss.75'] = np.quantile(shf_bulk[(abs(time_bulk - time_near) <= t_filter) & (shf_bulk > 0)],0.75)
    
    p_df_2 = pd.DataFrame({"class": ['ECOR'], "time":[time_near*3600]}, index=[time_near])
    p_df_2['hfls']   = lhf_ecor[(abs(time_ecor - time_near) <= t_filter) & (lhf_ecor > 0)].mean()
    p_df_2['hfls.25'] = np.quantile(lhf_ecor[(abs(time_ecor - time_near) <= t_filter) & (lhf_ecor > 0)],0.25)
    p_df_2['hfls.75'] = np.quantile(lhf_ecor[(abs(time_ecor - time_near) <= t_filter) & (lhf_ecor > 0)],0.75)
    p_df_2['hfss']    = shf_ecor[(abs(time_ecor - time_near) <= t_filter) & (shf_ecor > 0)].mean()
    p_df_2['hfss.25'] = np.quantile(shf_ecor[(abs(time_ecor - time_near) <= t_filter) & (shf_ecor > 0)],0.25)
    p_df_2['hfss.75'] = np.quantile(shf_ecor[(abs(time_ecor - time_near) <= t_filter) & (shf_ecor > 0)],0.75)
    
    return pd.concat([p_df,p_df_2])
    
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
    
    R_cp = 0.286 
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
    
    df_col2['ta'] = (df_col2['theta'])*(1000./df_col2['pf'])**(-R_cp)
    
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

def load_sims_2d(path,var_vec_2d,t_shift = 0,keyword='',subfolder=''):
    
    direc = pathlib.Path(path)
    NCFILES = list(direc.rglob("*nc"))
    NCFILES_STR = [str(p) for p in pathlib.Path(path).rglob('*.nc')]
    
    ## variables that only have time as dimension
    print('Loading variables: f(time,x,y)') 
    df_col2 = pd.DataFrame()
    count = 0
    count_con = 0
    for fn in NCFILES:
        #print(NCFILES_STR[count])
        if (keyword in NCFILES_STR[count]) and (subfolder in NCFILES_STR[count]):
            
            print(fn)
            label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
            group = "/".join(label_items)

            ncdata = xr.open_dataset(fn)
            ncdata['simulation']=group
            #print(ncdata)
            if count_con == 0:
                df_col2 = ncdata
            else:
                df_col2 = xr.concat([df_col2,ncdata],dim='simulation')
            count_con += 1
        count+=1 
    return df_col2

            
def load_sims_2d_slow(path,var_vec_2d,t_shift = 0,keyword='',subfolder=''):
    
    direc = pathlib.Path(path)
    NCFILES = list(direc.rglob("*nc"))
    NCFILES_STR = [str(p) for p in pathlib.Path(path).rglob('*.nc')]
    
    ## variables that only have time as dimension
    print('Loading variables: f(time,x,y)') 
    df_col2 = pd.DataFrame()
    count = 0
    for fn in NCFILES:
        #print(NCFILES_STR[count])
        if (keyword in NCFILES_STR[count]) and (subfolder in NCFILES_STR[count]):
            
            ds    = nc.Dataset(fn)
            time  = ds.variables['time'][:]
            x_vec = ds.variables['x'][:]
            y_vec = ds.variables['y'][:]

            label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
            group = "/".join(label_items)

            for xi in range(len(x_vec)-1):
                for yi in range(len(y_vec)-1):
                
                    p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time, "x": x_vec[xi], "y": y_vec[xi]}, index=time/3600) 
                    #print(p_df2)
                    for vv in var_vec_2d:
                        if vv in ds.variables:
                            p_df2[vv] = ds.variables[vv][:,xi,yi]                            
                            if (xi==0) & (yi==0) & (p_df2[vv].isna().sum() > 0): print(vv + ' shows NAN values in ' + str(fn))   
                        else:
                            if(ii==0): print(vv + ' not found in ' + str(fn))
                            p_df2[vv] = np.NAN
                    df_col2 = pd.concat([df_col2,p_df2])
            
        count+=1 
    return df_col2
    

def load_sims(path,var_vec_1d,var_vec_2d,t_shift = 0,keyword='',make_gray = 0,drop_t0=True,diag_zi_ctt=False,diag_qltot=False,diag_qitot=False,QTHRES=1.0e-6,subfolder='',ignore='placeholder'):
    
    ## load ERA5 data along trajectory
    ## __input__
    ## path..........directory (scanning all subdirectories)
    ## var_vec_1d....variables with time dependence
    ## var_vec_2d....variables with time and height dependence
    ## t_shift.......time shift prior to ice edge
    ## keyword.......search for subset of sims within path
    ## make_gray.....flag for gray appearance in plots
    ## drop_t0.......drop first time (initialization) when variables are null
    ## diag_zi_ctt...diagnose inversion height (zi) and cloud-top temperature (ctt)
    ## diag_qltot....diagnose total liquid water mixing ratio
    ## diag_qitot....diagnose total frozen water mixing ratio
    ## QTHRES........threshold to diagnose cloud-top height
    ## subfolder.....additional keyword to limit search results
    ## ignore........additional keyword to eliminate from search results
    
    direc = pathlib.Path(path)
    NCFILES = list(direc.rglob("*nc"))
    NCFILES_STR = [str(p) for p in pathlib.Path(path).rglob('*.nc')]
    #print(NCFILES_STR)
    
    ## variables that only have time as dimension
    print('Loading variables: f(time)')
    df_col = pd.DataFrame()
    count = 0
    for fn in NCFILES:
        #print(NCFILES_STR[count])
        if (keyword in NCFILES_STR[count]) and (subfolder in NCFILES_STR[count]):
            if ignore in NCFILES_STR[count]:
                count+=1
                continue
            #print('ding')
            print(fn)
            ds = nc.Dataset(fn)
            #print(ds)
            time = ds.variables['time'][:]
            # drop first time
            if drop_t0:
                if time[0] == 0.:
                    t0 = 1
                else:
                    t0 = 0
            else:
                t0 = 0
            time = time[t0:]
            #cwp  = ds.variables['cwp'][:]
            #rwp  = ds.variables['rwp'][:]

            label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
            #label_items = label_items[0:(len(label_items)-1)]
            group = "/".join(label_items)

            #p_df = pd.DataFrame({"class": [group]* len(time), "time":time, "cwp": cwp, "rwp": rwp},index=time/3600)
            p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
            for vv in var_vec_1d:
                if vv in ds.variables:
                    p_df[vv] = ds.variables[vv][t0:]
                    if p_df[vv].isna().sum() > 0: print(vv + ' shows NAN values in ' + str(fn))  
                else:
                    print(vv + ' not found in ' + str(fn))
                    p_df[vv] = np.NAN

            ds.close()
            df_col = pd.concat([df_col,p_df])
        #print(df_col)
            
        count+=1
        
    ## variables that have time and height as dimensions
    print('Loading variables: f(time,height)')
    df_col2 = pd.DataFrame()
    count = 0
    for fn in NCFILES:
        if (keyword in NCFILES_STR[count]) and (subfolder in NCFILES_STR[count]):
            if ignore in NCFILES_STR[count]:
                count+=1
                continue
            print(fn)
            ds = nc.Dataset(fn)
            time = ds.variables['time'][t0:]
            zf   = ds.variables['zf'][t0:]
            pa   = ds.variables['pa'][t0:]
            qv   = ds.variables['qv'][t0:,:]
            zf_ndim = zf.ndim
            pa_ndim = pa.ndim
            #print(len(zf))

            label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
            group = "/".join(label_items)

            if(zf_ndim > 1):
                zf = zf[1,:]
            #if (zf_ndim > 1) & (pa_ndim > 1):
            #    print('---either pa or zf should be 1-dimensional---')
            
            for ii in range(len(zf)-1):
                if zf_ndim > 1:
                    ## accounting for changing heights for SCMs that have constant pressure grid
                    p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600) 
                    p_df2['zf'] = ds.variables['zf'][t0:,ii]
                else:
                    ## simpler treatment for LES-like constant altitude grid
                    p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time, "zf": zf[ii]}, index=time/3600) 
                
                #print(ds.variables['zf'][t0:,ii])
                #print(p_df2)
                for vv in var_vec_2d:
                    if vv in ds.variables:
                        #if ii==0: 
                        #    print(vv)
                        #    print(ds.variables[vv])
                        #    print(len(ds.variables[vv][t0:][ii]))
                        #if ii == 0:
                        #    print(ds.variables[vv][t0:])
                        if(zf_ndim>1) & ((vv=='pa') | (vv=='pe')):
                            if(ds.variables[vv][t0:].ndim>1): ## some report both zf and pa as 2D fields
                                #p_df2[vv] = ds.variables[vv][t0:][0][ii]
                                p_df2[vv] = ds.variables[vv][t0:,ii]
                            else:
                                p_df2[vv] = ds.variables[vv][t0:][ii]
                        else:
                            #print(ds.variables[vv].shape)
                            #p_df2[vv] = ds.variables[vv][t0:,:][ii,:]
                            ## account for CM1 where dimensions are swapped
                            if ds.variables[vv].shape[0] != (len(zf)+1):
                                p_df2[vv] = ds.variables[vv][t0:,:][:,ii]
                            else:
                                p_df2[vv] = ds.variables[vv][:,t0:][ii,:]
                        #if ii==0: print(p_df2[vv])
                        if (ii==0) & (p_df2[vv].isna().sum() > 0): print(vv + ' shows NAN values in ' + str(fn))   
                        
                        #if (ii==0):
                        #    print(p_df2[vv])
                    else:
                        if(ii==0): print(vv + ' not found in ' + str(fn))
                        p_df2[vv] = np.NAN
                df_col2 = pd.concat([df_col2,p_df2])
            
        count+=1 
    
    
    lv = 2500*1000 #J/kg
    li = 2800*1000 #J/kg
    cp = 1.006*1000#J/kg/K
    
    ## a simple inversion height and corresponding cloud-top temperature
    if(diag_zi_ctt):
        print('computing inversion height, cloud-top height, and cloud-top temperature')
        if(('qlc' in df_col2.columns) & ('qic' in df_col2.columns)):  
            print('using liquid(-ice) potential temperature')
        df_col['zi'] = np.nan
        df_col['cth'] = np.nan
        df_col['ctt'] = np.nan
        for cc in np.unique(df_col['class']):
            print(cc)
            df_sub  = df_col.loc[df_col['class']==cc]
            df_sub2 = df_col2.loc[df_col2['class']==cc]
            if 'ta' in df_col2.columns and 'theta' in df_col2.columns:
                for tt in df_sub['time']:  
                    #zi_step = df_sub.loc[df_sub['time'] == tt,'zi']
                    ## diagnosing inversion height from theta profiles
                    if(('qlc' in df_col2.columns) & ('qic' in df_col2.columns)):  
                        theta_step = df_sub2.loc[df_sub2['time'] == tt,['zf','theta','qlc','qlr','qic','qis','qig']]
                        theta_step['qic'] = theta_step['qic'].fillna(0)
                        theta_step['qis'] = theta_step['qis'].fillna(0)
                        theta_step['qig'] = theta_step['qig'].fillna(0)
                        theta_step['theta'] = theta_step['theta'] - lv/cp*(theta_step['qlc'] + theta_step['qlr']) - li/cp*(theta_step['qic']+theta_step['qis']+theta_step['qig'])
                        theta_step['qcond_tot'] = theta_step['qlc'] + theta_step['qlr'] + theta_step['qic']+theta_step['qis']+theta_step['qig']
                    elif(('qlc' in df_col2.columns) & ('qlr' in df_col2.columns)):                          
                        theta_step = df_sub2.loc[df_sub2['time'] == tt,['zf','theta','qlc','qlr']]
                        if theta_step['qlr'].isna().sum() == 0:
                            theta_step['theta'] = theta_step['theta'] - lv/cp*(theta_step['qlc'] + theta_step['qlr'])
                        else:
                            theta_step['theta'] = theta_step['theta'] - lv/cp*theta_step['qlc']
                        theta_step['qcond_tot'] = theta_step['qlc'] + theta_step['qlr']
                    else:
                        theta_step = df_sub2.loc[df_sub2['time'] == tt,['zf','theta']]
                        theta_step['qcond_tot'] = 0
                    cth = np.max(theta_step.loc[theta_step['qcond_tot'] > QTHRES]['zf'])
                    if not theta_step.empty:
                        zi_step = zi_diagnose(theta_step)
                    ## obtaining corresponding temperature at that level
                        ta_step = df_sub2.loc[df_sub2['time'] == tt,['zf','ta']]
                        ta_step['zf_diff'] = np.abs(ta_step['zf'] - cth) #zi_step)
                        df_col.loc[(df_col['class']==cc) & (df_col['time']==tt),'cth'] = cth
                        df_col.loc[(df_col['class']==cc) & (df_col['time']==tt),'zi'] = zi_step
                        df_col.loc[(df_col['class']==cc) & (df_col['time']==tt),'ctt'] = min(ta_step.loc[ta_step.zf_diff == ta_step.zf_diff.min(),'ta'], default=np.NAN) - 273.15
                    else:
                        df_col.loc[(df_col['class']==cc) & (df_col['time']==tt),'cth'] = np.nan
                        df_col.loc[(df_col['class']==cc) & (df_col['time']==tt),'zi'] = np.nan
                        df_col.loc[(df_col['class']==cc) & (df_col['time']==tt),'ctt'] = np.nan
    df_col['time']  = df_col['time'] + t_shift*3600.
    df_col2['time'] = df_col2['time'] + t_shift*3600.
    
    ## obtain lwp if lwpc and lwpr are available
    if 'lwpr' in df_col.columns and 'lwpc' in df_col.columns:
        df_col['lwp'] = df_col['lwpr'] + df_col['lwpc']
    
    if diag_qltot:
        df_col2['qltot'] = df_col2['qlc'] + df_col2['qlr']
    
    if diag_qltot:
        df_col2['qitot'] = df_col2['qic'] + df_col2['qis'] + df_col2['qig']
    
    if(make_gray == 1):        
        df_col['colflag']  = 'gray'
        df_col2['colflag'] = 'gray'
    else:
        df_col['colflag']  = 'col'
        df_col2['colflag'] = 'col'        
    
    return df_col,df_col2

#def zi_diagnose(df_sub_2d):
#        
#    deriv_vec = pd.Series(df_sub_2d['theta']).diff() / pd.Series(df_sub_2d['zf']).diff()
#    
#    return df_sub_2d.loc[deriv_vec == deriv_vec.max(),'zf']


def zi_diagnose(df_sub_2dd):
        
    df_sub_2dd.index = df_sub_2dd['zf']
    df_sub_2dd['zfm'] = df_sub_2dd['zf'] - pd.Series(df_sub_2dd['zf']).diff()/2
    deriv_vec = pd.Series(df_sub_2dd['theta']).diff() / pd.Series(df_sub_2dd['zf']).diff()
    
    return df_sub_2dd.loc[deriv_vec.idxmax(),'zfm']

def zi_diagnose_slow(df_sub_2d):
    
    theta_step = df_sub_2d
    theta_step['d_zf'] = np.nan
    theta_step['d_th'] = np.nan
    theta_step['thm'] = np.nan
    theta_step['zfm'] = np.nan
    h_before = np.nan
    theta_before = np.nan
    for hh in theta_step['zf']:
        theta_step.loc[(theta_step['zf']==hh),'d_zf'] = hh - h_before
        theta_step.loc[(theta_step['zf']==hh),'d_th'] = theta_step[(theta_step['zf']==hh)]['theta'] - theta_before
        theta_step.loc[(theta_step['zf']==hh),'thm'] = (theta_step[(theta_step['zf']==hh)]['theta'] + theta_before)/2 
        theta_step.loc[(theta_step['zf']==hh),'zfm'] = (theta_step[(theta_step['zf']==hh)]['zf'] + h_before)/2 
        h_before = theta_step[(theta_step['zf']==hh)]['zf']
        theta_before = theta_step[(theta_step['zf']==hh)]['theta']
    theta_step['deriv'] = theta_step['d_th']/theta_step['d_zf']
    print(theta_step)
    
    return theta_step.loc[theta_step.deriv == theta_step.deriv.max(),'zfm']

def plot_1d(df_col,var_vec,**kwargs):
    
    ## plot variables with time dependence
    ## __input__
    ## df_col.......data frame containing simulations, reanalysis, and/or observations
    ## var_vec......variables with time dependence
    ## t0...........starting plot time (h relative to ice edge)
    ## t1...........end plot time (h relative to ice edge)
    ## longnames....full variable name
    ## units........variable units
    ## plot_colors..list of colors for line plots
    ## plot_ls......list of line styles for line plots
    
    ############################
    ######## SET KWARGS ########
    ############################
    if 't1' not in kwargs:
        t1 = 18.
    else:
        t1 = kwargs.get('t1')
    
    if 't0' not in kwargs:
        t0 = -2.
    else:
        t0 = kwargs.get('t0')
    
    if 'longnames' in kwargs and 'units' in kwargs:
        longnames = kwargs.get('longnames')
        units = kwargs.get('units')
    
    if 'plot_colors' not in kwargs:
        plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442","#808080","#FF00FF","#FF0000", "#00FF00", "#0000FF"]
    else:
        plot_colors = kwargs.get('plot_colors')
        
    if 'plot_ls' not in kwargs:
        plot_ls = ['-','-','-','-','-','-','-','-','-']
    else:
        plot_ls = kwargs.get('plot_ls')
    
    ############################
    ######## MAKE PLOTS ########
    ############################
    t0 = t0*3600. # convert h to s
    t1 = t1*3600.
    
    ## 1D plots
    plot_symbol = ['+','x','s','o','D','1','2','3']
    
    counter = 0
    counter_symbol = 0
    counter_plot = 0
        
    fig, axs = plt.subplots(len(var_vec),1,figsize=(5.5,1 + 2*len(var_vec)))
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
            if (label=='MAC-LWP') | (label=='MODIS') | (label=='VIIRS') | (label=='CERES') | (label=='SENTINEL') | (label=='KAZR (Kollias)')| (label=='KAZR (Clough)')| (label=='CALIOP')| (label=='ATMS')| (label=='RADFLUX') | (label=='Bulk') | (label=='ECOR')| (label=='CARRA'):
                obj.scatter(df.time/3600,df[var_vec[ii]],label=label,c='k',marker=plot_symbol[counter_symbol])
                #print(label)
                #print(df[var_vec[ii]])
                if (label=='MAC-LWP') | (label=='VIIRS') | (label=='MODIS') | (label=='CERES')| (label=='SENTINEL') | (label=='KAZR (Kollias)')| (label=='KAZR (Clough)')| (label=='CALIOP')| (label=='RADFLUX')| (label=='Bulk') | (label=='ECOR') | (label=='CARRA'):
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
                    obj.plot(df.time/3600,df[var_vec[ii]],label=label,c=plot_colors[counter_plot],ls=plot_ls[counter_plot],zorder=2)
            obj.grid(alpha=0.2)
            # set units string
            if 'longnames' in kwargs and 'units' in kwargs:
                if (len(longnames)>0) & (counter==0):
                    if units[ii] == 1:
                        unit_str = " [-]"
                    else:
                        unit_str = " [" + str(units[ii]) + "]"
                    obj.text(.01, .99, longnames[ii]+unit_str, ha='left', va='top', transform=obj.transAxes)
        counter +=1
        if not df['colflag'].unique() == 'gray':  counter_plot +=1
        if (label=='MAC-LWP') | (label=='MODIS') | (label=='VIIRS') | (label=='CERES') | (label=='SENTINEL') | (label=='KAZR (Kollias)')| (label=='KAZR (Clough)')| (label=='CALIOP')| (label=='ATMS')| (label=='RADFLUX')| (label=='Bulk') | (label=='ECOR') | (label=='CARRA'): counter_plot -=1    
    i_count = 0

    if len(var_vec) > 1:
        for ax in axs.flat:
            ax.set(xlabel='Time (h)', ylabel=var_vec[i_count])
            #ax.set_xlim([np.min(df_col.time)/3600 - 0.5, np.max(df_col.time)/3600 + 0.5])
            ax.set_xlim(t0/3600. - 0.5, t1/3600. + 0.5)
            ax.ticklabel_format(axis='y', useOffset=False)
            i_count += 1
        
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
    else:
        axs.set(xlabel='Time (h)', ylabel=var_vec[i_count])
            
    # Add a legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if len(df_col.groupby('class'))>7:
        frac = 0.4*2/len(var_vec)
    else:
        frac = 0.4/len(var_vec)
    fig.legend(by_label.values(), by_label.keys(),loc='upper center',ncol=2, bbox_to_anchor=(0.5, 1.0 + frac))
    
    fig.tight_layout()
    
    w1 = 1/abs(1.01 - len(var_vec))
    w2 = 1/abs(18.01 - len(var_vec))
    ww1 = w1/(w1 + w2)
    ww2 = w2/(w1 + w2)
    top_offset = -0.2*ww1 + 0.20*ww2
        
    #fig.subplots_adjust(top=0.85 + top_offset) #base + top_offset)
    
    plt.show()


def plot_2d(df_col2,var_vec,times,**kwargs):
    
    ## plot variables with time and height dependence
    ## __input__
    ## df_col2....data frame containing simulations, reanalysis, and/or observations
    ## var_vec....variables with time dependence
    ## times......list with hours of interest
    ## z_max......maximum altitude for plotting (meters)
    ## units........variable units
    ## plot_colors..list of colors for line plots
    ## plot_ls......list of line styles for line plots
    
    #print(len(df_col2.groupby('class')))
    ############################
    ######## SET KWARGS ########
    ############################
    if 'z_max' not in kwargs:
        z_max = 6000.
    else:
        z_max = kwargs.get('z_max')
    
    if 'units' in kwargs:
        units = kwargs.get('units')
    
    if 'plot_colors' not in kwargs:
        plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442","#808080","#FF00FF","#FF0000", "#00FF00", "#0000FF"]
    else:
        plot_colors = kwargs.get('plot_colors')
        
    if 'plot_ls' not in kwargs:
        plot_ls = ['solid','dotted','dashed','dashdot']
    else:
        plot_ls = kwargs.get('plot_ls')
    
    ###################################
    ######## COMPUTE WINDSPEED ########
    ###################################
    if 'ws' in var_vec:
        if  'ua' in df_col2.columns and 'va' in df_col2.columns:
            print('Computing wind speed')
            df_col2['ws'] = np.sqrt(df_col2['ua']**2 + df_col2['va']**2)
        else:
            print('Please include ua and va!')
            
    ########################################
    ######## COMPUTE WIND DIRECTION ########
    ########################################
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
    
    ############################
    ######## MAKE PLOTS ########
    ############################
    counter = 0
    fig, axs = plt.subplots(len(var_vec),len(times),figsize=(2*len(times),2 + 2*len(var_vec)))
    for tt in range(len(times)):
        df_sub = df_col2[round(df_col2.time) == times[tt]*3600.]      
        counter_col = 0 
        counter_line = 0
        for label, df in df_col2.groupby('class'):
            #df = df[round(df.time) == times[tt]*3600.]
            ## allow wiggleroom to accomodate uneven model output (suggested by TomiRaatikainen)
            df = df[abs(round(df.time)-times[tt]*3600.)<2.]
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
                if('colflag' not in df.columns):
                    df['colflag'] = 'col'
                if(len(df['colflag'].unique()) > 1):
                    df = df[df['colflag'] == 'col']
                if(df['colflag'].unique() == 'gray'):
                    obj.plot(df[var_vec[ii]],df.zf,label=label,c='gray',zorder=1,linewidth=3,alpha=0.7)
                else:
                    pcol = plot_colors[counter_col]
                    pline = 'solid'
                    if(label=='ERA5'): pcol='black'
                    if(label[0:5]=='Radio'): 
                        pcol='grey'
                        pline=plot_ls[counter_line]
                    obj.plot(df[var_vec[ii]],df.zf,label=label,c=pcol,ls=pline,zorder=2)
                obj.grid(alpha=0.2)
                obj.set_ylim([0, z_max])
                if ii==0:
                    obj.set_title(str(times[tt])+'h')
                # set units string
                if 'units' in kwargs:
                    if units[ii] == 1:
                        unit_str = " [-]"
                    else:
                        unit_str = " [" + str(units[ii]) + "]"
                else:
                    unit_str = ''
                if tt==0:
                    obj.set(ylabel='Altitude (m)', xlabel=var_vec[ii] + unit_str)
                else:
                    plt.setp(obj.get_yticklabels(), visible=False)
                counter +=1
            if not df['colflag'].unique() == 'gray': counter_col +=1
            if (label=='ERA5') or (label[0:5]=='Radio'): counter_col -=1
            if label[0:5]=='Radio': counter_line +=1
                
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))   
    #if len(df_col2.groupby('class'))>8:
    #    #print('here')
    #    frac = 2*0.4/len(var_vec)
    #else:
    #    frac = 0.4/len(var_vec)
    frac = np.max([0.4*len(df_col2.groupby('class'))/6/len(var_vec),0.4/len(var_vec)])
    #print(frac)
    fig.legend(handles, labels, loc = 'upper center', ncol=2,bbox_to_anchor=(0.5, 1.0 + frac))
    
    fig.tight_layout()
    
    plt.figure(figsize=(10,6))
    plt.show()
