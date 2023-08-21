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
    

def load_viirs(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'viirs_2020-03-13_satdat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    data = data.loc[abs(data['tdiff']) <= t_filter]
    data['time'] = (data['time.rel'] + t_off)*3600.
    data['zi'] = data['cth']
    data.index = data['time']
     
    data['class'] = data['sat']
    return data


def load_modis(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'modis_2020-03-13_satdat.csv'
        t_off = 18.
    
    data = pd.read_csv(PATH + file)
    data = data.loc[abs(data['tdiff']) <= t_filter]
    data['time'] = (data['time.rel'] + t_off)*3600.
    data['zi'] = data['cth']
    data.index = data['time']
     
    data['class'] = data['sat']
    return data


def load_maclwp(case='20200313',t_filter = 1.,PATH='../../data_files/'):
    
    ## load coincident MAC-LWP retrievals (Elsaesser et al., 2017)
    ## __input__
    ## case........string of COMBLE date
    ## t_filter....time window around arrival of trajectory (hours)
    ## PATH........directory
    
    if case == '20200313':
        file = 'maclwp_2020-03-13_satdat2.csv'
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
    data_mac['class'] = data_mac['sat']
    return data_mac


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


def load_sims(path,var_vec_1d,var_vec_2d,t_shift = 0,keyword=''):
    
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
            label_items = label_items[0:(len(label_items)-1)]
            group = "/".join(label_items)

            #p_df = pd.DataFrame({"class": [group]* len(time), "time":time, "cwp": cwp, "rwp": rwp},index=time/3600)
            p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
            for vv in var_vec_1d:
                p_df[vv] = ds.variables[vv][:]

            ds.close()
            df_col = pd.concat([df_col,p_df])
            
        count+=1
        
    ## variables that have time and height as dimensions
    df_col2 = pd.DataFrame()
    count = 0
    for fn in NCFILES:
        if keyword in NCFILES_STR[count]:
            print(fn)
            ds = nc.Dataset(fn)
            time = ds.variables['time'][:]
            zf   = ds.variables['zf'][:]
            qv   = ds.variables['qv'][:,:]

            label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
            label_items = label_items[0:(len(label_items)-1)]
            group = "/".join(label_items)

            for ii in range(len(zf)):
                p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time, "zf": zf[ii]}, index=time/3600)      
                for vv in var_vec_2d:
                    p_df2[vv] = ds.variables[vv][:,:][:,ii]
                df_col2 = pd.concat([df_col2,p_df2])
            
        count+=1
            
    df_col['time']  = df_col['time'] + t_shift*3600.
    df_col2['time'] = df_col2['time'] + t_shift*3600.
    
    return df_col,df_col2


def plot_1d(df_col,var_vec):
    
    ## plot variables with time dependence
    ## __input__
    ## df_col.....data frame containing simulations, reanalysis, and/or observations
    ## var_vec....variables with time dependence
    
    ## 1D plots
    plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442"]
    
    counter = 0
    if 'lwp' in var_vec:
        if  'rwp' in df_col.columns and 'cwp' in df_col.columns:
            print('Computing Liquid Water Path')
            df_col['lwp'] = df_col['rwp'] + df_col['cwp']
        else:
            print('Please include rwp and cwp!')
        
    fig, axs = plt.subplots(len(var_vec),1,figsize=(5,1 + 2*len(var_vec)))
    for label, df in df_col.groupby('class'):
        if label=='MAC-LWP':
            df['lwp'] = df['lwp_bu']
        for ii in range(len(var_vec)):
            if len(var_vec) == 1:
                obj = axs
            else:
                obj = axs[ii]
            if (label=='MAC-LWP') | (label=='MODIS') | (label=='VIIRS'):
                obj.scatter(df.time/3600,df[var_vec[ii]],label=label)
            else:
                obj.plot(df.time/3600,df[var_vec[ii]],label=label)
            obj.grid(alpha=0.2)
        counter +=1
    
    i_count = 0

    if len(var_vec) > 1:
        for ax in axs.flat:
            ax.set(xlabel='Time (h)', ylabel=var_vec[i_count])
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
    
    if len(var_vec) < 2:
        top_offset = -0.2
    else:
        top_offset = 0.0
        
    fig.subplots_adjust(top=0.85 + top_offset)
    
    plt.show()


def plot_2d(df_col2,var_vec,times,z_max = 6000.):
    
    ## plot variables with time and height dependence
    ## __input__
    ## df_col2....data frame containing simulations, reanalysis, and/or observations
    ## var_vec....variables with time dependence
    ## times......list with hours of interest
    ## z_max......maximum altitude for plotting (meters)
    
    plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442"]
    
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
            df_col2['wd'] = np.arctan2(df_col2['ua']/df_col2['ws'],df_col2['va']/df_col2['ws'])*180/np.pi

            ## bring over on one side
            df_col2.loc[df_col2['wd'] < 0,'wd'] = df_col2.loc[df_col2['wd'] < 0,'wd'] + 360
        else:
            print('Please include ua and va!')

    ## apply altitude filter
    df_col2 = df_col2[df_col2['zf'] < (z_max + 500)]
    
    counter = 0
    fig, axs = plt.subplots(len(var_vec),len(times),figsize=(2*len(times),2 + 2*len(var_vec)))
    for tt in range(len(times)):
        df_sub = df_col2[round(df_col2.time) == times[tt]*3600.]    
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
                obj.plot(df[var_vec[ii]],df.zf,label=label)
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
                
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 1.1),ncol=2)
    
    fig.tight_layout()
    
    if len(var_vec) < 3:
        top_offset = -0.1
    elif len(var_vec) < 2:
        top_offset = -0.2
    else:
        top_offset = 0.0
        
    fig.subplots_adjust(top=0.9 + top_offset)
    
    plt.figure(figsize=(10,6))
    plt.show()

