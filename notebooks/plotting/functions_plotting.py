import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import glob
import os


def sat_pres(x):
    ## Bolton (1980), return in hPa
    return 6.112*np.exp(17.67*x/(243.5 + x))
    
## load radiosonde obs
# scan for all available ones
def load_rs(case='20200313',t_filter = 1.,PATH='../../data_files/'):

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
    
    ## load ERA5 file
    if case =='20200313':
        fn = PATH + 'theta_temp_rh_sh_uvw_sst_along_trajectory_era5ml_28h_end_2020-03-13-18.nc'
        t_offset = 18.

    print(fn)
    ds = nc.Dataset(fn)
    
    ## extract 1D and 2D fields
    var_vec_1d = ['SST']
    var_vec_2d = ['U','V','W','Theta','GEOS_HT']
    var_vec_2d_trans = ['ua','va','w','theta','zf']
    group = 'ERA5'
    
    time = (ds.variables['Time'][:] + t_offset)*3600
    pf = ds.variables['Pressure'][:] 
    
    p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
    for vv in var_vec_1d:
        p_df[vv] = ds.variables[vv][:]
       
    df_col2 = pd.DataFrame()
    for ii in range(len(pf)):
        p_df2 = pd.DataFrame({"class": [group]* len(time), "time":time, "pf": pf[ii]}, index=time/3600)      
        for vv in var_vec_2d:
            vv_trans = var_vec_2d_trans[var_vec_2d.index(vv)]
            p_df2[vv_trans] = ds.variables[vv][:,:][:,ii]
        df_col2 = pd.concat([df_col2,p_df2])
        
    ds.close()

    return p_df,df_col2        


def load_sims(path,var_vec_1d,var_vec_2d):
    
    #print(direc)
    #print(var_vec)
    direc = pathlib.Path(path)
    NCFILES = list(direc.rglob("*nc"))
    
    ## variables that only have time as dimension
    df_col = pd.DataFrame()
    for fn in NCFILES:
        print(fn)
        ds = nc.Dataset(fn)
        time = ds.variables['time'][:]
        lwp  = ds.variables['lwp'][:]
        rwp  = ds.variables['rwp'][:]
        
        label_items = [x for x in fn.parts + direc.parts if x not in direc.parts]
        label_items = label_items[0:(len(label_items)-1)]
        group = "/".join(label_items)
        
        #p_df = pd.DataFrame({"class": [group]* len(time), "time":time, "lwp": lwp, "rwp": rwp},index=time/3600)
        p_df = pd.DataFrame({"class": [group]* len(time), "time":time}, index=time/3600)
        for vv in var_vec_1d:
            p_df[vv] = ds.variables[vv][:]
        
        ds.close()
        df_col = pd.concat([df_col,p_df])
        
    ## variables that have time and height as dimensions
    df_col2 = pd.DataFrame()
    for fn in NCFILES:
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
            
    return df_col,df_col2


def plot_1d(df_col,var_vec):

    ## 1D plots
    plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442"]
    
    counter = 0
    
    fig, axs = plt.subplots(len(var_vec),1,figsize=(5,1 + 2*len(var_vec)))
    for label, df in df_col.groupby('class'):
        for ii in range(len(var_vec)):
            if len(var_vec) == 1:
                obj = axs
            else:
                obj = axs[ii]
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
    fig.legend(by_label.values(), by_label.keys(),loc='upper center')

    fig.tight_layout()
    
    if len(var_vec) < 2:
        top_offset = -0.2
    else:
        top_offset = 0.0
        
    fig.subplots_adjust(top=0.85 + top_offset)
    #                    wspace=0.0,
    #                    hspace=0.4)
    
    #fig.figure(figsize=(5,4*len(var_vec)))
    plt.show()


def plot_2d(df_col2,var_vec,times,z_max = 5000.):
    
    plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442"]

    ## apply altitude filter
    df_col2 = df_col2[df_col2['zf'] < (z_max + 500)]
    
    counter = 0
    fig, axs = plt.subplots(len(var_vec),len(times),figsize=(2*len(times),2 + 2*len(var_vec)))
    for tt in range(len(times)):
        df_sub = df_col2[round(df_col2.time) == times[tt]*3600.]    
        for label, df in df_col2.groupby('class'):
            df = df[round(df.time) == times[tt]*3600.]  
            for ii in range(len(var_vec)):                
                if len(var_vec) == 1 & len(times) == 1:
                    obj = axs
                elif len(var_vec) == 1:
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
    fig.legend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 1.1))
    
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

