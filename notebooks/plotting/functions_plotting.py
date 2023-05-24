import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import glob
import os

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
    fig, axs = plt.subplots(len(var_vec),1)
    for label, df in df_col.groupby('class'):
        for ii in range(len(var_vec)):
            axs[ii].plot(df.time/3600,df[var_vec[ii]],label=label)
        counter +=1
    
    i_count = 0
    for ax in axs.flat:
        ax.set(xlabel='Time (h)', ylabel=var_vec[i_count])
        i_count += 1
        
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    # Add a legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc='upper center')
    
    plt.show()


def plot_2d(df_col2,var_vec,times):
    
    plot_colors = ["#E69F00", "#56B4E9", "#009E73","#0072B2", "#D55E00", "#CC79A7","#F0E442"]
    
    counter = 0
    fig, axs = plt.subplots(len(var_vec),len(times))
    for tt in range(len(times)):
        df_sub = df_col2[round(df_col2.time) == times[tt]*3600.]    
        for label, df in df_sub.groupby('class'):
            for ii in range(len(var_vec)):
                axs[ii,tt].plot(df[var_vec[ii]],df.zf,label=label)
                if ii==0:
                    axs[ii,tt].set_title(str(times[tt])+'h')
                if tt==0:
                    axs[ii,tt].set(ylabel='Altitude (km)', xlabel=var_vec[ii])
                else:
                    axs[ii,tt].set(xlabel=var_vec[ii])
                counter +=1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 1.1))
    
    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    
    plt.show()

