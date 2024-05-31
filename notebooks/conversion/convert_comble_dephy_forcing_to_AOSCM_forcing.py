import numpy as np
import xarray as xr
import pandas as pd
import sys


dscmbl = xr.open_dataset('comble-mip/notebooks/forcing/COMBLE_INTERCOMPARISON_FORCING_V2.4.nc',decode_times=False)
length_of_traj = len(dscmbl.time.data)
print(length_of_traj)
dsscm = xr.open_dataset('/home/x_micka/comppy/runtime/scm-classic/PAPA/data/oifs/fixed/fixed_11_500hPa_new_new.nc',decode_times=False).isel(time=slice(0,length_of_traj))

ds_vertlay = xr.open_dataset('/proj/bolinc/users/x_micka/LAGRANTO_runs/convertdir/vertlay_era5.nc',decode_times=False)


# constants
R=287.0
Cp=1.005e3
Lv=2400e3
Lc=2.5152e6
g=9.81
T0 = 273.15
Rv = 461.5
eps=0.622

t_cmbl = np.squeeze(dscmbl.temp.data)
q_cmbl = np.squeeze(dscmbl.qv.data)
u_cmbl = np.squeeze(dscmbl.u.data)
v_cmbl = np.squeeze(dscmbl.v.data)

ts_cmbl = dscmbl.ts.data[0]
height_cmbl = dscmbl.lev.data

t_scm = np.zeros(dsscm.t.data.shape[1])
q_scm = np.zeros(dsscm.q.data.shape[1])
u_scm = np.zeros(dsscm.u.data.shape[1])
v_scm = np.zeros(dsscm.v.data.shape[1])
ug_scm = np.zeros(dsscm.ug.data.shape)
vg_scm = np.zeros(dsscm.vg.data.shape)

t_scm[1:] = t_cmbl
q_scm[1:] = q_cmbl

print(t_cmbl)
print(ts_cmbl)
t_scm[0] = np.interp(10,np.array([0,height_cmbl[0]]),np.array([np.squeeze(ts_cmbl),np.squeeze(t_cmbl[0])]))
u_scm[0] = np.interp(10,np.array([0,height_cmbl[0]]),np.array([0,np.squeeze(u_cmbl[0])]))
v_scm[0] = np.interp(10,np.array([0,height_cmbl[0]]),np.array([0,np.squeeze(v_cmbl[0])]))
q_scm[0] = q_scm[1]


virt_temp=t_scm*((q_scm+eps)/(eps*(1+q_scm)))
ll = dsscm.t.shape[1]
pressure_h=ds_vertlay.hyai.data+ds_vertlay.hybi.data*np.squeeze(dscmbl.ps.data)
pressure_f=0.5*(pressure_h[0:-1]+pressure_h[1::])
plnpr=np.zeros(137)
pdelp=np.zeros(137)
prdelp=np.zeros(137)
palph=np.zeros(137)
height_h=np.zeros(138)
height_f=np.zeros(137)
for hh in range(1,137):
    plnpr[hh]=np.log(pressure_h[hh+1]/pressure_h[hh])
    pdelp[hh]=pressure_h[hh+1]-pressure_h[hh]
    prdelp[hh]=1/pdelp[hh]
    palph[hh]=1-pressure_h[hh]*prdelp[hh]*plnpr[hh]

for hh in range(136,137-ll-1,-1):
    height_h[hh]=height_h[hh+1]+R*virt_temp[hh-(137-ll)]*plnpr[hh]
    height_f[hh]=height_h[hh+1]+R*virt_temp[hh-(137-ll)]*palph[hh]
    dsscm['height_h'].data[0,:] = height_h/g
    dsscm['height_f'].data[0,:] = height_f/g
    dsscm['pressure_h'].data[0,:] = pressure_h
    dsscm['pressure_f'].data[0,:] = pressure_f

print('height:',dsscm['height_f'].data[0,:])

dsscm['t'].data[:,:] = 0 
dsscm['q'].data[:,:] = 0 
dsscm['u'].data[:,:] = 0
dsscm['v'].data[:,:] = 0 
dsscm['omega'].data[:,:] = 0
dsscm['ug'].data[:,:] = 0
dsscm['vg'].data[:,:] = 0
dsscm['pot_temperature'].data[:,:] = 0
dsscm['ql'].data[:,:] = 0
dsscm['qi'].data[:,:] = 0
dsscm['qr'].data[:,:] = 0
dsscm['qsn'].data[:,:] = 0
dsscm['cloud_fraction'].data[:,:] = 0
dsscm['pot_temp_e'].data[:,:] = 0
dsscm['dry_st_energy'].data[:,:] = 0
dsscm['moist_st_energy'].data[:,:] = 0
dsscm['relative_humidity'].data[:,:] = 0
dsscm['tadv'].data[:,:] = 0
dsscm['qadv'].data[:,:] = 0
dsscm['cladv'].data[:,:] = 0
dsscm['ciadv'].data[:,:] = 0
dsscm['ccadv'].data[:,:] = 0
dsscm['csadv'].data[:,:] = 0
dsscm['cradv'].data[:,:] = 0
dsscm['uadv'].data[:,:] = 0
dsscm['vadv'].data[:,:] = 0
dsscm['lsm'].data[:] = 0
dsscm['q_skin'].data[:] = 0
dsscm['mom_rough'].data[:] = 0
dsscm['heat_rough'].data[:] = 0

dsscm['t'].data[0,:] = np.flipud(np.interp(np.flipud(dsscm.height_f.data[0,:]),height_cmbl,t_cmbl))
dsscm['q'].data[0,:] = np.flipud(np.interp(np.flipud(dsscm.height_f.data[0,:]),height_cmbl,q_cmbl))
dsscm['u'].data[0,:] = np.flipud(np.interp(np.flipud(dsscm.height_f.data[0,:]),height_cmbl,u_cmbl))
dsscm['v'].data[0,:] = np.flipud(np.interp(np.flipud(dsscm.height_f.data[0,:]),height_cmbl,v_cmbl))

# time-space loop
for i in range(length_of_traj):
    ug_cmbl = np.squeeze(dscmbl.ug.data[i,:])
    vg_cmbl = np.squeeze(dscmbl.vg.data[i,:])
    ug_scm[i,1:] = ug_cmbl
    vg_scm[i,1:] = vg_cmbl
    ug_scm[i,0] = ug_scm[i,1]
    vg_scm[i,0] = vg_scm[i,1]
    dsscm['ug'].data[i,:] = np.flipud(np.interp(np.flipud(dsscm.height_f.data[0,:]),height_cmbl,ug_cmbl))
    dsscm['vg'].data[i,:] = np.flipud(np.interp(np.flipud(dsscm.height_f.data[0,:]),height_cmbl,vg_cmbl))

dsscm['open_sst'].data[:] = np.squeeze(dscmbl.ts.data)
dsscm['t_skin'].data[:] = np.squeeze(dscmbl.ts.data)
dsscm['ps'].data[:] = np.squeeze(dscmbl.ps.data)
dsscm['lat'].data[:] = np.squeeze(dscmbl.lat.data)
dsscm['lon'].data[:] = np.squeeze(dscmbl.lon.data)
dsscm['time'].data[:] = np.squeeze(dscmbl.time.data)


UTC = 22
time_sel = length_of_traj
hour0 = UTC
time_incr = np.ones(int(np.ceil(time_sel/24)))*3600
time_incr = time_incr[:,np.newaxis] * np.arange(24)
time_incr = np.reshape(time_incr,[time_incr.shape[0]*time_incr.shape[1]])
hour = np.zeros(time_incr.shape)
hour[0:-hour0] = time_incr[hour0:]
hour[-hour0::] = time_incr[0:hour0]
hour = hour[0:time_sel]

dsscm['hour'].data =  hour
dsscm['second'].data[:] = dsscm.hour.data
print(dsscm['hour'].data[0]/3600) 
print(dsscm['height_f'].data[-1,:])
print(dsscm['pressure_f'].data[-1,:])

dsscm.to_netcdf('/proj/bolinc/users/x_micka/comble-mip/notebooks/forcing/COMBLE_scm_in.nc')
