# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:22:38 2021

@author: Steve Kim
"""

from pr2_utils import *
import numpy as np
import matplotlib.pyplot as plt

lidar_TS, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
fog_TS, fog_data = read_data_from_csv('data/sensor_data/fog.csv')
fog_data_z = fog_data[:,2] # yaw at time fog_TS
encoder_TS, encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')

#align sensors by indices to use in loop - lidar is base
fog_sync_indices = []
encoder_sync_indices = []
prev_indexf = 0
prev_indexe = 0
for ts in lidar_TS:
  tempf = np.argmin(np.abs(ts - fog_TS[prev_indexf:prev_indexf+15]))
  prev_indexf += tempf
  fog_sync_indices.append(prev_indexf)
  tempe = np.argmin(np.abs(ts - encoder_TS[prev_indexe:prev_indexe+15]))
  prev_indexe += tempe
  encoder_sync_indices.append(prev_indexe)

#initialize map and particles
MAP, particles = init_map()
Plot = np.zeros((MAP['map'].shape[0],MAP['map'].shape[1],3),dtype=np.uint8)
particles_movement = []

#lidar angles
lidar_angles = np.linspace(-5, 185, 286) / 180 * np.pi
copy_angles = lidar_angles.copy()

#cum sum of theta
cumulative_yaw = np.cumsum(fog_data_z)
#velocity
left_wheel = (encoder_data[:,0][1:] - encoder_data[:,0][:-1]) * (np.pi * 0.623479) /4096
right_wheel = (encoder_data[:,1][1:] - encoder_data[:,1][:-1]) * (np.pi * 0.622806) / 4096

left_wheel_velocities = left_wheel / ((encoder_TS[1:] - encoder_TS[:-1]).T ) * 1e9
right_wheel_velocities = right_wheel /((encoder_TS[1:] - encoder_TS[:-1]).T) *  1e9


#yaw data init
for i in range(len(fog_data)):
  pass
x_d = 0
y_d = 0
for scan in range(1, len(lidar_data)):
  #get sync-ed sensor measurements
  lidar_scan_time = lidar_TS[scan]
  fog_scan_time_idx = fog_sync_indices[scan]
  fog_scan_time_idx_prev = fog_sync_indices[scan-1]
  encoder_scan_time_idx = encoder_sync_indices[scan]
  encoder_scan_time_idx_prev = encoder_sync_indices[scan-1]
  # set orientation of particles
  particles_orientation = np.copy(particles['orientation'])[:, np.argmax(particles['weights'])]
  particles_movement.append(particles_orientation)

  #get motion model
  velocity =( left_wheel_velocities[scan] + right_wheel_velocities[scan]) / 2
  theta = cumulative_yaw[fog_scan_time_idx]
  delta_theta = cumulative_yaw[fog_scan_time_idx] - cumulative_yaw[fog_scan_time_idx_prev]
  tau = (lidar_TS[scan] - lidar_TS[scan-1])/1.0e9
  #lidar
  ranges = lidar_data[scan] #first row
  lidar_scan = lidar_data[scan]
  idxValid = np.logical_and((ranges < 75),(ranges > 2))
  lidar_scan_valid = ranges[idxValid]
  valid_angles =  copy_angles[idxValid.T]
  #change lidar scans from polar to cartesian. Lidar scans give distance per angle,
  #and we have angle defined above
  #do all rotations
  x_lidar,y_lidar = polar_to_cartesian(lidar_scan_valid[:,np.newaxis], valid_angles[:,np.newaxis])
  fog = lidar2fog(x_lidar, y_lidar)
  if scan == 1:
    world = fog2world(0,particles_orientation,fog)[:2]
    update_log_odds(MAP,world[0:2,:],particles_orientation[0:2])
    continue
  world = fog2world(theta, particles_orientation, fog)[:2]
  update_log_odds(MAP, world, particles_orientation)
  motion_update(particles, delta_theta, theta, tau, velocity)
  #update particles
  update_weights(MAP, lidar_scan, copy_angles, particles)
  if scan % 1000 == 0 or scan==len(lidar_data)-1 or scan == 1:
    print('step: ',scan)
    # plt.savefig('figure_{}.png'.format(scan), dpi = 150)
    plotting(MAP, particles_movement, world, Plot,scan)
  elif scan==len(lidar_data)-1:
    plotting(MAP, particles_movement, world, Plot,scan)



