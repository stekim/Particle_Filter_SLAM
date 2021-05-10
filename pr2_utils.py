import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.special import softmax

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def compute_stereo():
  path_l = 'data/image_left.png'
  path_r = 'data/image_right.png'

  image_l = cv2.imread(path_l, 0)
  image_r = cv2.imread(path_r, 0)

  image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
  image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

  image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
  image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

  # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
  stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9)
  disparity = stereo.compute(image_l_gray, image_r_gray)

  fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
  ax1.imshow(image_l)
  ax1.set_title('Left Image')
  ax2.imshow(image_r)
  ax2.set_title('Right Image')
  ax3.imshow(disparity, cmap='gray')
  ax3.set_title('Disparity Map')
  plt.show()


def read_data_from_csv(filename):
  '''
  INPUT
  filename        file address

  OUTPUT
  timestamp       timestamp of each observation
  data            a numpy array containing a sensor measurement in each row
  '''
  data_csv = pd.read_csv(filename, header=None)
  data = data_csv.values[:, 1:]
  timestamp = data_csv.values[:, 0]
  return timestamp, data


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT
  im              the map
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)
  xs,ys           physical x,y,positions you want to evaluate "correlation"

  OUTPUT
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))



def test_mapCorrelation():
  _, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
  angles = np.linspace(-5, 185, 286) / 180 * np.pi
  ranges = lidar_data[0, :]

  # take valid indices
  indValid = np.logical_and((ranges < 80),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]

  # init MAP
  MAP = {}
  MAP['res']   = 2 #meters
  MAP['xmin']  = -1000  #meters
  MAP['ymin']  = -1000
  MAP['xmax']  =  1000
  MAP['ymax']  =  1000
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

  #import pdb
  #pdb.set_trace()

  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)

  # convert position in the map frame here
  Y = np.stack((xs0,ys0))

  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

  # build an arbitrary map
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood],yis[indGood]]=1

  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  x_range = np.arange(-0.4,0.4+0.1,0.1)
  y_range = np.arange(-0.4,0.4+0.1,0.1)



  print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
  ts = tic()
  c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
  print(c)
  toc(ts,"Map Correlation")

  c_ex = np.array([[ 4.,  6.,  6.,  5.,  8.,  6.,  3.,  2.,  0.],
                   [ 7.,  5., 11.,  8.,  5.,  8.,  5.,  4.,  2.],
                   [ 5.,  7., 11.,  8., 12.,  5.,  2.,  1.,  5.],
                   [ 6.,  8., 13., 66., 33.,  4.,  3.,  3.,  0.],
                   [ 5.,  9.,  9., 63., 55., 13.,  5.,  7.,  4.],
                   [ 1.,  1., 11., 15., 12., 13.,  6., 10.,  7.],
                   [ 2.,  5.,  7., 11.,  7.,  8.,  8.,  6.,  4.],
                   [ 3.,  6.,  9.,  8.,  7.,  7.,  4.,  4.,  3.],
                   [ 2.,  3.,  2.,  6.,  8.,  4.,  5.,  5.,  0.]])

  if np.sum(c==c_ex) == np.size(c_ex):
	  print("...Test passed.")
  else:
	  print("...Test failed. Close figures to continue tests.")

  #plot original lidar points
  fig1 = plt.figure()
  plt.plot(xs0,ys0,'.k')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Laser reading")
  plt.axis('equal')

  #plot map
  fig2 = plt.figure()
  plt.imshow(MAP['map'],cmap="hot");
  plt.title("Occupancy grid map")

  #plot correlation
  fig3 = plt.figure()
  ax3 = fig3.gca(projection='3d')
  X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
  ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
  plt.title("Correlation coefficient map")

  plt.show()

def show_map(MAP):
  _, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
  angles = np.linspace(-5, 185, 286) / 180 * np.pi
  ranges = lidar_data[0, :]

  # take valid indices
  indValid = np.logical_and((ranges < 75),(ranges > 2))
  ranges = ranges[indValid]
  angles = angles[indValid]
  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)

  # convert position in the map frame here
  # Y = np.stack((xs0,ys0))

  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

  # build an arbitrary map
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood],yis[indGood]]=1

  # x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  # y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  # x_range = np.arange(-0.4,0.4+0.1,0.1)
  # y_range = np.arange(-0.4,0.4+0.1,0.1)
  #plot original lidar points
  # fig1 = plt.figure()
  plt.plot(xs0,ys0,'.k')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Laser reading")
  plt.axis('equal')

  #plot map
  # fig2 = plt.figure()
  plt.imshow(MAP['map'],cmap="hot");
  plt.title("Occupancy grid map")

  plt.show()

def show_lidar():
  _, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
  angles = np.linspace(-5, 185, 286) / 180 * np.pi
  ranges = lidar_data[0, :]
  plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(80)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()

#### my utlity functions

def init_map():
    MAP = {}
    lidar_const = 0.8
    MAP['res']   = 1 #meters
    MAP['xmin']  = -50  #meters
    MAP['ymin']  = -50
    MAP['xmax']  =  1500
    MAP['ymax']  =  1500
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
    MAP['occupied'] = np.log(lidar_const)
    MAP['open'] = np.log(1-lidar_const)
    #log odds for occupied point
    MAP['occ_d'] = np.log(16)
    MAP['free_d'] = -np.log(16)
    #particles
    particles = {}
    particles['qty'] = 30 # number of particles
    particles['weights'] = np.ones(particles['qty'])/particles['qty']
    particles['orientation'] = np.zeros((3, particles['qty']))

    return MAP, particles

def update_map_log_odds(MAP, world, orientation):
  cells = world2cells()

def world2cells(points,Map):
    xis = np.ceil((points[0,:] - Map['xmin']) / Map['res'] ).astype(np.int16)-1
    yis = np.ceil((-points[1,:] - Map['ymin']) / Map['res'] ).astype(np.int16)-1

    return np.vstack((xis,yis))


def lidar2fog( x_lidar, y_lidar):
  # Lidar sensor (LMS511) extrinsic calibration parameter from vehicle
  # RPY(roll/pitch/yaw = XYZ extrinsic, degree),
  # R(rotation matrix),
  # T(translation matrix)
  # RPY: 142.759 0.0584636 89.9254s
  # R: 0.00130201 0.796097 0.605167
  #    0.999999 -0.000419027 -0.00160026
  #   -0.00102038 0.605169 -0.796097
  # T: 0.8349 -0.0126869 1.76416
  # rotation_matrix = [[0.00130201, 0.796097, 0.605167],
  #                    [0.999999, -0.000419027, -0.00160026],
  #                    [-0.00102038, 0.605169, -0.796097]]
  # translation = [0.8349, -0.0126869, 1.76416]  #lidar to vehicle
  b_T_l =np.array([[0.00130201, 0.796097, 0.605167, 0.8349],
          [0.999999, -0.000419027, -0.00160026, -0.0126869],
          [-0.00102038, 0.605169, -0.796097, 1.76416],
          [0, 0, 0, 1]])  #lecture 6 slide 23
  # print('here is x_lidar shape', x_lidar.shape)
  x_lidar = x_lidar.T
  y_lidar = y_lidar.T
  position = np.vstack((x_lidar, y_lidar))
  position = np.vstack((position, np.zeros((1,x_lidar.shape[1])),np.ones((1,x_lidar.shape[1] ))))
  body = np.dot(b_T_l, position)

  rotation_matrix2 = np.array([[1, 0, 0, -0.335 ],
                    [0, 1, 0, -0.0350],
                    [0, 0, 1, 0.78],
                    [0, 0, 0, 1]
                    ])

  to_fog = np.dot(rotation_matrix2, body)

  return to_fog

def fog2world(theta, particles_orientation, to_fog):
  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, particles_orientation[0]],
                    [np.sin(theta), np.cos(theta), 0, particles_orientation[1]],
                    [0, 0, 1, 0.78],
                    [0, 0, 0, 1]
                    ])
  to_world = np.dot(rotation_matrix,to_fog )
  return to_world

def update_log_odds(Map,world,orientation):
    # tstart = tic()
    cells= world2cells(world,Map)
    orientation=world2cells(orientation[:,np.newaxis],Map)
    occupied=np.hstack((cells,orientation))
    occupied=np.array([occupied]).T
    Map['map'][cells[1,:],cells[0,:]]=Map['map'][cells[1,:],cells[0,:]]+2*Map['occ_d']-Map['free_d']
    new_grid_contour=np.zeros(Map['map'].shape)
    cv2.drawContours(image=new_grid_contour, contours = [(occupied.astype(int))], contourIdx = -1, color = Map['free_d'], thickness=1)
    Map['map'] += new_grid_contour.astype(dtype='int8')
    Map['map'][Map['map']>120] = 120
    Map['map'][Map['map']<-120] = -120
    # toc(tstart, 'update log odds')

def encoder_distance(encoder_data, encoder_TS, encoder_idx, prev_encoder_idx):
  # :param encoder_data: ticks from the wheels
  # :param encoder_idx: which encoder entry we're currently looking at, synced with lidar
  # Encoder resolution: 4096
  # Encoder left wheel diameter: 0.623479
  # Encoder right wheel diameter: 0.622806
  # Encoder wheel base: 1.52439
  # :return:  distance traveled left and right wheels, type float
  left_encoder_count = encoder_data[encoder_idx][0] - encoder_data[prev_encoder_idx][0]
  right_encoder_count = encoder_data[encoder_idx][1] - encoder_data[prev_encoder_idx][1]
  delta_tau  = encoder_TS[encoder_idx] - encoder_TS[prev_encoder_idx]
  left_circumference = np.pi*0.623479
  right_circumference = np.pi*0.622806
  resolution = 4096
  d_left =left_circumference*left_encoder_count/resolution
  d_right =right_circumference*right_encoder_count/resolution
  return d_left, d_right, delta_tau

def polar_to_cartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def motion_update(Particles, delta_theta, theta, tau, velocity):
  # tstart = tic()
  for i in range(Particles['qty']):
      x = Particles['orientation'][0,i] + tau * velocity*np.cos(theta)
      y = Particles['orientation'][1,i] + tau * velocity*np.sin(theta)
      theta = Particles['orientation'][2,i] + delta_theta
      xt=abs(x)
      yt=abs(y)
      if xt>0 or yt>0:
          noise_x = np.random.normal(0,.0002,1)
          noise_y=np.random.normal(0,.0002,1)
          noise_theta=np.random.normal(0,0.0001,1)
      else:
          noise_x=0
          noise_y=0
          noise_theta=0
      Particles['orientation'][0,i] = x + noise_x
      Particles['orientation'][1,i] = y + noise_y
      Particles['orientation'][2,i] = theta + noise_theta

def update_weights(MAP, lidar_scan, lidar_angles, Particles):
  x_l, y_l = polar_to_cartesian(lidar_scan, lidar_angles)
  lidar_scan = lidar_scan[np.newaxis,:]
  lidar_scan=np.vstack((x_l,y_l,np.zeros((1,lidar_scan.shape[1]))))
  lidar_scan=np.vstack((lidar_scan,np.ones((1,lidar_scan.shape[1]))))
  orientation = Particles['orientation']
  lidar = np.zeros((4,lidar_scan.shape[1],Particles['qty']))
  b_T_l =np.array([[0.00130201, 0.796097, 0.605167, 0.8349],
             [0.999999, -0.000419027, -0.00160026, -0.0126869],
             [-0.00102038, 0.605169, -0.796097, 1.76416],
             [0, 0, 0, 1]])
  # start_time = tic()
  for i in range(Particles['qty']):
    w_T_b = np.array([[np.cos(orientation[2,i]), -np.sin(orientation[2,i]),0,orientation[0,i]],
                        [np.sin(orientation[2,i]),np.cos(orientation[2,i]),0,orientation[1,i]],
                   [0, 0, 1, 0.78],
                   [0, 0, 0, 1]
                   ])
    to_world = np.dot(w_T_b, b_T_l )
    # print('shape of lidar_scan', lidar_scan.shape)
    # print('shape of to_world', to_world.shape)
    temp_lidar = np.matmul(to_world, lidar_scan  )
    #filtering out ground reads
    # not_floor = temp_lidar[2] > 0.1
    # temp_lidar=temp_lidar[:, not_floor]
    lidar[:,:,i] = temp_lidar
    '''finding the correlation'''
    #creating a binary map of the log-odds map
  new_map = np.where(MAP['map'] > 0, 1, 0)
  corr = np.zeros(Particles['qty'])
  for i in range(Particles['qty']):
      occ = world2cells(lidar[:2,:,i], MAP)
      corr[i] = np.sum(new_map[occ[1],occ[0]])
  # toc(start_time, 'correlation pre softmax ')
  corr = softmax(corr)
  Particles['weights']= corr
  n_eff =1/np.sum(np.square(Particles['weights']))
  n_thresh=0.6*100
  if n_eff< n_thresh:
      # print('running resampling')
      particle_resampling(Particles)

def particle_resampling (Particles):
  # start_time = tic()
  new_particle=np.ones(Particles['orientation'].shape)
  uniform_dist=np.random.uniform(0,1/Particles['qty'])
  j = 0;
  c = Particles['weights'][0]
  for k in range(0,Particles['qty']):
      beta=uniform_dist+((k)/Particles['qty'])
      while (beta>c and c<=np.sum(Particles['weights'])):
          j=j+1
          c=c+Particles['weights'][j]
      new_particle[:,k]=Particles['orientation'][:,j]
  Particles['orientation']=new_particle
  Particles['weights']=np.ones(Particles['qty'])/Particles['qty']

def plotting(Map, Trajectory, Lidar, Plot, i):
  occ = Map['map']>0
  free = Map['map']<0
  none=Map['map']==0
  Plot[occ] = [139,0,0]
  Plot[free] = [245,245,245]
  Plot[none] = [169,169,169]
  traj = np.asarray(Trajectory)[:,:2]
  traj_pixel = world2cells(traj.T, Map)

  Plot[traj_pixel[-1,1],traj_pixel[-1,0]] = [0,0,255] #red color for particle
  # toc(start_time, 'what we think is taking a long time. traj_pixel[0] and traj_pixel[1] are ' +str(traj_pixel[-1,1]) + str(traj_pixel[-1,0]) )
  plt.imshow(Plot)
  # plt.scatter(traj_pixel[0],traj_pixel[1],color='r',s=0.00002)
  plt.savefig(r'C:\\Users\\s\\Google Drive\\UCSD GRADUATE CLASSES\\2021 WINTER\\ECE 276A\\pr2\\ECE276A_PR2\\code'+str(i)+'.png',dpi=1100)
  plt.show()

# if __name__ == '__main__':
  # compute_stereo()
  # show_lidar()
  # test_mapCorrelation()
  # test_bresenham2D()

