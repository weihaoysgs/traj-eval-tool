import os
import yaml
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from colorama import init, Fore

import traj_lib.trajectory_utils as traj_utils
import traj_lib.trajectory_loading as traj_loading
import traj_lib.results_writer as res_writer
import traj_lib.compute_trajectory_errors as traj_err
import traj_lib.align_utils as au
import traj_lib.plot_utils as pu
import traj_lib.transformations as tf

init(autoreset=True)
rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

class Trajectory:
  def __init__(self, traj_dir='', platform='', alg_name='', dataset_name='',
                 align_type='sim3', align_num_frames=-1,
                 preset_boxplot_distances=[]):
    self.traj_dir = traj_dir
    # TODO: add other parameters
    self.align_type = align_type
    self.align_num_frames = align_num_frames
    
    self.data_aligned = False
    self.abs_errors_calculated = False

    self.load_trajectory()

  def load_trajectory(self):
    self.t_es, self.p_es, self.q_es, self.t_gt, self.p_gt, self.q_gt =\
            traj_loading.load_stamped_dataset(self.traj_dir)
  
  def align_trajectory(self):
    if (self.data_aligned):
      print("Trajectory already aligned.")
      return
    
    print("Aliging the trajectory estimate to the groundtruth...")
    print("Alignment type is {0}.".format(self.align_type))
    
    n = int(self.align_num_frames)
    if n < 0.0:
      print('To align all frames.')
      n = len(self.p_es)
    else:
      print('To align trajectory using ' + str(n) + ' frames.')

    self.trans = np.zeros((3,))
    self.rot = np.eye(3)
    self.scale = 1.0

    self.scale, self.rot, self.trans = au.alignTrajectory(
                self.p_es, self.p_gt, self.q_es, self.q_gt, 
                self.align_type, self.align_num_frames)

    print("Alignment done.")
    print("Align scale: {0}".format(self.scale))
    print("Align rotation: \n", self.rot)
    print("Align translation: ", self.trans)
    
    self.p_es_aligned = np.zeros(np.shape(self.p_es))
    self.q_es_aligned = np.zeros(np.shape(self.q_es))

    for i in range(np.shape(self.p_es)[0]):
      self.p_es_aligned[i, :] = self.scale * self.rot.dot(self.p_es[i, :]) + self.trans
      q_es_R = self.rot.dot(tf.quaternion_matrix(self.q_es[i, :])[0:3, 0:3])
      q_es_T = np.identity(4)
      q_es_T[0:3, 0:3] = q_es_R
      self.q_es_aligned[i, :] = tf.quaternion_from_matrix(q_es_T)
    self.data_aligned = True
    print("Trajectory alignment done.")

  def compute_absolute_error(self):
    if self.abs_errors_calculated:
      print("Absolute errors already calculated")
      return
    print('Calculating RMSE...')
    self.align_trajectory()
    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc =\
                traj_err.compute_absolute_error(self.p_es_aligned,
                                                self.q_es_aligned,
                                                self.p_gt,
                                                self.q_gt)
    stats_trans = res_writer.compute_statistics(e_trans)
    stats_rot = res_writer.compute_statistics(e_rot)
    stats_scale = res_writer.compute_statistics(e_scale_perc)
    print(Fore.GREEN + "Stats translation RMSE")
    res_writer.print_format_stats(stats_trans)
    print(Fore.GREEN + "Stats rotation RMSE")
    res_writer.print_format_stats(stats_rot)
    print(Fore.GREEN + "Stats scale RMSE")
    res_writer.print_format_stats(stats_scale)

def main():
  parser = argparse.ArgumentParser(
        description='''Analyze trajectory estimate in a folder.''')
  parser.add_argument(
        'traj_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
  args = parser.parse_args()

  trajectory = Trajectory(args.traj_dir);
  trajectory.align_trajectory()
  trajectory.compute_absolute_error()

  fig = plt.figure(figsize=(6, 5.5))
  ax = fig.add_subplot(111, aspect='equal',
                        xlabel='x [m]', ylabel='y [m]')
  pu.plot_trajectory_side(ax, trajectory.p_es_aligned, 'b', 'Estimate')
  pu.plot_trajectory_side(ax, trajectory.p_gt, 'm', 'Groundtruth')
  pu.plot_aligned_side(ax, trajectory.p_es_aligned, trajectory.p_gt,
                      trajectory.align_num_frames)
  plt.legend(loc=1, borderaxespad=0.)
  fig.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
  print("eval trajectory")

