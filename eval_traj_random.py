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

def generate_random_colors(length):
    return np.random.rand(length, 1)

class Trajectory:
  def __init__(self, traj_dir='', align_type='sim3', align_num_frames=-1, gt_traj_path='', estimate_traj_path=''):
    self.traj_dir = traj_dir
    # TODO: add other parameters
    self.align_type = align_type
    self.align_num_frames = align_num_frames
    
    self.data_aligned = False
    self.abs_errors = {}

    self.gt_traj_path = gt_traj_path
    self.est_traj_path = estimate_traj_path

    self.load_trajectory()

  def load_trajectory(self):
    # self.t_es, self.p_es, self.q_es, self.t_gt, self.p_gt, self.q_gt =\
    #         traj_loading.load_stamped_dataset(self.traj_dir)
    self.t_es, self.p_es, self.q_es, self.t_gt, self.p_gt, self.q_gt =\
            traj_loading.load_stamped_dataset_from_file(self.gt_traj_path, self.est_traj_path)

    self.accum_distances = traj_utils.get_distance_from_start(self.p_gt)
    self.traj_length = self.accum_distances[-1]

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
    if self.abs_errors:
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
    # print(Fore.GREEN + "Stats rotation RMSE")
    # res_writer.print_format_stats(stats_rot)
    # print(Fore.GREEN + "Stats scale RMSE")
    # res_writer.print_format_stats(stats_scale)

    self.abs_errors['abs_e_trans'] = e_trans
    self.abs_errors['abs_e_trans_stats'] = stats_trans

    self.abs_errors['abs_e_trans_vec'] = e_trans_vec

    self.abs_errors['abs_e_rot'] = e_rot
    self.abs_errors['abs_e_rot_stats'] = stats_rot

    self.abs_errors['abs_e_ypr'] = e_ypr

    self.abs_errors['abs_e_scale_perc'] = e_scale_perc
    self.abs_errors['abs_e_scale_stats'] = stats_scale

def simple_draw_legend(ax):
  # Simplify legends and avoid duplication
  handles, labels = ax.get_legend_handles_labels()
  unique_labels = list(dict.fromkeys(labels))  
  unique_handles = [handles[labels.index(lbl)] for lbl in unique_labels]
  ax.legend(unique_handles, unique_labels, loc=1)
  
def plot_trajectory_3d(ax, pos, color, name, alpha=1.0):
  # ax.grid(ls='--', color='0.7')
  ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
        color=color, linestyle='-', 
        alpha=alpha, label=name)

def plot_3d_traj(eval_traj_list, traj_name_list, save_dir=""):
 
  fig = plt.figure(figsize=(6, 5.5))
  colors = generate_random_colors(len(traj_name_list))
  ax = fig.add_subplot(111, projection='3d',
                      xlabel='x [m]', 
                      ylabel='y [m]',
                      zlabel='z [m]')
  ax.set_xlabel('X [m]', fontsize=14)
  ax.set_ylabel('Y [m]', fontsize=14)
  ax.set_zlabel('Z [m]', fontsize=14)

  ax.grid(ls='--', color='0.7')
  ax.plot(eval_traj_list[0].p_gt[:, 0], eval_traj_list[0].p_gt[:, 1], eval_traj_list[0].p_gt[:, 2], 'm', 
    linestyle='--', alpha=1.0, label='Groundtruth')

  # draw all trajectories
  for i, traj in enumerate(eval_traj_list):
      ax.plot(traj.p_es_aligned[:, 0], 
              traj.p_es_aligned[:, 1],  
              traj.p_es_aligned[:, 2],  
              # color=colors[i], 
              linestyle='-', 
              alpha=0.7, 
              label=traj_name_list[i])

  simple_draw_legend(ax)
  
  fig.tight_layout()
  plt.show()
  fig.savefig(save_dir + "/3d_traj_compare.pdf", bbox_inches="tight")

def plot_2d_traj_xyz(eval_traj_list, traj_name_list, save_dir=""):
  alpha = 0.7
  legend_font_size = 14
  colors = generate_random_colors(len(traj_name_list))
  fig = plt.figure(figsize=(6, 5.5))

  ax = fig.add_subplot(311)
  ax.grid(ls='--', color='0.7')
  ax.set_xlabel('accum_distances [m]', fontsize=14)
  ax.set_ylabel('X [m]', fontsize=14)
  ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_gt[:, 0], 'm', linestyle='--', alpha=1.0, label='Groundtruth')
  for i, trajectory in enumerate(eval_traj_list):
    ax.plot(eval_traj_list[i].accum_distances, trajectory.p_es_aligned[:, 0], colors[i], linestyle='-', alpha=alpha, label=traj_name_list[i])

  simple_draw_legend(ax)
  
  ax = fig.add_subplot(312)
  ax.grid(ls='--', color='0.7')
  ax.set_xlabel('accum_distances [m]', fontsize=14)
  ax.set_ylabel('Y [m]', fontsize=14)
  ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_gt[:, 1], 'm', linestyle='--', alpha=1.0, label='Groundtruth')
  for i, trajectory in enumerate(eval_traj_list):
    ax.plot(eval_traj_list[i].accum_distances, trajectory.p_es_aligned[:, 1], colors[i], linestyle='-', alpha=alpha, label=traj_name_list[i])
  
  simple_draw_legend(ax)
  
  ax = fig.add_subplot(313)
  ax.grid(ls='--', color='0.7')
  ax.set_xlabel('accum_distances [m]', fontsize=14)
  ax.set_ylabel('Z [m]', fontsize=14)
  ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_gt[:, 2], 'm', linestyle='--', alpha=1.0, label='Groundtruth')
  for i, trajectory in enumerate(eval_traj_list):
    ax.plot(eval_traj_list[i].accum_distances, trajectory.p_es_aligned[:, 2], colors[i], linestyle='-', alpha=alpha, label=traj_name_list[i])

  simple_draw_legend(ax)
  
  fig.tight_layout()
  plt.show()
  fig.savefig(save_dir + "/3d_traj_xyz_compare.pdf", bbox_inches="tight")


def plot_2d_traj(eval_traj_list, traj_name_list, save_dir=""):
    alpha = 0.7
    legend_font_size = 14
    
    assert len(eval_traj_list) == len(traj_name_list), \
        "eval_traj_list和traj_name_list长度必须相同"
    
    colors = generate_random_colors(len(traj_name_list))
    
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, aspect='equal', xlabel='x [m]', ylabel='y [m]')
    ax.set_xlabel('X [m]', fontsize=legend_font_size)
    ax.set_ylabel('Y [m]', fontsize=legend_font_size)
    ax.grid(ls='--', color='0.7')
    
    ax.plot(eval_traj_list[0].p_gt[:, 0], eval_traj_list[0].p_gt[:, 1], 'm', linestyle='--', alpha=1.0, label='Groundtruth')
    
    for i, trajectory in enumerate(eval_traj_list):
      ax.plot(trajectory.p_es_aligned[:, 0], trajectory.p_es_aligned[:, 1], colors[i], linestyle='-', alpha=alpha, label=traj_name_list[i])
    
    simple_draw_legend(ax)
  
    fig.tight_layout()
    plt.show()

    fig.savefig(save_dir + "/2d_traj_xy_compare.pdf", bbox_inches="tight")

    
def main():
  parser = argparse.ArgumentParser(
      description='''Analyze trajectory estimate in a folder.''')
  parser.add_argument(
      'traj_dir', type=str,
      help="Folder containing the groundtruth and the estimate.")
  parser.add_argument(
      'output_dir', type=str,
      help="Folder containing the groundtruth and the estimate.")
  args = parser.parse_args()

  eval_traj_list = []
  traj_name_list = []

  traj_list = os.listdir(os.path.join(args.traj_dir, "estimate_traj"))
  gt_traj_path = os.path.join(args.traj_dir, "groundtruth.txt")
  print("Groundtruth trajectory: {0}".format(gt_traj_path))
  if not os.path.exists(gt_traj_path):
      print(Fore.RED + "Groundtruth trajectory not exists.")
      return
  for traj_file in traj_list:
      print("Processing trajectory: {0}".format(traj_file))
      estimate_traj_path = os.path.join(args.traj_dir, "estimate_traj", traj_file)
      trajectory = Trajectory(
          args.traj_dir,
          align_type="sim3",
          align_num_frames=-1,
          gt_traj_path=gt_traj_path,
          estimate_traj_path=estimate_traj_path,
      )
      trajectory.compute_absolute_error()
      eval_traj_list.append(trajectory)
      # using the file name as the legend
      traj_name_list.append(os.path.splitext(traj_file)[0])

  plot_3d_traj(eval_traj_list=eval_traj_list, traj_name_list=traj_name_list, save_dir=args.output_dir)
  plot_2d_traj_xyz(eval_traj_list=eval_traj_list, traj_name_list=traj_name_list, save_dir=args.output_dir)
  plot_2d_traj(eval_traj_list=eval_traj_list, traj_name_list=traj_name_list, save_dir=args.output_dir)
  
if __name__ == "__main__":
  main()
