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

  # colors = generate_random_colors(len(traj_name_list))
  colors = ['b', 'g', 'r']
  fig = plt.figure(figsize=(6, 5.5))
  ax = fig.add_subplot(111, aspect='equal',
                        xlabel='x [m]', ylabel='y [m]')
  # pu.plot_trajectory_side(ax, eval_traj_list[0].p_gt, 'm', 'Groundtruth')
  ax.set_xlabel('X [m]', fontsize=14)
  ax.set_ylabel('Y [m]', fontsize=14)

  ax.grid(ls='--', color='0.7')
  ax.plot(eval_traj_list[0].p_gt[:, 0], eval_traj_list[0].p_gt[:, 1], 'm', 
              linestyle='--', alpha=1.0, label='Groundtruth')

  for i, trajectory in enumerate(eval_traj_list):
    pu.plot_trajectory_side(ax, trajectory.p_es_aligned, colors[i], traj_name_list[i], 0.6)
    # pu.plot_trajectory_side(ax, trajectory.p_es_aligned, 'b', traj_name_list[i])
    # pu.plot_aligned_side(ax, trajectory.p_es_aligned, trajectory.p_gt, trajectory.align_num_frames, colors[i])
  plt.legend(loc=1)
  fig.tight_layout()
  fig.savefig(args.output_dir + "/multi_traj_compare.pdf", bbox_inches="tight")

  # index = 0
  # fig = plt.figure(figsize=(8, 2.5))
  # ax = fig.add_subplot(111, xlabel='Distance [m]', ylabel='X [m]',
  #     xlim=[0, eval_traj_list[0].accum_distances[-1]])
  # ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_gt[:, index], 'm', linestyle='--', alpha=1.0,
  #         label = "Groundtruth")
  # ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_es_aligned[:, index], colors[0], alpha=0.7,
  #         label = traj_name_list[0])
  # ax.plot(eval_traj_list[1].accum_distances, eval_traj_list[1].p_es_aligned[:, index], colors[1], alpha=0.7,
  #         label = traj_name_list[1])
  # ax.plot(eval_traj_list[2].accum_distances, eval_traj_list[2].p_es_aligned[:, index], colors[2], alpha=0.7,
  #         label = traj_name_list[2])

  # plt.legend(loc=1)
  # fig.tight_layout()
  # fig.savefig(args.output_dir + "/x.pdf", bbox_inches="tight")


  # index = 1
  # fig = plt.figure(figsize=(8, 2.5))
  # ax = fig.add_subplot(111, xlabel='Distance [m]', ylabel='Y [m]',
  #     xlim=[0, eval_traj_list[0].accum_distances[-1]])
  # ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_gt[:, index], 'm', linestyle='--', alpha=1.0,
  #         label = "Groundtruth")
  # ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_es_aligned[:, index], colors[0], alpha=0.7,
  #         label = traj_name_list[0])
  # ax.plot(eval_traj_list[1].accum_distances, eval_traj_list[1].p_es_aligned[:, index], colors[1], alpha=0.7,
  #         label = traj_name_list[1])
  # ax.plot(eval_traj_list[2].accum_distances, eval_traj_list[2].p_es_aligned[:, index], colors[2], alpha=0.7,
  #         label = traj_name_list[2])

  # plt.legend(loc=1)
  # fig.tight_layout()
  # fig.savefig(args.output_dir + "/y.pdf", bbox_inches="tight")

  # index = 2
  # fig = plt.figure(figsize=(8, 2.5))
  # ax = fig.add_subplot(111, xlabel='Distance [m]', ylabel='Z [m]',
  #     xlim=[0, eval_traj_list[0].accum_distances[-1]])
  # ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_gt[:, index], 'm', linestyle='--', alpha=1.0,
  #         label = "Groundtruth")
  # ax.plot(eval_traj_list[0].accum_distances, eval_traj_list[0].p_es_aligned[:, index], colors[0], alpha=0.7,
  #         label = traj_name_list[0])
  # ax.plot(eval_traj_list[1].accum_distances, eval_traj_list[1].p_es_aligned[:, index], colors[1], alpha=0.7,
  #         label = traj_name_list[1])
  # ax.plot(eval_traj_list[2].accum_distances, eval_traj_list[2].p_es_aligned[:, index], colors[2], alpha=0.7,
  #         label = traj_name_list[2])

  # plt.legend(loc=1)
  # fig.tight_layout()
  # fig.savefig(args.output_dir + "/z.pdf", bbox_inches="tight")

  plt.show()

if __name__ == "__main__":
  main()
