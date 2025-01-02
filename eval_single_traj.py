import os
import yaml
import pickle
import argparse

import numpy as np

import traj_lib.trajectory_utils as traj_utils
import traj_lib.trajectory_loading as traj_loading
# import traj_lib.results_writer as res_writer
# import traj_lib.compute_trajectory_errors as traj_err
import traj_lib.align_utils as au

import traj_lib.transformations as tf

class Trajectory:
  def __init__(self, traj_dir='', platform='', alg_name='', dataset_name='',
                 align_type='sim3', align_num_frames=-1,
                 preset_boxplot_distances=[]):
    self.traj_dir = traj_dir

    self.load_trajectory()

  def load_trajectory(self):
    self.t_es, self.p_es, self.q_es, self.t_gt, self.p_gt, self.q_gt =\
            traj_loading.load_stamped_dataset(self.traj_dir)
    pass

def main():
  parser = argparse.ArgumentParser(
        description='''Analyze trajectory estimate in a folder.''')
  parser.add_argument(
        'traj_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
  args = parser.parse_args()

  trajectory = Trajectory(args.traj_dir);



if __name__ == "__main__":
  main()
  print("eval trajectory")

