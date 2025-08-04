import os
import sys
from colorama import Fore, init
import traj_lib.trajectory_loading as traj_loading
import traj_lib.align_utils as au
import traj_lib.compute_trajectory_errors as traj_err
import traj_lib.results_writer as res_writer
import traj_lib.transformations as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

init(autoreset=True)

def add_results_table_page(results_table, pdf):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    df = pd.DataFrame(results_table, columns=["Sequence", "Translation RMSE (m)"])
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    pdf.savefig(fig)
    plt.close(fig)


def plot_trajectory_2d(p_gt, p_est_aligned, seq_name, rmse, pdf):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(ls='--', color='0.7')

    ax.plot(p_gt[:, 0], p_gt[:, 1], 'm-', label='Groundtruth')
    ax.plot(p_est_aligned[:, 0], p_est_aligned[:, 1], 'b-', label='Estimate')

    ax.set_title(f"{seq_name} (XY Plane) | Trans. RMSE: {rmse:.4f} m")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.axis('equal')
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)


def plot_aligned_trajectory(p_gt, p_est_aligned, seq_name, rmse, pdf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2], label='Ground Truth', color='blue')
    ax.plot(p_est_aligned[:, 0], p_est_aligned[:, 1], p_est_aligned[:, 2], label='Estimated', color='red')

    ax.set_title(f"{seq_name} | Trans. RMSE: {rmse:.4f} m")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.grid(True)
    pdf.savefig(fig)
    plt.close(fig)


def evaluate_single_sequence(gt_path, est_path, align_type='se3', align_num_frames=-1):
    print(Fore.CYAN + f"==> Evaluating sequence: {os.path.basename(gt_path)}")
  
    t_gt, p_est, q_est, t_est, p_gt, q_gt = traj_loading.load_stamped_dataset_from_file(gt_path,est_path)

    print(f"Loaded {len(p_est)} estimated poses, {len(p_gt)} groundtruth poses.")

    print("Aligning trajectory...")
    scale, rot, trans = au.alignTrajectory(p_est, p_gt, q_est, q_gt, align_type, align_num_frames)

    p_est_aligned = np.array([scale * rot.dot(p) + trans for p in p_est])
    q_est_aligned = np.zeros_like(q_est)
    for i in range(len(q_est)):
        q_R = rot @ tf.quaternion_matrix(q_est[i])[:3, :3]
        q_T = np.eye(4)
        q_T[:3, :3] = q_R
        q_est_aligned[i] = tf.quaternion_from_matrix(q_T)

    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = traj_err.compute_absolute_error(
        p_est_aligned, q_est_aligned, p_gt, q_gt
    )

    stats_trans = res_writer.compute_statistics(e_trans)
    stats_rot = res_writer.compute_statistics(e_rot)
    stats_scale = res_writer.compute_statistics(e_scale_perc)

    print(Fore.GREEN + "[Translation RMSE]")
    res_writer.print_format_stats(stats_trans)
    # print(Fore.GREEN + "[Rotation RMSE]")
    # res_writer.print_format_stats(stats_rot)
    # print(Fore.GREEN + "[Scale Error (%)]")
    # res_writer.print_format_stats(stats_scale)

    return {
        "trans": stats_trans,
        "rot": stats_rot,
        "scale": stats_scale,
        "p_gt": p_gt,
        "p_est_aligned": p_est_aligned
    }


def evaluate_all_sequences(gt_dir, est_dir, pdf = None):
    print(Fore.YELLOW + f"Evaluating all sequences in: {est_dir}")
    seq_files = sorted([f for f in os.listdir(est_dir) if f.endswith('.txt')])
    results_table = []

    results = {}
    for fname in seq_files:
        seq_name = fname.split('.')[0]
        gt_path = os.path.join(gt_dir,seq_name, "data.tum")
        est_path = os.path.join(est_dir, fname)

        if not os.path.exists(gt_path):
            print(Fore.RED + f"Groundtruth for {fname} not found, skipping.")
            continue

        stats = evaluate_single_sequence(gt_path, est_path)
        results[fname] = stats
        
        if pdf is not None:
          rmse = stats["trans"]["rmse"]
          plot_aligned_trajectory(stats["p_gt"], stats["p_est_aligned"], seq_name, rmse, pdf)
          plot_trajectory_2d(stats["p_gt"], stats["p_est_aligned"], seq_name, rmse, pdf)
          results_table.append((seq_name, rmse))
    
    if pdf is not None:
      add_results_table_page(results_table, pdf)

    return results

def main():
    if len(sys.argv) < 4:
        print(Fore.RED + "Usage: python eval_euroc_all.py <groundtruth_dir> <estimate_dir> <pdf_path>")
        sys.exit(1)

    gt_dir = sys.argv[1]
    est_dir = sys.argv[2]
    pdf_path = sys.argv[3]

    with PdfPages(pdf_path) as pdf:
      results = evaluate_all_sequences(gt_dir, est_dir, pdf)


    # print(Fore.YELLOW + "\n=== Summary of Results ===")
    # for seq, res in results.items():
    #     print(f"\n{seq}")
    #     print("Translation RMSE:")
    #     res_writer.print_format_stats(res["trans"])
        # print("Rotation RMSE:")
        # res_writer.print_format_stats(res["rot"])
        # print("Scale Error:")
        # res_writer.print_format_stats(res["scale"])

if __name__ == "__main__":
    main()
