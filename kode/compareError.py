import matplotlib.pyplot as plt
import numpy as np
import cv2
import rotationFix as RF
from dataset_configs import datasets

#made with help from CoPilot

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


selected = "experiment_006"  
cfg = datasets[selected]
scale = int(cfg["hz"] / 30)
offset = cfg["offset"]
duration_of_video = cfg["duration_of_video"]
file_name1 = f"airporttestfiles\\" + selected + "output.csv"
file_name2 = f"csvfiles\\" + selected + "pose_transformed.csv"



# file_name1 = r"airporttestfiles\experiment_005output.csv"
val1 = np.loadtxt(file_name1, delimiter=',',skiprows=7+offset, max_rows=duration_of_video*scale)  # OptiTrack
# file_name2 = r"csvfiles\experiment_005pose_transformed.csv"
val2 = np.loadtxt(file_name2, delimiter=',')
def analyze_marker(marker_id):
    if marker_id < 10:
        marker_id2 = marker_id + 10
    else:
        marker_id2 = marker_id - 10

    val2_id1 = val2[val2[:, 7] == marker_id]
    val2_id2 = val2[val2[:, 7] == marker_id2]
    common_times = np.intersect1d(val2_id2[:, 0], val2_id1[:, 0])
    val2_id1_common = val2_id1[np.isin(val2_id1[:, 0], common_times)]
    pose_time = val2_id1_common[:, 0] * scale + offset
    opti_time = val1[:, 0]
    common_times2, idx_opti, idx_pose = np.intersect1d(opti_time, pose_time, return_indices=True)
    
    aligned_opti = val1[idx_opti]
    aligned_pose = val2_id1_common[idx_pose]
    error_xyz = aligned_opti[:, 1:4] - aligned_pose[:, 1:4]
    z_distance = aligned_opti[:, 3]

    return error_xyz, z_distance, marker_id, aligned_opti
def percentage_error_metrics_by_z(error_xyz, aligned_opti, z_values, bin_size=1.0, z_min=0.0, z_max=7.0):
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    results = {}

    def safe_div(numerator, denominator):
        return (numerator / denominator * 100) if denominator != 0 else np.nan

    print("==== Percentage Position Error Metrics per Z Bin ====")
    print("Bin Range (m) | %Mean X | %Mean Y | %Mean Z |  %STD X |  %STD Y |  %STD Z")
    print("-" * 70)

    for i in range(len(bins) - 1):
        z_start, z_end = bins[i], bins[i + 1]
        mask = (z_values >= z_start) & (z_values < z_end)
        err_bin = error_xyz[mask]
        ref_bin = aligned_opti[mask, 1:4]
    

        if err_bin.shape[0] == 0:
            values = ['nan'] * 6
            # rmse_pct = mean_pct = mae_pct = (np.nan, np.nan, np.nan)
            mean_pct = std_pct = (np.nan, np.nan, np.nan)
        else:
            # rmse = np.sqrt(np.mean(err_bin**2, axis=0))
            mean = np.mean(err_bin, axis=0)
            # mae = np.mean(np.abs(err_bin), axis=0)
            std = np.std(err_bin, axis=0)

            ref_mean_abs = np.mean(np.abs(ref_bin), axis=0)
            # rmse_pct = [safe_div(r, ref) for r, ref in zip(rmse, ref_mean_abs)]
            mean_pct = [safe_div(m, ref) for m, ref in zip(mean, ref_mean_abs)]
            # mae_pct = [safe_div(m, ref) for m, ref in zip(mae, ref_mean_abs)]
            std_pct = [safe_div(s, ref) for s, ref in zip(std, ref_mean_abs)]

            values = [f"{v:.1f}" if not np.isnan(v) else "nan" for v in (*mean_pct, *std_pct)]

        bin_label = f"{z_start:.1f}-{z_end:.1f}"
        print(f"{bin_label:<14}|{''.join(f'{val:>8} ' for val in values)}")

        results[(z_start, z_end)] = {
            # "rmse_pct": tuple(rmse_pct),
            "mean_pct": tuple(mean_pct),
            # "mae_pct": tuple(mae_pct),
            "std_pct": tuple(std_pct)
        }

    return results
def analyze_errors(error_xyz, z_distance, marker_id, z_bins=None, plot_metrics=True, print_metrics=True, print_avg=True):
    if z_bins is None:
        z_bins = np.arange(0, 8, 1)
    num_bins = len(z_bins) - 1

    # rmse_x_bins, rmse_y_bins, rmse_z_bins = [], [], []
    mean_x_bins, mean_y_bins, mean_z_bins = [], [], []
    # mae_x_bins, mae_y_bins, mae_z_bins = [], [], []
    std_x_bins, std_y_bins, std_z_bins = [], [], []

    for i in range(num_bins):
        in_bin = (z_distance >= z_bins[i]) & (z_distance < z_bins[i+1])
        errors_in_bin = error_xyz[in_bin]
        if errors_in_bin.size == 0:
            # rmse_x_bins.append(np.nan)
            # rmse_y_bins.append(np.nan)
            # rmse_z_bins.append(np.nan)

            mean_x_bins.append(np.nan)
            mean_y_bins.append(np.nan)
            mean_z_bins.append(np.nan)

            # mae_x_bins.append(np.nan)
            # mae_y_bins.append(np.nan)
            # mae_z_bins.append(np.nan)

            std_x_bins.append(np.nan)
            std_y_bins.append(np.nan)
            std_z_bins.append(np.nan)
            continue

        # rmse_x_bins.append(np.sqrt(np.mean(errors_in_bin[:, 0] ** 2)))
        # rmse_y_bins.append(np.sqrt(np.mean(errors_in_bin[:, 1] ** 2)))
        # rmse_z_bins.append(np.sqrt(np.mean(errors_in_bin[:, 2] ** 2)))

        mean_x_bins.append(np.mean(errors_in_bin[:, 0]))
        mean_y_bins.append(np.mean(errors_in_bin[:, 1]))
        mean_z_bins.append(np.mean(errors_in_bin[:, 2]))

        # mae_x_bins.append(np.mean(np.abs(errors_in_bin[:, 0])))
        # mae_y_bins.append(np.mean(np.abs(errors_in_bin[:, 1])))
        # mae_z_bins.append(np.mean(np.abs(errors_in_bin[:, 2])))

        std_x_bins.append(np.std(errors_in_bin[:, 0]))
        std_y_bins.append(np.std(errors_in_bin[:, 1]))
        std_z_bins.append(np.std(errors_in_bin[:, 2]))

    z_bin_centers = (z_bins[:-1] + z_bins[1:]) / 2

    # rmse_x = np.sqrt(np.mean(error_xyz[:, 0] ** 2))
    # rmse_y = np.sqrt(np.mean(error_xyz[:, 1] ** 2))
    # rmse_z = np.sqrt(np.mean(error_xyz[:, 2] ** 2))

    mean_x = np.mean(error_xyz[:, 0])
    mean_y = np.mean(error_xyz[:, 1])
    mean_z = np.mean(error_xyz[:, 2])

    # mae_x = np.mean(np.abs(error_xyz[:, 0]))
    # mae_y = np.mean(np.abs(error_xyz[:, 1]))
    # mae_z = np.mean(np.abs(error_xyz[:, 2]))

    std_x = np.std(error_xyz[:, 0])
    std_y = np.std(error_xyz[:, 1])
    std_z = np.std(error_xyz[:, 2])

    # Convert lists to arrays
    # rmse_x_arr = np.array(rmse_x_bins)
    # rmse_y_arr = np.array(rmse_y_bins)
    # rmse_z_arr = np.array(rmse_z_bins)

    mean_x_arr = np.array(mean_x_bins)
    mean_y_arr = np.array(mean_y_bins)
    mean_z_arr = np.array(mean_z_bins)

    # mae_x_arr = np.array(mae_x_bins)
    # mae_y_arr = np.array(mae_y_bins)
    # mae_z_arr = np.array(mae_z_bins)

    std_x_arr = np.array(std_x_bins)
    std_y_arr = np.array(std_y_bins)
    std_z_arr = np.array(std_z_bins)
    
    # Print metrics
    if print_metrics:
        print(f"==== Position Error Metrics for Marker {marker_id} ====")
        print("==== Position Error Metrics per Z Bin ====")
        print(f"{'Bin Range (m)':<12} | {'Mean X':>7} | {'Mean Y':>7} | {'Mean Z':>7} | {'STD X':>7} | {'STD Y':>7} | {'STD Z':>7}")
        print("-" * 80)
        for i in range(num_bins):
            bin_range = f"{z_bins[i]:.1f}-{z_bins[i+1]:.1f}"
            print(f"{bin_range:<12} | "
                f"{mean_x_arr[i]:7.3f} | {mean_y_arr[i]:7.3f} | {mean_z_arr[i]:7.3f} | "
                f"{std_x_arr[i]:7.3f} | {std_y_arr[i]:7.3f} | {std_z_arr[i]:7.3f}")
        print("|" * 100)
        

    # Plot Mean and STD for X, Y, Z
    metrics = ['X', 'Y', 'Z']
    # mean_vals = [mean_x, mean_y, mean_z]
    # std_vals = [std_x, std_y, std_z]

    x = np.arange(len(metrics))
    width = 0.2  # Adjusted width for 2 bars

    # # Plot 1: Error vs Z distance
    # fig2 = plt.figure()
    # fig2.suptitle(f"Marker: {marker_id}")
    # plt.plot(z_distance, error_xyz[:, 0], '.r', label='X error')
    # plt.plot(z_distance, error_xyz[:, 1], '.g', label='Y error')
    # plt.plot(z_distance, error_xyz[:, 2], '.b', label='Z error')
    # plt.xlabel('OptiTrack Z Position (m)')
    # plt.ylabel('Position Error (m)')
    # plt.title(f'Pose vs OptiTrack Position Error marker {marker_id}')
    # plt.savefig('position_error.png')
    # plt.legend()
    # plt.grid(True)

    # Plot 2: Mean and STD over Z bins for X, Y, Z using error bars
    fig, ax = plt.subplots()
    fig.suptitle(f"Marker: {marker_id} - Mean and STD of Error by Z Bin")
    offset = 0.08  # horizontal offset for each error bar group
    ax.errorbar(z_bin_centers - offset, mean_x_arr, yerr=std_x_arr, fmt='-o', color='r', label='X', capsize=5)
    ax.errorbar(z_bin_centers, mean_y_arr, yerr=std_y_arr, fmt='-o', color='g', label='Y', capsize=5)
    ax.errorbar(z_bin_centers + offset, mean_z_arr, yerr=std_z_arr, fmt='-o', color='b', label='Z', capsize=5)
    ax.set_xlabel('OptiTrack Z Position (m)')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Mean Error with STD over Z Bins')
    ax.legend()
    ax.grid(True)

    # # Plot 3: Mean error with error bars (STD) vs Z bin centers ("I" bars, i.e., error bars)
    # # Plot 3: Mean error over time with shaded ±1 std (and ±2 std for 95% confidence)
    # fig3 = plt.figure()
    # fig3.suptitle(f"Marker: {marker_id} - Error Distribution by Z Bin (Box Plot)")

    # # Collect errors per bin for each axis
    # errors_x, errors_y, errors_z = [], [], []
    # for i in range(len(z_bins) - 1):
    #     in_bin = (z_distance >= z_bins[i]) & (z_distance < z_bins[i+1])
    #     errors_in_bin = error_xyz[in_bin]
    #     # If no data in bin, append empty array for consistent plotting
    #     if errors_in_bin.size == 0:
    #         errors_x.append([])
    #         errors_y.append([])
    #         errors_z.append([])
    #     else:
    #         errors_x.append(errors_in_bin[:, 0])
    #         errors_y.append(errors_in_bin[:, 1])
    #         errors_z.append(errors_in_bin[:, 2])

    # box_positions = np.arange(len(z_bin_centers))
    # box_width = 0.25

    # # Plot each boxplot and keep the artists for legend
    # plt.boxplot(errors_x, positions=box_positions - box_width, widths=box_width, patch_artist=True,
    #                     boxprops=dict(facecolor='r', alpha=0.4), medianprops=dict(color='r'), tick_labels=['']*len(z_bin_centers))
    # plt.boxplot(errors_y, positions=box_positions, widths=box_width, patch_artist=True,
    #                     boxprops=dict(facecolor='g', alpha=0.4), medianprops=dict(color='g'), tick_labels=['']*len(z_bin_centers))
    # plt.boxplot(errors_z, positions=box_positions + box_width, widths=box_width, patch_artist=True,
    #                     boxprops=dict(facecolor='b', alpha=0.4), medianprops=dict(color='b'), tick_labels=['']*len(z_bin_centers))

    # plt.xticks(box_positions, [f"{z_bins[i]:.1f}-{z_bins[i+1]:.1f}" for i in range(len(z_bin_centers))], rotation=45)
    # plt.xlabel('OptiTrack Z Position (m)')
    # plt.ylabel('Position Error (m)')
    # plt.title('Position Error Distribution by Z Bin (Box Plot)')
    # # Use proxy artists for correct legend colors
    # legend_handles = [
    #     Patch(facecolor='r', edgecolor='r', alpha=0.4, label='X'),
    #     Patch(facecolor='g', edgecolor='g', alpha=0.4, label='Y'),
    #     Patch(facecolor='b', edgecolor='b', alpha=0.4, label='Z')
    # ]
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    # Return the computed metrics if you want to use them later
    [plt.figure(num).savefig(fr"datapictures\errorImages\marker{marker_id}\{selected}"+"_"+f"{i}.png") for i, num in enumerate(plt.get_fignums())]
    if plot_metrics:
        plt.show()
    plt.close('all')
    # Print average mean and std over bins if called from main
    return {
        # "rmse": (rmse_x, rmse_y, rmse_z),
        "mean": (mean_x, mean_y, mean_z),
        # "mae": (mae_x, mae_y, mae_z),
        # "rmse_bins": (rmse_x_arr, rmse_y_arr, rmse_z_arr),
        "mean_bins": (mean_x_arr, mean_y_arr, mean_z_arr),
        # "mae_bins": (mae_x_arr, mae_y_arr, mae_z_arr),
        "std": (std_x, std_y, std_z),
        "std_bins": (std_x_arr, std_y_arr, std_z_arr),
        "z_bin_centers": z_bin_centers
    }
    # Analyze rotation errors for each marker in loop_list and loop_list+10
def analyze_rotation_error(marker_id):
    if marker_id < 10:
        marker_id2 = marker_id + 10
    else:
        marker_id2 = marker_id - 10

    val2_id1 = val2[val2[:, 7] == marker_id]
    val2_id2 = val2[val2[:, 7] == marker_id2]
    common_times = np.intersect1d(val2_id2[:, 0], val2_id1[:, 0])
    val2_id1_common = val2_id1[np.isin(val2_id1[:, 0], common_times)]
    pose_time = val2_id1_common[:, 0] * scale + offset
    opti_time = val1[:, 0]
    common_times2, idx_opti, idx_pose = np.intersect1d(opti_time, pose_time, return_indices=True)
    aligned_opti = val1[idx_opti]
    aligned_pose = val2_id1_common[idx_pose]
    # Rotation columns: 4,5,6 (assuming Euler angles or rotation vector)
    rot_error = aligned_opti[:, 4:7] - aligned_pose[:, 4:7]
    z_distance = aligned_opti[:, 3]
    return rot_error, z_distance, marker_id

    

loop_list = [0, 1, 2, 3, 8]
nfold_Std = []
aruco_std = []

for i in loop_list:
    # Analyze marker i (Aruco)
    error_xyz1, z_distance1, marker_id1, aligned_opti1 = analyze_marker(i)
    metrics1 = analyze_errors(error_xyz1, z_distance1, marker_id1, plot_metrics=False, print_metrics=False)
    std_bins1 = np.stack(metrics1["std_bins"], axis=0)  # shape: (3, num_bins)
    mean_bins1 = np.stack(metrics1["mean_bins"], axis=0)  # shape: (3, num_bins)
    aruco_std.append(std_bins1)
    if i == 0:
        aruco_mean_bins = mean_bins1
    # Analyze marker i+10 (N-fold)
    error_xyz2, z_distance2, marker_id2, aligned_opti2 = analyze_marker(i + 10)
    metrics2 = analyze_errors(error_xyz2, z_distance2, marker_id2, plot_metrics=False, print_metrics=False)
    std_bins2 = np.stack(metrics2["std_bins"], axis=0)  # shape: (3, num_bins)
    mean_bins2 = np.stack(metrics2["mean_bins"], axis=0)  # shape: (3, num_bins)
    nfold_Std.append(std_bins2)
    if i == 0:
        nfold_mean_bins = mean_bins2
    z_bin_centers = metrics1["z_bin_centers"]  # Save for plotting



aruco_rot_mean = []
aruco_rot_std = []
nfold_rot_mean = []
nfold_rot_std = []

for i in loop_list:
    # ArUco
    rot_error1, z_distance_rot1, _ = analyze_rotation_error(i)
    z_bins = np.arange(0, 8, 1)
    num_bins = len(z_bins) - 1
    mean_rot_bins1 = []
    std_rot_bins1 = []
    for j in range(num_bins):
        in_bin = (z_distance_rot1 >= z_bins[j]) & (z_distance_rot1 < z_bins[j+1])
        errors_in_bin = rot_error1[in_bin]
        if errors_in_bin.size == 0:
            mean_rot_bins1.append([np.nan, np.nan, np.nan])
            std_rot_bins1.append([np.nan, np.nan, np.nan])
        else:
            mean_rot_bins1.append(np.mean(errors_in_bin, axis=0))
            std_rot_bins1.append(np.std(errors_in_bin, axis=0))
    aruco_rot_mean.append(mean_rot_bins1)
    aruco_rot_std.append(std_rot_bins1)

    # N-fold
    rot_error2, z_distance_rot2, _ = analyze_rotation_error(i + 10)
    mean_rot_bins2 = []
    std_rot_bins2 = []
    for j in range(num_bins):
        in_bin = (z_distance_rot2 >= z_bins[j]) & (z_distance_rot2 < z_bins[j+1])
        errors_in_bin = rot_error2[in_bin]
        if errors_in_bin.size == 0:
            mean_rot_bins2.append([np.nan, np.nan, np.nan])
            std_rot_bins2.append([np.nan, np.nan, np.nan])
        else:
            mean_rot_bins2.append(np.mean(errors_in_bin, axis=0))
            std_rot_bins2.append(np.std(errors_in_bin, axis=0))
    nfold_rot_mean.append(mean_rot_bins2)
    nfold_rot_std.append(std_rot_bins2)

# Convert to arrays: (num_markers, num_bins, 3)
aruco_rot_mean = np.array(aruco_rot_mean)
aruco_rot_std = np.array(aruco_rot_std)
nfold_rot_mean = np.array(nfold_rot_mean)
nfold_rot_std = np.array(nfold_rot_std)

# Average across markers: (num_bins, 3)
avg_aruco_rot_mean = np.nanmean(aruco_rot_mean, axis=0)
avg_aruco_rot_std = np.nanmean(aruco_rot_std, axis=0)
avg_nfold_rot_mean = np.nanmean(nfold_rot_mean, axis=0)
avg_nfold_rot_std = np.nanmean(nfold_rot_std, axis=0)

# Plot mean and std for rotation (Aruco)
plt.figure()
plt.errorbar(z_bin_centers, avg_aruco_rot_mean[:, 1], yerr=avg_aruco_rot_std[:, 1], fmt='-o', color='r', label='Aruco Rot X',capsize=5)
plt.errorbar(z_bin_centers - 0.1, avg_aruco_rot_mean[:, 0], yerr=avg_aruco_rot_std[:, 0], fmt='-o', color='b', label='Aruco Rot Z',capsize=5)
plt.errorbar(z_bin_centers + 0.1, avg_aruco_rot_mean[:, 2], yerr=avg_aruco_rot_std[:, 2], fmt='-o', color='g', label='Aruco Rot Y',capsize=5)
plt.xlabel('OptiTrack Z Position (m)')
plt.ylabel('Rotation Error (deg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"datapictures/errorImages/rot/aruco_rotation_mean_std_{selected}.png")
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.close()


# Plot mean and std for rotation (N-fold)
plt.figure()
plt.errorbar(z_bin_centers, avg_nfold_rot_mean[:, 1], yerr=avg_nfold_rot_std[:, 1], fmt='-o', color='r', label='N-fold Rot X',capsize=5)
plt.errorbar(z_bin_centers - 0.1, avg_nfold_rot_mean[:, 0], yerr=avg_nfold_rot_std[:, 0], fmt='-o', color='b', label='N-fold Rot Z',capsize=5)
plt.errorbar(z_bin_centers + 0.1, avg_nfold_rot_mean[:, 2], yerr=avg_nfold_rot_std[:, 2], fmt='-o', color='g', label='N-fold Rot Y',capsize=5)
plt.xlabel('OptiTrack Z Position (m)')
plt.ylabel('Rotation Error (deg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig(f"datapictures/errorImages/rot/nfold_rotation_mean_std_{selected}.png")

# Convert to arrays after the loop
nfold_Std = np.array(nfold_Std)    # shape: (num_markers, 3, num_bins)
aruco_std = np.array(aruco_std)    # shape: (num_markers, 3, num_bins)

avg_std_n = np.nanmean(nfold_Std, axis=0).T      # shape: (num_bins, 3)
avg_std_aruco = np.nanmean(aruco_std, axis=0).T  # shape: (num_bins, 3)

offset = 0.2

# Plot only STD (not mean) for all markers in loop_list (aruco_std) as error bars
plt.figure()
plt.errorbar(z_bin_centers - offset, avg_std_aruco[:, 0], yerr=np.nanstd(aruco_std[:, 0, :], axis=0), fmt='o', color='r', label='Aruco X Std', capsize=5)
plt.errorbar(z_bin_centers, avg_std_aruco[:, 1], yerr=np.nanstd(aruco_std[:, 1, :], axis=0), fmt='o', color='g', label='Aruco Y Std', capsize=5)
plt.errorbar(z_bin_centers + offset, avg_std_aruco[:, 2], yerr=np.nanstd(aruco_std[:, 2, :], axis=0), fmt='o', color='b', label='Aruco Z Std', capsize=5)

# Add mean for marker 0 (aruco)
if 'aruco_mean_bins' in locals():
    plt.plot(z_bin_centers - offset, aruco_mean_bins[0], '--', color='r', label='Aruco Mean X (marker 0)')
    plt.plot(z_bin_centers, aruco_mean_bins[1], '--', color='g', label='Aruco Mean Y (marker 0)')
    plt.plot(z_bin_centers + offset, aruco_mean_bins[2], '--', color='b', label='Aruco Mean Z (marker 0)')

plt.xlabel('OptiTrack Z Position (m)')
plt.ylabel('Average STD Error (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.ylim(-0.5,2.5) #special case for experiment_003, due to 6-7 having trouble
plt.savefig(f"datapictures/errorImages/std/average_std_aruco_{selected}.png")

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.close()




# Plot only STD (not mean) for all markers in loop_list+10 (nfold_Std) as error bars
plt.figure()
plt.errorbar(z_bin_centers - offset, avg_std_n[:, 0], yerr=np.nanstd(nfold_Std[:, 0, :], axis=0), fmt='o', color='r', label='N-fold X Std', capsize=5)
plt.errorbar(z_bin_centers, avg_std_n[:, 1], yerr=np.nanstd(nfold_Std[:, 1, :], axis=0), fmt='o', color='g', label='N-fold Y Std', capsize=5)
plt.errorbar(z_bin_centers + offset, avg_std_n[:, 2], yerr=np.nanstd(nfold_Std[:, 2, :], axis=0), fmt='o', color='b', label='N-fold Z Std', capsize=5)

# Add mean for marker 10 (nfold)
if 'nfold_mean_bins' in locals():
    plt.plot(z_bin_centers - offset, nfold_mean_bins[0], '--', color='r', label='N-fold Mean X (marker 10)')
    plt.plot(z_bin_centers, nfold_mean_bins[1], '--', color='g', label='N-fold Mean Y (marker 10)')
    plt.plot(z_bin_centers + offset, nfold_mean_bins[2], '--', color='b', label='N-fold Mean Z (marker 10)')

plt.xlabel('OptiTrack Z Position (m)')
plt.ylabel('Average STD Error (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(xlim)
plt.ylim(ylim)
# plt.ylim(-0.5,2.5) special case for experiment_003, due to 6-7 having trouble
plt.savefig(f"datapictures/errorImages/std/average_std_nfold_{selected}.png")


# Compute pairwise differences between nfold_Std and aruco_std for each marker in loop_list
pair_diffs = np.abs(nfold_Std) - np.abs(aruco_std)  # shape: (num_markers, 3, num_bins)
avg_pair_diff = np.nanmean(pair_diffs, axis=0)  # shape: (3, num_bins)

print("==== Average STD & Rotation STD Difference (ArUco - N-fold) per Z Bin ====")
print(f"{'Bin Range (m)':<14} | {'ΔSTD X':>10} | {'ΔSTD Y':>10} | {'ΔSTD Z':>10} | {'ΔRot STD X':>12} | {'ΔRot STD Y':>12} | {'ΔRot STD Z':>12}")
print("-" * 80)
rot_std_pair_diff = np.abs(avg_aruco_rot_std) - np.abs(avg_nfold_rot_std)  # shape: (num_bins, 3)
for i, center in enumerate(z_bin_centers):
    bin_range = f"{center - 0.5:.1f}-{center + 0.5:.1f}"
    dx, dy, dz = (avg_pair_diff[0, i] * -1, avg_pair_diff[1, i] * -1, avg_pair_diff[2, i] * -1)
    # Rotation: x is [1], y is [0], z is [2]
    drx, dry, drz = rot_std_pair_diff[i][1], rot_std_pair_diff[i][0], rot_std_pair_diff[i][2]
    print(f"{bin_range:<14} | {dx:10.4f} | {dy:10.4f} | {dz:10.4f} | {drx:12.4f} | {dry:12.4f} | {drz:12.4f}")

print("==== Average STD per Z Bin (Aruco) ====")
print(f"{'Bin Range (m)':<14} | {'STD X':>10} | {'STD Y':>10} | {'STD Z':>10} | {'Rot STD X':>10} | {'Rot STD Y':>10} | {'Rot STD Z':>10}")
print("-" * 80)
for i, center in enumerate(z_bin_centers):
    bin_range = f"{center - 0.5:.1f}-{center + 0.5:.1f}"
    sx, sy, sz = avg_std_aruco[i]
    rsy, rsx, rsz = avg_aruco_rot_std[i]  # Note: [Y, X, Z] order in rotation
    print(f"{bin_range:<14} | {sx:10.4f} | {sy:10.4f} | {sz:10.4f} | {rsx:10.4f} | {rsy:10.4f} | {rsz:10.4f}")

print("\n==== Average STD per Z Bin (N-fold) ====")
print(f"{'Bin Range (m)':<14} | {'STD X':>10} | {'STD Y':>10} | {'STD Z':>10} | {'Rot STD X':>10} | {'Rot STD Y':>10} | {'Rot STD Z':>10}")
print("-" * 80)
for i, center in enumerate(z_bin_centers):
    bin_range = f"{center - 0.5:.1f}-{center + 0.5:.1f}"
    sx, sy, sz = avg_std_n[i]
    rsy, rsx, rsz = avg_nfold_rot_std[i]  # Note: [Y, X, Z] order in rotation
    print(f"{bin_range:<14} | {sx:10.4f} | {sy:10.4f} | {sz:10.4f} | {rsx:10.4f} | {rsy:10.4f} | {rsz:10.4f}")

# print("\n==== Mean Error (Bias) per Z Bin (Aruco, marker 0) ====")
# print(f"{'Bin Range (m)':<14} | {'Mean X':>10} | {'Mean Y':>10} | {'Mean Z':>10} | {'Rot Mean X':>12} | {'Rot Mean Y':>12} | {'Rot Mean Z':>12}")
# print("-" * 80)
# for i, center in enumerate(z_bin_centers):
#     bin_range = f"{center - 0.5:.1f}-{center + 0.5:.1f}"
#     mx, my, mz = aruco_mean_bins[:, i]
#     # Rotation mean for marker 0 (aruco)
#     rmy, rmx, rmz = avg_aruco_rot_mean[i]  # Note: [Y, X, Z] order
#     print(f"{bin_range:<14} | {mx:10.4f} | {my:10.4f} | {mz:10.4f} | {rmx:12.4f} | {rmy:12.4f} | {rmz:12.4f}")

# print("\n==== Mean Error (Bias) per Z Bin (N-fold, marker 10) ====")
# print(f"{'Bin Range (m)':<14} | {'Mean X':>10} | {'Mean Y':>10} | {'Mean Z':>10} | {'Rot Mean X':>12} | {'Rot Mean Y':>12} | {'Rot Mean Z':>12}")
# print("-" * 80)
# for i, center in enumerate(z_bin_centers):
#     bin_range = f"{center - 0.5:.1f}-{center + 0.5:.1f}"
#     mx, my, mz = nfold_mean_bins[:, i]
#     # Rotation mean for marker 10 (nfold)
#     rmy, rmx, rmz = avg_nfold_rot_mean[i]  # Note: [Y, X, Z] order
#     print(f"{bin_range:<14} | {mx:10.4f} | {my:10.4f} | {mz:10.4f} | {rmx:12.4f} | {rmy:12.4f} | {rmz:12.4f}")


# plt.show()
plt.close('all')
















