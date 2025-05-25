import matplotlib.pyplot as plt
import numpy as np
import cv2
import rotationFix as RF
from dataset_configs import datasets
from matplotlib.patches import Patch


selected = "experiment_005"  
cfg = datasets[selected]
scale = 8
offset = 873
duration_of_video = cfg["duration_of_video"]
file_name1 = f"airporttestfiles\\" + selected + "output.csv"
file_name2 = f"csvfiles\\" + selected + "pose_transformed.csv"



# file_name1 = r"airporttestfiles\experiment_005output.csv"
val1 = np.loadtxt(file_name1, delimiter=',',skiprows=7+offset, max_rows=duration_of_video*scale)  # OptiTrack
# file_name2 = r"csvfiles\experiment_005pose_transformed.csv"
val2 = np.loadtxt(file_name2, delimiter=',')
def analyze_marker(marker_id):
    val2_del = val2[val2[:,7] == marker_id]
    pose_time = val2_del[:, 0] * scale + offset  # Convert pose frame count to OptiTrack timebase
    opti_time = val1[:, 0]  # Already in OptiTrack timebase
    common_times, idx_opti, idx_pose = np.intersect1d(opti_time, pose_time, return_indices=True)
    aligned_opti = val1[idx_opti]
    aligned_pose = val2_del[idx_pose]
    error_xyz = aligned_opti[:, 1:4] - aligned_pose[:, 1:4]  # (X, Y, Z)

    z_distance = aligned_opti[:, 3]  # Z position from OptiTrack
    return error_xyz, z_distance, marker_id, aligned_opti
def analyze_errors(error_xyz, z_distance, marker_id, z_bins=None):
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
    print(f"==== Position Error Metrics for Marker {marker_id} ====")
    print("==== Position Error Metrics per Z Bin ====")
    print(f"{'Bin Range (m)':<12} | {'Mean X':>7} | {'Mean Y':>7} | {'Mean Z':>7} | {'STD X':>7} | {'STD Y':>7} | {'STD Z':>7}")
    print("-" * 80)
    for i in range(num_bins):
        bin_range = f"{z_bins[i]:.1f}-{z_bins[i+1]:.1f}"
        print(f"{bin_range:<12} | "
              f"{mean_x_arr[i]:7.3f} | {mean_y_arr[i]:7.3f} | {mean_z_arr[i]:7.3f} | "
              f"{std_x_arr[i]:7.3f} | {std_y_arr[i]:7.3f} | {std_z_arr[i]:7.3f}")

    # Plot Mean and STD for X, Y, Z
    metrics = ['X', 'Y', 'Z']
    # mean_vals = [mean_x, mean_y, mean_z]
    # std_vals = [std_x, std_y, std_z]

    x = np.arange(len(metrics))
    width = 0.2  # Adjusted width for 2 bars

    # Bar chart for error metrics including STD
    # fig1 = plt.figure()
    # fig1.suptitle(f"Marker: {marker_id}")
    # # plt.bar(x - 0.5*width, mean_vals, width, label='Mean')
    # # plt.bar(x + 0.5*width, std_vals, width, label='STD')
    # plt.bar(x, [mean_x, mean_y, mean_z], width, label='Mean')
    # plt.bar(x + width, [std_x, std_y, std_z], width, label='STD')
    # plt.xticks(x, metrics)
    # plt.ylabel('Error (m)')
    # plt.title('Position Error Metrics (X, Y, Z)')
    # plt.legend()
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.savefig('position_error_metrics.png')

    # Plot 2: Error vs Z distance
    fig2 = plt.figure()
    fig2.suptitle(f"Marker: {marker_id}")
    plt.plot(z_distance, error_xyz[:, 0], '.r', label='X error')
    plt.plot(z_distance, error_xyz[:, 1], '.g', label='Y error')
    plt.plot(z_distance, error_xyz[:, 2], '.b', label='Z error')
    plt.xlabel('OptiTrack Z Position (m)')
    plt.ylabel('Position Error (m)')
    plt.title(f'Pose vs OptiTrack Position Error marker {marker_id}')
    plt.savefig('position_error.png')
    plt.legend()
    plt.grid(True)

    # Plot 3: Mean error with error bars (STD) vs Z bin centers ("I" bars, i.e., error bars)
    # Plot 3: Mean error over time with shaded ±1 std (and ±2 std for 95% confidence)
    fig3 = plt.figure()
    fig3.suptitle(f"Marker: {marker_id} - Error Distribution by Z Bin (Box Plot)")

    # Collect errors per bin for each axis
    errors_x, errors_y, errors_z = [], [], []
    for i in range(len(z_bins) - 1):
        in_bin = (z_distance >= z_bins[i]) & (z_distance < z_bins[i+1])
        errors_in_bin = error_xyz[in_bin]
        # If no data in bin, append empty array for consistent plotting
        if errors_in_bin.size == 0:
            errors_x.append([])
            errors_y.append([])
            errors_z.append([])
        else:
            errors_x.append(errors_in_bin[:, 0])
            errors_y.append(errors_in_bin[:, 1])
            errors_z.append(errors_in_bin[:, 2])

    box_positions = np.arange(len(z_bin_centers))
    box_width = 0.25

    # Plot each boxplot and keep the artists for legend
    plt.boxplot(errors_x, positions=box_positions - box_width, widths=box_width, patch_artist=True,
                        boxprops=dict(facecolor='r', alpha=0.4), medianprops=dict(color='r'), tick_labels=['']*len(z_bin_centers))
    plt.boxplot(errors_y, positions=box_positions, widths=box_width, patch_artist=True,
                        boxprops=dict(facecolor='g', alpha=0.4), medianprops=dict(color='g'), tick_labels=['']*len(z_bin_centers))
    plt.boxplot(errors_z, positions=box_positions + box_width, widths=box_width, patch_artist=True,
                        boxprops=dict(facecolor='b', alpha=0.4), medianprops=dict(color='b'), tick_labels=['']*len(z_bin_centers))

    plt.xticks(box_positions, [f"{z_bins[i]:.1f}-{z_bins[i+1]:.1f}" for i in range(len(z_bin_centers))], rotation=45)
    plt.xlabel('OptiTrack Z Position (m)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Error Distribution by Z Bin (Box Plot)')
    # Use proxy artists for correct legend colors
    legend_handles = [
        Patch(facecolor='r', edgecolor='r', alpha=0.4, label='X'),
        Patch(facecolor='g', edgecolor='g', alpha=0.4, label='Y'),
        Patch(facecolor='b', edgecolor='b', alpha=0.4, label='Z')
    ]
    plt.legend(handles=legend_handles)
    plt.grid(axis='y')
    plt.tight_layout()
   
        


    # Plot 4: Bar chart for Mean vs Z bin
    # labels = [f'{z_bins[i]}-{z_bins[i+1]}m' for i in range(num_bins)]
    # x = np.arange(num_bins)
    # width = 0.25
    # fig4, axs = plt.subplots(2, 1, sharex=True)
    # # Mean subplot
    # fig4.suptitle(f"Marker: {marker_id}")
    # axs[0].bar(x - width/2, mean_x_arr, width, label='X', color='r')
    # axs[0].bar(x + width/2, mean_y_arr, width, label='Y', color='g')
    # axs[0].bar(x + 1.5*width, mean_z_arr, width, label='Z', color='b')
    # axs[0].set_title('Mean Error vs Z Distance Bins')
    # axs[0].set_ylabel('Mean Error (m)')
    # axs[0].legend()
    # axs[0].grid(axis='y')

    # # STD subplot
    # axs[1].bar(x - width/2, std_x_arr, width, label='X', color='r')
    # axs[1].bar(x + width/2, std_y_arr, width, label='Y', color='g')
    # axs[1].bar(x + 1.5*width, std_z_arr, width, label='Z', color='b')
    # axs[1].set_title('STD vs Z Distance Bins')
    # axs[1].set_xlabel('OptiTrack Z Position (m)')
    # axs[1].set_ylabel('STD (m)')
    # axs[1].legend()
    # axs[1].grid(axis='y')
    # # Set x-ticks to bin labels
    # axs[1].set_xticks(x)
    # axs[1].set_xticklabels(labels, rotation=45, ha='right')

    # Plot 3: line plot for Mean and STD vs Z bin centers
    # fig3 = plt.figure()
    # fig3.suptitle(f"Marker: {marker_id}")
    # plt.plot(z_bin_centers, mean_x_arr, 'r--s', label='Mean X')
    # plt.plot(z_bin_centers, mean_y_arr, 'g--s', label='Mean Y')
    # plt.plot(z_bin_centers, mean_z_arr, 'b--s', label='Mean Z')

    # plt.plot(z_bin_centers, std_x_arr, 'r:', label='STD X')
    # plt.plot(z_bin_centers, std_y_arr, 'g:', label='STD Y')
    # plt.plot(z_bin_centers, std_z_arr, 'b:', label='STD Z')

    # plt.xlabel('OptiTrack Z Position (m)')
    # plt.ylabel('Error (m)')
    # plt.title('Position Error Metrics vs Z Distance')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # Return the computed metrics if you want to use them later
    [plt.figure(num).savefig(fr"datapictures\errorImages\marker{marker_id}\test{i}.png") for i, num in enumerate(plt.get_fignums())]
    plt.show()
    plt.close('all')
    return {
        # "rmse": (rmse_x, rmse_y, rmse_z),
        "mean": (mean_x, mean_y, mean_z),
        # "mae": (mae_x, mae_y, mae_z),
        # "rmse_bins": (rmse_x_arr, rmse_y_arr, rmse_z_arr),
        "mean_bins": (mean_x_arr, mean_y_arr, mean_z_arr),
        # "mae_bins": (mae_x_arr, mae_y_arr, mae_z_arr),
        "z_bin_centers": z_bin_centers
    }
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



# Run for marker 0
marker = 0
marker2 = marker + 10
error_xyz, z_distance,marker_id, aligned_opti = analyze_marker(marker2)
metrics = analyze_errors(error_xyz, z_distance, marker_id)
# metrics_by_z = percentage_error_metrics_by_z(error_xyz, aligned_opti, z_distance)
print("|" * 100)
# Run for marker 10
error_xyz, z_distance, marker_id, aligned_opti = analyze_marker(marker)
metrics = analyze_errors(error_xyz, z_distance,marker_id)
# metrics_by_z = percentage_error_metrics_by_z(error_xyz, aligned_opti, z_distance)













