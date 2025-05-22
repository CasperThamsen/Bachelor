import matplotlib.pyplot as plt
import numpy as np
import cv2
import rotationFix as RF
from dataset_configs import datasets


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

    # Then run the rest of your code below for this error_xyz and z_distance...
    # (Bins, error computations, print statements, plots, etc.)
    
    # For example, just print RMSE for this marker:
    rmse_x = np.sqrt(np.mean(error_xyz[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(error_xyz[:, 1] ** 2))
    rmse_z = np.sqrt(np.mean(error_xyz[:, 2] ** 2))
    print(f"Marker {marker_id} RMSE X: {rmse_x:.4f}, Y: {rmse_y:.4f}, Z: {rmse_z:.4f}")
    # You can extend with full analysis and plotting or return the values
    
    # Return values or data for combined plotting if needed
    return error_xyz, z_distance, marker_id, aligned_opti
def analyze_errors(error_xyz, z_distance,marker_id, z_bins=None):
    if z_bins is None:
        z_bins = np.arange(0, 8, 1)
    num_bins = len(z_bins) - 1

    rmse_x_bins, rmse_y_bins, rmse_z_bins = [], [], []
    mean_x_bins, mean_y_bins, mean_z_bins = [], [], []
    mae_x_bins, mae_y_bins, mae_z_bins = [], [], []

    for i in range(num_bins):
        in_bin = (z_distance >= z_bins[i]) & (z_distance < z_bins[i+1])
        errors_in_bin = error_xyz[in_bin]
        if errors_in_bin.size == 0:
            rmse_x_bins.append(np.nan)
            rmse_y_bins.append(np.nan)
            rmse_z_bins.append(np.nan)

            mean_x_bins.append(np.nan)
            mean_y_bins.append(np.nan)
            mean_z_bins.append(np.nan)

            mae_x_bins.append(np.nan)
            mae_y_bins.append(np.nan)
            mae_z_bins.append(np.nan)
            continue

        rmse_x_bins.append(np.sqrt(np.mean(errors_in_bin[:, 0] ** 2)))
        rmse_y_bins.append(np.sqrt(np.mean(errors_in_bin[:, 1] ** 2)))
        rmse_z_bins.append(np.sqrt(np.mean(errors_in_bin[:, 2] ** 2)))

        mean_x_bins.append(np.mean(errors_in_bin[:, 0]))
        mean_y_bins.append(np.mean(errors_in_bin[:, 1]))
        mean_z_bins.append(np.mean(errors_in_bin[:, 2]))

        mae_x_bins.append(np.mean(np.abs(errors_in_bin[:, 0])))
        mae_y_bins.append(np.mean(np.abs(errors_in_bin[:, 1])))
        mae_z_bins.append(np.mean(np.abs(errors_in_bin[:, 2])))

    z_bin_centers = (z_bins[:-1] + z_bins[1:]) / 2

    rmse_x = np.sqrt(np.mean(error_xyz[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(error_xyz[:, 1] ** 2))
    rmse_z = np.sqrt(np.mean(error_xyz[:, 2] ** 2))
    rmse_total = np.sqrt(np.mean(np.sum(error_xyz ** 2, axis=1)))

    mean_x = np.mean(error_xyz[:, 0])
    mean_y = np.mean(error_xyz[:, 1])
    mean_z = np.mean(error_xyz[:, 2])

    mae_x = np.mean(np.abs(error_xyz[:, 0]))
    mae_y = np.mean(np.abs(error_xyz[:, 1]))
    mae_z = np.mean(np.abs(error_xyz[:, 2]))
    mae_total = np.mean(np.linalg.norm(error_xyz, axis=1))

    # Convert lists to arrays
    rmse_x_arr = np.array(rmse_x_bins)
    rmse_y_arr = np.array(rmse_y_bins)
    rmse_z_arr = np.array(rmse_z_bins)

    mean_x_arr = np.array(mean_x_bins)
    mean_y_arr = np.array(mean_y_bins)
    mean_z_arr = np.array(mean_z_bins)

    mae_x_arr = np.array(mae_x_bins)
    mae_y_arr = np.array(mae_y_bins)
    mae_z_arr = np.array(mae_z_bins)

    # Print metrics
    print(f"==== Position Error Metrics for Marker {marker_id} ====")
    print(f"RMSE X: {rmse_x:.4f} m, Mean X: {mean_x:.4f} m, MAE X: {mae_x:.4f} m")
    print(f"RMSE Y: {rmse_y:.4f} m, Mean Y: {mean_y:.4f} m, MAE Y: {mae_y:.4f} m")
    print(f"RMSE Z: {rmse_z:.4f} m, Mean Z: {mean_z:.4f} m, MAE Z: {mae_z:.4f} m")

    print("==== Position Error Metrics per Z Bin ====")
    print(f"{'Bin Range (m)':<12} | {'RMSE X':>7} | {'RMSE Y':>7} | {'RMSE Z':>7} | {'Mean X':>7} | {'Mean Y':>7} | {'Mean Z':>7} | {'MAE X':>7} | {'MAE Y':>7} | {'MAE Z':>7}")
    print("-" * 92)
    for i in range(num_bins):
        bin_range = f"{z_bins[i]:.1f}-{z_bins[i+1]:.1f}"
        print(f"{bin_range:<12} | "
              f"{rmse_x_arr[i]:7.3f} | {rmse_y_arr[i]:7.3f} | {rmse_z_arr[i]:7.3f} | "
              f"{mean_x_arr[i]:7.3f} | {mean_y_arr[i]:7.3f} | {mean_z_arr[i]:7.3f} | "
              f"{mae_x_arr[i]:7.3f} | {mae_y_arr[i]:7.3f} | {mae_z_arr[i]:7.3f}")

    # Plot RMSE, Mean, and MAE for X, Y, Z
    metrics = ['X', 'Y', 'Z']
    rmse_vals = [rmse_x, rmse_y, rmse_z]
    mean_vals = [mean_x, mean_y, mean_z]
    mae_vals = [mae_x, mae_y, mae_z]

    x = np.arange(len(metrics))
    width = 0.25

    # Plot 1: Bar chart for error metrics
    fig1 = plt.figure()
    fig1.suptitle(f"Marker: {marker_id}")
    plt.bar(x - width, rmse_vals, width, label='RMSE')
    plt.bar(x, mean_vals, width, label='Mean')
    plt.bar(x + width, mae_vals, width, label='MAE')
    plt.xticks(x, metrics)
    plt.ylabel('Error (m)')
    plt.title('Position Error Metrics (X, Y, Z)')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('position_error_metrics.png')

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


    # Plot 4: Bar chart for RMSE, Mean, and MAE vs Z bin
    labels = [f'{z_bins[i]}-{z_bins[i+1]}m' for i in range(num_bins)]
    x = np.arange(num_bins)
    width = 0.25
    fig4, axs = plt.subplots(3, 1, sharex=True)
    # RMSE subplot
    fig4.suptitle(f"Marker: {marker_id}")
    axs[0].bar(x - width, rmse_x_arr, width, label='X', color='r')
    axs[0].bar(x, rmse_y_arr, width, label='Y', color='g')
    axs[0].bar(x + width, rmse_z_arr, width, label='Z', color='b')
    axs[0].set_title('RMSE vs Z Distance Bins')
    axs[0].set_ylabel('RMSE (m)')
    axs[0].legend()
    axs[0].grid(axis='y')

    # Mean subplot
    axs[1].bar(x - width, mean_x_arr, width, label='X', color='r')
    axs[1].bar(x, mean_y_arr, width, label='Y', color='g')
    axs[1].bar(x + width, mean_z_arr, width, label='Z', color='b')
    axs[1].set_title('Mean Error vs Z Distance Bins')
    axs[1].set_ylabel('Mean Error (m)')
    axs[1].legend()
    axs[1].grid(axis='y')

    # MAE subplot
    axs[2].bar(x - width, mae_x_arr, width, label='X', color='r')
    axs[2].bar(x, mae_y_arr, width, label='Y', color='g')
    axs[2].bar(x + width, mae_z_arr, width, label='Z', color='b')
    axs[2].set_title('MAE vs Z Distance Bins')
    axs[2].set_xlabel('OptiTrack Z Position (m)')
    axs[2].set_ylabel('MAE (m)')
    axs[2].legend()
    axs[2].grid(axis='y')
    # Set x-ticks to bin labels
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=45, ha='right')

    # Plot 3: line plot for RMSE, Mean, and MAE vs Z bin centers
    fig3 = plt.figure()
    fig3.suptitle(f"Marker: {marker_id}")
    plt.plot(z_bin_centers, rmse_x_arr, 'r-o', label='RMSE X')
    plt.plot(z_bin_centers, rmse_y_arr, 'g-o', label='RMSE Y')
    plt.plot(z_bin_centers, rmse_z_arr, 'b-o', label='RMSE Z')

    plt.plot(z_bin_centers, mean_x_arr, 'r--s', label='Mean X')
    plt.plot(z_bin_centers, mean_y_arr, 'g--s', label='Mean Y')
    plt.plot(z_bin_centers, mean_z_arr, 'b--s', label='Mean Z')

    plt.plot(z_bin_centers, mae_x_arr, 'r:', label='MAE X')
    plt.plot(z_bin_centers, mae_y_arr, 'g:', label='MAE Y')
    plt.plot(z_bin_centers, mae_z_arr, 'b:', label='MAE Z')

    plt.xlabel('OptiTrack Z Position (m)')
    plt.ylabel('Error (m)')
    plt.title('Position Error Metrics vs Z Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Return the computed metrics if you want to use them later
    [plt.figure(num).savefig(fr"datapictures\errorImages\marker{marker_id}\test{i}.png") for i, num in enumerate(plt.get_fignums())]
    plt.show()
    plt.close('all')
    return {
        "rmse": (rmse_x, rmse_y, rmse_z, rmse_total),
        "mean": (mean_x, mean_y, mean_z),
        "mae": (mae_x, mae_y, mae_z, mae_total),
        "rmse_bins": (rmse_x_arr, rmse_y_arr, rmse_z_arr),
        "mean_bins": (mean_x_arr, mean_y_arr, mean_z_arr),
        "mae_bins": (mae_x_arr, mae_y_arr, mae_z_arr),
        "z_bin_centers": z_bin_centers
    }
def percentage_error_metrics_by_z(error_xyz, aligned_opti, z_values, bin_size=1.0, z_min=0.0, z_max=7.0):
    """
    Calculates RMSE, MAE, and mean error as percentage of mean abs ground truth,
    for X, Y, Z per Z-distance bin, and prints a formatted results table.

    Parameters:
        error_xyz (np.ndarray): Nx3 array of position errors.
        aligned_opti (np.ndarray): Nx4+ array of ground truth data (XYZ in columns 1:4).
        z_values (np.ndarray): Length-N array of Z distances.
        bin_size (float): Width of each Z bin.
        z_min (float): Min Z bin value.
        z_max (float): Max Z bin value.

    Returns:
        dict: Metrics per bin.
    """
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    results = {}

    def safe_div(numerator, denominator):
        return (numerator / denominator * 100) if denominator != 0 else np.nan

    print("==== Percentage Position Error Metrics per Z Bin ====")
    print("Bin Range (m) | %RMSE X | %RMSE Y | %RMSE Z | %Mean X | %Mean Y | %Mean Z |  %MAE X |  %MAE Y |  %MAE Z")
    print("-" * 98)

    for i in range(len(bins) - 1):
        z_start, z_end = bins[i], bins[i + 1]
        mask = (z_values >= z_start) & (z_values < z_end)
        err_bin = error_xyz[mask]
        ref_bin = aligned_opti[mask, 1:4]

        if err_bin.shape[0] == 0:
            values = ['nan'] * 9
            rmse_pct = mean_pct = mae_pct = (np.nan, np.nan, np.nan)
        else:
            rmse = np.sqrt(np.mean(err_bin**2, axis=0))
            mean = np.mean(err_bin, axis=0)
            mae = np.mean(np.abs(err_bin), axis=0)

            ref_mean_abs = np.mean(np.abs(ref_bin), axis=0)
            rmse_pct = [safe_div(r, ref) for r, ref in zip(rmse, ref_mean_abs)]
            mean_pct = [safe_div(m, ref) for m, ref in zip(mean, ref_mean_abs)]
            mae_pct = [safe_div(m, ref) for m, ref in zip(mae, ref_mean_abs)]

            values = [f"{v:.1f}" if not np.isnan(v) else "nan" for v in (*rmse_pct, *mean_pct, *mae_pct)]

        bin_label = f"{z_start:.1f}-{z_end:.1f}"
        print(f"{bin_label:<14}|{''.join(f'{val:>8} ' for val in values)}")

        results[(z_start, z_end)] = {
            "rmse_pct": tuple(rmse_pct),
            "mean_pct": tuple(mean_pct),
            "mae_pct": tuple(mae_pct)
        }

    return results



# Run for marker 0
marker = 1
marker2 = marker + 10
error_xyz, z_distance,marker_id, aligned_opti = analyze_marker(marker2)
metrics = analyze_errors(error_xyz, z_distance, marker_id)
metrics_by_z = percentage_error_metrics_by_z(error_xyz, aligned_opti, z_distance)
print("|" * 100)
# Run for marker 10
error_xyz, z_distance, marker_id, aligned_opti = analyze_marker(marker)
metrics = analyze_errors(error_xyz, z_distance,marker_id)
metrics_by_z = percentage_error_metrics_by_z(error_xyz, aligned_opti, z_distance)













