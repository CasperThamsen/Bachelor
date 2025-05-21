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
val2_del = val2[val2[:,7] == 10]
val3 = val1[::8]
pose_time = val2_del[:, 0] * scale + offset  # Convert pose frame count to OptiTrack timebase
opti_time = val1[:, 0]  # Already in OptiTrack timebase
common_times, idx_opti, idx_pose = np.intersect1d(opti_time, pose_time, return_indices=True)
aligned_opti = val1[idx_opti]
aligned_pose = val2_del[idx_pose]
error_xyz = aligned_opti[:, 1:4] - aligned_pose[:, 1:4]  # (X, Y, Z)

z_distance = aligned_opti[:, 3]  # Z position from OptiTrack

# RMSE (Root Mean Square Error) size of errors
rmse_x = np.sqrt(np.mean(error_xyz[:, 0] ** 2))
rmse_y = np.sqrt(np.mean(error_xyz[:, 1] ** 2))
rmse_z = np.sqrt(np.mean(error_xyz[:, 2] ** 2))
rmse_total = np.sqrt(np.mean(np.sum(error_xyz ** 2, axis=1)))

# Mean Error (Bias towards a certain direction?)
mean_x = np.mean(error_xyz[:, 0])
mean_y = np.mean(error_xyz[:, 1])
mean_z = np.mean(error_xyz[:, 2])

# Mean Absolute Error (Average error ignoring direction)
mae_x = np.mean(np.abs(error_xyz[:, 0]))
mae_y = np.mean(np.abs(error_xyz[:, 1]))
mae_z = np.mean(np.abs(error_xyz[:, 2]))
mae_total = np.mean(np.linalg.norm(error_xyz, axis=1))

# Print all metrics
print("==== Position Error Metrics ====")
print(f"RMSE X: {rmse_x:.4f} m, Mean X: {mean_x:.4f} m, MAE X: {mae_x:.4f} m")
print(f"RMSE Y: {rmse_y:.4f} m, Mean Y: {mean_y:.4f} m, MAE Y: {mae_y:.4f} m")
print(f"RMSE Z: {rmse_z:.4f} m, Mean Z: {mean_z:.4f} m, MAE Z: {mae_z:.4f} m")

# Plot RMSE, Mean, and MAE for X, Y, Z
metrics = ['X', 'Y', 'Z']
rmse_vals = [rmse_x, rmse_y, rmse_z]
mean_vals = [mean_x, mean_y, mean_z]
mae_vals = [mae_x, mae_y, mae_z]

x = np.arange(len(metrics))
width = 0.25

# Plot 1: Bar chart for error metrics
fig1 = plt.figure()
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
plt.plot(z_distance, error_xyz[:, 0], '.r', label='X error')
plt.plot(z_distance, error_xyz[:, 1], '.g', label='Y error')
plt.plot(z_distance, error_xyz[:, 2], '.b', label='Z error')
plt.xlabel('OptiTrack Z Position (m)')
plt.ylabel('Position Error (m)')
plt.title('Pose vs OptiTrack Position Error')
plt.savefig('position_error.png')
plt.legend()
plt.grid(True)

# plot 3: plot RMSE, Mean, and MAE on the scatter plot 1
# plt.plot(z_distance, np.full_like(z_distance, rmse_x), 'r-', label='RMSE X error')
# plt.plot(z_distance, np.full_like(z_distance, rmse_y), 'g-', label='RMSE Y error')
# plt.plot(z_distance, np.full_like(z_distance, rmse_z), 'b-', label='RMSE Z error')
# plt.plot(z_distance, np.full_like(z_distance, mae_x), 'r:', label='MAE X error')
# plt.plot(z_distance, np.full_like(z_distance, mae_y), 'g:', label='MAE Y error')
# plt.plot(z_distance, np.full_like(z_distance, mae_z), 'b:', label='MAE Z error')
# plt.plot(z_distance, np.full_like(z_distance, mean_x), 'r--', label='Mean X error')
# plt.plot(z_distance, np.full_like(z_distance, mean_y), 'g--', label='Mean Y error')
# plt.plot(z_distance, np.full_like(z_distance, mean_z), 'b--', label='Mean Z error')
# plt.xlabel('OptiTrack Z Position (m)')
# plt.ylabel('Position Error (m)')
# plt.title('Pose vs OptiTrack Position Error with Metrics')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()



plt.show()  # Show both figures and wait until they are closed





