import numpy as np
import MarkerTracker as MarkerTracker
import cv2
import icecream as ic

def compute_distance_matrix(marker_positions,number_of_markers):
    """
    Compute distance matrix for a given set of marker positions (x, y).
    marker_positions: List of (x, y) coordinates of the markers.
    """
    distances_matrix = np.zeros((number_of_markers, number_of_markers))  # Initialize square matrix
    
    for i in range(number_of_markers):
        for j in range(i + 1, number_of_markers):
            distance = np.sqrt((marker_positions[i].x - marker_positions[j].x)**2 + 
                        (marker_positions[i].y - marker_positions[j].y)**2)
            distances_matrix[i][j] = distance
            distances_matrix[j][i] = distance

    return distances_matrix

def numerate_markers_distance(poses):

    distance_between_markers = [[] for _ in range(len(poses))]
    for i in range(len(poses)):
        for j in range(len(poses)):
            if i != j:
                distance_between_markers[i].append(
                    np.sqrt((poses[i].x - poses[j].x)**2 + (poses[i].y - poses[j].y)**2)
                )
    summed_distances = [sum(distance) for distance in distance_between_markers]
    
    sorted_indices = sorted(range(len(poses)), key=lambda i: summed_distances[i])
    sorted_poses = [poses[i] for i in sorted_indices]
    
    return sorted_poses

def main():
    mt = MarkerTracker.MarkerTracker(order=4,  # number of shaded regions
                                      kernel_size=30,
                                      scale_factor=100)
    mt.track_marker_with_missing_black_leg = False
    
    img = cv2.imread('findrelation.jpg')
    mt.locate_marker_init(frame=img[:, :, 1])
    poses, number_of_markers = mt.detect_multiple_markers(frame=img[:, :, 1])
    sorted_poses = numerate_markers_distance(poses)
    distance_matrix = compute_distance_matrix(sorted_poses,number_of_markers)

    
    print("Distance Matrix:")
    print(distance_matrix)

    valid_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    valid_distances = valid_distances[~np.isinf(valid_distances)]  # Remove 'inf' values
    # valid_distances = np.sort(valid_distances)
    
    # Normalize distances using the smallest distance as a base
    base_distance = valid_distances[0]
    expected_ratios = valid_distances / base_distance
    
    formatted_ratios = ", ".join(f"{ratio:.6f}" for ratio in expected_ratios)
    print("Expected Distance Ratios (formatted):")
    print(formatted_ratios)

main()
