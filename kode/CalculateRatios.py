import numpy as np
import MarkerTracker as MarkerTracker
import cv2

def compute_distance_matrix(marker_positions,number_of_markers):
    """
    Compute distance matrix for a given set of marker positions (x, y).
    marker_positions: List of (x, y) coordinates of the markers.
    """
    distances_matrix = np.zeros((number_of_markers, number_of_markers))  # Initialize square matrix
    
    for i in range(number_of_markers):
        for j in range(i + 1, number_of_markers):
            distance = np.sqrt((marker_positions[i][0] - marker_positions[j][0])**2 + 
                        (marker_positions[i][1] - marker_positions[j][1])**2)
            distances_matrix[i][j] = distance
            distances_matrix[j][i] = distance

    return distances_matrix

def main():
    mt = MarkerTracker.MarkerTracker(order=4,  # number of shaded regions
                                      kernel_size=30,
                                      scale_factor=100)
    mt.track_marker_with_missing_black_leg = False
    
    img = cv2.imread('findrelation.jpg')
    mt.locate_marker_init(frame=img[:, :, 1])
    poses, number_of_markers = mt.detect_multiple_markers(frame=img[:, :, 1])
    marker_positions = [(pose.x, pose.y) for pose in poses]
    distance_matrix = compute_distance_matrix(marker_positions)
    
    print("Distance Matrix:")
    print(distance_matrix)

    valid_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    valid_distances = valid_distances[~np.isinf(valid_distances)]  # Remove 'inf' values
    valid_distances = np.sort(valid_distances)
    
    # Normalize distances using the smallest distance as a base
    base_distance = valid_distances[0]
    expected_ratios = valid_distances / base_distance
    
    print("Expected Distance Ratios:")
    print(expected_ratios)

main()
