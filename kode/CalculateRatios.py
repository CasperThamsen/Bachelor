import numpy as np
import MarkerTracker as MarkerTracker
import cv2
import icecream as ic

def compute_normalized_distances(sorted_poses, number_of_markers):
    distances = []
    for i in range(number_of_markers):
        for j in range(i + 1, number_of_markers):
            distance = np.sqrt((sorted_poses[i].x - sorted_poses[j].x)**2 + (sorted_poses[i].y - sorted_poses[j].y)**2)
            distances.append(distance)
    base_distance = min(distances)
    normalized_distances = [distance / base_distance for distance in distances]
    return normalized_distances

#Can be remade to sort in a different way, etc using direction vectors to find the markers in an anti-clockwise manner.
def desired_marker_order(poses):
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
    
    img = cv2.imread('findrelation2.jpg')
    mt.locate_marker_init(frame=img[:, :, 1])
    poses, number_of_markers = mt.detect_multiple_markers(frame=img[:, :, 1])
    sorted_poses = desired_marker_order(poses)
    normalized_distances = compute_normalized_distances(sorted_poses, number_of_markers)

    expected_ratios = normalized_distances
    
    formatted_ratios = ", ".join(f"{ratio:.6f}" for ratio in expected_ratios)
    print("Expected Distance Ratios (formatted):")
    print(formatted_ratios)

main()
