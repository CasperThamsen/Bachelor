import numpy as np
import MarkerTracker as MarkerTracker
import cv2
import icecream as ic

def compute_normalized_distances(desired_order, number_of_markers):
    distances = []
    for i in range(number_of_markers):
        for j in range(i + 1, number_of_markers):
            distance = np.sqrt((desired_order[i].x - desired_order[j].x)**2 + (desired_order[i].y - desired_order[j].y)**2)
            distances.append(distance)
    base_distance = min(distances)
    normalized_distances = [distance / base_distance for distance in distances]
    return normalized_distances

#Can be remade to sort in a different way, etc using direction vectors to find the markers in an anti-clockwise manner.
def sort_by_sum(poses):
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

def order_anticlockwise(sorted_poses,poses):
    desired_order = []
    middle_marker = sorted_poses[0]
    top_left_corner = sorted_poses[1]
    desired_order.append(middle_marker)
    desired_order.append(top_left_corner)

    ref_vector = np.array([top_left_corner.x - middle_marker.x, top_left_corner.y - middle_marker.y])
    remaining_markers = [pose for pose in sorted_poses if pose not in desired_order]
    angles = [(pose, angle_between_vectors(ref_vector, np.array([pose.x - middle_marker.x, pose.y - middle_marker.y]))) for pose in remaining_markers]
    
    angles.sort(key=lambda x: x[1])
    for marker, _ in angles:
        desired_order.append(marker)
    print("desired_order", desired_order)
    return desired_order

def angle_between_vectors(v1, v2):
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    return angle if angle >= 0 else angle + 2 * np.pi


def main():
    #PICTURE PATH HERE
    img = cv2.imread(r"ratio_pictures\lufthavn.png")
    #PICTURE PATH HERE



    mt = MarkerTracker.MarkerTracker(order=4,  # number of shaded regions
                                      kernel_size=20,
                                      scale_factor=1)
    mt.track_marker_with_missing_black_leg = False
    mt.locate_marker_init(frame=img[:, :, 1])
    poses, number_of_markers = mt.detect_multiple_markers(frame=img[:, :, 1])
    sorted_poses = sort_by_sum(poses)
    desired_order = order_anticlockwise(sorted_poses,poses)
    normalized_distances = compute_normalized_distances(desired_order, number_of_markers)

    expected_ratios = normalized_distances
    
    formatted_ratios = ", ".join(f"{ratio:.6f}" for ratio in expected_ratios)
    print("Expected Distance Ratios (formatted):")
    print(formatted_ratios)

    img_pairs_copy = img.copy()
    for i, pose in enumerate(poses):
        cv2.putText(img_pairs_copy, str(i), (int(pose.x+20), int(pose.y+20)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
    for i, pose in enumerate(sorted_poses):
        cv2.putText(img_pairs_copy, str(i), (int(pose.x-20), int(pose.y-20)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0, 255), 2)
    for i, pose in enumerate(desired_order):
        cv2.putText(img_pairs_copy, str(i), (int(pose.x), int(pose.y)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 2)
    cv2.putText(img_pairs_copy, "Green: detected order", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_pairs_copy, "Red: sorted order", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_pairs_copy, "Blue: desired order", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.namedWindow("Sorted Markers", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sorted Markers", 1280, 720)
    cv2.imshow("Sorted Markers", img_pairs_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.savez('validation_ratios.npz', ratios=expected_ratios)

main()
