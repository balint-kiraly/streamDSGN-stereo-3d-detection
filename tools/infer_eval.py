import json
import time
from pathlib import Path
import numpy as np
import os
import requests
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from tqdm import tqdm

#scene = 1
KITTI_BASE = Path("data/kitti_tracking/training")
API_URL = "http://localhost:5000/predict"

# Custom Label Reading Function
def read_label(label_file_path, idx):
    """
    Reads a label file and returns a list of annotations in the form of a dictionary.
    """
    annotations = []

    with open(label_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            # Extracting each value from the line based on the KITTI label format
            frame_idx = int(parts[0])  # Frame index
            if frame_idx == idx:
                object_id = int(parts[1])  # Object ID
                object_type = parts[2]  # Object type (car, pedestrian, etc.)
                truncation = float(parts[3])  # Truncation
                occlusion = int(parts[4])  # Occlusion
                alpha = float(parts[5])  # Observation angle of object
                left = float(parts[6])  # Left bound of the bounding box
                top = float(parts[7])  # Top bound of the bounding box
                right = float(parts[8])  # Right bound of the bounding box
                bottom = float(parts[9])  # Bottom bound of the bounding box
                height = float(parts[10])  # Object height
                width = float(parts[11])  # Object width
                length = float(parts[12])  # Object length
                x = float(parts[13])  # X position of object in 3D space
                y = float(parts[14])  # Y position of object in 3D space
                z = float(parts[15])  # Z position of object in 3D space
                rotation_y = float(parts[16])  # Rotation along Y-axis

                annotation = {
                    'frame_idx': frame_idx,
                    'object_id': object_id,
                    'object_type': object_type,
                    'truncation': truncation,
                    'occlusion': occlusion,
                    'alpha': alpha,
                    'bounding_box': [left, top, right, bottom],
                    'dimensions': [height, width, length],
                    'location': [x, y, z],
                    'rotation_y': rotation_y
                }
                annotations.append(annotation)

    return annotations


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    The boxes are represented as [left, top, right, bottom].
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def rotation_matrix_y(angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])


def get_3d_box_corners(h, w, l, x, y, z, ry):
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    R = rotation_matrix_y(ry)
    corners_3d = R @ corners
    corners_3d += np.array([[x], [y], [z]])
    return corners_3d.T


def compute_3d_box_volume(corners):
    hull = ConvexHull(corners)
    return hull.volume


def compute_3d_iou(box1, box2):
    import trimesh

    mesh1 = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=box1, process=False))
    mesh2 = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=box2, process=False))

    inter_mesh = mesh1.intersection(mesh2)

    inter_vol = inter_mesh.volume if inter_mesh.is_volume else 0.0
    vol1 = mesh1.volume
    vol2 = mesh2.volume

    union_vol = vol1 + vol2 - inter_vol
    iou = inter_vol / union_vol if union_vol > 0 else 0.0
    return iou


def iou_3d_from_params(b1, b2):
    corners1 = get_3d_box_corners(*b1[0:3], *b1[3:6], b1[6])
    corners2 = get_3d_box_corners(*b2[0:3], *b2[3:6], b2[6])
    return compute_3d_iou(corners1, corners2)


# Custom Precision and Recall Calculation
def calculate_precision_recall(ground_truths, detections, iou_threshold):
    """
    Calculates precision and recall based on ground truths and detections.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for gt in ground_truths:
        matched = False
        for det in detections:
            box1 = gt['dimensions'] + gt['location'] + [gt['rotation_y']]
            box2 = det['dimensions'] + det['location'] + [det['rotation_y']]
            box2 = [box2[1], box2[0], box2[2], box2[3], box2[4], box2[5], box2[6]] # Pcdet format is in different order
            iou = iou_3d_from_params(box1, box2)
            if iou >= iou_threshold:
                true_positives += 1
                matched = True
                break
        if not matched:
            false_negatives += 1

    false_positives = len(detections) - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0

    return true_positives, false_positives, false_negatives, precision, recall


# Custom mAP Calculation Function
def compute_cls_metrics(ground_truths, detections, metrics):
    """
    Computes precision, recall, and average precision for each class.
    """

    for cls in metrics.keys():
        cls_gt = [gt for gt in ground_truths if gt['object_type'] == cls]
        cls_det = [det for det in detections if det['object_type'] == cls]

        for iou_threshold in metrics[cls].keys():
            true_positives, false_positives, false_negatives, precision, recall = calculate_precision_recall(cls_gt, cls_det, float(iou_threshold))
            metrics[cls][str(iou_threshold)]['precisions'].append(precision)
            metrics[cls][str(iou_threshold)]['recalls'].append(recall)
            metrics[cls][str(iou_threshold)]['TP'] += true_positives
            metrics[cls][str(iou_threshold)]['FP'] += false_positives
            metrics[cls][str(iou_threshold)]['FN'] += false_negatives

    return metrics


# Testing Custom Functions
def evaluate_scene(scene, metrics=None):
    """
    Evaluates the detection performance for a given scene.
    """
    label_file_path = os.path.join(KITTI_BASE, "label_02", f'{scene:04d}.txt')
    image_dir = KITTI_BASE / "image_02" / f"{scene:04d}"

    image_files = sorted(list(image_dir.glob(f"*.png")))

    latencies = []
    if metrics is None:
        metrics = {"Car": {"0.2": {"TP": 0, "FP": 0, "FN": 0, "precisions": [], "recalls": [] },
                                "0.1": {"TP": 0, "FP": 0, "FN": 0, "precisions": [], "recalls": [] }},
                   "Pedestrian": {"0.2": {"TP": 0, "FP": 0, "FN": 0, "precisions": [], "recalls": [] },
                                "0.1": {"TP": 0, "FP": 0, "FN": 0, "precisions": [], "recalls": [] }},
                   "Cyclist": {"0.2": {"TP": 0, "FP": 0, "FN": 0, "precisions": [], "recalls": [] },
                                "0.1": {"TP": 0, "FP": 0, "FN": 0, "precisions": [], "recalls": [] }}}


    for image_file in tqdm(image_files):
        idx = int(image_file.stem)
        imgL_path = KITTI_BASE / "image_02" / f"{scene:04d}" / f"{idx:06d}.png"
        imgR_path = KITTI_BASE / "image_03" / f"{scene:04d}" / f"{idx:06d}.png"
        calib_path = KITTI_BASE / "calib" / f"{scene:04d}.txt"

        files = {
            "left_image": open(imgL_path, "rb"),
            "right_image": open(imgR_path, "rb"),
            "calib": open(calib_path, "rb"),
        }

        start = time.time()
        response = requests.post(API_URL, files=files)
        end = time.time()
        latency = end - start
        latencies.append(latency)

        if response.status_code != 200:
            print(f"Request failed for {idx}: {response.status_code}")
            continue

        prediction = response.json()
        ground_truths = read_label(label_file_path, idx)

        metrics = compute_cls_metrics(ground_truths, prediction['objects'], metrics)

    # Calculate average metrics
    for object_type, thresholds in metrics.items():
        print(f"\nMetrics for {object_type}:")

        for threshold, data in thresholds.items():
            TP = data["TP"]
            FP = data["FP"]
            FN = data["FN"]
            precisions = data["precisions"]
            recalls = data["recalls"]

            # Calculate precision and recall (if the lists are non-empty)
            avg_precision = np.mean(precisions) if precisions else 0.0
            avg_recall = np.mean(recalls) if recalls else 0.0

            # Print TP, FP, FN, Precision, Recall for each threshold
            print(f"Threshold: {threshold}")
            print(f"  TP: {TP}, FP: {FP}, FN: {FN}")
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall: {avg_recall:.4f}")

    avg_latency = np.mean(latencies)
    print(f"Average Latency: {avg_latency:.4f} seconds")

    return metrics, avg_latency

def create_diagrams():
    """
    Creates diagrams for the evaluation metrics.
    """
    # Scene range
    scene_range = range(21)

    # Object classes and thresholds
    object_classes = ["Car", "Pedestrian", "Cyclist"]
    thresholds = ["0.1", "0.2"]

    # Prepare structure for storing average precisions
    avg_precisions = {thr: {cls: [] for cls in object_classes} for thr in thresholds}

    # Load and process metrics
    for scene in scene_range:
        filepath = f"output/eval/scene_{scene}_metrics.json"
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            scene_metrics = json.load(f)
        for cls in object_classes:
            for thr in thresholds:
                precisions = scene_metrics[cls][thr]["precisions"]
                avg_precision = sum(precisions) / len(precisions) if precisions else 0
                avg_precisions[thr][cls].append(avg_precision)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey='all')

    colors = {"Car": "blue", "Pedestrian": "green", "Cyclist": "red"}

    for i, thr in enumerate(thresholds):
        ax = axes[i]
        for cls in object_classes:
            ax.plot(scene_range, avg_precisions[thr][cls], label=cls, color=colors[cls], marker='o')
        ax.set_title(f"Average Precision @ IoU {thr}")
        ax.set_xlabel("Scene")
        ax.set_ylabel("Average Precision")
        ax.set_xticks(scene_range)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    Path("output/eval/diagrams").mkdir(parents=True, exist_ok=True)
    plt.savefig("output/eval/diagrams/avg_precision_by_scene.png", dpi=300)
    plt.show()

    # Load latencies
    with open("output/eval/latencies.json", "r") as f:
        latencies = json.load(f)

    # Convert latencies to numpy array for easier manipulation
    latencies = np.array(latencies)
    latencies = latencies * 1000  # Convert to milliseconds

    print(f"Average Latency: {np.mean(latencies):.4f} ms")
    # Create the latency plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(latencies)), latencies, marker='o', color='purple', label="Average Latency per Scene")

    plt.title("Average Inference Latency per Scene")
    plt.xlabel("Scene")
    plt.ylabel("Latency (ms)")
    plt.xticks(range(len(latencies)))
    plt.grid(True)
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig("output/eval/diagrams/latency_by_scene.png", dpi=300)
    plt.show()


def inference_evaluation():
    # Ensure the API is running before executing this script
    avg_latencies = []
    Path("output/eval").mkdir(parents=True, exist_ok=True)

    for scene in range(0, 21):
        print(f"Evaluating scene {scene}...")
        metrics, avg_latency = evaluate_scene(scene)

        with open(f"output/eval/scene_{scene}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        avg_latencies.append(avg_latency)

    with open(f"output/eval/latencies.json", "w") as f:
        json.dump(avg_latencies, f, indent=4)

if __name__ == "__main__":
    #create_diagrams()


# count objects
    # Scene range
    scene_range = range(21)

    # Object classes and thresholds
    object_classes = ["Car", "Pedestrian", "Cyclist"]
    thresholds = ["0.1", "0.2"]

    # Prepare structure for storing average precisions
    counts = {cls: [] for cls in object_classes}

    # Load and process metrics
    for scene in scene_range:
        filepath = f"output/eval/scene_{scene}_metrics.json"
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            scene_metrics = json.load(f)
        for cls in object_classes:
            tp = scene_metrics[cls]["0.2"]["TP"]
            fn = scene_metrics[cls]["0.2"]["FN"]
            counts[cls].append(tp + fn)

    colors = {"Car": "blue", "Pedestrian": "green", "Cyclist": "red"}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in object_classes:
        ax.plot(scene_range, counts[cls], label=cls, marker='o', color=colors[cls])
    ax.set_title("Object Count per Scene")
    ax.set_xlabel("Scene")
    ax.set_ylabel("Object Count")
    ax.set_xticks(scene_range)
    ax.set_ylim(0, max(max(counts[cls]) for cls in object_classes) + 5)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    Path("output/eval/diagrams").mkdir(parents=True, exist_ok=True)
    plt.savefig("output/eval/diagrams/object_count_by_scene.png", dpi=300)
    plt.show()