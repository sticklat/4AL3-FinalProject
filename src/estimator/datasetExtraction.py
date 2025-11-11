import os
import pandas as pd
import torch
import torch.nn.functional as F


def load_vehicle_annotations(folder_path, save_encoded=False, filepath=None):
    """
    Reads KITTI-style label text files and extracts:
    xmin, xmax, ymin, ymax, label, and distance (z in camera coordinates).

    Returns both a DataFrame and PyTorch tensors with one-hot encoded labels.
    """
    records = []

    # Collect all .txt files in numerical order
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

    for file in files:
        file_id = os.path.splitext(file)[0]  # e.g. "000000"
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                label = parts[0]
                if label == 'DontCare':
                    continue  # skip unlabeled/ignored regions

                # bounding box coords
                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])

                # z-coordinate = distance from camera
                distance = float(parts[13])

                records.append({
                    'file_id': file_id,
                    'label': label,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'distance': distance
                })

    df = pd.DataFrame(records)

    # --- One-hot encode labels ---
    unique_labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    # df['label_id'] = df['label'].map(label_to_idx)
    
    
    # Convert to integer indices first
    label_indices = torch.tensor(df['label'].map(label_to_idx).values, dtype=torch.long)
    # Convert to one-hot encoding
    num_classes = len(unique_labels)
    labels_onehot = F.one_hot(label_indices, num_classes=num_classes) #.float()

    # --- Convert features to PyTorch tensors ---
    features = torch.tensor(
        df[['xmin', 'xmax', 'ymin', 'ymax']].values, dtype=torch.float32
    )
    features = torch.cat((features, labels_onehot), dim=1)  # concatenate one-hot label to be a feature
    
    distances = torch.tensor(df['distance'].values, dtype=torch.float32).unsqueeze(1)
    

    if save_encoded:
        if filepath == None:
            torch.save({
                "features": features,
                "labels": labels_onehot,
                "label_map": label_to_idx
            }, os.path.join(folder_path, "vehicle_dataset.pt"))
            print(f"Saved encoded dataset to {folder_path}/vehicle_dataset.pt")
        else:
            torch.save({
                "features": features,
                "labels": labels_onehot,
                "label_map": label_to_idx
            }, filepath)
            print(f"Saved encoded dataset to {filepath}")

    return df, features, distances, label_to_idx


if __name__ == "__main__":
    # Example usage
    df, X, y, label_map = load_vehicle_annotations("dataset/training/label_2", save_encoded=True, filepath="dataset/vehicle_dataset.pt")

    # print(df.head())
    # print(X[:5])
    # print(y[:5])
    # print(label_map)
    # print(X.shape, y.shape)