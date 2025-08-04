import tonic
import torch
import tonic.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tonic.collation import PadTensors
from tonic import DiskCachedDataset
import numpy as np
from PIL import Image

class CIFAR10DVSData:
    """
    A class to handle the CIFAR10DVS dataset preparation for different SNN
    training methodologies.

    This class centralizes the data download and provides separate pipelines
    for creating:
    1. Frame-based datasets (suitable for CNN-to-SNN conversion, e.g., Akida).
    2. Spike-tensor-based datasets (suitable for direct SNN training, e.g., Lava-dl Slayer).
    """

    def __init__(self, save_to='./datasets'):
        """
        Initializes the data handler.

        Args:
            save_to (str): The directory path where the dataset will be
                           downloaded and stored. Defaults to './data'.
        """
        self.save_to = save_to
        self.sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
        self.labels = ['Airplanes','Automobile', 'Birds', 'Cats', 'Deer', 'Dogs', 'Frogs', 'Horses', 'Ships', 'Trucks']
        print(f"CIFAR10DVS Data handler initialized. Data will be saved to '{self.save_to}'")

    def check_and_download(self):
        print("Downloading CIFAR10DVS dataset (if not already present)...")
        tonic.datasets.CIFAR10DVS(save_to=self.save_to)
        print("Download check complete.")

    def get_frame_dataloader(self, batch_size=32, n_time_bins=48, resize_shape=(32, 32), train_ratio=0.7):

        # Custom transform to average over channel and time dimensions
        class AggregateOverTimeAndChannel:
            def __call__(self, frames):
                # frames: [channels, time_steps, H, W]
                # Average over channel (dim=0) and time (dim=1)
                avg_img = frames.mean(axis=0).mean(axis=0)  # shape: [H, W]
                # Resize to (32, 32) using PIL for better quality
                img = Image.fromarray(avg_img.astype(np.float32))
                img = img.resize(resize_shape, resample=Image.BILINEAR)
                img = np.array(img)
                return img[None, :, :]  # [1, 32, 32]
        
        # Define the transformation pipeline to convert events to frames
        frame_transform = transforms.Compose([
            transforms.ToFrame(
                sensor_size=self.sensor_size,
                n_time_bins=n_time_bins
            ),
            AggregateOverTimeAndChannel()
        ])

        # Load the dataset with the specified transform
        dataset = tonic.datasets.CIFAR10DVS(
            save_to=self.save_to,
            transform=frame_transform
        )

        cached_dataset = DiskCachedDataset(dataset, cache_path=f"{self.save_to}/cache/cifar10_dvs_frames_tb_{n_time_bins}_shape_{resize_shape[0]}_{resize_shape[1]}/")

        # Define the split ratios
        val_ratio = (1 - train_ratio) / 2
        test_ratio = val_ratio  # Assuming equal split for validation and test

        # Calculate the number of samples for each set
        dataset_size = len(cached_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = int(test_ratio * dataset_size)

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size]
        )

        print(f"Splited dataset into:")
        print(f" - Training set: {len(train_dataset)} samples")
        print(f" - Validation set: {len(val_dataset)} samples")
        print(f" - Test set: {len(test_dataset)} samples")
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=PadTensors(batch_first=True), shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=PadTensors(batch_first=True), shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=PadTensors(batch_first=True), shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader

    def get_voxel_grid_dataloader(self, batch_size=32, train_ratio=0.7, num_time_bins=300):
        """
        Creates and returns a dataset of raw events.

        This is ideal for direct SNN training frameworks like Lava-dl slayer,
        which have their own optimized methods for converting raw event streams
        into sparse spike tensors. The training script will handle the final
        conversion using tools like `slayer.io`.
        """

        transform = tonic.transforms.Compose([
            tonic.transforms.NumpyAsType(int)
        ])

        dataset = tonic.datasets.CIFAR10DVS(
            save_to=self.save_to,
            transform=transform
        )

        cached_dataset = DiskCachedDataset(dataset, cache_path=f"{self.save_to}/cache/cifar10_dvs_spike/")

        # Define the split ratios
        val_ratio = (1 - train_ratio) / 2
        test_ratio = val_ratio  # Assuming equal split for validation and test

        # Calculate the number of samples for each set
        dataset_size = len(cached_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = int(test_ratio * dataset_size)

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size]
        )

        print(f"Splited dataset into:")
        print(f" - Training set: {len(train_dataset)} samples")
        print(f" - Validation set: {len(val_dataset)} samples")
        print(f" - Test set: {len(test_dataset)} samples")

        # Convert datasets to spike tensor format
        train_dataset = self.transform_to_spike_tensor_dataset(train_dataset, num_time_bins)
        val_dataset = self.transform_to_spike_tensor_dataset(val_dataset, num_time_bins)
        test_dataset = self.transform_to_spike_tensor_dataset(test_dataset, num_time_bins)
        
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=PadTensors(batch_first=True))
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=PadTensors(batch_first=True))
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=PadTensors(batch_first=True))

        return train_dataloader, val_dataloader, test_dataloader