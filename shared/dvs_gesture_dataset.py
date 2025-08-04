import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic.collation import PadTensors
from tonic import DiskCachedDataset

class DVSGestureData:
    """
    A class to handle the DVSGesture dataset preparation for different SNN
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
        self.sensor_size = tonic.datasets.DVSGesture.sensor_size
        print(f"DVSGestureData handler initialized. Data will be saved to '{self.save_to}'")

    def check_and_download(self):
        print("Downloading DVSGesture dataset (if not already present)...")
        tonic.datasets.DVSGesture(save_to=self.save_to, train=True)
        tonic.datasets.DVSGesture(save_to=self.save_to, train=False)
        print("Download check complete.")

    def get_frame_based_dataloader(self, train=True, time_window=10000, batch_size=32):

        # Define the transformation pipeline to convert events to frames
        frame_transform = transforms.Compose([
            transforms.ToFrame(
                sensor_size=self.sensor_size,
                time_window=time_window
            )
        ])

        # Load the dataset with the specified transform
        dataset = tonic.datasets.DVSGesture(
            save_to=self.save_to,
            train=train,
            transform=frame_transform
        )

        cached_dataset = DiskCachedDataset(dataset, cache_path=f"{self.save_to}/cache/dvs_gesture_frames_{'train' if train else 'test'}_tw{time_window}/")
        cached_dataloader = DataLoader(cached_dataset, batch_size=batch_size, collate_fn=PadTensors(batch_first=True))

        return cached_dataloader

    def get_spike_tensor_dataset(self, train=True):
        """
        Creates and returns a dataset of raw events.

        This is ideal for direct SNN training frameworks like Lava-dl slayer,
        which have their own optimized methods for converting raw event streams
        into sparse spike tensors. The training script will handle the final
        conversion using tools like `slayer.io`.

        Args:
            train (bool): If True, returns the training dataset. Otherwise,
                          returns the test dataset. Defaults to True.

        Returns:
            torch.utils.data.Dataset: The dataset with denoised raw events.
        """
        print(f"Preparing SPIKE-based dataset (train={train})...")
        
        # Define a minimal transformation pipeline, usually just denoising.
        # The final tensor conversion will be done in the training loop.
        event_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000)
        ])

        dataset = tonic.datasets.DVSGesture(
            save_to=self.save_to,
            train=train,
            transform=event_transform # Note: No ToFrame transform
        )

        return dataset