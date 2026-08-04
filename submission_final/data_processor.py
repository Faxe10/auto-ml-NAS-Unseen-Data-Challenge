import torch
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    """
        PyTorch dataset wrapper for image data.

        The input data is converted to float tensors and reshaped to the
        channel-first format [N, C, H, W] when necessary.
    """

    def __init__(self, x, y, transform=None):
        self.x = torch.as_tensor(x).float()

        # Add a channel dimension for grayscale datasets stored as [N, H, W].
        if len(self.x.shape) == 3:
            self.x = torch.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1], self.x.shape[2]))

        # Labels are optional because the test dataset may not contain targets.
        if y is None:
            self.y = None
        else:
            self.y = torch.tensor(y).long()

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        im = self.x[idx]
        if self.transform is not None:
            im = self.transform(im)
        if self.y is None:
            return im
        return im, self.y[idx]


class DataProcessor:
    """
    Prepares training, validation, and test datasets for the NAS pipeline.

    Processing includes:
    - Converting input arrays to PyTorch datasets
    - Computing channel-wise normalization statistics
    - Estimating a suitable batch size based on available GPU memory
    - Creating the corresponding data loaders
    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def _estimate_batch_size(self, channels, height, width, n_train, n_valid,
                          safety_fraction=0.6, bytes_per_pixel=8,
                          activation_multiplier=42, min_batch=2, max_batch_cap=1024):
        """
        Estimate a suitable batch size based on currently available GPU memory.

        The raw input size of one sample is estimated from its number of
        channels and spatial dimensions. During training, a CNN requires
        substantially more memory for feature maps, activations, gradients,
        parameters, and optimizer states.

        The activation multiplier approximates this additional memory usage.
        The safety fraction reserves part of the available GPU memory for
        CUDA overhead, memory fragmentation, and other processes.

        If no CUDA device is available, a conservative CPU batch size is used.
        """

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            # GPU memory is irrelevant on CPU. Use a conservative batch size
            # to avoid excessive RAM consumption and long iteration times.
            return min(max_batch_cap, 256, n_train, n_valid) if min(n_train, n_valid) > 0 else 1

        try:
            # Query the memory that is currently available on the active GPU.
            free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
        except Exception:
            # Fall back to the total device memory if the free-memory query
            # is unavailable in the installed PyTorch or CUDA version.
            free_bytes = torch.cuda.get_device_properties(DEVICE).total_memory
            total_bytes = free_bytes

        # Keep part of the available memory unused as a safety margin.
        usable_bytes = free_bytes * safety_fraction

        # Approximate the memory required by one sample during training.
        bytes_per_sample = channels * height * width * bytes_per_pixel * activation_multiplier
        # Prevent division by zero for malformed metadata.
        bytes_per_sample = max(bytes_per_sample, 1)

        estimated_batch = int(usable_bytes // bytes_per_sample)
        batch_size = max(min_batch, min(estimated_batch, max_batch_cap))

        print(f"🧮 Available VRAM: {free_bytes / 1e9:.2f}GB of {total_bytes / 1e9:.2f}GB | "
            f"Estimated batch size: {batch_size} (input: {channels}x{height}x{width})")

        return batch_size

    def process(self):
        """
        Process all dataset splits and return their data loaders.

        Returns:
            tuple:
                train_loader, validation_loader, test_loader
        """

        # Metadata input shapes are expected in the format [N, C, H, W].
        _, channels, orig_h, orig_w = self.metadata['input_shape']

        transform_list = []
        target_h, target_w = orig_h, orig_w

        # Resize large square images to limit memory consumption.
        # Non-square images are currently kept at their original resolution.
        if orig_h == orig_w:
            if orig_h > 256:
                target_h, target_w = 256, 256
                transform_list.append(transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR))

        # Create a temporary tensor to calculate normalization statistics.
        tmp_x = torch.as_tensor(self.train_x).float()
        if len(tmp_x.shape) == 3:
            tmp_x = torch.reshape(tmp_x, (tmp_x.shape[0], 1, tmp_x.shape[1], tmp_x.shape[2]))

        # Apply the same resizing operation before calculating statistics.
        if transform_list:
            resizer = transform_list[0]
            tmp_x = resizer(tmp_x)

        # Compute the mean and standard deviation independently for each channel.
        mean = torch.mean(tmp_x, dim=[0, 2, 3]).tolist()
        std = torch.std(tmp_x, dim=[0, 2, 3])
        # Avoid division by zero for constant or nearly constant channels.
        std = torch.clamp(std, min=1e-6).tolist()

        # Release the temporary tensor as early as possible because it may
        # occupy a considerable amount of system memory.
        del tmp_x

        # Apply channel-wise normalization after the optional resize.
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        pipeline = transforms.Compose(transform_list)

        # Create datasets using the same preprocessing pipeline for all splits.
        train_ds = Dataset(self.train_x, self.train_y, transform=pipeline)
        valid_ds = Dataset(self.valid_x, self.valid_y, transform=pipeline)
        test_ds = Dataset(self.test_x, None, transform=pipeline)

        n_train = len(train_ds)
        n_valid = len(valid_ds)

        # Estimate the largest reasonable batch size for the processed
        # input dimensions and currently available GPU memory.
        max_batch = self._estimate_batch_size(
            channels=channels, height=target_h, width=target_w,
            n_train=n_train, n_valid=n_valid
        )

        batch_size = min(max_batch, n_train, n_valid) if min(n_train, n_valid) > 0 else 1
        batch_size = max(batch_size, 2)

        # Create data loaders. Only the training data is shuffled.
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # Update the metadata so that the NAS model receives the actual
        # dimensions produced by the preprocessing pipeline.
        self.metadata['input_shape'] = (self.metadata['input_shape'][0], channels, target_h, target_w)

        return train_loader, valid_loader, test_loader
