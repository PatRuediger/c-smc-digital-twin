import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import scipy.ndimage

class CSRNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, sigma=4):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
                               Assumes a structure like:
                               - root_dir/
                                 - images/
                                   - train/
                                   - val/
                                 - points_annotations/
                                   - train/
                                   - val/
            split (string): 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
            sigma (float): Sigma for Gaussian kernel in density map.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.sigma = sigma

        self.image_dir = os.path.join(root_dir, 'images', split)
        self.points_dir = os.path.join(root_dir, 'points_annotations', split)

        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load points
        points_name = os.path.splitext(img_name)[0] + '.csv'
        points_path = os.path.join(self.points_dir, points_name)
        
        points = []
        if os.path.exists(points_path):
            points_df = pd.read_csv(points_path)
            points = points_df[['x', 'y']].values

        # Generate density map
        # The model's output is 1/8th of the input size
        output_stride = 8
        density_map = self.generate_density_map(image.size, points, output_stride)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert density map to tensor
        density_map = torch.from_numpy(density_map).unsqueeze(0) # Add channel dimension

        return image, density_map

    def generate_density_map(self, image_size, points, output_stride):
        """Generates a density map downsampled by the output_stride."""
        width, height = image_size
        
        # Create a map at the original image size
        full_size_map = np.zeros((height, width), dtype=np.float32)
        for x, y in points:
            if 0 <= x < width and 0 <= y < height:
                full_size_map[int(y), int(x)] = 1
        
        # Apply Gaussian filter
        full_size_map = scipy.ndimage.gaussian_filter(full_size_map, sigma=self.sigma, mode='constant')
        
        # Downsample the map
        # This is a simple form of downsampling by summing blocks.
        # A more robust way might involve interpolation, but this is common.
        target_height = height // output_stride
        target_width = width // output_stride
        
        downsampled_map = np.zeros((target_height, target_width), dtype=np.float32)
        for r in range(target_height):
            for c in range(target_width):
                start_r, end_r = r * output_stride, (r + 1) * output_stride
                start_c, end_c = c * output_stride, (c + 1) * output_stride
                downsampled_map[r, c] = np.sum(full_size_map[start_r:end_r, start_c:end_c])
                
        return downsampled_map
