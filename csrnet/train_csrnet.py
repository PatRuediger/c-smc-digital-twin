import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import tqdm
import numpy as np
import csv

from csrnet.csrnet_model import CSRNet
from csrnet.csrnet_dataset import CSRNetDataset

def train_csrnet():
    # --- Configuration ---
    dataset_root = 'methods_comparison/csrnet_data'
    output_dir = 'methods_comparison/checkpoints'

    # Training parameters
    epochs = 100
    batch_size = 16 # Adjust based on your GPU memory
    learning_rate = 1e-5

    # Device configuration: cuda > mps (Apple Silicon) > cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets and dataloaders
    train_dataset = CSRNetDataset(root_dir=dataset_root, split='train', transform=transform)
    val_dataset = CSRNetDataset(root_dir=dataset_root, split='val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Data loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    # --- Model, Loss, and Optimizer ---
    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    best_mae = float('inf')
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV Logging ---
    csv_path = os.path.join(output_dir, 'results.csv')
    csv_header = ['epoch', 'train_loss', 'val_mae']
    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, density_maps in progress_bar:
            images = images.to(device)
            density_maps = density_maps.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, density_maps)
            epoch_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch+1} Training Loss: {epoch_loss / len(train_loader):.4f}")

        # --- Validation Phase ---
        model.eval()
        epoch_mae = 0.0
        
        with torch.no_grad():
            progress_bar_val = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, density_maps in progress_bar_val:
                images = images.to(device)
                
                # Forward pass
                pred_density_maps = model(images)
                
                # Calculate predicted count and ground truth count
                pred_count = pred_density_maps.detach().cpu().sum()
                gt_count = density_maps.detach().cpu().sum()
                
                # Update Mean Absolute Error (MAE)
                epoch_mae += abs(pred_count - gt_count)

        epoch_mae /= len(val_dataset)
        print(f"Epoch {epoch+1} Validation MAE: {epoch_mae:.2f}")

        # Save results to CSV
        train_loss = epoch_loss / len(train_loader)
        with open(csv_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch + 1, train_loss, epoch_mae.item()])

        # Save the best model
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            best_model_path = os.path.join(output_dir, 'csrnet_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved with MAE: {best_mae:.2f}")

    print("✅ Training complete.")
    print(f"Best validation MAE: {best_mae:.2f}")
    print(f"Best model saved at: {best_model_path}")

if __name__ == '__main__':
    train_csrnet()
