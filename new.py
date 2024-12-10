import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# UNet Autoencoder
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder: Downsampling
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, pool=False)

        # Decoder: Upsampling
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 256, pool=False)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 128, pool=False)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 64, pool=False)

        # Final Layer
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Decoder
        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = self.dec1(dec1)

        # Final output layer
        out = self.final(dec1)
        return out

# Dataset
class NormalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Anomaly Detection Function
def detect_anomaly(image, model, threshold=0.1):
    model.eval()
    with torch.no_grad():
        reconstructed_image = model(image.unsqueeze(0)) 
        
        # Calculate pixel-wise reconstruction error
        reconstruction_error = torch.mean((reconstructed_image - image.unsqueeze(0)) ** 2, dim=1)
        reconstruction_error = reconstruction_error.squeeze().cpu().numpy()
    
    # Normalize reconstruction error for visualization
    error_normalized = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min())
    
    # Create binary mask based on threshold
    anomaly_mask = error_normalized > threshold
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    original_img = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # # Reconstructed image
    # plt.subplot(1, 3, 2)
    # reconstructed_img = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # plt.imshow(reconstructed_img)
    # plt.title("Reconstructed Image")
    # plt.axis('off')
    
    # Anomaly Mask
    plt.subplot(1, 2, 2)
    plt.imshow(error_normalized, cmap='hot')
    plt.title("Anomaly Mask")
    # plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print anomaly detection result
    is_anomaly = anomaly_mask.any()
    print(f"Anomaly detected: {is_anomaly}")
    
    return is_anomaly

# Training Function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs): 
        model.train()
        total_loss = 0.0
        for images in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)  # MSE loss between input and reconstructed image
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Main Execution
def main():
    # Dataset and DataLoader
    dataset = NormalImageDataset(
        root_dir='/Users/2chenglengzai/Desktop/fyp/mvtec_anomaly_detection/bottle/train/good', 
        transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instantiate model, loss function, and optimizer
    model = UNetAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    # train_model(model, train_loader, criterion, optimizer)

    # Test with an anomalous image
    test_image = Image.open("/Users/2chenglengzai/Desktop/fyp/mvtec_anomaly_detection/bottle/test/good/000.png").convert("RGB")
    test_image = transform(test_image)
    detect_anomaly(test_image, model, threshold=0.1)

if __name__ == "__main__":
    main()