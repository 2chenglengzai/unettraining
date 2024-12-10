import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Dataset class for loading images
class BottleDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = torch.zeros((256, 256), dtype=torch.float32) 
        if self.transform:
            image = self.transform(image)
        return image, label


# Transformations for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = self.conv_block(1024, 2048)

        self.up1 = self.upconv_block(2048, 1024) 
        self.up2 = self.upconv_block(1024, 512)
        self.up3 = self.upconv_block(512, 256)
        self.up4 = self.upconv_block(256, 128)
        self.up5 = self.upconv_block(128, 64) 

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4)) 

        bottleneck = self.bottleneck(self.pool(enc5))

        dec1 = self.up1(bottleneck) + enc5
        dec2 = self.up2(dec1) + enc4
        dec3 = self.up3(dec2) + enc3
        dec4 = self.up4(dec3) + enc2
        dec5 = self.up5(dec4) + enc1

        return self.out_conv(dec5)


def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])  
    tensor = tensor * std[:, None, None] + mean[:, None, None] 
    return tensor.clamp(0, 1) 

def train(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    print("Training complete.")
    return model

def test(model, image_path, transform):
    model.eval()
    model.load_state_dict(torch.load("unet_bottle_segmentation.pth"))
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image).unsqueeze(0) 
    with torch.no_grad():
        output = torch.sigmoid(model(image)) 
        binary_mask = (output > 0.5).float() 
    return image.squeeze(), binary_mask.squeeze()


def visualize_results(input_image, predicted_mask, save_path="predicted_mask.png"):
    predicted_mask_np = (predicted_mask.numpy() * 255).astype(np.uint8)

    cv2.imwrite(save_path, predicted_mask_np)
    print(f"Predicted mask saved to {save_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_image.permute(1, 2, 0).numpy())
    axes[0].set_title("Input Image")
    axes[1].imshow(predicted_mask_np, cmap="gray")
    axes[1].set_title("Predicted Mask")
    plt.show()


def main():
    image_path = "/Users/2chenglengzai/Desktop/fyp/mvtec_anomaly_detection/bottle/test/broken_large/000.png"
    model = UNet()
    model_weights = torch.load("/Users/2chenglengzai/Desktop/fyp/segmentation/unet_bottle_segmentation.pth", weights_only=True)
    model.load_state_dict(model_weights) 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_image, prediction = test(model, image_path, transform)
    input_image = denormalize(input_image)
    visualize_results(input_image, prediction)


if __name__ == "__main__":
    model = UNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    dataset = BottleDataset(
        image_dir="/Users/2chenglengzai/Desktop/fyp/mvtec_anomaly_detection/bottle/train/good",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = train(model, dataloader, criterion, optimizer, epochs=10)
    torch.save(model.state_dict(), "unet_bottle_segmentation.pth")    
    main()
