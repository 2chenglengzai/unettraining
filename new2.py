import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_names = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_names[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out


def train_model(model, dataloader, criterion, optimizer, device, epochs=10, save_path="unet_model.pth"):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def test_model(model, dataloader, device, load_path="unet_model.pth"):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    print(f"Model loaded from {load_path}")
    all_images = []
    all_outputs = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            all_images.append(images.cpu())
            all_outputs.append(outputs.cpu())
    return torch.cat(all_images, 0), torch.cat(all_outputs, 0)


# if __name__ == "__main__":
#     train_image_folder = "/Users/2chenglengzai/Desktop/fyp/mvtec_anomaly_detection/bottle/train/good"
#     test_image_folder = "/Users/2chenglengzai/Desktop/fyp/mvtec_anomaly_detection/bottle/test/broken_small"
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
#     train_dataset = ImageDataset(train_image_folder, transform=transform)
#     train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     test_dataset = ImageDataset(test_image_folder, transform=transform)
#     test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Unet(3, 3).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     # train_model(model, train_dataloader, criterion, optimizer, device, epochs=10, save_path="unet_model.pth")
#     test_images, reconstructed_images = test_model(model, test_dataloader, device, load_path="unet_model.pth")
#     test_images = test_images.numpy().transpose(0, 2, 3, 1)
#     reconstructed_images = reconstructed_images.numpy().transpose(0, 2, 3, 1)

#     for i in range(5): 
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 3, 1)
#         plt.title("Test Input Image")
#         plt.imshow(test_images[i]) 
#         plt.subplot(1, 3, 2)
#         plt.title("Reconstructed Image")
#         plt.imshow(reconstructed_images[i]) 

#         difference = np.abs(test_images[i] - reconstructed_images[i]) 
#         plt.subplot(1, 3, 3)
#         plt.title("Difference (Optional)")
#         plt.imshow(np.mean(difference, axis=-1), cmap="gray") 
#         plt.colorbar()

#         plt.show()



