import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Define your UNet model here (same as in your previous code)
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, amplification_factor=100):
        super(UNet, self).__init__()
        self.amplification_factor = amplification_factor
        
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = block(in_channels, 64)
        self.encoder2 = block(64, 128)
        self.encoder3 = block(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        self.middle = block(256, 512)
        
        self.upconv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.decoder3 = block(512, 256)
        self.upconv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.decoder2 = block(256, 128)
        self.upconv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.decoder1 = block(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Apply amplification factor
        x = x * self.amplification_factor
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        
        middle = self.middle(self.pool(enc3))
        
        dec3 = nn.functional.interpolate(middle, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = self.center_crop(enc3, dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = nn.functional.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = self.center_crop(enc2, dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = nn.functional.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = self.center_crop(enc1, dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        y = self.out_conv(dec1)
        return y
    
    def center_crop(self, enc, dec):
        _, _, H, W = enc.size()
        _, _, h, w = dec.size()
        x1 = (W - w) // 2
        y1 = (H - h) // 2
        return enc[:, :, y1:y1+h, x1:x1+w]

# Dataset class for test images
class TestDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(input_dir))  # Ensure sorted order for consistency

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = Image.open(input_path).convert("RGB")
        
        if self.transform:
            input_image = self.transform(input_image)
        
        return input_image, img_name

# Function to predict images and save them
def predict_images(model, dataloader, output_dir):
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for batch_idx, (inputs, img_names) in enumerate(dataloader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        
        for i in range(outputs.size(0)):
            output_img = outputs[i].cpu().detach()
            output_img = transforms.ToPILImage()(output_img).convert("RGB")
            output_img = output_img.resize((600, 400))
            output_path = os.path.join(output_dir, img_names[i])
            output_img.save(output_path)
            print(f"Saved: {output_path}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
input_dir = 'test/low/'
output_dir = 'test/predicted/'

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = TestDataset(input_dir=input_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
model = UNet(in_channels=3, out_channels=3).to(device)

# Load pretrained weights
weights_path = 'best_final_unet_final.pth'
model.load_state_dict(torch.load(weights_path, map_location=device))

# Predict and save images
predict_images(model, test_dataloader, output_dir)
