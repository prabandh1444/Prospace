from email.mime import image
import os
import random
from PIL import Image, ImageDraw
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def create_dataset(width, height, num_samples):
    dataset = []
    for i in range(num_samples):
        num_shapes = random.randint(5, 15)
        # Light colour shapes
        background_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)
        mask = np.zeros((height, width), dtype=np.uint8)
        for _ in range(num_shapes):
            # choose shape randomly
            shape = random.choice(["circle", "square", "triangle"])
            # randomly choose two vertices
            x1, y1 = random.randint(0, width-176), random.randint(0, height-176)
            t = random.randint(64, 176)
            x2, y2 = x1 + t, y1 + t
            # Dark colour shapes
            shape_color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            if shape == "circle":
                radius = random.randint(4, 11)
                draw.ellipse([x1, y1, x2, y2], fill=shape_color)
                mask_circle = Image.new("L", (width, height), 0)
                draw_circle_mask = ImageDraw.Draw(mask_circle)
                draw_circle_mask.ellipse([x1, y1, x2, y2], fill=255)
                mask = np.maximum(mask, np.array(mask_circle))
            elif shape == "square":
                draw.rectangle([x1, y1, x2, y2], fill=shape_color)
                mask_square = Image.new("L", (width, height), 0)
                draw_square_mask = ImageDraw.Draw(mask_square)
                draw_square_mask.rectangle([x1, y1, x2, y2], fill=255)
                mask = np.maximum(mask, np.array(mask_square))
            elif shape == "triangle":
                # change orientation of the triangle 
                rotation_angle = random.uniform(0, 360)
                vertices = [(x1, y2), (x2, y2), ((x1 + x2) // 2, y1)]
                rotated_vertices = [(math.cos(math.radians(rotation_angle)) * (x - ((x1 + x2) // 2)) - math.sin(math.radians(rotation_angle)) * (y - y1) + ((x1 + x2) // 2),
                                    math.sin(math.radians(rotation_angle)) * (x - ((x1 + x2) // 2)) + math.cos(math.radians(rotation_angle)) * (y - y1) + y1) for x, y in vertices]
                draw.polygon(rotated_vertices, fill=shape_color)
                mask_triangle = Image.new("L", (width, height), 0)
                draw_triangle_mask = ImageDraw.Draw(mask_triangle)
                draw_triangle_mask.polygon(rotated_vertices, fill=255)
                mask = np.maximum(mask, np.array(mask_triangle))
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0  
        mask_tensor = torch.tensor(mask).unsqueeze(0).float() / 255.0  
        dataset.append((img_tensor, mask_tensor))

    return dataset
class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

def train_model(model, train_loader, num_epochs=15, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch_list = []
    loss_list = []
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        epoch_list.append(epoch + 1)
        loss_list.append(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    plt.plot(epoch_list, loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'plot.png')
    plt.savefig(save_path)
    plt.close()

def test_model(model, test_loader):
    model.eval()
    i = 1
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            visualize(inputs[0], targets[0], predictions[0],i)
            i = i+1


def visualize(input_img, target_mask, prediction_mask,i):
    # print(input_img.shape)
    # print(target_mask.shape)
    # print(prediction_mask.shape)
    os.makedirs('results', exist_ok=True)
    input_img_pil = TF.to_pil_image(input_img)
    target_mask_np = target_mask.numpy().squeeze()
    prediction_mask_np = prediction_mask.numpy().squeeze()
    target_mask_pil = Image.fromarray((target_mask_np * 255).astype(np.uint8), mode='L')
    prediction_mask_pil = Image.fromarray((prediction_mask_np * 255).astype(np.uint8), mode='L')
    composite_image = Image.new('RGB', (input_img_pil.width * 3, input_img_pil.height))
    composite_image.paste(input_img_pil, (0, 0))
    composite_image.paste(target_mask_pil, (input_img_pil.width, 0))
    composite_image.paste(prediction_mask_pil, (input_img_pil.width * 2, 0))
    #print(composite_image.shape)
    composite_image.save(f'results/save-{i}.png')


if __name__ == "__main__":
    random.seed(0) 
    width = 1024
    height = 1024
    samples = 100
    test_samples = 5
    dataset = create_dataset(width, height, samples)
    test_dataset = create_dataset(width,height,test_samples)
    random.shuffle(dataset)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = SimpleSegmentationModel()
    train_model(model, train_loader, num_epochs=15, learning_rate=0.01)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)
    test_model(model,test_loader)

