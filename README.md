# Prospace

This repository contains implementation of image segmentation model supported by <a href=https://gymnasium.farama.org](https://pytorch.org/docs/stable/index.html/>pytorch</a> library.
# Usage
* clone this repository into your local machine
<pre>
<code>
git clone https://github.com/prabandh1444/Prospace.git
</code>
</pre>   
* Usage in python 3.10
<pre>
<code>
pip install torchvision
pip install pillow
pip3 install numpy
pip3 install mathplotlib
</code>
</pre>

# How to Run:
* Running in Python 3.10
<pre>
<code>
python3 draw.py
</code>
</pre>

# Code Walkthrough:
Each notebook file contains the following 3 classes:

## Dataset Creation
This contains implementation of creation of dataset which contain dark shapes on light background
```python
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
```

## Neural Network model
Contains implementation of the CNN which takes 1024x1024x1024 RGB image and outputs 1024x1024x1 mask
```python
def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x
```

## AgentDQN
This contains the implementation of  training loop and the test loop ( Cross Binary Entropy loass and Adam Optimizer)
```python
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
```
### Hyperparameters
* Try to change them as you please
```python
random.seed(0) 
width = 1024
height = 1024
samples = 100
test_samples = 5
```
# Metrics and results
### Training loss
Below shows traing Loss values and plot of {loss vs epoch}
![Screenshot from 2024-02-05 18-47-11](https://github.com/prabandh1444/Prospace/assets/111416767/08dd4a20-bcfe-4224-95bd-5068fa5dc26f)
![plot](https://github.com/prabandh1444/Prospace/assets/111416767/f1a7f2bc-d9d6-46f4-a4cf-54df36d41103)
### Visualization
* First column respresnts the actual image
* Second column represnents the actual mask
* Third coulmn represents the predicted mask
![save-1](https://github.com/prabandh1444/Prospace/assets/111416767/2ae78ee8-f7df-4cbd-9c51-356ee77c9359)
![save-2](https://github.com/prabandh1444/Prospace/assets/111416767/264afafc-0ffb-44d0-a0b8-36eed556b398)
![save-3](https://github.com/prabandh1444/Prospace/assets/111416767/4fae6650-6634-4bcf-add4-e343208118ad)
![save-4](https://github.com/prabandh1444/Prospace/assets/111416767/3dcfd93d-b63c-4cc7-8edb-78d15481e5a8)
![save-5](https://github.com/prabandh1444/Prospace/assets/111416767/347328f0-e19f-41ff-84f5-69e36eb854af)


# REFS:
  https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/?utm_source=blog&utm_medium=introduction-image-segmentation-techniques-python

  https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/?utm_source=blog&utm_medium=computer-vision-implementing-mask-r-cnn-image-segmentation
