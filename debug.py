import cv2
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.yolo import Model
from utils.general import check_dataset

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """Plots one bounding box on image img"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

def _make_grid(nx=20, ny=20, i=0):
    """Generates a mesh grid for anchor boxes."""
    d = detect_module.anchors[i].device
    t = detect_module.anchors[i].dtype
    shape = (1, detect_module.na, int(ny), int(nx), 2)  # grid shape
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = (detect_module.anchors[i] * detect_module.stride[i]).view((1, detect_module.na, 1, 1, 2)).expand(shape)
    return grid, anchor_grid

# Load hyperparameters
device = 'cpu'
hyp_path = 'data/hyps/hyp.scratch-low.yaml'
with open(hyp_path, errors="ignore") as f:
    hyp = yaml.safe_load(f)  # load hyps dict

# Load model configuration and data
cfg = 'models/yolov5n_nuscenes.yaml'
data = 'data/nuscenes.yaml'
data_dict = check_dataset(data)
nc = int(data_dict["nc"])  # number of classes

# Create model
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)

# Get anchors from the model
anchors = model.model[-1].anchors

# [TODO] Draw anchors
# Load and preprocess image
image_path = 'datasets/nuscenes/test/images/n008-2018-08-01-15-34-25-0400__CAM_FRONT__1533152214512404.jpg'  # 이미지 경로 설정
image = cv2.imread(image_path)
assert image is not None, 'Image not found'
img_h, img_w = image.shape[:2]

# Preprocess image for model input
img_size = 416  # input image size
img = cv2.resize(image, (img_size, img_size))
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, 0)
img = torch.from_numpy(img).float().to(device) / 255.0

# Draw anchor boxes on the image
detect_module = model.model[-1]

stride = detect_module.stride
nl = detect_module.nl  # number of detection layers
na = detect_module.na  # number of anchors

# Define colors for different anchors
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

step = 13  # grid cell step
for i in range(nl):
    nx, ny = int(img_size // stride[i]), int(img_size // stride[i])  # number x, y grid points
    grid, anchor_grid = _make_grid(nx, ny, i)

    # Draw anchor boxes
    for y in range(0, ny, step):
        for x in range(0, nx, step):
            for a in range(na):
                # Calculate anchor box coordinates
                x_center = (grid[0, a, y, x, 0] + 0.5) * stride[i]
                y_center = (grid[0, a, y, x, 1] + 0.5) * stride[i]
                w = anchor_grid[0, a, y, x, 0]
                h = anchor_grid[0, a, y, x, 1]
                x1 = int((x_center - w / 2).item())
                y1 = int((y_center - h / 2).item())
                x2 = int((x_center + w / 2).item())
                y2 = int((y_center + h / 2).item())

                # Convert coordinates to image size
                x1 = int(x1 * img_w / img_size)
                y1 = int(y1 * img_h / img_size)
                x2 = int(x2 * img_w / img_size)
                y2 = int(y2 * img_h / img_size)

                # Draw anchor box
                plot_one_box([x1, y1, x2, y2], image, color=colors[a % len(colors)], label=f'Anchor {a}', line_thickness=1)

# Display image with anchor boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Anchors on Image')
plt.savefig('./anchors_on_image.png')