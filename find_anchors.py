import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

labels = []

with open('/home/yuseung/datasets/kaist-rgbt/train-all-04.txt', 'r') as file:
    lines = file.readlines()
    for idx, line in enumerate(lines):
        img_path = line.strip().replace("{}", "")
        file_name = os.path.basename(img_path).replace('.jpg', '')
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

        with open(label_path, 'r') as lbl_file:
            for anno_id, line in enumerate(lbl_file):
                category_id, x, y, w, h, occlusion = map(float, line.split())
                labels.append([x, y, w, h])
                
labels = np.array(labels)
            
# Extract width and height of bounding boxes in labels
widths = labels[:, 2] * 640  # width
heights = labels[:, 3] * 512  # height
boxes = np.column_stack((widths, heights))

# K-means clustering
n_clusters = 9  # number of anchors
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(boxes)
anchors = kmeans.cluster_centers_

print("Optimal anchors: ", anchors)

plt.figure(figsize=(10, 6))
plt.scatter(boxes[:, 0], boxes[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', s=50, alpha=0.6, edgecolor='k')
plt.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='x', s=200, label='Anchors')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.title('Bounding Box Width and Height Clustering')
plt.legend()
plt.grid(True)
plt.savefig('./optimal_anchors.png')