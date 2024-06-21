import random
import os
import json

def create_kaist_json(val_txt_path, output_json_path):
    categories = [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "cyclist"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "person?"}
    ]
    images = []
    annotations = []

    with open(val_txt_path, 'r') as file:
        lines = file.readlines()
        id = 0
        for idx, line in enumerate(lines):
            img_path = line.strip().replace("{}", "")
            file_name = os.path.basename(img_path).replace('.jpg', '')
            image_id = idx
            width, height = 640, 512

            images.append({"image_name": file_name, "id": image_id, "width": width, "height": height})

            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

            with open(label_path, 'r') as lbl_file:
                for anno_id, line in enumerate(lbl_file):
                    category_id, x, y, w, h, occlusion = map(float, line.split())
                    bbox = [x * width, y * height, w * width, h * height]
                    annotations.append({
                        "image_name": file_name,
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": bbox,
                        "iscrowd": 0,
                        "occlusion": int(occlusion),
                        "width": bbox[2],
                        "height": bbox[3],
                        "id": id
                    })
                    id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)
    print(f"KAIST_annotation.json has been created: {output_json_path}")

def split_dataset(base_path, file_name, train_ratio=0.8):
    # Construct the full path to the file
    file_path = os.path.join(base_path, file_name)
    
    # Read the lines from the file
    with open(file_path, 'r') as file:
        image_paths = file.readlines()

    # Strip newline characters and shuffle the list
    image_paths = [line.strip() for line in image_paths]
    random.shuffle(image_paths)

    # Calculate the split index
    split_index = int(len(image_paths) * train_ratio)

    # Split the data
    train_images = sorted(image_paths[:split_index])  # Sort the training images
    val_images = sorted(image_paths[split_index:])    # Sort the validation images

    # Write the training images to train-split.txt in the specified directory
    with open(os.path.join(base_path, 'train-split.txt'), 'w') as file:
        for path in train_images:
            file.write(path + '\n')

    # Write the validation images to val-split.txt in the specified directory
    with open(os.path.join(base_path, 'val-split.txt'), 'w') as file:
        for path in val_images:
            file.write(path + '\n')

    print(f"Written {len(train_images)} dataset to {os.path.join(base_path, 'train-split.txt')}")
    print(f"Written {len(val_images)} dataset to {os.path.join(base_path, 'val-split.txt')}")
        
    return os.path.join(base_path, 'val-split.txt')  # Return path to validation split

if __name__ == "__main__":
    val_txt_path = split_dataset('./datasets/kaist-rgbt', 'train-all-04.txt')
    create_kaist_json(val_txt_path, './utils/eval/KAIST_annotation.json')




