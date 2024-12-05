import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os, glob


class OpenDataset(data.Dataset):

    type_name = ['clock', 'cover_plate', 'insulator', 'silicone_bucket']

    def __init__(self,
                root, 
                transform,
                target_transform,
                detect=False, 

                ) -> None:
        super().__init__()
        self.root = root
        self.detect = detect
        self.transform = transform
        self.target_transform = target_transform

        self.data_all = {}
        self.cls_names = [tname.replace('_', ' ') for tname in self.type_name]
        for tname in self.type_name:
            item_dir = os.path.join(root, tname)
            img_paths = glob.glob(os.path.join(item_dir, "*/*.jpg"))
            for img_path in img_paths:
                img_index = int(img_path.split("/")[-1].split(".")[0])
                if 'mask' in img_path:
                    anomaly_tag = 1
                    mask_path = img_path.replace("jpg", "png")
                else:
                    anomaly_tag, mask_path = 0, None
                self.data_all[img_index] = \
                    {"img_path": img_path, "cls_name": tname.replace('_', ' '), "anomaly": anomaly_tag, "mask_path": mask_path, "boxes": []}

            if detect:
                # detect_item_dir = os.path.join(detect_dir, tname)
                detect_item_file = os.path.join(item_dir, "loc.txt")
                with open(detect_item_file) as f:
                    item_list = eval("[" + f.read().replace("\n", ",") + "]")
                    for fname, loc in item_list:
                        index = int(fname.split("dino_")[0])
                        loc = [int(max(0, loc[0])), int(max(0, loc[1])), int(loc[2]), int(loc[3])]
                        if index in self.data_all:
                            self.data_all[index]["boxes"].append(loc)
                

    def __len__(self):
        return len(self.data_all)
    
    def get_cls_names(self):
        return self.cls_names
            
    def __getitem__(self, index):
        data = list(self.data_all.values())[index]
        img_path, cls_name, mask_path, anomaly, boxes = \
            data['img_path'], data['cls_name'], data['mask_path'], data['anomaly'], data["boxes"]
        img = Image.open(img_path)
        img_mask = Image.open(mask_path) if mask_path else Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        cropped_images = []
            
        if self.detect:
            for box in boxes:
                cropped_image = img.crop(box)
                cropped_images.append(self.transform(cropped_image) if self.transform else cropped_image)
            img = torch.tensor(np.array(img).transpose(2, 0, 1) / 255.)
        else:
            img = self.transform(img) if self.transform else img
            
        img_mask = self.target_transform(img_mask) if self.target_transform else img_mask            
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': img_path, "boxes": boxes, "cropped": cropped_images}
        
        