# ============================================
__author__ = "jhhuh"
__maintainer__ = "jhhuh"
# ============================================

import os
from torch.utils import data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

CUSTOM_CLASS_LIST = ['__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

class CustomDataset(data.Dataset):

    def __init__(self, trainset_dir, valset_dir, transform=None, target_transform=None, keep_difficult=False, is_training=True):
        '''
        Dataset for CustomSet
        :param root_dir: the Trainset Path
        :param transform: Image transforms
        :param target_transform: Box transforms
        :param keep_difficult: Keep difficult or not
        :param is_training: True if Training else False
        '''
        super(CustomDataset, self).__init__()
        self.trainset_dir = trainset_dir
        self.valset_dir = valset_dir
        if is_training:
            search_paths = os.path(trainset_dir)
        else:
            search_paths = os.path(valset_dir)

        self.ids = CustomDataset._read_image_ids(search_paths)

        self.data_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self.CLASSES = CUSTOM_CLASS_LIST

        self.class_dict = {class_name: i for i, class_name in enumerate(self.CLASSES)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                image_path = os.path.splitext(line.rstrip().split('/')[-1])[0]
                ids.append(image_path)
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, self.split, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_id):
        image_file = f"/data/yper_data/converted_img/resized_{image_id}.jpg"
        image = Image.open(str(image_file)).convert("RGB")
        image = np.array(image)
        return image
