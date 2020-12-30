# ============================================
__author__ = "jhhuh"
__maintainer__ = "jhhuh"
# ============================================

import os
from torch.utils import data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

CUSTOM_CLASS_LIST = ['__background__', 'car_front', 'car_etc', 'car_back', 'car_plate', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '가', '거', '경', '고',
            '구', '기', '나', '너', '노', '누', '다', '더', '도', '두', '라', '러', '로', '루',
            '리', '마', '머', '모', '무', '바', '배', '버', '보', '부', '사', '서', '소', '수',
            '시', '아', '어', '오', '우', '울', '임', '자', '장', '저', '조', '주', '청', '하',
            '허', '호']

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

        # self.data_dir = root_dir
        # self.split = split
        self.transform = transform
        self.target_transform = target_transform
        # self.keep_difficult = keep_difficult
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
    
    # extract bbox from xml
    def _extract_bbox(search_object):
        for bbox in search_object :
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
        return x1, y1, x2, y2

    def _get_annotation(self, image_id):
        annotation_file = f"/data/yper_data/converted_xml/{image_id}.xml"
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        boxes = []
        labels = []
        main_body = []

        for object in root.findall('smr_object') :
            class_name = object.find('name').text
            if class_name in ['car_front', 'car_etc', 'car_back'] :
                x1, y1, x2, y2 = _extract_bbox(object.findall('bndbox'))

                main_body = [x1, y1, x2, y2]
                boxes.append(main_body)

                labels.append(self.class_dict[class_name])
                break
            else :
                continue
        for object in root.findall('smr_object') :
            class_name = object.find('name').text
            if class_name == 'car_plate' :
                lp_label = object.find('pose').text
                x1, y1, x2, y2 = _extract_bbox(object.findall('bndbox'))
                try :
                    if (x1 < main_body[0]) or (x2 > main_body[2]) or (y1 < main_body[1]) or (y2 > main_body[3]) :
                        continue
                    elif lp_label in list(self.class_names) :
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_dict[lp_label])
                except :
                    if lp_label in list(self.class_names) :
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_dict[lp_label])
                    else :
                        continue
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_id):
        image_file = f"/data/yper_data/converted_img/resized_{image_id}.jpg"
        image = Image.open(str(image_file)).convert("RGB")
        image = np.array(image)
        return image
