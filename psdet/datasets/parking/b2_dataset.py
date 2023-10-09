import json
import math
import numpy as np
from PIL import Image
import scipy.io as sio
from pathlib import Path
from torchvision import transforms as T

from psdet.datasets.base import BaseDataset
from psdet.datasets.registry import DATASETS
from psdet.utils.precision_recall import calc_average_precision, calc_precision_recall

from .process_data import boundary_check, overlap_check, rotate_centralized_marks, rotate_image, generalize_marks
from .utils import match_marking_points, match_slots


def cross_product(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    cross = np.cross(v1, v2)
    return cross


@DATASETS.register
class B2Dataset(BaseDataset):

    def __init__(self, cfg, logger=None):
        super(B2Dataset, self).__init__(cfg=cfg, logger=logger)
        
        assert(self.root_path.exists())
        
        print(self.root_path)
        
        # TODO: 要划分 train 和 val 集
        if cfg.mode == 'train':
            data_dir = self.root_path / 'avm_0331'
        elif cfg.mode == 'val':
            data_dir = self.root_path / 'avm_0331'
        
        assert(data_dir.exists())
        
        # TODO:
        self.anno_files = []
        for p in data_dir.glob('*.json'):
            # load label
            # data = sio.loadmat(str(p))
            # slots = np.array(data['slots'])
            # if slots.size > 0:
                #if slots[0][3] == 90:
                # self.mat_files.append(str(p))
            self.anno_files.append(str(p))
        
        self.anno_files.sort()
        
        if cfg.mode == 'train': 
            # data augmentation
            self.image_transform = T.Compose([T.ColorJitter(brightness=0.1, 
                contrast=0.1, saturation=0.1, hue=0.1), T.ToTensor()])
        else:
            self.image_transform = T.Compose([T.ToTensor()])

        if self.logger:
            self.logger.info('Loading B2 {} dataset with {} samples'.format(cfg.mode, len(self.anno_files)))
       
    def __len__(self):
        return len(self.anno_files)
    
    def __getitem__(self, idx):
        anno_file = Path(self.anno_files[idx]) 
        # load label
        with open(anno_file, 'r') as fp:
            data = json.load(fp)
        
        # TODO: 
        # marks: (N, 2)
        # slots: idx1, idx2, categ1, degree
        # categ1: 90 degree （垂直车位、平行车位）
        # categ2: 67 degree
        marks = []
        data = data['annotations']
        for d in data:
            point_xs = d['polygon'][::2]
            point_ys = d['polygon'][1::2]
            points = np.stack([point_xs, point_ys], axis=1)
            cross = cross_product(points[0], points[1], points[2])
            
            entry_pt = d['entry']
            if cross > 0:  # 顺时针
                x1, y1, x2, y2 = entry_pt
            elif cross < 0:  # 逆时针
                x2, y2, x1, y1 = entry_pt
            marks.append([x1, y1])
            marks.append([x2, y2])
        marks = np.array(marks)
        f = 600 / 3440
        marks = marks * f
        slots = []
        # 目前都是垂直和平行车位
        for i in range(1, len(marks)+1, 2):
            slots.append([i, i+1, 1, 90])
        slots = np.array(slots)
        
        # print(slots.shape, marks.shape)
        
        assert slots.size > 0
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)
        if len(slots.shape) < 2:
            slots = np.expand_dims(slots, axis=0)

        num_points = marks.shape[0]
        max_points = self.cfg.max_points
        # assert max_points >= num_points
        if max_points < num_points:
            slots = slots[slots[:,0] <= max_points]
            slots = slots[slots[:,1] <= max_points]
            marks = marks[:max_points,:]
            num_points = marks.shape[0]
            # print('max_points: ', max_points)
            # print('num_points: ', num_points)

        # centralize (image size = 600 x 600)
        marks[:,0:4] -= 300.5
        
        img_file = str(self.anno_files[idx]).replace('.json', '.png')
        image = Image.open(img_file)
        image = image.resize((512,512), Image.BILINEAR)
        
        marks = generalize_marks(marks)
        image = self.image_transform(image)
         
        # make sample with the max num points
        marks_full = np.full((max_points, marks.shape[1]), 0.0, dtype=np.float32)
        marks_full[:num_points] = marks
        match_targets = np.full((max_points, 2), -1, dtype=np.int32)
        
        for slot in slots:
            match_targets[slot[0] - 1, 0] = slot[1] - 1
            match_targets[slot[0] - 1, 1] = 0 # 90 degree slant

        input_dict = {
                'marks': marks_full,
                'match_targets': match_targets,
                'npoints': num_points,
                'frame_id': idx,
                'image': image
                }
        
        return input_dict 

    def generate_prediction_dicts(self, batch_dict, pred_dicts):
        pred_list = []
        pred_slots = pred_dicts['pred_slots']
        for i, slots in enumerate(pred_slots):
            single_pred_dict = {}
            single_pred_dict['frame_id'] = batch_dict['frame_id'][i]
            single_pred_dict['slots'] = slots
            pred_list.append(single_pred_dict)
        return pred_list
     
    def evaluate_point_detection(self, predictions_list, ground_truths_list):
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_marking_points)
        average_precision = calc_average_precision(precisions, recalls)
        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Point detection: average_precision {}'.format(average_precision))

    def evaluate_slot_detection(self, predictions_list, ground_truths_list):
                
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_slots)
        average_precision = calc_average_precision(precisions, recalls)

        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Slot detection: average_precision {}'.format(average_precision))
