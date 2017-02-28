"""
Processor is in charge of task-specific processing

A processor is in charge of:
1. Given a Folder, generate crops of parts
2. Import external labels into frames/parts
"""

from db2 import *
from huva import rfcn
import cv2
import os
from glob import glob

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
class CarBBoxProcessor:
    """
    To load first batch:
    1. init_folder one by one
    2. run_rfcn one by one
    3. path_small_crop_folder(folder, True)
    4. load_small_crop_csv

    To handle semi-supervised:
    1. db.add_folder one by one
    2. iterate:
       1. run auto_label on the folders one by one
       2. select a subset of cars that have no label, manually label them
       3. train a model on all labelled cars
    """
    def __init__(self, divisible_by=5):
        # divisible_by used for filtering frames
        self.divisible_by = divisible_by

    def path_small_crop_folder(self, folder, create=False):
        full_path = os.path.join(folder.absolute_path, 'small_crops')
        if create:
            create_folder(full_path)
        return full_path

    def path_big_crop_folder(self, folder, create=False):
        full_path = os.path.join(folder.absolute_path, 'big_crops')
        if create:
            create_folder(full_path)
        return full_path

    def init_folder(self, folder_abs_path, db):
        assert os.path.exists(folder_abs_path)
        folder = Folder(folder_abs_path, db, processor=self)
        db.add_folder(folder)
        fpaths = glob(folder_abs_path+'/*.jpg')
        fpaths = sorted(fpaths)
        for fpath in fpaths:
            fname = fpath.split('/')[-1]
            frame = Frame(folder, fname)
            folder.add_frame(frame)
        return folder



    def frame_selector(self, frame):
        if len(frame.parts) > 0: return False
        """ all the frames are named {header}_{frameidx}.jpg """
        frame_idx = int(frame.path.split('.')[0].split('_')[-1])
        if frame_idx % self.divisible_by !=0: return False
        return True

    def run_rfcn(self, folder):
        rfcn.make()
        key = 'run_rfcn_already' 
        assert key not in folder.attrs
        folder.attrs[key] = True
        for frame in folder.frames:
            assert len(frame.parts)==0
            # we are not recording whether the frame is selected, only if rfcn has been run on it
            if not self.frame_selector(frame): 
                continue
            car_bboxes = rfcn.get_car_bboxes(rfcn.net, frame.absolute_path())
            frame.attrs['rfcn_run']=True
            frame.parts = []
            for bbox_idx, car_bbox in enumerate(car_bboxes):
                bbox = BBox(frame, car_bbox)
                frame.add_part(bbox)
            print(frame.path)

    def write_small_crops(self, folder):
        """ write all BBox directly under any frame """
        crop_folder_path = self.path_small_crop_folder(folder, True)
        for frame in folder.frames:
            frame_img = cv2.imread(frame.absolute_path())
            for bbox in frame.parts:
                crop_path_full = os.path.join(crop_folder_path, bbox.unique_name())
                bbox_crop = bbox.get_crop(frame_img)
                cv2.imwrite(crop_path_full, bbox_crop)

    def load_small_crop_csv(self, folder):
        crop_folder_path = self.path_small_crop_folder(folder)
        assert os.path.exists(crop_folder_path)
        csv_name = os.path.join(crop_folder_path, 'via_region_data.csv')
        """
        Returns filename_to_box: {'287_0102_0.jpg': [] or [x,y,width,height]}
        """
        fname_to_box = rfcn.read_csv(crop_folder_path)
        for fname, bbox in fname_to_box.iteritems():
            # if doesn't contain plate_bbox, just skip
            if len(bbox) != 4: continue
            x,y,w,h = bbox
            x1,y1,x2,y2 = x,y,x+w,y+h
            """
            translate fname to frame_idx, part_idx
            two formats:
            1. {folder_idx}#{frame_idx}#{part_idx}.jpg
            2. {header}_{frame_idx+1}_{part_idx}.jpg
            """
            name_clean = fname.split('.')[0]
            if '#' in name_clean: # first format
                folder_idx, frame_idx, part_idx = map(int, name_clean.split('#'))
                assert folder.folder_id == folder_idx
            else: # second format
                header, frame_idx, part_idx = name_clean.split('_')
                frame_idx = int(frame_idx) - 1 # frames start at 1, convert to 0-based
                part_idx  = int(part_idx)
            assoc_frame = folder.frames[frame_idx]   # frame
            car_bbox  = assoc_frame.parts[part_idx]  # car bbox
            assert isinstance(car_bbox, BBox)
            car_x, car_y, _, _ = car_bbox.bbox
            plate_bbox= BBox(car_bbox, (x1+car_x,y1+car_y,x2+car_x,y2+car_y))
            car_bbox.add_part(plate_bbox)


import cv2
import numpy as np
from matplotlib import pyplot as plt
cmap = plt.get_cmap('jet')
def heat_all_plates(frame):
    img = cv2.imread(frame.absolute_path())
    label = np.zeros((img.shape[0], img.shape[1]), np.float32)
    for car_bbox in frame.parts:
        for plate_bbox in car_bbox.parts:
            x1,y1,x2,y2 = plate_bbox.bbox
            label[y1:y2, x1:x2] = 1
    jet = (cmap(label)[:,:,[0,1,2]] * 255).astype(np.uint8)
    plt.imshow(img/2 + jet/2)
    plt.show()



batch1_folders = [7287,7288,7289,7290]
batch1_folders = ['{}_frames'.format(i) for i in batch1_folders]

batch2_folders = [7293,7294,7295,7296,7297,7298,7299,7300]
batch2_folders = ['{}_frames'.format(i) for i in batch2_folders]

batch3_folders = [7308,7309,7310,7310,7312,7313,7314,7315]
batch3_folders = ['{}_frames'.format(i) for i in batch3_folders]

