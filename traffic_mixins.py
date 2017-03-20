
from huva import rfcn
import cv2
import os
from glob import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
cmap = plt.get_cmap('jet')

from db2 import BBox, Part, Frame, Folder, Batch, Database, inject_mixin





""" Folder """
class TrafficFolderMixin:

    def data_batch(self, val=None):
        if val is None:
            return self.attrs['data_batch']
        assert isinstance(val, int)
        self.attrs['data_batch'] = val

    def divisible_by(self, val=None):
        if val is None:
            return self.attrs['frame_selected_divisible_by']
        self.attrs['frame_selected_divisible_by'] = val

    def run_rfcn_already(self, val=None):
        if val is None:
            return 'run_rfcn_already' in self.attrs
        if val:
            self.attrs['run_rfcn_already'] = True
        else:
            del self.attrs['run_rfcn_already']

    def update_format(self):
        self.divisible_by(self.processor.divisible_by)
        self.processor = None

    def run_rfcn(self):
        rfcn.make()
        assert not self.run_rfcn_already()
        self.run_rfcn_already(True)
        for frame in self.frames:
            assert len(frame.parts)==0
            # run rfcn on the frame only if the frame_idx is divisible by self.divisible_by()
            frame_idx = int(frame.path.split('.')[0].split('_')[-1])
            if frame_idx % self.divisible_by()!=0: 
                continue
            car_bboxes = rfcn.get_car_bboxes(rfcn.net, frame.absolute_path())
            frame.set_run_rfcn()
            frame.parts = []
            for bbox_idx, car_bbox in enumerate(car_bboxes):
                bbox = BBox(frame, 'car', car_bbox)
                frame.add_part(bbox)
            print(frame.path)

    @staticmethod
    def create_folder_load_jpgs(folder_abs_path, db):
        """
        1. create the folder
        2. add folder to database
        3. load all jpg files under folder as frames
        """
        assert os.path.exists(folder_abs_path)
        """ create """
        folder = Folder(db, 'traffic_folder', folder_abs_path)
        db.add_folder(folder)
        """ load jpgs """
        fpaths = glob(folder_abs_path+'/*.jpg')
        fpaths = sorted(fpaths)
        for fpath in fpaths:
            fname = fpath.split('/')[-1]
            frame = Frame(folder, fname)
            folder.add_frame(frame)
        return folder

    @staticmethod
    def create_folder_load_jpgs_run_rfcn(folder_abs_path, db, data_batch, divisible_by):
        """
        1. create folder, add to database, load all jpgs
        2. set data_batch and divisible_by
        3. run rfcn to detect cars
        4. load saved labels if this is batch1
        """
        folder = TrafficFolderMixin.create_folder_load_jpgs(folder_abs_path, db)
        folder.data_batch(data_batch)
        folder.divisible_by(divisible_by)
        folder.run_rfcn()
        if data_batch==1:
            TrafficFolderMixin.load_csv_for_batch1_folder(folder)
        return folder

    @staticmethod
    def load_csv_for_batch1_folder(folder):
        """
        load folder.absolute_path/small_crops/via_region_data.csv
        """
        assert folder.data_batch() == 1
        crop_folder_path = os.path.join(folder.absolute_path, 'small_crops')
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
            format: {header}_{frame_idx+1}_{part_idx}.jpg
            """
            name_clean = fname.split('.')[0]
            header, frame_idx, part_idx = name_clean.split('_')
            frame_idx = int(frame_idx) - 1 # frames start at 1, convert to 0-based
            part_idx  = int(part_idx)
            assoc_frame = folder.frames[frame_idx]   # frame
            car_bbox  = assoc_frame.parts[part_idx]  # car bbox
            assert isinstance(car_bbox, BBox)
            car_x, car_y, _, _ = car_bbox.bbox
            plate_bbox= BBox(car_bbox, 'plate', (x1+car_x,y1+car_y,x2+car_x,y2+car_y))
            plate_bbox.label_type('manual')
            car_bbox.add_part(plate_bbox)
        """
        For the cars that didn't get a plate label, we need to label them as 'no plate'
        """
        for frame in folder.frames:
            for car_bbox in frame.parts:
                assert isinstance(car_bbox, BBox)
                assert car_bbox.typename == 'car'
                if len([p for p in car_bbox.parts if p.typename=='plate']) == 0:
                    plate_bbox = BBox(car_bbox, 'plate', (0,0,0,0))
                    plate_bbox.label_type('none')
                    car_bbox.add_part(plate_bbox)

inject_mixin('traffic_folder', Folder, TrafficFolderMixin)





""" Frame """
class TrafficFrameMixin:
    def has_run_rfcn(self):
        return 'rfcn_run' in self.attrs
    def set_run_rfcn(self):
        self.attrs['rfcn_run'] = True
    def heat_all_plates(self):
        img = cv2.imread(self.absolute_path())
        label = np.zeros((img.shape[0], img.shape[1]), np.float32)
        for car_bbox in self.parts:
            for plate_bbox in car_bbox.parts:
                x1,y1,x2,y2 = plate_bbox.bbox
                label[y1:y2, x1:x2] = 1
        jet = (cmap(label)[:,:,[0,1,2]] * 255).astype(np.uint8)
        plt.imshow(img/2 + jet/2)
        plt.show()
inject_mixin('traffic_frame', Frame, TrafficFrameMixin)




""" CarBBox """
class CarBBoxMixin:
    def car_type(self, val=None):
        if val is None:
            if 'cartype' not in self.attrs:
                return 'car'
            else:
                return self.attrs['cartype']
        assert val in ['car', 'bus', 'truck']
        self.attrs['cartype'] = val
inject_mixin('car', BBox, CarBBoxMixin)




""" PlateBBox """
class PlateBBoxMixin:
    def label_type(self, val=None):
        """ 
        auto    : automatically labelled by self.attrs['automodel']
        manual  : manually labelled, plate present
        none    : manually labelled, plate absent
        """
        if val is None:
            return self.attrs['type']
        assert label_type in ['auto', 'manual', 'none']
        self.attrs['type'] = label_type

    def auto_model(self, val=None):
        assert self.type=='auto'
        if val is None:
            return self.attrs['automodel']
        self.attrs['automodel'] = val

    def has_auto_sequence(self):
        return 'plate_auto_sequence' in self.attrs

    def auto_sequence(self, val=None):
        if val is None:
            return self.attrs['plate_auto_sequence']
        """ val is [(x,y,char)] """
        assert type(val)==list and all([len(blob)==3 for blob in val])
        self.attrs['plate_auto_sequence'] = val

    def auto_sequence_confirmed(self, val=None):
        if val is None:
            return 'plate_auto_sequence_confirmed' in self.attrs
        if val:
            self.attrs['plate_auto_sequence_confirmed'] = True
        else:
            del self.attrs['plate_auto_sequence_confirmed']

inject_mixin('plate', BBox, PlateBBoxMixin)




""" PlateBatch """
class PlateBatchMixin:
    def output_batch(self):
        """
        output all units of a self into self.output_folder
        """
        output_folder = self.output_folder
        for unit in self.units:
            # Right now we need every unit to be a BBox, otherwise we don't know how to output the image
            assert isinstance(unit, BBox)
            assert unit.typename=='car'
            parent = unit.parent
            # find the frame to which unit belongs
            if not isinstance(parent, Frame):
                parent = parent.parent
            frame = parent
            frame_img = cv2.imread(frame.absolute_path())
            output_path = os.path.join(output_folder, unit.unique_name(with_jpg=True))
            cv2.imwrite(output_path, unit.get_crop(frame_img))

    def load_batch_label(self):
        """
        Load via_region_data.csv from self.output_folder
        put the BBox into source units
        """
        db = self.db
        fname_to_box = rfcn.read_csv(self.output_folder)
        folders = db.get_folders()
        for fname, bbox in fname_to_box.iteritems():
            # if doesn't contain plate_bbox, just skip
            if len(bbox) != 4: continue
            """ fname of the form {folder_id}#{frame_idx}#{part_idx}.jpg """
            name_clean = fname.split('.')[0]
            folder_id, frame_idx, part_idx = map(int, name_clean.split('#'))
            folder = db.get_folder_by_id(folder_id)
            car_bbox = folder.frames[frame_idx].parts[part_idx]
            assert isinstance(car_bbox, BBox)
            assert car_bbox.typename == 'car'
            assert self.has_unit(car_bbox)
            assert len(car_bbox.get_typed_parts('plate'))==0, \
                    '{} already has plate label'.format(car_bbox.unique_name)
            x,y,w,h = bbox
            x1,y1,x2,y2 = x,y,x+w,y+h
            car_x, car_y, _, _ = car_bbox.bbox
            plate_bbox= BBox(car_bbox, 'plate', (x1+car_x,y1+car_y,x2+car_x,y2+car_y))
            plate_bbox.label_type('manual')
            car_bbox.add_part(plate_bbox)
        """
        For the cars that didn't get a plate label, label them as 'no palte'
        """
        for car_bbox in self.units:
            assert isinstance(car_bbox, BBox)
            assert car_bbox.typename == 'car'
            if len(car_bbox.get_typed_parts('plate')) == 0:
                plate_bbox = BBox(car_bbox, 'plate', (0,0,0,0))
                plate_bbox.attrs['type'] = 'none'
                car_bbox.add_part(plate_bbox)
inject_mixin('platebatch', Batch, PlateBatchMixin)





class TrafficDatabaseMixin:
    pass
inject_mixin('traffic_database', Database, TrafficDatabaseMixin)












""" Folder loading logic """

def num_to_foldername(num):
    return '{}_frames'.format(num)

batch1_folders = map(num_to_foldername, [7287,7288,7289,7290])
batch2_folders = map(num_to_foldername, [7293,7294,7295,7296,7297,7298,7299,7300])
batch3_folders = map(num_to_foldername, [7308,7309,7310,7312,7313,7314,7315])
batch4_folders = map(num_to_foldername, [7320,7322,7323,7325,7326,7327,7328])
batch5_folders = map(num_to_foldername, [7332,7335,7336,7337,7340,7374,7384])
batch6_folders = map(num_to_foldername, [7417,7418,7420,7421,7422,7423,7424])


def assert_folders_exist(folders):
    for folder_name in folders:
        assert os.path.exists(os.path.join(data_root, folder_name))

def init_batchN(db, data_batch, divisible_by, folders):
    assert_folders_exist(folders)
    for folder_name in folders:
        folder = load_new_folder(db, folder_name, cbp, data_batch)

def init_batch1(db):
    init_batchN(db, 1, 5, batch1_folders) # every 5 frames
def init_batch2(db):
    init_batchN(db, 2, 5, batch2_folders) # every 5 frames
def init_batch3(db):
    init_batchN(db, 3, 2, batch3_folders) # every 2 frames
def init_batch4(db):
    init_batchN(db, 4, 3, batch4_folders) # every 3 frames
def init_batch5(db):
    init_batchN(db, 5, 3, batch5_folders) # every 3 frames
def init_batch6(db):
    init_batchN(db, 6, 3, batch6_folders) # every 3 frames

