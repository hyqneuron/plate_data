"""
We are organizing our data as thus: (Folder, Frame, Part)
-- Folder: a folder is a unit of organized data. We put a sequence of frames under the same folder
-- Folders are informally organized into batches, for the sake of easier organization
-- Every folder has a processor, designated by string name

Tasks we'll pay particular attention to include:
1. Extracting crops from a folder
2. Putting information annotated on crops back into frame's data
3. Naming every frame/part in a unique manner

"""

import os
import re
import cPickle


class Folder:
    def __init__(self, absolute_path, database, attrs=None, processor=None):
        if attrs is None: attrs={}
        self.absolute_path = absolute_path
        self.folder_id = None
        self.attrs = attrs
        self.frames = []
        self.frame_names = []
        self.processor = processor # determines how to filter for frames in a folder, and bunch other stuffs

    def unique_name(self):
        return str(self.folder_id)

    def add_frame(self, frame):
        assert frame.folder == self
        if frame not in self.frames and frame.path not in self.frame_names:
            self.frames.append(frame)
            self.frame_names.append(frame.path)

class Frame:
    def __init__(self, folder, img_name, parts=None, attrs=None):
        """
        make sure img_name is clean, doesn't contain '/', and contain at most one dot
        """
        if attrs is None: attrs={}
        if parts is None: parts=[]
        assert '/' not in img_name
        assert len([m.start() for m in re.finditer('\.', img_name)]) <= 1
        self.folder = folder
        self.path = img_name   # path is relative to folder.absolute_path
        assert os.path.exists(self.absolute_path())
        self.parts = parts
        self.attrs = attrs
    def absolute_path(self):
        return os.path.join(self.folder.absolute_path, self.path)
    def unique_name(self):
        """ We assume frame uses index into folder.frames as unique identifier """
        idx = self.folder.frames.index(self)
        return '{}#{}'.format(self.folder.unique_name(), idx)
    def add_part(self, part):
        assert part.parent == self
        if part not in self.parts: self.parts.append(part)

class Part:
    def __init__(self, parent, parts=None, attrs=None):
        if attrs is None: attrs={}
        if parts is None: parts=[]
        self.parent = parent # parent is either a Frame or a Part
        self.parts  = parts
        self.attrs  = attrs
    def unique_name(self):
        """ We assume index into parent.parts is unique identifier, this means once a part is added, it can't be removed """
        idx = self.parent.parts.index(self)
        return '{}#{}'.format(self.parent.unique_name(), idx)

    def add_part(self, part):
        assert part.parent == self
        if part not in self.parts: self.parts.append(part)

class BBox(Part):
    def __init__(self, parent, bbox_raw, parts=None, attrs=None):
        Part.__init__(self, parent, parts=parts, attrs=attrs)
        # bbox_raw in the form of x1,y1,x2,y2, taken w.r.t. owner Frame, not owner Part
        assert len(bbox_raw) == 4
        self.bbox = bbox_raw
    def xyxy(self):
        return self.bbox
    def xywh(self):
        x1,y1,x2,y2 = self.bbox
        w = x2-x1
        h = y2-y1
        return x1,y1,w,h
    def get_crop(self, frame_img):
        # frame_img is numpy image of the owner frame
        x1,y1,x2,y2 = self.bbox
        return frame_img[y1:y2, x1:x2]


class Database:
    """
    Data graph:
    - Database keeps all folders
    - Each Folder keeps its own Frames
    - Each Frame keeps its own Parts
    - Each Part keeps its own (sub)Parts
    """
    def __init__(self):
        self.folder_registry = {} # maps folder.absolute_path to folder
    def add_folder(self, folder):
        assert folder.folder_id == None
        assert folder.absolute_path not in self.folder_registry
        max_folder_id = max([0]+[_folder.folder_id for _folder in self.folder_registry.values()])
        folder.folder_id = max_folder_id + 1
        self.folder_registry[folder.absolute_path] = folder
        return folder
    def get_folder(self, absolute_path):
        return self.folder_registry[absolute_path]
    def get_folder_by_id(self, folder_id):
        for k,folder in self.folder_registry.iteritems():
            if folder.folder_id == folder_id:
                return folder
        assert False, 'folder_id={} not found'.format(folder_id)
    def get_folders(self):
        return self.folder_registry.values()

db = None

data_root = os.path.dirname(os.path.realpath(__file__))
path_db = os.path.join(data_root, 'db2.pkl')
def save():
    global db
    assert isinstance(db, Database)
    cPickle.dump(db, open(path_db, 'wb'))

def load():
    global db
    db = cPickle.load(open(path_db, 'rb'))

# always load database when module is loaded
if os.path.exists(path_db):
    load()



