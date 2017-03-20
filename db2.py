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
import cPickle
import functools

class AcceptMixin:
    """
    AcceptMixin 

        AcceptMixin is designed to allow mixins to be injected into a class.  The primary classes (Folder, Frame, etc.)
        within this file are task-agnostic, in the sense that they contain nothing but data. We isolate task-specific
        logic into separate mixin classes.

        For example, BBox is just a simple data structure that contains bounding box information. In the task of license
        plate recognition (LPR), we use BBox to indicate the type/source of the label, the number sequence of the plate
        and so on. All of that logic, which is specific to LPR, is contained within CarBBoxMixin and PlateBBoxMixin,
        which exist in a separate file (traffic_mixinx.py).

        How it works:

        A PrimaryClass is a subclass of AcceptMixin. E.g. Folder, Frame, Part, Batch, Database. Every instance of a
        PrimaryClass shall contain a 'typename' str member, indicating the mixin with which this instance is associated.
        For example, in our LPR task, a BBox whose typename=='car' is associated with CarBBoxMixin, whereas a BBox whose
        typename=='plate' is associated with the PlateBBoxMixin.

        In a sense, the 'typename' indicates the specific subtype/subclass of the instance. This allows multiple
        instances with different specific types to co-exist in our data. We could have used python's own subclassing
        machinery to do this kind of polymorphism, but that would make data migration a nightmare so I decided to roll
        my own. Another benefit of this approach is ease of serialization and strong access control for a journal system
        (for future, of course).

        """
    def __getattr__(self, name):
        if name.startswith('__'):
            return object.__getattr__(name)
        if (self.typename in self.__class__.__dict__['typename_to_mixin']
                and name in self.__class__.typename_to_mixin[self.typename].__dict__):
            method = self.__class__.typename_to_mixin[self.typename].__dict__[name]
            return functools.partial(method, self)
        return AttributeError('{} instance has no attribute {}'.format(self.__class__.__name__, name))

    def has_typename(self, typename):
        return typename in self.__class__.typename_to_mixin

def inject_mixin(typename, injectable_class, mixin_class):
    assert typename not in injectable_class.typename_to_mixin
    injectable_class.typename_to_mixin[typename] = mixin_class

class Folder(AcceptMixin):
    typename_to_mixin = {}
    def __init__(self, db, typename, absolute_path, attrs=None, processor=None):
        if attrs is None: attrs={}
        assert isinstance(db, Database)
        assert self.has_typename(typename)
        self.db = db
        self.typename = typename
        self.absolute_path = absolute_path
        self.folder_id = None
        self.attrs = attrs
        self.frames = []
        self.frame_names = []
        self.processor = processor # determines how to filter for frames in a folder, and bunch other stuffs

    def unique_name(self, with_jpg=False):
        """ 
        unique_name() gives a string, used to uniquely identify a folder. 
        When we take crops of frames, we prefix the folder's unique_name to the output filename of that crop, so that
        using the crop's filename alone we can identify
        1. the folder
        2. the frame
        3. the part
        When outputing a frame, we name it as {folder_id}#{frame_idx}.jpg
        When outputing a crop image for a part, we name it as {folder_id}#{frame_idx}#{part_idx}.jpg
        When outputting a deeply embedded part, we name it as {folder_id}#{frame_idx}#{part_idx}...#{part_idx}.jpg
        Note: the '#' character is used as delimitor
        """
        unique_name = str(self.folder_id)
        if with_jpg: unique_name = unique_name + '.jpg'
        return unique_name

    def add_frame(self, frame):
        assert isinstance(frame, Frame)
        assert frame.folder == self
        if frame not in self.frames and frame.path not in self.frame_names:
            self.frames.append(frame)
            self.frame_names.append(frame.path)
    """
    def is_format_updated(self):
        return all([key in self.__dict__ for key in ['db', 'typename']])

    def update_format(self, db=None):
        if 'db' not in self.__dict__:
            assert isinstance(db, Database)
            self.db = db
        if 'typename' not in self.__dict__:
            self.typename = 'traffic_folder'
    """

class HasParts:
    def add_part(self, part):
        assert isinstance(part, Part)
        assert part.parent == self
        if part not in self.parts: self.parts.append(part)
    def get_typed_parts(self, typename):
        return [part for part in self.parts if part.typename == typename]

class Frame(AcceptMixin, HasParts):
    typename_to_mixin = {}
    def __init__(self, folder, typename, img_path, parts=None, attrs=None):
        """
        make sure img_path is clean, doesn't contain '/', and contain at most one dot
        """
        if attrs is None: attrs={}
        if parts is None: parts=[]
        assert '/' not in img_path
        assert len([s for s in img_path if s=='.']) <= 1
        assert isinstance(folder, Folder)
        assert self.has_typename(typename)
        self.folder = folder   # folder containing this frame
        self.typename = typename
        self.path = img_path   # path is relative to folder.absolute_path
        assert os.path.exists(self.absolute_path())
        self.parts = parts     # things that this image contain
        self.attrs = attrs     # informal attributes


    def absolute_path(self):
        return os.path.join(self.folder.absolute_path, self.path)

    def unique_name(self, with_jpg=False):
        """ We assume frame uses index into folder.frames as unique identifier """
        frame_idx = self.folder.frames.index(self)
        unique_name = '{}#{}'.format(self.folder.unique_name(), frame_idx)
        if with_jpg: unique_name = unique_name + '.jpg'
        return unique_name

    """
    def is_format_updated(self):
        return all([key in self.__dict__ for key in ['typename']])

    def update_format(self):
        if 'typename' not in self.__dict__:
            self.typename = 'traffic_frame'
    """


class Part(AcceptMixin, HasParts):
    typename_to_mixin = {}

    def __init__(self, parent, typename, parts=None, attrs=None):
        if attrs is None: attrs={}
        if parts is None: parts=[]
        assert self.has_typename(typename)
        self.parent = parent # parent is either a Frame or a Part
        self.typename = typename     # name is more like 'type', identifies what this part is
        self.parts  = parts  # subparts within this part
        self.attrs  = attrs  # informal attributes

    def unique_name(self, with_jpg=False):
        """ We assume index into parent.parts is unique identifier, this means once a part is added, it can't be removed """
        part_idx = self.parent.parts.index(self)
        unique_name ='{}#{}'.format(self.parent.unique_name(), part_idx)
        if with_jpg: unique_name = unique_name + '.jpg'
        return unique_name

    """
    def is_format_updated(self):
        return all([key in self.__dict__ for key in ['typename']])

    def update_format(self):
        if 'typename' not in self.__dict__:
            self.typename = self.name
            del self.__dict__['name']
    """


class BBox(Part):
    typename_to_mixin = {}
    """
    BBox is a specific kind of Part. Most of the time we'll be using BBox instead of the generic Part class
    In the future we can add more subtypes of Part
    """
    def __init__(self, parent, typename, bbox_raw, parts=None, attrs=None):
        Part.__init__(self, parent, typename, parts=parts, attrs=attrs)
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

class Point(Part):
    typename_to_mixin = {}
    def __init__(self, parent, typename, (x,y), parts=None, attrs=None):
        Part.__init__(self, parent, typename, parts=parts, attrs=attrs)
        self.x = x
        self.y = y
    def xy(self):
        return self.x, self.y

class Batch(AcceptMixin):
    typename_to_mixin = {}
    """
    This class is for iterative labelling.
    Steps as following:
    1. We put a set of Parts or Frames into a batch
    1. We output this batch into a folder
    2. We manually label this batch
    3. We load the manual labels back into those Parts or Frames
    4. We 
    """
    def __init__(self, db, typename, batch_name, output_folder):
        # use name:str to uniquely identify a batch
        self.batch_name = batch_name
        assert os.path.exists(output_folder), 'output_folder needs to be absolute path and should already exist'
        self.output_folder = output_folder
        # part.attrs[batch_typename] identifies the batch of type batch_typename to which part belongs
        # a Part cannot be added to a batch if part.attrs[batch.typename] already exists
        assert isinstance(db, Database)
        assert self.has_typename(typename)
        self.db = db
        self.typename = typename
        self.unique_names = []
        self.units = []

    def add_unit(self, unit):
        assert self.typename not in unit.attrs, 'part/frame already has batch_key:{}'.format(self.typename)
        assert unit.unique_name() not in self.unique_names, 'part already present in this batch'
        unit.attrs[self.typename] = self
        self.units.append(unit)

    def has_unit(self, unit):
        return (unit in self.units and 
                self.typename in unit.attrs and 
                unit.attrs[self.typename] == self)

    """
    def is_format_updated(self):
        return all([key in self.__dict__ for key in ['db', 'typename']])

    def update_format(self, db=None):
        if 'db' not in self.__dict__:
            assert db is not None
            self.db = db
        if 'typename' not in self.__dict__:
            self.typename = self.batch_key
            del self.__dict__['batch_key']
    """

class Database(AcceptMixin):
    typename_to_mixin = {}
    """
    Data graph:
    - Database keeps all folders
    - Each Folder keeps its own Frames
    - Each Frame keeps its own Parts
    - Each Part keeps its own (sub)Parts
    """
    def __init__(self, typename, data_root):
        assert self.has_typename(typename)
        self.typename = typename
        self.data_root = data_root
        self.folder_registry = {} # maps folder.absolute_path to folder
        self.batches = {} # maps batch.batch_name to batch


    def save(self, filename='db2.pkl'):
        save_path = os.path.join(self.data_root, filename)
        cPickle.dump(self, open(save_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(db_path):
        db = cPickle.load(open(db_path, 'rb'))
        assert isinstance(db, Database)
        return db

    """ folder """

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

    def get_folder_by_lastpath(self, lastpath):
        matches = []
        for abs_path, folder in self.folder_registry.iteritems():
            if lastpath == abs_path.split('/')[-1]:
                matches.append(folder)
        return matches

    def get_folders(self):
        return self.folder_registry.values()

    """ batch """

    def add_batch(self, batch):
        assert isinstance(batch, Batch)
        assert batch.batch_name not in self.batches
        self.batches[batch.batch_name] = batch

    def get_batches(self):
        return self.batches.values()

    def get_batch_by_name(self, batch_name):
        return self.batches[batch_name]

    """ parts """

    def get_all_parts(self):
        all_parts = []
        def extract_parts(unit):
            for part in unit.parts:
                all_parts.append(part)
                extract_parts(part)
        for folder in self.get_folders():
            for frame in folder.frames:
                extract_parts(frame)
        return all_parts

    def get_all_typed_parts(self, typename):
        return [part for part in self.get_all_parts() if part.typename==typename]

    """
    def is_format_updated(self):
        return all([key in self.__dict__ for key in ['data_root','typename']])

    def update_format(self):
        self.typename = "traffic_database"
    """


"""
def check_formats(db):
    folders = db.get_folders()
    frames  = [frame for folder in folders for frame in folder.frames]
    parts   = db.get_all_parts()
    batches = db.get_batches()
    print('folders', all([a.is_format_updated() for a in folders]))
    print('frames',  all([a.is_format_updated() for a in frames]))
    print('parts',   all([a.is_format_updated() for a in parts]))
    print('batches', all([a.is_format_updated() for a in batches]))
    print('db',      db.is_format_updated())

"""

