import os
from glob import glob
from shutil import copyfile

folder = '7300_frames'

filenames = glob(folder + '/crops/*.jpg')
for filename in filenames:
    split = filename.split('_')
    if (int(split[2]) % 5) == 0:
        copyfile(filename, folder + '/crops_filtered/%s' %'_'.join([split[1][-3:], split[2], split[3]]))


