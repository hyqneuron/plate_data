
# db2

`db2.py` has primary data class declaration
`traffic_mixins.py` has mixin classes specific to `License Plate Recognition`

## How to pre-process a video

`makeframes.sh` allows you to extract frames from a video, and put them in a folder. You should put video files at the
root of this folder. Video should be named `MVI_xxxx.MOV`

Then, run `./makeframes.sh xxxx` where xxxx is the 4 digit in the MOV file. This will create `xxxx_frames` folder, with
frames taken from the MOV file stored within. Frames are taken at 1fps. Modify `-r` parameter in `makeframes.sh` to
change framerate.

## How to use the database

Under your script's root directory, create a softlink `data` to the directory where db2 is contained. Then,

```python
from data import db2
from data import traffic_mixins.py # that is, if you need the LPR mixins to be loaded

# create a new database
db = db2.Database(data_root='path_to_root_of_database') # data_root is where the pickle file, db2.pkl, will be stored

# save the database
db.save() # defaults to data_root/db2.pkl

# load from an existing database
db = db2.Database.load('path_to_saved_db2.pkl')
```

`db2.py` is very short and quite readable. `traffic_mixins.py` demonstrates how to use stuffs inside db2.py
