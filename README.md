
# car plate database

`db2.py` has basic data class declaration
`db2processor.py` has data manipulation logic

## How to pre-process a video

You should put video files at the root of this folder. Video should be named `MVI_xxxx.MOV`

Then, run `./makeframes.sh xxxx` where xxxx is the 4 digit in the MOV file. This will create `xxxx_frames` folder, with
frames taken from the MOV file stored within. Frames are taken at 1fps. Modify `-r` parameter in `makeframes.sh` to
change framerate.

## How to use the database
ASK YQ [TBD]
