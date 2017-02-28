#!/bin/bash

folder="$1"_frames
movie=MVI_"$1".MOV
cd $folder
ffmpeg -i ../"$movie" -r 1 -qscale:v 2 '$'"$1"_%04d.jpg
cd ..
