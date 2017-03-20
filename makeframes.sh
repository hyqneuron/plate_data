#!/bin/bash

folder="$1"_frames
movie=MVI_"$1".MOV
mkdir -p $folder
cd $folder
ffmpeg -i ../"$movie" -r 3 -qscale:v 2 '$'"$1"_%04d.jpg
cd ..
