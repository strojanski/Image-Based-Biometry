#!/bin/bash

# Convert $1.png to a grayscale image and save as $1.gray
magick "$1.png" -depth 8 -type Grayscale -compress none -colorspace Gray -strip "$1.gray"
echo "$1.png" "$1.gray"
# Create WSQ from the RAW image, -raw_in W,H,bits,DPI, just keep in mind that pcasys needs more than 300
cwsq 0.75 wsq "$1.gray" -raw_in 256,256,8,500
echo "$1.gray"

# Compute minutiae on WSQ
mindtct "$1.wsq" "$1"
