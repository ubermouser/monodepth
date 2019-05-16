#!/usr/bin/env python3

import datetime
import os
import numpy as np
import sys
from glob import glob

TRAIN_SEQUENCES = {
    '2011_09_26/2011_09_26_drive_0001_sync',
    '2011_09_26/2011_09_26_drive_0002_sync',
    '2011_09_26/2011_09_26_drive_0020_sync',
    '2011_09_26/2011_09_26_drive_0023_sync',
    '2011_09_26/2011_09_26_drive_0035_sync',
    '2011_09_26/2011_09_26_drive_0039_sync',
    '2011_09_26/2011_09_26_drive_0048_sync',
    '2011_09_26/2011_09_26_drive_0052_sync',
    '2011_09_26/2011_09_26_drive_0060_sync',
    '2011_09_26/2011_09_26_drive_0061_sync',
    '2011_09_26/2011_09_26_drive_0064_sync',
    '2011_09_26/2011_09_26_drive_0079_sync',
    '2011_09_26/2011_09_26_drive_0086_sync',
    '2011_09_26/2011_09_26_drive_0087_sync',
    '2011_09_26/2011_09_26_drive_0091_sync',
    '2011_09_26/2011_09_26_drive_0093_sync',
    '2011_09_26/2011_09_26_drive_0095_sync',
    '2011_09_26/2011_09_26_drive_0106_sync',
    '2011_09_26/2011_09_26_drive_0113_sync',
    '2011_09_26/2011_09_26_drive_0117_sync',
    '2011_09_28/2011_09_28_drive_0001_sync',
    '2011_09_29/2011_09_29_drive_0026_sync',
    '2011_09_30/2011_09_30_drive_0016_sync',
    '2011_09_30/2011_09_30_drive_0018_sync',
    '2011_09_30/2011_09_30_drive_0020_sync',
    '2011_09_30/2011_09_30_drive_0027_sync',
    '2011_09_30/2011_09_30_drive_0028_sync',
    '2011_09_30/2011_09_30_drive_0033_sync',
    '2011_09_30/2011_09_30_drive_0034_sync',
    '2011_10_03/2011_10_03_drive_0027_sync',
    '2011_10_03/2011_10_03_drive_0034_sync',
    '2011_10_03/2011_10_03_drive_0042_sync',
}
GLOB_EXPRESSION = os.path.join("image_02", "data", "*.jpg")
IMAGE_EXPRESSION = os.path.join("image_02", "data", "%s.jpg")
OP_IMAGE_EXPRESSION = os.path.join("image_03", "data", "%s.jpg")
OXTS_EXPRESSION = os.path.join("oxts", "data", "%s.txt")
TIME_EXPRESSION = os.path.join("oxts", "timestamps.txt")
DELTA_TIMESTEP = 2
# camera 3 is always 0.54 meters to the right of camera 2. They have the same height, 1.65m
# TODO: use lat-lon-alt + roll/pitch/yaw to compute absolute pose difference between two images

def main(data_directory):
    lines = []
    for sequence in TRAIN_SEQUENCES:
        abspath = os.path.join(data_directory, sequence, GLOB_EXPRESSION)
        frames = sorted(glob(abspath))

        oxts_expr = os.path.join(sequence, OXTS_EXPRESSION)
        image_expr = os.path.join(sequence, IMAGE_EXPRESSION)
        timestamps = open(os.path.join(data_directory, sequence, TIME_EXPRESSION)).readlines()
        timestamps = [datetime.datetime.strptime(timestr[:-4], "%Y-%m-%d %H:%M:%S.%f").timestamp() for timestr in timestamps]

        for i_last_frame in range(DELTA_TIMESTEP, len(frames)):
            first_frame = frames[i_last_frame - DELTA_TIMESTEP]
            last_frame = frames[i_last_frame]

            first_frame_number = os.path.splitext(os.path.basename(first_frame))[0]
            first_frame = image_expr % first_frame_number
            first_oxts = oxts_expr % first_frame_number

            last_frame_number = os.path.splitext(os.path.basename(last_frame))[0]
            last_frame = image_expr % last_frame_number
            last_oxts = oxts_expr % last_frame_number

            delta_timestamp = timestamps[i_last_frame] - timestamps[i_last_frame - DELTA_TIMESTEP]

            #assert os.path.exists(first_frame)
            #assert os.path.exists(first_oxts)
            #assert os.path.exists(last_frame)
            #assert os.path.exists(last_oxts)
            lines.append("%s %s %s %s %s" % (first_frame, first_oxts, last_frame, last_oxts, delta_timestamp))
    np.random.shuffle(lines)

    for line in lines:
        print(line)


if __name__ == '__main__':
    main(sys.argv[1])
