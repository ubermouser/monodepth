#!/usr/bin/env python3

import datetime
import os
import numpy as np
import sys
from glob import glob

import functools

import tqdm

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
MAXIMUM_FRAMEDELTA = 10
MINIMUM_DISTANCE = 0.5
MAXIMUM_DISTANCE = 3.

# camera 3 is always 0.54 meters to the right of camera 2. They have the same height, 1.65m
# TODO: use lat-lon-alt + roll/pitch/yaw to compute absolute pose difference between two images


@functools.lru_cache(maxsize=10000)
def read_oxts(oxts_path):
    with open(oxts_path) as oxts:
        oxts_strs = oxts.read().split()

        return np.asarray(
            [float(oxts_strs[i]) for i in [8, 9, 10, 14, 15, 16, 20, 21, 22]],
            dtype=np.float64
        )


def compute_delta_position_angle(first_oxts, second_oxts, first_time, second_time, alt_camera):
    delta_time = second_time - first_time

    oxts = (first_oxts + second_oxts) / 2.

    delta_position = oxts[0:3] * delta_time + oxts[3:6] * (delta_time ** 2)
    delta_angle = oxts[6:9] * delta_time

    if alt_camera:
        delta_position += [0., -0.54, 0.]

    return delta_position, delta_angle


def compute_distance(delta_position):
    return np.sqrt(np.sum(delta_position ** 2))


def main(data_directory):
    lines = []
    excluded_distance = 0
    for sequence in tqdm.tqdm(TRAIN_SEQUENCES):
        abspath = os.path.join(data_directory, sequence, GLOB_EXPRESSION)
        #frames = sorted(glob(abspath))

        oxts_expr = os.path.join(data_directory, sequence, OXTS_EXPRESSION)
        image_expr = os.path.join(sequence, IMAGE_EXPRESSION)
        alt_image_expr = os.path.join(sequence, OP_IMAGE_EXPRESSION)
        timestamps = open(os.path.join(data_directory, sequence, TIME_EXPRESSION)).readlines()
        timestamps = [datetime.datetime.strptime(timestr[:-4], "%Y-%m-%d %H:%M:%S.%f").timestamp() for timestr in timestamps]

        for i_first_frame in range(len(timestamps)):
                second_frame_range = range(
                    i_first_frame, min(i_first_frame + MAXIMUM_FRAMEDELTA, len(timestamps)))
                for i_second_frame in second_frame_range:
                    first_frame_number = '%010d' % i_first_frame
                    second_frame_number = '%010d' % i_second_frame

                    first_oxts = read_oxts(oxts_expr % first_frame_number)
                    second_oxts = read_oxts(oxts_expr % second_frame_number)

                    first_timestamp = timestamps[i_first_frame]
                    second_timestamp = timestamps[i_second_frame]

                    for alt_camera, alt_expr in [(False, image_expr), (True, alt_image_expr)]:
                        delta_position, delta_angle = compute_delta_position_angle(
                            first_oxts, second_oxts, first_timestamp, second_timestamp, alt_camera)
                        distance = compute_distance(delta_position)

                        if MINIMUM_DISTANCE <= distance <= MAXIMUM_DISTANCE:
                            first_frame = image_expr % first_frame_number
                            second_frame = alt_expr % second_frame_number

                            assert os.path.exists(os.path.join(data_directory, first_frame))
                            assert os.path.exists(os.path.join(data_directory, second_frame))

                            lines.append(
                                "%s %s %.12f %.12f %.12f %.12f %.12f %.12f" % (
                                first_frame, second_frame, delta_position[0], delta_position[1],
                                delta_position[2], delta_angle[0], delta_angle[1], delta_angle[2])
                             )
                        else:
                            excluded_distance += 1

    np.random.shuffle(lines)

    print("%d total, %d excluded by distance" % (len(lines), excluded_distance), file=sys.stderr)

    for line in lines:
        print(line)


if __name__ == '__main__':
    main(sys.argv[1])
