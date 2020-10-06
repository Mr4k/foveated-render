import matplotlib
matplotlib.use("TkAgg")

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import default_rng

from scipy.spatial import cKDTree

import cv2

IN_IMAGE_WIDTH = 420
IN_IMAGE_HEIGHT = 236

rng = default_rng(10)
# * np.array([50, 50 * IN_IMAGE_HEIGHT / IN_IMAGE_WIDTH])
"""vals = np.concatenate((
    rng.standard_normal((2, int(NUM_PIXELS_TO_DRAW / 2))) * np.reshape(np.array([120, 120 * IN_IMAGE_HEIGHT / IN_IMAGE_WIDTH]), (2, 1)),
    rng.standard_normal((2, int(NUM_PIXELS_TO_DRAW / 2))) * np.reshape(np.array([50, 50 * IN_IMAGE_HEIGHT / IN_IMAGE_WIDTH]), (2, 1))
), axis=1)"""

def draw_circle(center, r, already_filled, vals):
    total_pixels = 0
    for x in range(already_filled.shape[0]):
        for y in range(already_filled.shape[1]):
            print(x, y)
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < r * r:
                already_filled[x, y] = 1
                total_pixels += 1
                vals.append((x, y))
    return total_pixels

def draw_from_gaussian_distribution_without_repeats(rng, mean, variance, already_filled):
    coords = (rng.standard_normal(2) * variance + mean).astype(int)
    while coords[0] < 0 or coords[1] < 0 or coords[0] >= already_filled.shape[0] or coords[1] >= already_filled.shape[1] or already_filled[coords[0], coords[1]] == 1:
        coords = (rng.standard_normal(2) * variance + mean).astype(int)
    already_filled[coords[0], coords[1]] = 1
    return coords.astype(float)

center = np.array([IN_IMAGE_WIDTH / 2, IN_IMAGE_HEIGHT / 2])
pixel_budget = IN_IMAGE_WIDTH * IN_IMAGE_HEIGHT / 10

vals = []
already_filled = np.zeros((IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT))

pixel_budget -= draw_circle(center, 40, already_filled, vals)
print(pixel_budget)

variance = np.array([40, IN_IMAGE_HEIGHT / IN_IMAGE_WIDTH * 40])
for i in range(int(pixel_budget / 2)):
    vals.append(draw_from_gaussian_distribution_without_repeats(rng, center, variance, already_filled))

variance = np.array([120, IN_IMAGE_HEIGHT / IN_IMAGE_WIDTH * 120])
for i in range(int(pixel_budget / 2)):
    vals.append(draw_from_gaussian_distribution_without_repeats(rng, center, variance, already_filled))
vals = np.vstack(vals).T

cv2.imwrite('mask.png', np.uint8(already_filled * 255).T)

plt.scatter(x = vals[0], y = vals[1], s = 1) 
plt.xlim(0, IN_IMAGE_WIDTH)
plt.ylim(0, IN_IMAGE_HEIGHT)
plt.show()

# one which maps pixels to closest filled out pixels
# maybe we can pack more pixels in if we get trickier
# if we want to make it toroidal use boxsize=np.array([IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT]
kdt = cKDTree(vals.T, leafsize=30)

# all hail numpy magic
xs = np.arange(0, IN_IMAGE_WIDTH)
ys = np.arange(0, IN_IMAGE_HEIGHT)
xs = np.kron(xs, np.ones(IN_IMAGE_HEIGHT))
ys = np.tile(ys, IN_IMAGE_WIDTH)
source_pixels = np.column_stack((xs, ys))

def create_dst_map_from_indexes(name, indexes, weights):
    # for now just weigh neighbor contributions by distance
    # there are definitely bad cases here (such as when one 
    # point is directly past another point in the same direction)
    index_map = np.uint16(np.reshape(indexes, (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT, -1)))
    weight_map = np.reshape(weights, (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT, -1))

    # channels are in order of least significant bit
    # could make this much cleaner... (but the rest of the code is pretty bad rn)
    channel1 = IN_IMAGE_HEIGHT - vals[1][index_map] - 1
    channel2 = np.right_shift(np.uint16(vals[0][index_map]), 8)
    # this is a pretty atrocious hack b/c only width > 256, we pack stuff the second half of the channel here
    channel3 = np.bitwise_and(np.uint16(vals[0][index_map]), 255)
    channel4 = weight_map * 255

    dst_pix_map = np.dstack((channel1, channel2, channel3, channel4))
    dst_pix_map = np.swapaxes(dst_pix_map, 0, 1)
    print(dst_pix_map)

    plt.imshow(dst_pix_map)
    plt.show()

    cv2.imwrite(name, (dst_pix_map).astype(np.uint8))

def test_reconstruction(src_map_name, dst_map_names):
    src_pix_map = cv2.imread(src_map_name) / 255.0
    dst_pix_maps = list(map(lambda name: cv2.imread(name, cv2.IMREAD_UNCHANGED), dst_map_names))
    plt.imshow(src_pix_map)
    plt.show()

    # test that the decoding works
    im = np.zeros((IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, 3))
    for i in range(0,IN_IMAGE_HEIGHT):
        for j in range(0, IN_IMAGE_WIDTH):
            for dst_pix_map in dst_pix_maps:
                x = dst_pix_map[i, j][0]
                y = dst_pix_map[i, j][1]
                w = dst_pix_map[i, j][2] / 255.0
                im[i,j][0:2] += src_pix_map[x,y][0:2] * w
    plt.imshow(im)
    plt.show()

distances, indexes = kdt.query(source_pixels, k = 4)

# there has to be a better numpy way of doing this...
inverted_d = 1.0/distances
inverted_d = np.nan_to_num(inverted_d, 1000000)
normalizing_consts = inverted_d.sum(axis=1)
weights = np.array([
    inverted_d[:, 0] / normalizing_consts,
    inverted_d[:, 1] / normalizing_consts,
    inverted_d[:, 2] / normalizing_consts,
    inverted_d[:, 3] / normalizing_consts
])

create_dst_map_from_indexes('dst_map_1.png', indexes[:, 0], weights[0, :])
create_dst_map_from_indexes('dst_map_2.png', indexes[:, 1], weights[1, :])
create_dst_map_from_indexes('dst_map_3.png', indexes[:, 2], weights[2, :])
create_dst_map_from_indexes('dst_map_4.png', indexes[:, 3], weights[3, :])

#test_reconstruction('src_map.png', ['dst_map_1.png', 'dst_map_2.png', 'dst_map_3.png', 'dst_map_4.png'])
