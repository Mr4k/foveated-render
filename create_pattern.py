import matplotlib
matplotlib.use("TkAgg")

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import default_rng

from sklearn.neighbors import KDTree

import cv2

IN_IMAGE_WIDTH = 512
IN_IMAGE_HEIGHT = 228

OUT_IMAGE_SIZE = 128
NUM_PIXELS_TO_DRAW = OUT_IMAGE_SIZE * OUT_IMAGE_SIZE

rng = default_rng(10)
vals = rng.standard_normal((2, NUM_PIXELS_TO_DRAW))

x_scale = 100
vals[0] *= x_scale
vals[1] *= x_scale * IN_IMAGE_HEIGHT / IN_IMAGE_WIDTH
vals[0] += IN_IMAGE_WIDTH / 2
vals[1] += IN_IMAGE_HEIGHT / 2

# bad way of dealing with out of bounds pixels
vals[0] %= IN_IMAGE_WIDTH
vals[1] %= IN_IMAGE_HEIGHT

plt.scatter(x = vals[0], y = vals[1], s = 1) 
plt.xlim(0, IN_IMAGE_WIDTH)
plt.ylim(0, IN_IMAGE_HEIGHT)
plt.show()

# one which maps pixels to closest filled out pixels
# maybe we can pack more pixels in if we get trickier
kdt = KDTree(vals.T, leaf_size=30, metric='euclidean')

# all hail numpy magic
xs = np.arange(0, IN_IMAGE_WIDTH)
ys = np.arange(0, IN_IMAGE_HEIGHT)
xs = np.kron(xs, np.ones(IN_IMAGE_HEIGHT))
ys = np.tile(ys, IN_IMAGE_WIDTH)
source_pixels = np.column_stack((xs, ys))

def create_src_map(name):
    src_pix_map = np.reshape(np.swapaxes(vals, 0, 1), (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE, -1))
    src_pix_map = np.dstack((src_pix_map, np.zeros((OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))))

    cv2.imwrite(name, (src_pix_map * 255).astype(np.uint8))

    plt.imshow(src_pix_map)
    plt.show()


def create_dst_map_from_indexes(name, indexes, weights):
    # for now just weigh neighbor contributions by distance
    # there are definitely bad cases here (such as when one 
    # point is directly past another point in the same direction)
    index_map = np.uint16(np.reshape(indexes, (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT, -1)))
    weight_map = np.reshape(weights, (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT, - 1))

    # channels are in order of least significant bit
    # could make this much cleaner... (but the rest of the code is pretty bad rn)
    channel1 = np.bitwise_and(index_map // OUT_IMAGE_SIZE, 255)
    channel2 = np.bitwise_and(index_map % OUT_IMAGE_SIZE, 255)
    channel3 = weight_map * 255

    dst_pix_map = np.dstack((channel1, channel2, channel3))
    dst_pix_map = np.swapaxes(dst_pix_map, 0, 1)
    print(dst_pix_map)

    vals[0] /= IN_IMAGE_WIDTH
    vals[1] /= IN_IMAGE_HEIGHT

    plt.imshow(dst_pix_map)
    plt.show()

    # test that the decoding works
    """im = np.zeros((228, 512, 3))
    for i in range(228):
        for j in range(512):
            x = dst_pix_map[i, j][0]
            y = dst_pix_map[i, j][1]
            im[i,j][0:2] = src_pix_map[x, y][0:2]
    plt.imshow(im)
    plt.show()"""

    cv2.imwrite(name, (dst_pix_map).astype(np.uint8))

def test_reconstruction(src_map_name, dst_map_names):
    src_pix_map = cv2.imread(src_map_name) / 255.0
    dst_pix_maps = list(map(lambda name: cv2.imread(name, cv2.IMREAD_UNCHANGED), dst_map_names))
    plt.imshow(src_pix_map)
    plt.show()

    # test that the decoding works
    im = np.zeros((228, 512, 3))
    for i in range(0,228):
        for j in range(0, 512):
            for dst_pix_map in dst_pix_maps:
                x = dst_pix_map[i, j][0]
                y = dst_pix_map[i, j][1]
                w = dst_pix_map[i, j][2] / 255.0
                im[i,j][0:2] += src_pix_map[x,y][0:2] * w
    plt.imshow(im)
    plt.show()

distances, indexes = kdt.query(source_pixels, k = 2)

create_src_map('src_map.png')

# there has to be a better numpy way of doing this...
normalizing_consts = distances.sum(axis=1)
weights = np.array([
    distances[:, 0] / normalizing_consts,
    distances[:, 1] / normalizing_consts
])

create_dst_map_from_indexes('dst_map_1.png', indexes[:, 0], weights[0, :])
create_dst_map_from_indexes('dst_map_2.png', indexes[:, 1], weights[1, :])

test_reconstruction('src_map.png', ['dst_map_1.png', 'dst_map_2.png'])
