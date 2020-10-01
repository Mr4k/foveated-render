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

rng = default_rng()
vals = rng.standard_normal((2, NUM_PIXELS_TO_DRAW))

x_scale = 100
vals[0] *= x_scale
vals[1] *= x_scale * IN_IMAGE_HEIGHT / IN_IMAGE_WIDTH
vals[0] += IN_IMAGE_WIDTH / 2
vals[1] += IN_IMAGE_HEIGHT / 2

# bad way of dealing with out of bounds pixels
vals[0] %= IN_IMAGE_WIDTH
vals[1] %= IN_IMAGE_HEIGHT
print('max?', np.max(vals[0]))

plt.scatter(x = vals[0], y = vals[1], s = 1) 
plt.xlim(0, IN_IMAGE_WIDTH)
plt.ylim(0, IN_IMAGE_HEIGHT)
plt.show()

# export two map images
# one which maps points to pixels

# one which maps pixels to closest filled out pixels
# maybe we can pack more pixels in if we get trickier

kdt = KDTree(vals.T, leaf_size=30, metric='euclidean')

# all hail numpy magic
xs = np.arange(0, IN_IMAGE_WIDTH)
ys = np.arange(0, IN_IMAGE_HEIGHT)
xs = np.kron(xs, np.ones(IN_IMAGE_HEIGHT))
ys = np.tile(ys, IN_IMAGE_WIDTH)
source_pixels = np.column_stack((xs, ys))

print(source_pixels)
distances, indexes = kdt.query(source_pixels, k = 1)
print(source_pixels[10])
print('closests to 0, 10', vals[:, indexes[10]], '@ index', indexes[10])
print(source_pixels[2])
print('closests to 0, 2', vals[:, indexes[2]], '@ index', indexes[2])
print(source_pixels[0])
print('closests to 0, 0', vals[:, indexes[0]], '@ index', indexes[0])
print("")
print("")

# for now just weigh neighbor contributions by distance
# there are definitely bad cases here (such as when one 
# point is directly past another point in the same direction)

index_map = np.uint16(np.reshape(indexes[:, 0], (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT, -1)))
print('dtype', index_map.dtype)

# channels are in order of least significant bit
# could make this much cleaner... (but the rest of the code is pretty bad rn)
channel1 = np.bitwise_and(index_map, 255)
channel2 = np.bitwise_and(np.right_shift(index_map, 8), 255)
channel3 = np.bitwise_and(np.right_shift(index_map, 16), 255)

dst_pix_map = np.dstack((channel1, channel2, channel3))
print('hello')
print(dst_pix_map)
dst_pix_map = np.swapaxes(dst_pix_map, 0, 1)

print('???', dst_pix_map[0, 0])
print('???', dst_pix_map[0, 2])
print('???', dst_pix_map[0, 10])

print('max?', np.max(vals[0]))
#vals[0] /= IN_IMAGE_WIDTH
#vals[1] /= IN_IMAGE_HEIGHT
print(np.swapaxes(vals, 0, 1))

vals[0] /= IN_IMAGE_WIDTH
vals[1] /= IN_IMAGE_HEIGHT

src_pix_map = np.reshape(np.swapaxes(vals, 0, 1), (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE, -1))
src_pix_map = np.dstack((src_pix_map, np.zeros((OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))))
print('max?', np.max(src_pix_map))

plt.imshow(src_pix_map)
plt.show()

plt.imshow(dst_pix_map)
plt.show()

print('test')
d = dst_pix_map[0,10]
"""print('attempt index', d)
print('closest to 0, 10', src_pix_map[int(d[0] * OUT_IMAGE_SIZE), int(d[1] * OUT_IMAGE_SIZE)][0:2] * (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT))

d = dst_pix_map[0,2]
print('closest to 0, 2', src_pix_map[int(d[0] * OUT_IMAGE_SIZE), int(d[1] * OUT_IMAGE_SIZE)][0:2] * (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT))

d = dst_pix_map[0,0]
print('closest to 0, 0', src_pix_map[int(d[0] * OUT_IMAGE_SIZE), int(d[1] * OUT_IMAGE_SIZE)][0:2] * (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT))

d = dst_pix_map[105,0]
print('closest to 105, 0', src_pix_map[int(d[0] * OUT_IMAGE_SIZE), int(d[1] * OUT_IMAGE_SIZE)][0:2] * (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT))
#print(src_pix_map[0,0][:2] * (512, 228))

d = dst_pix_map[105, 200]
print('closest to 105, 200', src_pix_map[int(d[0] * OUT_IMAGE_SIZE), int(d[1] * OUT_IMAGE_SIZE)][0:2] * (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT))
"""

# test that the decoding works
print(dst_pix_map.dtype)
dst_pix_map = dst_pix_map.astype(np.uint16)
#dst_pix_map = np.float64(dst_pix_map * 32768)/32768
im = np.zeros((228, 512, 3))
for i in range(228):
    for j in range(512):
        a = dst_pix_map[i, j][0]
        a = a << 16
        d = (dst_pix_map[i, j][0]) | (dst_pix_map[i, j][1] << 8) | (dst_pix_map[i, j][2] << 16) 
        im[i,j][0:2] = src_pix_map[int(d // OUT_IMAGE_SIZE), int(d % OUT_IMAGE_SIZE)][0:2]
plt.imshow(im)
plt.show()

old_dst_map = dst_pix_map

print('pix_src', src_pix_map)
cv2.imwrite("src_map.png", (src_pix_map * 255).astype(np.uint8))

cv2.imwrite("dst_map.png", (dst_pix_map).astype(np.uint8))

import cv2
src_pix_map = cv2.imread("src_map.png") / 255.0
dst_pix_map = cv2.imread("dst_map.png", cv2.IMREAD_UNCHANGED)
print('typeee??', dst_pix_map.dtype)

plt.imshow(src_pix_map)
plt.show()

# test that the decoding works
dst_pix_map = dst_pix_map.astype(np.uint16)
#dst_pix_map = np.float64(dst_pix_map * 32768)/32768
im = np.zeros((228, 512, 3))
for i in range(3):
    for j in range(3):
        d = (dst_pix_map[i, j][0]) | (dst_pix_map[i, j][1] << 8) | (dst_pix_map[i, j][2] << 16) 
        im[i,j][0:2] = src_pix_map[
            int(d // OUT_IMAGE_SIZE),
            int(d % OUT_IMAGE_SIZE)][0:2]
        print(i, j, d)
plt.imshow(im)
plt.show()
