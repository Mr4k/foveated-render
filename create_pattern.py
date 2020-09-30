import matplotlib
matplotlib.use("TkAgg")

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import default_rng

from sklearn.neighbors import KDTree

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

distances, indexes = kdt.query(source_pixels, k = 2)

# for now just weigh neighbor contributions by distance
# there are definitely bad cases here (such as when one 
# point is directly past another point in the same direction)
sample_image_x_coords = (indexes[:, 0] % OUT_IMAGE_SIZE) / OUT_IMAGE_SIZE
sample_image_y_coords = (indexes[:, 0] // OUT_IMAGE_SIZE) / OUT_IMAGE_SIZE
print(np.max(sample_image_y_coords))

dst_pix_map = np.reshape(np.column_stack((sample_image_x_coords, sample_image_y_coords)),
 (IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT, -1))

dst_pix_map = np.dstack((dst_pix_map, np.zeros((IN_IMAGE_WIDTH, IN_IMAGE_HEIGHT))))
dst_pix_map = np.swapaxes(dst_pix_map, 1, 0)

print(dst_pix_map.shape)
print(dst_pix_map[0, 0])

vals[0] /= IN_IMAGE_WIDTH
vals[1] /= IN_IMAGE_HEIGHT

src_pix_map = np.reshape(vals, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE, -1))
print(src_pix_map.shape)
src_pix_map = np.dstack((src_pix_map, np.zeros((OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))))

plt.imshow(src_pix_map)
plt.show()

plt.imshow(dst_pix_map)
plt.show()

src_map = Image.fromarray(np.uint8(np.clip(src_pix_map, 0, 1)*255))
src_map.save("src_map.png")

dst_map = Image.fromarray(np.uint8(np.clip(dst_pix_map, 0, 1)*255))
dst_map.save("dst_map.png")
