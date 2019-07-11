# 画洋流扩散图
# created by he in July 2019

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import animation
from PIL import Image


def update(f):
    print(f)
    global loc
    if f == 0:
        loc = loc_prime
    next_loc = np.zeros((m, n), dtype=np.float)
    for i in np.arange(m):
        for j in np.arange(n):
            next_loc[i, j] = calc_next_loc(np.array([i, j]), loc, direction)
    loc = next_loc / np.max(next_loc)
    im.set_array(loc)

    # save
    if save_image:
        if f % 3 == 0:
            image_data = plt.cm.coolwarm(loc) * 255
            image_data, _ = np.split(image_data, (-1, ), axis=2)
            image_data = image_data.astype(np.uint8).clip(0, 255)
            output = 'image'
            if not os.path.exists(output):
                os.mkdir(output)
            a = Image.fromarray(image_data, mode='RGB')
            a.save('%s%d.png' % (output, f))
        return [im]


def calc_next_loc(now, loc, direction):
    near_index = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1),  # 代表八个方向(相对于坐标)
                           (0, 1), (1, -1), (1, 0), (1, 1)])
    direction_index = np.array([7, 6, 5, 0, 4, 1, 2, 3])
    nn = now + near_index
    ii, jj = nn[:, 0], nn[:, 1]
    ii[ii >= m] = 0
    jj[jj >= n] = 0
    return np.dot(loc[ii, jj], direction[ii, jj, direction_index])


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=300, edgeitems=8)
    np.random.seed(0)

    save_image = False
    style = 'Sin'
    m, n = 50, 100
    direction = np.random.rand(m, n, 8)

    if style == 'Direct':
        direction[:, :, 0] = 10
    elif style == 'Sin':
        x = np.arange(n)
        y_d = np.cos(6*np.pi*x/n)
        theta = np.empty_like(x, dtype=np.int)
        theta[y_d > 0.5] = 1
        theta[~(y_d > 0.5) & (y_d > -0.5)] = 0
        theta[~(y_d > 0.5)] = 7
        direction[:, x.astype(np.int), theta] = 10
        direction[:, :] /= np.sum(direction[:, :])
        print(direction)

        loc = np.zeros((m, n), dtype=np.float)
        loc[25, 50] = 1
        loc_prime = np.empty_like(loc)
        loc_prime = loc

        fig = plt.figure(figure=(8, 6), faceclolor='w')
        im = plt.imshow(loc/np.max(loc), cmp='coolwarm')
        anim = animation.FuncAnimation(fig, update, frames=300, intervel=50, blit=True)
        plt.tight_layout(1.5)
        plt.show()
