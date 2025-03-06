import math

import numpy as np
from PIL import Image

img_mat = np.zeros((200, 200, 3), dtype=np.uint8)
img_mat[0:200,0:200,0] = 25
img_mat[0:200,0:200,1] = 25
img_mat[0:200,0:200,2] = 112

def dotted_line(image, x0, y0, x1, y1, color):
    count = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    xchange = False

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in np.arange(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

def bresanham(image, x0, y0, x1, y1, color):
    xchange = False

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1-y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in np.arange(x0, x1):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > (x1 - x0):
            derror -= 2*(x1-x0)
            y += y_update

for i in range(13):
    x0 = 100
    y0 = 100
    x1 = round(100 + 95 * np.cos(i * 2 * np.pi / 13))
    y1 = round(100 + 95 * np.sin(i * 2 * np.pi / 13))
    bresanham(img_mat, x0, y0, x1, y1, (255, 255, 255))

def baricenter (x0, y0, x1, y1, x2, y2, x, y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_matr, color, z_buffer):
    xmin = min(x0, x1, x2)
    xmax = max(x0, x1, x2)
    ymin = min(y0, y1, y2)
    ymax = max(y0, y1, y2)

    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0

    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = math.ceil(xmax)
    ymax = math.ceil(ymax)

    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            lam0, lam1, lam2 = baricenter(x0, y0, x1, y1, x2, y2, x, y)
            if (lam0 >= 0 and lam1>= 0 and lam2 >= 0):
                z_coord = lam0*z0 + lam1*z1 + lam2*z2
                if z_coord > z_buffer[y][x]:
                    continue
                else:
                    img_matr[y, x] = color
                    z_buffer[y][x] = z_coord

# draw_triangle(100.0, 100.0, 500.0, 500.0, 100.0, 1000.0, img_matr)
# img = Image.fromarray(img_mat, mode='RGB')
# img.save('img.jpg')

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])

def sekator_nelic_gran(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0,0,1]
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    return np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))