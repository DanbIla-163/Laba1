import numpy as np
from PIL import  Image, ImageOps
import Laba1 as gc
from Laba1 import sekator_nelic_gran

f = open("model_1.obj")

vectorv = []
vectorf = []
for line in f:
    v = line.split()

    if (v[0] == "v"):
        vectorv.append([float(v[1]), float(v[2]), float(v[3])])
    if (v[0] == "f"):
        v1 = v[1].split('/')[0]
        v2 = v[2].split('/')[0]
        v3 = v[3].split('/')[0]
        vectorf.append([int(v1), int(v2), int(v3)])

img_mat2 = np.zeros(shape=(2000, 2000, 3), dtype=np.uint8)
z_buffer_mat = np.full((2000, 2000), np.inf, dtype=np.float32)

color = (255, 255, 255)
for i in range(0,len(vectorf)):
    # if (i % 1000) == 0:
    #     color = ((10*i)%255, (20*i)%255, (30*i)%255)
    x0 = ((vectorv[vectorf[i][0]-1][0])*10000 + 1000)
    y0 = ((vectorv[vectorf[i][0]-1][1])*10000 + 500)
    z0 = ((vectorv[vectorf[i][0]-1][2])*10000)
    x1 = ((vectorv[vectorf[i][1]-1][0])*10000 + 1000)
    y1 = ((vectorv[vectorf[i][1]-1][1])*10000 + 500)
    z1 = ((vectorv[vectorf[i][1]-1][2])*10000)
    x2 = ((vectorv[vectorf[i][2]-1][0])*10000 + 1000)
    y2 = ((vectorv[vectorf[i][2]-1][1])*10000 + 500)
    z2 = ((vectorv[vectorf[i][2]-1][2])*10000)
    color = (255*gc.sekator_nelic_gran(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 255*gc.sekator_nelic_gran(x0, y0, z0, x1, y1, z1, x2, y2, z2))
    # gc.bresanham(img_mat2, x0, y0, x1, y1, color)
    # gc.bresanham(img_mat2, x1, y1, x2, y2, color)
    # gc.bresanham(img_mat2, x0, y0, x2, y2, color)
    if gc.sekator_nelic_gran(x0, y0, z0, x1, y1, z1, x2, y2, z2):
        gc.draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_mat2, color, z_buffer_mat)


img = Image.fromarray(img_mat2, mode="RGB")
img = ImageOps.flip(img)
img.save("img.jpg")

