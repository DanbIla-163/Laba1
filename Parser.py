import numpy as np
from PIL import  Image, ImageOps
import Laba1 as gc

#Моделька
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

#Соединить все вершины треугольника, нарисовать хранится в виде: (x1, x2, x3)(y1,y2,y3)(z1,z2,z3)

img_mat2 = np.zeros(shape=(2000, 2000, 3), dtype=np.uint8)

color = (255, 255, 255)
for i in range(0,len(vectorf)):
    if (i % 100) == 0:
        color = ((10*i)%255, (20*i)%255, (30*i)%255)

    x0 = int((vectorv[vectorf[i][0]-1][0])*10000 + 1000)
    y0 = int((vectorv[vectorf[i][0]-1][1])*10000 + 500)
    x1 = int((vectorv[vectorf[i][1]-1][0])*10000 + 1000)
    y1 = int((vectorv[vectorf[i][1]-1][1])*10000 + 500)
    x2 = int((vectorv[vectorf[i][2]-1][0])*10000 + 1000)
    y2 = int((vectorv[vectorf[i][2]-1][1])*10000 + 500)
    gc.bresanham(img_mat2, x0, y0, x1, y1, color)
    gc.bresanham(img_mat2, x1, y1, x2, y2, color)
    gc.bresanham(img_mat2, x0, y0, x2, y2, color)



# for fl in vectorv:
#     img_mat2[round(fl[1]*5000) + 250, round(fl[0]*5000)+ 500] = (255, 255, 255)
img = Image.fromarray(img_mat2, mode="RGB")
img = ImageOps.flip(img)
img.save("img.jpg")

