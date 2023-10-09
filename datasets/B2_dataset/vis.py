import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from pprint import pprint
import os

def cross_product(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    cross = np.cross(v1, v2)
    return cross

# with open(r'datasets\B2_dataset\avm_0331\750.json', 'r') as fp:
#     anno = json.load(fp)
# with open(r'datasets\B2_dataset\车位导出10.8\avm_new_26424\avm_dataset\avm_0331\600.json', 'r') as fp:
#     anno = json.load(fp)
    

# pprint(anno)
# pprint(anno.keys())

# img_name = anno['image']
# annos = anno['annotations']
# for a in annos:
#     print(a['polygon'])

# img_center = np.array([256, 256])

fig, ax = plt.subplots(ncols=1, nrows=1)
for i in range(570, 1470, 30):
    with open(os.path.join('车位导出10.8','avm_new_26424','avm_dataset','avm_0331', str(i)+'.json'), 'r') as fp:
        anno = json.load(fp)
    img_name = anno['image']
    annos = anno['annotations']
    ax.cla()
    ax.set_xlim(left=0, right=512)
    ax.set_ylim(top=0, bottom=512)
    img = cv2.imread(r"avm_0331" + f'\{img_name}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    
    for a in annos:
        ax.cla()
        ax.imshow(img)
        # if a['category'] == 0:
        #     x1, y1, x2, y2, x3, y3, x4, y4 = a['polygon']
        #     plt.scatter([x1, x4], [y1, y4])
        # elif a['category'] == 1:
        #     x1, y1, x2, y2, x3, y3, x4, y4 = a['polygon']
        #     plt.scatter([x1, x2], [y1, y2])
        point_xs = a['polygon'][::2]
        point_ys = a['polygon'][1::2]
        points = np.stack([point_xs, point_ys], axis=1)
        cross = cross_product(points[0], points[1], points[2])
        if cross > 0:  # 顺时针
            xs = np.array(a['entry'][::2])
            ys = np.array(a['entry'][1::2])
        elif cross < 0:  # 逆时针
            xs = np.array(a['entry'][::2])
            ys = np.array(a['entry'][1::2])
            xs = xs[::-1]
            ys = ys[::-1]
        points = np.stack([xs, ys], axis=1)
        f = 512 / 3440
        points *= f
        # dis = np.linalg.norm(points - img_center, axis=1)
        # entry_idx = np.argsort(dis)[:2].tolist()
        # print(dis, entry_idx)
        # entry_point = points[[*entry_idx], ...]
        # print(entry_point)
        ax.plot(points[:, 0], points[:, 1], c='b')
        ax.scatter(points[:, 0], points[:, 1], c='r')
        # xs = a['polygon'][::2]
        # ys = a['polygon'][1::2]
        # plt.scatter(points[:, 0], points[:, 1])
        for i in range(len(points)):
            ax.text(points[i, 0], points[i, 1], str(i))
        # break
        plt.pause(2)
    plt.pause(1)
plt.show()
