import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from pprint import pprint

with open(r'datasets\B2_dataset\avm_0331\1400.json', 'r') as fp:
    anno = json.load(fp)

pprint(anno)
pprint(anno.keys())

img_name = anno['image']
annos = anno['annotations']
for a in annos:
    print(a['polygon'])


img = cv2.imread(r"datasets\B2_dataset\avm_0331" + f'\{img_name}')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))
plt.figure()
plt.imshow(img)
for a in annos:
    # if a['category'] == 0:
    #     x1, y1, x2, y2, x3, y3, x4, y4 = a['polygon']
    #     plt.scatter([x1, x4], [y1, y4])
    # elif a['category'] == 1:
    #     x1, y1, x2, y2, x3, y3, x4, y4 = a['polygon']
    #     plt.scatter([x1, x2], [y1, y2])
    xs = np.array(a['polygon'][::2])
    ys = np.array(a['polygon'][1::2])
    points = np.stack([xs, ys], axis=1)
    print(points.shape)
    
    x1, y1, x2, y2 = a['entry']
    f = 512 / 3440
    plt.scatter([x1*f, x2*f], [y1*f, y2*f])
    # xs = a['polygon'][::2]
    # ys = a['polygon'][1::2]
    # plt.scatter(xs, ys)
    # for i in range(len(xs)):
    #     plt.text(xs[i], ys[i], str(i))
    break
plt.show()
