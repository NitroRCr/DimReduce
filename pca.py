from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
import json


def out_points(vectors, filename):
    length = len(vectors)
    if vectors.shape[1] == 3:
        out_points = [{'x': round(vectors[i][0], 5), 'y': round(vectors[i][1], 5),
                       'z': round(vectors[i][2], 5)} for i in range(length)]
    elif vectors.shape[1] == 2:
        out_points = [{'x': round(vectors[i][0], 2), 'y': CENTER_HEIGHT,
                       'z': round(vectors[i][1], 2)} for i in range(length)]
    output_f = open(filename, 'w')
    output_f.write(json.dumps(out_points))
    output_f.close()


OUTPUT_FILE_3D = 'pca_3d.json'
OUTPUT_FILE_2D = 'pca_2d.json'
RADIUS_3D = 180
CUBE_LEN = 5.625
CENTER_HEIGHT = -180

# 建立3维空间坐标
radius2d = (((RADIUS_3D*2/CUBE_LEN)**3)**0.5)*CUBE_LEN/2
axis3d = np.arange(-RADIUS_3D + CUBE_LEN/2, RADIUS_3D, CUBE_LEN)
vectors3d = np.zeros((len(axis3d)**3, 3))
index = 0
# 将所有坐标放入一个列表中
for y in axis3d:
    for x in axis3d:
        for z in axis3d:
            vector = [x, y, z]
            for dim in range(3):
                vectors3d[index][dim] = vector[dim]
            index += 1
# 输出三维坐标到文件
out_points(vectors3d, OUTPUT_FILE_3D)
# PCA降维
pca = PCA(n_components=2, whiten=True)
vectors2d = pca.fit_transform(vectors3d)
# 标准化，使点分布均匀
vectors2d = scale(vectors2d)
# 乘以系数，使点分散开
vectors2d *= (radius2d*2/(np.max(vectors2d) - np.min(vectors2d)))
# 输出二维坐标到文件
out_points(vectors2d, OUTPUT_FILE_2D)

