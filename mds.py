from sklearn.manifold import MDS
import numpy as np
import json

INPUT_FILE = 'earth_shell_5it.json'
OUTPUT_FILE = 'mds.json'
WIDTH = 1440
CENTER_HEIGHT = -180

input_f = open(INPUT_FILE)
in_points = json.loads(input_f.read())
input_f.close()

length = len(in_points)
in_arr = np.zeros((length, 3))
for i in range(length):
    indexes = ['x', 'y', 'z']
    for j in range(3):
        in_arr[i][j] = in_points[i][indexes[j]]

mds = MDS(n_components=2, metric=False)
out_arr = mds.fit_transform(in_arr)
out_arr *= WIDTH/(np.max(out_arr) - np.min(out_arr))
out_points = [{'x': round(out_arr[i][0], 2), 'y': CENTER_HEIGHT,
               'z': round(out_arr[i][1], 2), 'color': in_points[i]['color']} for i in range(length)]
output_f = open(OUTPUT_FILE, 'w')
output_f.write(json.dumps(out_points))
output_f.close()