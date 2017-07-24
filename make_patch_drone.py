import data
import cv2
import numpy as np
import os

drone_dir = data.get_drone_dir_all()

patches_dir = 'patches'

patch_size = 64

patch_stride = patch_size

for row in drone_dir:
	name = []
	curr_dataset_name = row[0]
	x_dir = row[1]

	x = np.array(cv2.imread(x_dir))

	xpath = '%s/%s/x' % (patches_dir, curr_dataset_name)

	os.makedirs(xpath)

	rows = x.shape[0]
	cols = x.shape[1]

	x_data = []
	num = []
	k = 0

	for i in range(0, rows, patch_stride):
		for j in range(0, cols, patch_stride):
			try:
				x_patch = np.array(x[i:i+patch_size, j:j+patch_size])
				if x_patch.shape != (patch_size, patch_size, 3):
					print('boundary! NO LOOK PASS')
				else:
					x_patch_0 = x_patch

					xname0 = '%s/NGII_Data_%s_%s_x_0.png' % (xpath, i, j)
					cv2.imwrite(xname0, x_patch_0)

					x_data.append(xname0)

					name.append(curr_dataset_name)
					num.append(k)
					k = k + 1
					print('NGII_Data_%s_%s_x_0.png done.' % (i, j))
			except Exception as e:
				print(e)

	try:
		data.insert_drone_patch(name, x_data, num)
	except Exception as e:
		print(e)
