import data
import cv2
import numpy as np
import os

ngii_dir = data.get_ngii_dir()

training_patches_dir = 'training_patches'

x_patch_size = 64
y_patch_size = 64

x_patch_stride = 10
y_patch_stride = 10

y_patch_ctr = 31

for row in ngii_dir:
	name = row[0]
	x_dir = row[1]
	y_dir = row[2]

	x = np.array(cv2.imread(x_dir))
	y = np.array(cv2.imread(y_dir))

	xpath = '%s/%s/x' % (training_patches_dir, name)
	ypath = '%s/%s/y' % (training_patches_dir, name)

	os.makedirs(xpath)
	os.makedirs(ypath)

	x_rows = x.shape[0]
	print(x_rows)
	x_cols = x.shape[1]
	y_rows = y.shape[0]
	y_cols = y.shape[1]

	x_data = []
	y_data = []
	y_label = []

	for i in range(0, x_rows, x_patch_stride):
		for j in range(0, x_cols, x_patch_stride):
			try:
				x_patch = np.array(x[i:i+x_patch_size, j:j+x_patch_size])

				if x_patch.shape != (x_patch_size, x_patch_size, 3):
					print('boundary! NO LOOK PASS')
				else:
					x_patch_0 = x_patch
					xname0 = '%s/NGII_Data_%s_%s_x_0.png' % (xpath, i, j)
					cv2.imwrite(xname0, x_patch_0)
					x_data.append(xname0)
					print('NGII_Data_%s_%s_x_0.png done.' % (i, j))
			except Exception as e:
				print(e)


	for i in range(0, y_rows, y_patch_stride):
		for j in range(0, y_cols, y_patch_stride):
			try:
				y_patch = np.array(y[i:i+y_patch_size, j:j+y_patch_size])

				if y_patch.shape != (y_patch_size, y_patch_size, 3):
					print('boundary! NO LOOK PASS')
				else:
					y_patch_0 = y_patch
					yname0 = '%s/NGII_Data_%s_%s_y_0.png' % (ypath, i, j)
					cv2.imwrite(yname0, y_patch_0)
					y_data.append(yname0)

					one_hot_element = y_patch_0[y_patch_ctr][y_patch_ctr]

					if one_hot_element[0] == 1:
						one_hot_enc = 'otherwise'
					elif one_hot_element[1] == 1:
						one_hot_enc = 'road'
					elif one_hot_element[2] == 1:
						one_hot_enc = 'building'
					else:
						one_hot_enc = 'otherwise'

					y_label.append(one_hot_enc)

					print('NGII_Data_%s_%s_y_0.png done, and it is %s' % (i, j, one_hot_enc))
			except Exception as e:
				print(e)

	try:
		data.insert_patch(name, x_data, y_data, y_label)
	except Exception as e:
		print(e)
