import data
import cv2
import numpy as np
import os

ngii_dir = data.get_ngii_dir()

training_patches_dir = 'training_patches'

x_patch_size = 64
y_patch_size = 64

x_patch_stride = 64
y_patch_stride = 64

y_patch_ctr = 31

for row in ngii_dir:
	name = []
	curr_dataset_name = row[0]
	x_dir = row[1]
	y_dir = row[2]

	x = np.array(cv2.imread(x_dir))
	y = np.array(cv2.imread(y_dir))

	xpath = '%s/%s/x' % (training_patches_dir, curr_dataset_name)
	ypath = '%s/%s/y' % (training_patches_dir, curr_dataset_name)

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

					M = cv2.getRotationMatrix2D((x_patch_0.shape[1]/2, x_patch_0.shape[0]/2), 90, 1)
					x_patch_90 = cv2.warpAffine(x_patch_0, M, (x_patch_0.shape[1], x_patch_0.shape[0]))
					x_patch_180 = cv2.warpAffine(x_patch_90, M, (x_patch_0.shape[1], x_patch_0.shape[0]))
					x_patch_270 = cv2.warpAffine(x_patch_180, M, (x_patch_0.shape[1], x_patch_0.shape[0]))

					xname0 = '%s/NGII_Data_%s_%s_x_0.png' % (xpath, i, j)
					cv2.imwrite(xname0, x_patch_0)
					xname90 = '%s/NGII_Data_%s_%s_x_90.png' % (xpath, i, j)
					cv2.imwrite(xname90, x_patch_90)
					xname180 = '%s/NGII_Data_%s_%s_x_180.png' % (xpath, i, j)
					cv2.imwrite(xname180, x_patch_180)
					xname270 = '%s/NGII_Data_%s_%s_x_270.png' % (xpath, i, j)
					cv2.imwrite(xname270, x_patch_270)

					x_data.append(xname0)
					x_data.append(xname90)
					x_data.append(xname180)
					x_data.append(xname270)

					for p in range(0, 4):
						name.append(curr_dataset_name)
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

					M = cv2.getRotationMatrix2D((y_patch_0.shape[1]/2, y_patch_0.shape[0]/2), 90, 1)
					y_patch_90 = cv2.warpAffine(y_patch_0, M, (y_patch_0.shape[1], y_patch_0.shape[0]))
					y_patch_180 = cv2.warpAffine(y_patch_90, M, (y_patch_0.shape[1], y_patch_0.shape[0]))
					y_patch_270 = cv2.warpAffine(y_patch_180, M, (y_patch_0.shape[1], y_patch_0.shape[0]))

					yname0 = '%s/NGII_Data_%s_%s_y_0.png' % (ypath, i, j)
					cv2.imwrite(yname0, y_patch_0)
					yname90 = '%s/NGII_Data_%s_%s_y_90.png' % (ypath, i, j)
					cv2.imwrite(yname90, y_patch_90)
					yname180 = '%s/NGII_Data_%s_%s_y_180.png' % (ypath, i, j)
					cv2.imwrite(yname180, y_patch_180)
					yname270 = '%s/NGII_Data_%s_%s_y_270.png' % (ypath, i, j)
					cv2.imwrite(yname270, y_patch_270)

					y_data.append(yname0)
					y_data.append(yname90)
					y_data.append(yname180)
					y_data.append(yname270)

					'''
					#Determine one hot encoding by center pixel
					one_hot_element = y_patch_0[y_patch_ctr][y_patch_ctr]
					if one_hot_element[0] == 1:
						one_hot_enc = 'otherwise'
					elif one_hot_element[1] == 1:
						one_hot_enc = 'road'
					elif one_hot_element[2] == 1:
						one_hot_enc = 'building'
					else:
						one_hot_enc = 'otherwise'
					'''

					#Determine one hot encoding by raster statistics
					y_patch_sample = y_patch_0[15:31, 15:31, :]

					sum_ch_0 = np.sum(y_patch_sample[:,:,0])
					sum_ch_1 = np.sum(y_patch_sample[:,:,1])
					sum_ch_2 = np.sum(y_patch_sample[:,:,2])

					one_hot_element = np.argmax([sum_ch_0, sum_ch_1, sum_ch_2])

					if one_hot_element == 0:
						one_hot_enc = 'otherwise'
					elif one_hot_element == 1:
						one_hot_enc = 'road'
					elif one_hot_element == 2:
						one_hot_enc = 'building'
					else:
						one_hot_enc = 'otherwise'

					for p in range(0, 4):
						y_label.append(one_hot_enc)

					print('NGII_Data_%s_%s_y_0.png done, and it is %s' % (i, j, one_hot_enc))
			except Exception as e:
				print(e)

	try:
		data.insert_patch(name, x_data, y_data, y_label)
	except Exception as e:
		print(e)
