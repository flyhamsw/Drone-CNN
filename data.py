import sqlite3
import os
import numpy as np
import random
import cv2

def get_db_connection():
	try:
		conn = sqlite3.connect('tensorOrtho.db')
	except Exception as e:
		print(e)

	cur = conn.cursor()

	return conn, cur

def insert_ngii_dataset():
	ngii_dataset_training_dir = 'ngii_dataset_training'
	ngii_dataset_test_dir = 'ngii_dataset_test'

	conn, cur = get_db_connection()
	dataset_training_names = os.listdir(ngii_dataset_training_dir)
	dataset_test_names = os.listdir(ngii_dataset_test_dir)

	cur.execute('delete from ngii_dir;')
	cur.execute('delete from patch_dir;')

	for name in dataset_training_names:
		ngii_x_dir = '%s/%s/x.png' % (ngii_dataset_training_dir, name)
		ngii_y_dir = '%s/%s/y.png' % (ngii_dataset_training_dir, name)
		cur.execute("insert into ngii_dir values ('%s', '%s', '%s', 'training');" % (name, ngii_x_dir, ngii_y_dir))

	for name in dataset_test_names:
		ngii_x_dir = '%s/%s/x.png' % (ngii_dataset_test_dir, name)
		ngii_y_dir = '%s/%s/y.png' % (ngii_dataset_test_dir, name)
		cur.execute("insert into ngii_dir values ('%s', '%s', '%s', 'test');" % (name, ngii_x_dir, ngii_y_dir))

	conn.commit()
	cur.close()
	conn.close()

def insert_drone_dataset():
	drone_dataset_dir = 'drone_dataset'

	conn, cur = get_db_connection()
	dataset_drone_names = os.listdir(drone_dataset_dir)

	cur.execute('delete from drone_dir;')

	for name in dataset_drone_names:
		drone_x_dir = '%s/%s/x.png' % (drone_dataset_dir, name)
		cur.execute("insert into drone_dir values ('%s', '%s');" % (name, drone_x_dir))

	conn.commit()
	cur.close()
	conn.close()

def get_ngii_dir_all():
	conn, cur = get_db_connection()
	cur.execute("select * from ngii_dir;")
	ngii_dir = cur.fetchall()
	cur.close()
	conn.close()
	return ngii_dir

def get_drone_dir_all():
	conn, cur = get_db_connection()
	cur.execute("select * from drone_dir;")
	drone_dir = cur.fetchall()
	cur.close()
	conn.close()
	return drone_dir

def get_ngii_dir(purpose):
	conn, cur = get_db_connection()
	cur.execute("select * from ngii_dir where purpose='%s';" % purpose)
	ngii_dir = cur.fetchall()
	cur.close()
	conn.close()
	return ngii_dir

def get_patch_dir(conn, cur, purpose, batch_size):
	cur.execute("select patch_dir.x_dir, patch_dir.y_dir, patch_dir.building, patch_dir.road, patch_dir.otherwise from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='%s' order by RANDOM() LIMIT %d;" % (purpose, batch_size))
	patch_dir = cur.fetchall()
	return patch_dir

def get_patch_all(conn, cur, purpose):
	cur.execute("select patch_dir.x_dir, patch_dir.y_dir from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='%s';" % purpose)
	patch_dir = cur.fetchall()
	random.shuffle(patch_dir)
	patch_filenames = []
	x_patch_filenames = []
	y_patch_filenames = []
	for row in patch_dir:
		patch_filenames.append((row[0], row[1]))
		x_patch_filenames.append(row[0])
		y_patch_filenames.append(row[1])
	return patch_filenames, x_patch_filenames, y_patch_filenames

def get_patch_dir_building(conn, cur, purpose, batch_size):
	cur.execute("select patch_dir.x_dir, patch_dir.y_dir, patch_dir.building, patch_dir.road, patch_dir.otherwise from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='%s' and patch_dir.building=1 order by RANDOM() LIMIT %d;" % (purpose, int(batch_size/2)))
	patch_dir = cur.fetchall()
	return patch_dir

def get_patch_dir_no_building(conn, cur, purpose, batch_size):
	cur.execute("select patch_dir.x_dir, patch_dir.y_dir, patch_dir.building, patch_dir.road, patch_dir.otherwise from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='%s' and patch_dir.building=0 order by RANDOM() LIMIT %d;" % (purpose, int(batch_size/2)))
	patch_dir = cur.fetchall()
	return patch_dir

def get_drone_patch_dir(conn, cur, start_idx, batch_size):
	cur.execute("select x_dir from drone_patch_dir where num between %d and %d" % (start_idx, start_idx + batch_size - 1))
	drone_patch_dir = cur.fetchall()
	return drone_patch_dir

def get_ohe(y_batch_fnames):
	conn, cur = get_db_connection()

	ohe_list = []

	for y_dir in y_batch_fnames:
		cur.execute("select building, road, otherwise from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where patch_dir.y_dir='%s';" % y_dir)
		ohe = cur.fetchall()

		building = ohe[0][0]
		road = ohe[0][1]
		otherwise = ohe[0][2]

		ohe = [building, road, otherwise]
		ohe_list.append(ohe)

	cur.close()
	conn.close()

	return ohe_list

def get_steps(batch_size):
	conn, cur = get_db_connection()
	cur.execute("select count(*) from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='training';")
	rows = cur.fetchall()
	steps = int(rows[0][0]/batch_size)

	print('%d steps / epoch' % steps)

	return steps

def get_patch_num(dataset_name):
	conn, cur = get_db_connection()
	cur.execute("select count(*) from drone_patch_dir where name='%s'" % dataset_name)
	num = cur.fetchall()
	return num[0][0]

def make_batch_from_patch_queue(batch_size, patch_queue):
	x_batch = []
	y_batch = []
	for i in range(0, batch_size):
		try:
			x_patch, y_patch = patch_queue.pop()
			x_batch.append(cv2.cvtColor(cv2.imread(x_patch), cv2.COLOR_BGR2RGB))
			y_batch.append(cv2.cvtColor(cv2.imread(y_patch), cv2.COLOR_BGR2RGB))
		except Exception as e:
			pass
	return x_batch, y_batch

def make_batch(conn, cur, purpose, batch_size, y_batch_interest=None):
	patch_dir = []

	if y_batch_interest == 'Building':
		patch_dir_building = get_patch_dir_building(conn, cur, purpose, batch_size)
		patch_dir_no_building = get_patch_dir_no_building(conn, cur, purpose, batch_size)
		for row in patch_dir_building:
			patch_dir.append(row)
		for row in patch_dir_no_building:
			patch_dir.append(row)
	else:
		patch_dir = get_patch_dir(conn, cur, purpose, batch_size)

	x_batch_image = []
	y_batch_ohe = []
	y_batch_image = []

	for i in range(0, len(patch_dir)):
		x_batch_image.append(cv2.imread(patch_dir[i][0]))
		if y_batch_interest == 'Building':
			if patch_dir[i][2] == 1:
				y_batch_ohe.append([1, 0])
			else:
				y_batch_ohe.append([0, 1])
		else:
			y_batch_ohe.append([patch_dir[i][2], patch_dir[i][3], patch_dir[i][4]])
			im = cv2.imread(patch_dir[i][1])
			im_ch_0 = im[:,:,0]
			im_ch_1 = im[:,:,1]
			im_ch_2 = im[:,:,2]
			im_merge = im_ch_0 + im_ch_1 * 1 + im_ch_2 * 2
			y_batch_image.append(im_merge)

	return x_batch_image, y_batch_ohe, y_batch_image

def make_batch_drone(conn, cur, start_idx, batch_size):
	patch_dir = get_drone_patch_dir(conn, cur, start_idx, batch_size)

	x_batch_fnames = []

	for i in range(0, len(patch_dir)):
		x_batch_fnames.append(patch_dir[i][0])

	x_batch = []

	for fname in x_batch_fnames:
		print(fname)
		x_batch.append(cv2.imread(fname))

	return x_batch

def insert_patch(name, x_data, y_data, y_label):
	conn, cur = get_db_connection()

	if len(x_data) > len(y_data):
		num_data = len(y_data)
	else:
		num_data = len(x_data)

	for i in range(0, num_data):
		curr_dataset_name = name[i]
		x_patch_dir = x_data[i]
		y_patch_dir = y_data[i]

		road = 1 if y_label[i] == 'road' else 0
		building = 1 if y_label[i] == 'building' else 0
		otherwise = 1 if y_label[i] == 'otherwise' else 0

		cur.execute("insert into patch_dir values ('%s', '%s', '%s', %r, %r, %r);" % (curr_dataset_name, x_patch_dir, y_patch_dir, building, road, otherwise))

	conn.commit()
	cur.close()
	conn.close()

def insert_drone_patch(name, x_data, num):
	conn, cur = get_db_connection()

	for i in range(0, len(x_data)):
		curr_dataset_name = name[i]
		x_patch_dir = x_data[i]
		curr_num = num[i]

		cur.execute("insert into drone_patch_dir values ('%s', '%s', %d);" % (curr_dataset_name, x_patch_dir, curr_num))

	conn.commit()
	cur.close()
	conn.close()

if __name__=='__main__':
	insert_ngii_dataset()
	insert_drone_dataset()
