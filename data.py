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
	ngii_dataset_dir = 'ngii_dataset'

	conn, cur = get_db_connection()
	dataset_names = os.listdir(ngii_dataset_dir)

	cur.execute('delete from ngii_dir;')
	cur.execute('delete from patch_dir;')

	for name in dataset_names:
		ngii_x_dir = '%s/%s/x.png' % (ngii_dataset_dir, name)
		ngii_y_dir = '%s/%s/y.png'%(ngii_dataset_dir, name)
		cur.execute("insert into ngii_dir values ('%s', '%s', '%s');" % (name, ngii_x_dir, ngii_y_dir))

	conn.commit()
	cur.close()
	conn.close()

def get_ngii_dir():
	conn, cur = get_db_connection()
	cur.execute('select * from ngii_dir;')
	ngii_dir = cur.fetchall()
	cur.close()
	conn.close()
	return ngii_dir

def get_patch_dir(name):
	conn, cur = get_db_connection()
	cur.execute("select x_dir, y_dir from patch_dir where name='%s';" % name)
	patch_dir = cur.fetchall()
	cur.close()
	conn.close()
	return patch_dir

def get_ohe(y_batch_fnames):
	conn, cur = get_db_connection()

	ohe_list = []

	for y_dir in y_batch_fnames:
		cur.execute("select building, road, otherwise from patch_dir where y_dir='%s';" % y_dir)
		ohe = cur.fetchall()


		building = ohe[0][0]
		road = ohe[0][1]
		otherwise = ohe[0][2]

		ohe = [building, road, otherwise]
		ohe_list.append(ohe)

	cur.close()
	conn.close()

	return ohe_list

def make_batch(name, batch_size):
	patch_dir = get_patch_dir(name)

	x_batch_fnames = []
	y_batch_fnames = []

	for i in random.sample(range(len(patch_dir)-1), batch_size):
		x_batch_fnames.append(patch_dir[i][0])
		y_batch_fnames.append(patch_dir[i][1])

	x_batch = []

	for fname in x_batch_fnames:
		x_batch.append(cv2.imread(fname))

	y_batch_ohe = get_ohe(y_batch_fnames)

	y_batch_image = []
	for fname in y_batch_fnames:
		y_batch_image.append(cv2.imread(fname))

	return x_batch, y_batch_image, y_batch_ohe

def insert_patch(name, x_data, y_data, y_label):
	conn, cur = get_db_connection()

	if len(x_data) > len(y_data):
		num_data = len(y_data)
	else:
		num_data = len(x_data)

	for i in range(0, num_data):
		x_patch_dir = x_data[i]
		y_patch_dir = y_data[i]

		road = 1 if y_label[i] == 'road' else 0
		building = 1 if y_label[i] == 'building' else 0
		otherwise = 1 if y_label[i] == 'otherwise' else 0

		cur.execute("insert into patch_dir values ('%s', '%s', '%s', %r, %r, %r);" % (name, x_patch_dir, y_patch_dir, building, road, otherwise))

	conn.commit()
	cur.close()
	conn.close()

if __name__=='__main__':
	insert_ngii_dataset()
