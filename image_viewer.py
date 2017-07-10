from matplotlib import pyplot as plt
import cv2
import data

ngii_dir = data.get_ngii_dir()
num_data = len(data.get_ngii_dir())

def label_coder(ohe):
    if ohe == [1, 0, 0]: label = 'otherwise'
    if ohe == [0, 1, 0]: label = 'road'
    if ohe == [0, 0, 1]: label = 'building'
    return label

for i in range(0, num_data):
    dataset_name = ngii_dir[i][0]
    print('Current Dataset: %s (num_data %d)' % (dataset_name, i))

    f, axarr = plt.subplots(2, 10)

    x_batch, y_batch_image, y_batch_ohe = data.make_batch(dataset_name, 20)

    for j in range(0, 20):
        axarr[0, j].imshow(x_batch[j]) if j < 10 else axarr[1, j-10].imshow(x_batch[j])
        axarr[0, j].set_title(label_coder(y_batch_ohe[j])) if j < 10 else axarr[1, j-10].set_title(label_coder(y_batch_ohe[j]))

    plt.show()
