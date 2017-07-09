from matplotlib import pyplot as plt
import cv2
import data

ngii_dir = data.get_ngii_dir()
num_data = len(data.get_ngii_dir())

def label_coder(ohe):
    if ohe == [1, 0, 0]: label = 'otherwise'
    if ohe == [0, 1, 0]: label = 'Road'
    if ohe == [0, 0, 1]: label = 'Building'
    return label

for i in range(0, num_data):
    dataset_name = ngii_dir[i][0]
    print('Current Dataset: %s (num_data %d)' % (dataset_name, i))

    f, axarr = plt.subplots(5)

    x_batch, y_batch,_,_ = data.make_batch(dataset_name, 5, 'ohe')

    for j in range(0, 5):
        axarr[j].imshow(x_batch[j])
        axarr[j].set_title(label_coder(y_batch[j]))

    plt.show()
