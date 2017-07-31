import models

batch_size = 128
model_list = []
model_list.append(models.Saito_label_bn('Saito_label_bn', 64, 0.0001, 0.1, 5000, 0.9, batch_size))

for model in model_list:
	print('============================================================\nCurrent Working Model: %s\n============================================================' % model.model_name)
	model.drone_prediction('MOST_PROBABLE_CLASS')
	model.drone_prediction('PROB_OF_INTEREST')