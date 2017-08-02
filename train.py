import models

batch_size = 128
model_list = []

model_list.append(models.Saito_label_bn('Saito_new_data', 64, 0.0001, 0.1, 10000, 0.9, batch_size))

for model in model_list:
	print('============================================================\nCurrent Working Model: %s\n============================================================' % model.model_name)
	model.train(50)
