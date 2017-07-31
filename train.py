import models

batch_size = 128
model_list = []

#model_list.append(models.Saito_label_bn('Saito_label_bn', 64, 0.0001, 0.1, 5000, 0.9, batch_size))
model_list.append(models.Saito_label_bn('Saito_label_bn_sum', 64, 0.0001, 0.1, 5000, 0.9, batch_size))
#model_list.append(models.VGG16_label('VGG16_label', 64, 0.0005, 0.1, 5000, 0.9, batch_size))

for model in model_list:
	print('============================================================\nCurrent Working Model: %s\n============================================================' % model.model_name)
	model.train(50)
