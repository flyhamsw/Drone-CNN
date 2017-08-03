import models

batch_size = 128
model_list = []
model_list.append(models.Saito_single_label_bn('Saito_single_label_bn', 64, 0.1, 0.1, 10000, 0.9, 128))

for model in model_list:
	print('============================================================\nCurrent Working Model: %s\n============================================================' % model.model_name)
	model.drone_prediction('PROB_OF_INTEREST', interest_label='Building')
