import models

batch_size = 128
model_list = []
model_list.append(models.Saito_single_label_bn('Saito_single_label_bn_large', 64, 0.1, 0.1, 5000, 0.9, 128))

for model in model_list:
	print('============================================================\nCurrent Working Model: %s\n============================================================' % model.model_name)
	model.drone_prediction('PROB_OF_INTEREST', window_sliding_stride=2, interest_label='Building')
