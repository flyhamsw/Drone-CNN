import models

batch_size = 128
model_dict = {'Saito_label_bn': models.Saito_label_bn(input_patch_size=64, lr_value=0.0001, lr_decay_rate=0.1, lr_decay_freq=5000, m_value=0.9, batch_size=batch_size), 'VGG16_label': models.VGG16_label(input_patch_size=64, lr_value=0.0005, lr_decay_rate=0.1, lr_decay_freq=5000, m_value=0.9, batch_size=batch_size)}

for model_name in model_dict:
	model_dict[model_name].train(1)
