import models

#model = models.Saito_label_bn('Saito_new_data', 64, 0.0001, 0.1, 10000, 0.9, 128)
#model = models.Saito_single_label_bn('Saito_single_label_bn_large', 64, 0.1, 0.1, 5000, 0.9, 128)
model = models.VGG16_deconv_single('VGG16_deconv_single', 224, 1e-2, 0.1, 5000, 0.9, 8)
print('============================================================\nCurrent Working Model: %s\n============================================================' % model.model_name)
model.train(500)
