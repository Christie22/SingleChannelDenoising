from keras.utils import plot_model
import os

import model_conv_dilation_leaky
import model_hsu_glass
import model_vanilla_resnet


path_root = 'graphs'

models = {
    'DilConv': model_conv_dilation_leaky,
    'HsuGlass': model_hsu_glass,
    'ResNet': model_vanilla_resnet
}

for mName in models:
    print(mName)


    print(path_root, mName)
    path_root2 = os.path.join(path_root, mName)
    print(path_root2)
    os.makedirs(path_root2, exist_ok=True)

    obj = models[mName].VaeModel(
        input_shape=(256,64,2),
        kernel_size=3,
        n_filters=256,
        n_intermediate_dim=128,
        n_latent_dim=6)
    model = obj.get_model()

    path_model = os.path.join(path_root2, 'model.png')
    plot_model(
            model,
            to_file=path_model,
            show_shapes=True,
            show_layer_names=True)

    path_enc = os.path.join(path_root2, 'encoder.png')
    plot_model(
            model.get_layer('vae_encoder'),
            to_file=path_enc,
            show_shapes=True,
            show_layer_names=True)

    path_dec = os.path.join(path_root2, 'decoder.png')
    plot_model(
            model.get_layer('vae_decoder'),
            to_file=path_dec,
            show_shapes=True,
            show_layer_names=True)

    print(path_model, path_enc, path_dec)

