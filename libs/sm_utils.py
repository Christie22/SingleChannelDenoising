# utilities
import os
import pandas as pd
from keras.models import load_model

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def gen_filename(params, extention=None, extra=None):
    params['extention'] = extention
    params['extra'] = extra
    return '_'.join([
        '{kernel_size}',
        '{n_filters}',
        '{n_intermediate_dim}',
        '{n_latent_dim}',
        '{epochs}',
        '{batch_size}' +
            ('_{extra}' if extra is not None else '') +
            ('.{extention}' if extention is not None else '')
    ]).format(**params)

def load_dataset(params):
    annotation_path = os.path.expanduser(os.path.join(params['annotation_path']))
    print('Reading annotation files from {}'.format(annotation_path))
    df = pd.read_csv(annotation_path, engine='python')
    # filenames = df['filepath'].values.tolist()
    # labels = df.drop(columns='filepath').to_dict('list')
    return df

def gen_model_filepath(params):
    model_dir = os.path.join('models', '{pre_processing}_{model}'.format(**params))
    create_folder(model_dir)
    model_name = gen_filename(params, 'h5')
    model_path = os.path.join(model_dir, model_name)
    return model_path

def load_model_vae(custom_objects, params):
    model_dir = os.path.join('models', '{pre_processing}_{model}'.format(**params))
    model_name = gen_filename(params, 'h5')
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path, custom_objects=custom_objects)
    # extract encoder from main model
    encoder = model.get_layer('vae_encoder')
    decoder = model.get_layer('vae_decoder')
    return encoder, decoder, model

def gen_logs_dir(params):
    logs_dir = os.path.join('logs', '{pre_processing}_{model}'.format(**params))
    create_folder(logs_dir)
    return logs_dir

def load_test_data(params):
    test_dir = os.path.join('test_data', '{pre_processing}_{model}'.format(**params))
    test_name = gen_filename(params, 'pkl')
    test_path = os.path.join(test_dir, test_name)
    df = pd.read_pickle(test_path)
    return df

def load_all_as_test_data(params):
    annotation_path = os.path.expanduser(os.path.join(params['annotation_path']))
    print('Reading annotation files from {}'.format(annotation_path))
    df = pd.read_csv(annotation_path, engine='python')
    return df

def store_history_data(history, params):
    # build path
    history_dir = os.path.join('history', '{pre_processing}_{model}'.format(**params))
    create_folder(history_dir)
    history_name = gen_filename(params, 'pkl')
    history_path = os.path.join(history_dir, history_name)
    # store data
    df = pd.DataFrame(history.history)
    df.to_pickle(history_path)
    print('')

def load_history_data(params):
    history_dir = os.path.join('history', '{pre_processing}_{model}'.format(**params))
    history_name = gen_filename(params, 'pkl')
    history_path = os.path.join(history_dir, history_name)
    return pd.read_pickle(history_path)

def gen_output_dir(params):
    output_dir = os.path.join('output', '{pre_processing}_{model}'.format(**params))
    output_curr_dir_name = gen_filename(params)
    output_curr_dir = os.path.join(output_dir, output_curr_dir_name)
    create_folder(output_curr_dir)
    return output_curr_dir

def store_history_plot(fig, params):
    output_dir = gen_output_dir(params)
    output_history_path = os.path.join(output_dir, 'history.png')
    fig.savefig(output_history_path)

def store_latent_space_plot(fig, params):
    output_dir = gen_output_dir(params)
    output_latent_space_path = os.path.join(output_dir, 'latent_space.png')
    fig.savefig(output_latent_space_path)

def store_manifold_plot(fig, params):
    output_dir = gen_output_dir(params)
    output_manifold_path = os.path.join(output_dir, 'manifold.png')
    fig.savefig(output_manifold_path)
