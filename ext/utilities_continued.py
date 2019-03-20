# missing imports!


######### v NOT VERIFIED OR USED v ############
### Functions

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


def gen_model_filepath(params):
    model_dir = os.path.join(
        'models', '{pre_processing}_{model}'.format(**params))
    create_folder(model_dir)
    model_name = gen_filename(params, 'h5')
    model_path = os.path.join(model_dir, model_name)
    return model_path


def gen_logs_dir(params):
    logs_dir = os.path.join(
        'logs', '{pre_processing}_{model}'.format(**params))
    create_folder(logs_dir)
    return logs_dir


def gen_output_dir(params):
    output_dir = os.path.join(
        'output', '{pre_processing}_{model}'.format(**params))
    output_curr_dir_name = gen_filename(params)
    output_curr_dir = os.path.join(output_dir, output_curr_dir_name)
    create_folder(output_curr_dir)
    return output_curr_dir


def load_model_vae(custom_objects, params):
    model_dir = os.path.join(
        'models', '{pre_processing}_{model}'.format(**params))
    model_name = gen_filename(params, 'h5')
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path, custom_objects=custom_objects)
    # extract encoder from main model
    encoder = model.get_layer('vae_encoder')
    decoder = model.get_layer('vae_decoder')
    return encoder, decoder, model


def load_test_data(params):
    test_dir = os.path.join(
        'test_data', '{pre_processing}_{model}'.format(**params))
    test_name = gen_filename(params, 'pkl')
    test_path = os.path.join(test_dir, test_name)
    df = pd.read_pickle(test_path)
    return df


def load_all_as_test_data(params):
    annotation_path = os.path.expanduser(
        os.path.join(params['annotation_path']))
    print('Reading annotation files from {}'.format(annotation_path))
    df = pd.read_csv(annotation_path, engine='python')
    return df


def load_history_data(params):
    history_dir = os.path.join(
        'history', '{pre_processing}_{model}'.format(**params))
    history_name = gen_filename(params, 'pkl')
    history_path = os.path.join(history_dir, history_name)
    return pd.read_pickle(history_path)

# Storing


def store_history_data(history, params):
    # build path
    history_dir = os.path.join(
        'history', '{pre_processing}_{model}'.format(**params))
    create_folder(history_dir)
    history_name = gen_filename(params, 'pkl')
    history_path = os.path.join(history_dir, history_name)
    # store data
    df = pd.DataFrame(history.history)
    df.to_pickle(history_path)
    print('')


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


# Arbitrarily chosen constant:
norm_coeff = 100

# Input's processing


def norm_channel(ch):
    #ch = 1.0 / (1.0 + np.exp(-norm_coeff*ch))
    return ch


def process_data(s):
    # remove a bin if odd number
    if s.shape[0] % 2 != 0:
        s = s[:-1]
    # split re/im
    re = np.real(s)
    im = np.imag(s)
    re = norm_channel(re)
    im = norm_channel(im)
    s = np.dstack((re, im))
    return s

# Output's inverse processing


def denorm_channel(ch):
    #ch = np.log(-ch / (ch - 1.0)) / norm_coeff
    return ch


def unprocess_data(s):
    # should return a complex spectrum
    # convert to complex:
    re = s[:, :, 0]
    im = s[:, :, 1]
    re = denorm_channel(re)
    im = denorm_channel(im)
    s = re + 1j*im
    # adding previously removed bin
    padding = np.zeros((1, *s.shape[1:]))
    s = np.concatenate((s, padding))
    return s

# Plotting

# Import predictions


def import_spectrogram(filepath):
    if os.path.isfile(filepath):
        print('Reading data from `{}`...'.format(filepath))
        data = np.load(filepath)
        print('Data loaded. Shape = {}'.format(data.shape))
        return data
    else:
        print('Wrong path: `{}`'.format(filepath))
        quit()


def plot_spectrogram(spect, title_name):
    spect_db = librosa.amplitude_to_db(
        np.abs(spect), ref=np.max)  # should it be power_to_db?
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spect_db, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title_name)
    plt.savefig('outputs/spectrogram_'+title_name+'.png')
