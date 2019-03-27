import click
from scripts.train import train as script_train
from scripts.results import results as script_results
from scripts.denoise import denoise as script_denoise

# pass models as class, parse model_name here (just like we do it snrs)


defaults = {
    'sr': 16000,
    'rir_path': '/data/riccardo_datasets/rirs/train/',
    'noise_snrs': '5 25',
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'frag_hop_length': 128,
    'frag_win_length': 256,
    'batch_size': 32,
    'epochs': 20,
    'model_path': '/data/riccardo_models/denoising/model_e{epoch}.h5',
    'history_path': None
}


@click.group()
@click.option('--cuda_device', type=str, default='2')
@click.pass_context
def cli(ctx, cuda_device):
    ctx.obj['cuda_device'] = cuda_device


# TRAIN 
@cli.command()
@click.pass_context
@click.argument('model_name', type=str)
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--sr', type=int, default=defaults['sr'])
@click.option('--rir_path', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=defaults['rir_path'])
@click.option('--noise_snrs', type=str, default=defaults['noise_snrs'])
@click.option('--n_fft', type=int, default=defaults['n_fft'])
@click.option('--hop_length', type=int, default=defaults['hop_length'])
@click.option('--win_length', type=int, default=defaults['win_length'])
@click.option('--frag_hop_length', type=int, default=defaults['frag_hop_length'])
@click.option('--frag_win_length', type=int, default=defaults['frag_win_length'])
@click.option('--batch_size', type=int, default=defaults['batch_size'])
@click.option('--epochs', type=int, default=defaults['epochs'])
@click.option('--model_path', type=click.Path(), default=defaults['model_path'])
@click.option('--history_path', type=click.Path(), default=defaults['history_path'])
@click.option('--force_cacheinit', is_flag=True, default=False)
def train(ctx,
          model_name, 
          dataset_path, 
          sr,
          rir_path, 
          noise_snrs,
          n_fft, 
          hop_length,
          win_length,
          frag_hop_length,
          frag_win_length,
          batch_size,
          epochs,
          model_path,
          history_path,
          force_cacheinit):
    noise_snrs_list = [int(n) for n in noise_snrs.split(' ')]
    script_train(model_name,
                 dataset_path,
                 sr,
                 rir_path,
                 noise_snrs_list,
                 n_fft,
                 hop_length,
                 win_length,
                 frag_hop_length,
                 frag_win_length,
                 batch_size,
                 epochs,
                 model_path,
                 history_path,
                 force_cacheinit,
                 ctx.obj['cuda_device'])


# RESULTS
@cli.command()
@click.pass_context
@click.argument('model_name', type=str)
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--sr', type=int, default=defaults['sr'])
@click.option('--rir_path', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=defaults['rir_path'])
@click.option('--noise_snrs', type=str, default=defaults['noise_snrs'])
@click.option('--n_fft', type=int, default=defaults['n_fft'])
@click.option('--hop_length', type=int, default=defaults['hop_length'])
@click.option('--win_length', type=int, default=defaults['win_length'])
@click.option('--frag_hop_length', type=int, default=defaults['frag_hop_length'])
@click.option('--frag_win_length', type=int, default=defaults['frag_win_length'])
@click.option('--batch_size', type=int, default=defaults['batch_size'])
def results(ctx,
            model_name,
            model_path,
            dataset_path,
            sr,
            rir_path,
            noise_snrs,
            n_fft,
            hop_length,
            win_length,
            frag_hop_length,
            frag_win_length,
            batch_size):
    noise_snrs_list = [int(n) for n in noise_snrs.split(' ')]
    script_results(model_name,
                   model_path,
                   dataset_path,
                   sr,
                   rir_path,
                   noise_snrs_list,
                   n_fft,
                   hop_length,
                   win_length,
                   frag_hop_length,
                   frag_win_length,
                   batch_size,
                   ctx.obj['cuda_device'])

# DENOISE
@cli.command()
@click.pass_context
@click.argument('model_name', type=str)
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('input_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('output_path', type=click.Path())
@click.option('--sr', type=int, default=defaults['sr'])
@click.option('--n_fft', type=int, default=defaults['n_fft'])
@click.option('--hop_length', type=int, default=defaults['hop_length'])
@click.option('--win_length', type=int, default=defaults['win_length'])
@click.option('--frag_hop_length', type=int, default=defaults['frag_hop_length'])
@click.option('--frag_win_length', type=int, default=defaults['frag_win_length'])
@click.option('--batch_size', type=int, default=defaults['batch_size'])
def denoise(ctx,
            model_name,
            model_path,
            input_path,
            output_path,
            sr,
            n_fft,
            hop_length,
            win_length,
            frag_hop_length,
            frag_win_length,
            batch_size):
    script_denoise(model_name,
                   model_path,
                   input_path,
                   output_path,
                   sr,
                   n_fft,
                   hop_length,
                   win_length,
                   frag_hop_length,
                   frag_win_length,
                   batch_size,
                   ctx.obj['cuda_device'])


if __name__ == "__main__":
    cli(obj={})
