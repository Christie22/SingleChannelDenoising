import click
from scripts.train import train as script_train
from scripts.results import results as script_results
from scripts.denoise import denoise as script_denoise

defaults = {
    'sr': 16000,
    'rir_path': None,  # '/data/riccardo_datasets/rirs/train/',
    'noise_snrs': '5,15,30',
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'frag_hop_length': 16,
    'frag_win_length': 32,
    'batch_size': 256,
    'epochs': 100,
    'model_destination': '/data/riccardo_models/denoising/model_e{epoch}.h5',
    'cuda_device': '2',
    'logs_path': './training_logs.json',
    'results_output_path': '../results/'
}


@click.group()
@click.option('--cuda_device', type=str, default=defaults['cuda_device'])
@click.pass_context
def cli(ctx, cuda_device):
    ctx.obj['cuda_device'] = cuda_device


# TRAIN 
@cli.command()
@click.pass_context
@click.argument('model_source', type=click.Path(exists=True, file_okay=True, dir_okay=False))
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
@click.option('--model_destination', type=click.Path(), default=defaults['model_destination'])
@click.option('--logs_path', type=click.Path(), default=defaults['logs_path'])
@click.option('--force_cacheinit', is_flag=True, default=False)
def train(ctx,
          model_source, 
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
          model_destination,
          logs_path,
          force_cacheinit):
    noise_snrs_list = [int(n) for n in noise_snrs.split(',')]
    script_train(model_source,
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
                 model_destination,
                 logs_path,
                 force_cacheinit,
                 ctx.obj['cuda_device'])


# RESULTS
@cli.command()
@click.pass_context
@click.argument('model_source', type=click.Path(exists=True, file_okay=True, dir_okay=False))
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
@click.option('--force_cacheinit', is_flag=True, default=False)
@click.option('--output_path', type=click.Path(), default=defaults['results_output_path'])
@click.option('--store_wavs', is_flag=True, default=False)
def results(ctx,
            model_source,
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
            force_cacheinit,
            output_path,
            store_wavs):
    noise_snrs_list = [int(n) for n in noise_snrs.split(',')]
    script_results(model_source,
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
                   force_cacheinit,
                   output_path,
                   store_wavs,
                   ctx.obj['cuda_device'])


# DENOISE
@cli.command()
@click.pass_context
@click.argument('model_source', type=click.Path(exists=True, file_okay=True, dir_okay=False))
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
            model_source,
            input_path,
            output_path,
            sr,
            n_fft,
            hop_length,
            win_length,
            frag_hop_length,
            frag_win_length,
            batch_size):
    script_denoise(model_source,
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
