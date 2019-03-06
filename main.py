import click
from scripts.train import train as script_train
from scripts.results import results as script_results
from scripts.encode import encode as script_encode
from scripts.decode import decode as script_decode


defaults = {
    'rir_path': '/data/riccardo_datasets/rirs/',
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'frag_hop_length': 128,
    'frag_win_length': 512,
    'batch_size': 4,
    'epochs': 20,
    'model_path': None,
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
@click.option('--rir_path', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=defaults['rir_path'])
@click.option('--n_fft', type=int, default=defaults['n_fft'])
@click.option('--hop_length', type=int, default=defaults['hop_length'])
@click.option('--win_length', type=int, default=defaults['win_length'])
@click.option('--frag_hop_length', type=int, default=defaults['frag_hop_length'])
@click.option('--frag_win_length', type=int, default=defaults['frag_win_length'])
@click.option('--batch_size', type=int, default=defaults['batch_size'])
@click.option('--epochs', type=int, default=defaults['epochs'])
@click.option('--model_path', type=click.Path(), default=None)
@click.option('--history_path', type=click.Path(), default=None)
def train(ctx, 
          model_name, 
          dataset_path, 
          rir_path, 
          n_fft, 
          hop_length,
          win_length,
          frag_hop_length,
          frag_win_length,
          batch_size,
          epochs,
          model_path,
          history_path):
    script_train(model_name,
                 dataset_path,
                 rir_path,
                 n_fft,
                 hop_length,
                 win_length,
                 frag_hop_length,
                 frag_win_length,
                 batch_size,
                 epochs,
                 model_path,
                 history_path,
                 ctx.obj['cuda_device'])


# RESULTS
@cli.command()
@click.pass_context
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--rows', type=int, default=defaults['rows'])
@click.option('--cols', type=int, default=defaults['cols'])
@click.option('--channels', type=int, default=defaults['channels'])
@click.option('--epochs', type=int, default=defaults['epochs'])
@click.option('--batch_size', type=int, default=defaults['batch_size'])
@click.option('--history_path', type=click.Path(), default=None)
def results(ctx,
            model_path,
            dataset_path,
            rows,
            cols,
            channels,
            epochs,
            batch_size,
            history_path):
    script_results(model_path,
                   dataset_path,
                   rows,
                   cols,
                   channels,
                   epochs,
                   batch_size,
                   history_path,
                   ctx.obj['cuda_device'])


if __name__ == "__main__":
    cli(obj={})
