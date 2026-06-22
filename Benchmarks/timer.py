"""
BoloGAN generation timing benchmark.

Usage:
    python timer.py <path to model directory> --n <number of events> --k <number of trials>
"""

import argparse
import json
import time
import numpy as np
import tensorflow as tf
import gc


import sys
TRAINPATH = "/afs/cern.ch/user/h/halhadda/BoloGAN-FCC/training"
sys.path.insert(0, TRAINPATH)

def gpu_sync():
    """Synchronize GPU operations to ensure accurate timing."""
    try:
        for dev in tf.config.list_physical_devices('GPU'):
            with tf.device(dev.name.replace('physical_device:', '')):
                _ = tf.constant(0).numpy()
    except Exception as e:
        raise RuntimeError(f"gpu_sync failed: {e}") from e

def configure_gpu():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

def LoadModel(path):
    configPath = path + "train/config.json"
    config = json.load(open(configPath))

    from model import WGANGP
    wgan = WGANGP(job_config=config['job_config'], hp_config=config['hp_config'], logger="notebook_session") # change logger back to __file__ for .py files
    import glob, os
    ckpts = sorted(
        int(os.path.basename(p).split('-')[1].split('.')[0])
        for p in glob.glob(os.path.join(path, 'selected/model-*.index'))
    )
    if not ckpts:
        raise RuntimeError(f'No checkpoints found in {path}')
    model_i = ckpts[-1]
    print(f'Using checkpoint: {model_i}')

    return wgan, model_i, config

def find_max_batch(wgan, model_i, high=500000, low = 1000):
    # NOT WORKING PROPERLY YET
    gpus = tf.config.list_physical_devices('GPU')
    HIGH = high
    if not gpus:
        print('No GPU found. Using CPU for benchmarking.')
        return high # it will run siquenally on CPU, so no need to find max batch size.
    
    # First check that low itself works
    try:
        _ = wgan.predict(model_i=model_i, labels=np.ones((low, 1), dtype=np.float32))
        gpu_sync()
    except Exception:
        raise RuntimeError(f'GPU OOM even at minimum batch size ({low}). '
                           f'Try reducing --low or freeing GPU memory.')
    
    best = low
    print(f'Finding max batch size between {low} and {high}...')
    while low < high:
        mid = (low + high) // 2
        print(f'Testing batch size: {mid}...')
        try:
            _ = wgan.predict(model_i=model_i, labels=np.ones((mid, 1), dtype=np.float32))
            gpu_sync()
            best = mid
            low = mid + 1
        except Exception:
            high = max(low, mid - 50000) # step down by 50k to speed up search, can be adjusted if needed

    if best < HIGH:
        print(f'Warning: GPU OOM at batch size {HIGH}. Benchmarking will use batch size {int(best * 0.7)}.')
    return int(best * 0.7)  # safety margin

def benchmark(wgan, model_i, nEvents, nTrials):
    # NOTE: Can not handle OOM 
    times = np.zeros(nTrials)
    # warm up
    labels = np.random.uniform(0, 1, size=(nEvents, 1)).astype(np.float32)
    _ = wgan.predict(model_i=model_i, labels=labels)
    gpu_sync()

    for i in range(nTrials):
        labels = np.random.uniform(0, 1, size=(nEvents, 1)).astype(np.float32)
        t0 = time.perf_counter()
        E_vox = wgan.predict(model_i=model_i, labels=labels)
        gpu_sync()
        t = time.perf_counter() - t0
        times[i] = t
        print(f'  Trial {i+1}: {t:.4f} s  ({t/nEvents*1e3:.3f} ms/event)')

    return times, nEvents                           


def main():
    configure_gpu()
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the model directory')
    parser.add_argument('--n',     type=int, default=5000, help='Number of events to generate per trial')
    parser.add_argument('--k',     type=int, default=5, help='Number of trials')
    args = parser.parse_args()


    wgan, model_i, config = LoadModel(args.path)

    # turn into an np array and plot timing distribution accross trials.
    times, nEvents = benchmark(wgan, model_i, args.n, args.k)

    print(f'\n{nEvents} events | {args.k} trials')
    print(f'  mean : {times.mean():.4f} s  ({times.mean()/nEvents*1e3:.3f} ms/event)')
    print(f'  std  : {times.std():.4f} s')
    print(f'  ev/s : {nEvents/times.mean():.1f}')

    # import matplotlib.pyplot as plt
    # particle = config['job_config']['particle']
    # plt.hist(times*1000, bins=50, alpha=0.7, histtype='step')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Frequency')
    # plt.title('Timing Distribution for Generation of {} Events ({})'.format(nEvents, particle))
    # plt.savefig('timing_distribution_{}_{}.png'.format(particle, nEvents))
    np.savetxt('/afs/cern.ch/user/h/halhadda/BoloGAN-FCC/Benchmarks/timing_results_{}_{}.csv'.format(config['job_config']['particle'], nEvents), times, header='Time (s)')
if __name__ == '__main__':
    main()