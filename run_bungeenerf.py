import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


SCENES = dict(zip(
    ('amsterdam', 'barcelona', 'bilbao', 'chicago', 'hollywood', 'pompidou', 'quebec', 'rome'),
    (21, 20, 17, 20, 16, 21, 20, 20),
))
ROOT = Path(__file__).resolve().parent


def command(data, output, gpu, port):
    return [sys.executable, str(ROOT / 'train.py'), '--eval',
            '-s', str(data), '-m', str(output), '-i', 'images', '-r', '-1',
            '--gpu', gpu, '--port', str(port), '--ratio', '1', '--appearance_dim', '0',
            '--fork', '2', '--base_layer', '10', '--visible_threshold', '0.0',
            '--dist2level', 'round', '--update_ratio', '0.2', '--progressive',
            '--levels', '-1', '--init_level', '-1', '--dist_ratio', '0.99',
            '--extra_ratio', '0.25', '--extra_up', '0.01', '--iterations', '40000',
            '--test_iterations', '40000', '--save_iterations', '40000',
            '--checkpoint_iterations', '999999', '--update_until', '25000',
            '--position_lr_max_steps', '40000', '--offset_lr_max_steps', '40000',
            '--mlp_opacity_lr_max_steps', '40000', '--mlp_cov_lr_max_steps', '40000',
            '--mlp_color_lr_max_steps', '40000', '--mlp_featurebank_lr_max_steps', '40000',
            '--appearance_lr_max_steps', '40000', '--kernel_size', '0.1',
            '--filter_3D_update_interval', '1000', '--anchor_contribution_pruning',
            '--contribution_reset_interval', '10000']


def verify(output, expected):
    for relative in ('results.json', 'per_view.json', 'point_cloud/iteration_40000/point_cloud.ply'):
        path = output / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Missing output: ' + str(path))
    metrics = json.loads((output / 'results.json').read_text())['ours_40000']
    if not all(math.isfinite(metrics[key]) for key in ('PSNR', 'SSIM', 'LPIPS')):
        raise RuntimeError('Non-finite final metrics')
    json.loads((output / 'per_view.json').read_text())
    for folder in ('renders', 'gt'):
        count = sum(p.is_file() for p in (output / 'test/ours_40000' / folder).iterdir())
        if count != expected:
            raise RuntimeError('{}: expected {}, found {}'.format(folder, expected, count))


def main():
    parser = argparse.ArgumentParser(description='BungeeNeRF 40K release configuration')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scene', choices=list(SCENES) + ['all'], default='all')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.gpu == '-1' or ',' in args.gpu:
        parser.error('--gpu must select one explicit GPU index or UUID')
    data_root, output_root = args.data_root.resolve(), args.output_root.resolve()
    for path in (data_root, output_root):
        if path == ROOT or ROOT in path.parents:
            parser.error('Data and output must be outside this repository')
    scenes = list(SCENES) if args.scene == 'all' else [args.scene]
    for scene in scenes:
        data, output = data_root / scene, output_root / scene
        if output.exists():
            parser.error('Refusing to overwrite ' + str(output))
        if not args.dry_run:
            if not (data / 'images').is_dir() or not (data / 'sparse/0').is_dir():
                parser.error('Missing images/ or sparse/0/ in ' + str(data))
    for scene in scenes:
        output = output_root / scene
        cmd = command(data_root / scene, output, args.gpu, args.port)
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu)
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
            verify(output, SCENES[scene])
            print('Verified: ' + scene, flush=True)


if __name__ == '__main__':
    main()
