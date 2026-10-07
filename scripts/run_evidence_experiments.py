"""Generate reproducible pilot/formal commands; execute only with --execute.

Pilot trains on TRAIN and evaluates dev2000. It is not a no-training experiment.
All arms start fresh from the supplied per-seed S0, never from a pilot winner.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def build_commands(args):
    checkpoints = {}
    for spec in args.s0:
        seed, path = spec.split("=", 1)
        seed = int(seed)
        if seed in checkpoints:
            raise ValueError("duplicate S0 seed")
        checkpoints[seed] = path
    arms = ([('A-frozen', 'conditioned', 'full', False, 0.0),
             ('A-joint', 'conditioned', 'full', True, 0.0),
             ('A-first', 'conditioned', 'first_only', args.base == 'joint', 0.0)]
            if args.phase == 'pilot' else
            [('A', 'conditioned', 'full', args.base == 'joint', 0.0),
             ('B', 'conditioned', 'full', args.base == 'joint', args.evidence_weight),
             ('C', 'agnostic_matched', 'full', args.base == 'joint', args.evidence_weight)])
    if args.phase == 'pilot' and args.slotwise:
        arms.append(('A-slotwise', 'conditioned', 'slotwise', args.base == 'joint', 0.0))
    steps = args.steps or (250 if args.phase == 'pilot' else 3000)
    if steps < 1 or args.evidence_weight <= 0:
        raise ValueError("positive steps and formal evidence weight required")
    commands = []
    for seed, checkpoint in sorted(checkpoints.items()):
        for label, query, stage, joint, weight in arms:
            output = str(Path(args.out_dir)/args.phase/f'seed{seed}'/label)
            command = [sys.executable, 'src/train.py', '--preset', 'pisco_evidence_projector',
                       '--train_file', args.train_file, '--eval_files', 'dev='+args.dev_file,
                       '--cache_dir', args.cache_dir, '--generator_path', args.pisco_path,
                       '--s0_checkpoint', checkpoint, '--seed', str(seed), '--steps', str(steps),
                       '--projector_query_mode', query, '--evidence_stage', stage,
                       '--evidence_loss_weight', str(weight), '--evidence_max_len', str(args.evidence_max_len),
                       '--eval_max_samples', '2000', '--eval_every_samples', '2000',
                       '--eval_every', str(min(steps, 125 if args.phase == 'pilot' else 250)),
                       '--select_metric', 'f1', '--out_dir', output, '--tag', args.phase+'-'+label]
            if joint:
                command.append('--evidence_base_trainable')
            commands.append(dict(seed=seed, arm=label, command=command))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'formal'], default='pilot')
    parser.add_argument('--s0', action='append', required=True, metavar='SEED=CHECKPOINT')
    for flag in ('train_file', 'dev_file', 'cache_dir', 'pisco_path', 'out_dir'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--base', choices=['frozen', 'joint'], default='frozen')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--evidence_weight', type=float, default=0.1)
    parser.add_argument('--evidence_max_len', type=int, default=128)
    parser.add_argument('--slotwise', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    commands = build_commands(args)
    if args.execute:
        for path in [args.train_file, args.dev_file, *[s.split('=', 1)[1] for s in args.s0]]:
            if not Path(path).is_file():
                parser.error('missing input file: '+path)
        for item in commands:
            output = Path(item['command'][item['command'].index('--out_dir')+1])
            if output.exists() and any(output.iterdir()):
                parser.error('refusing to append to an existing run: '+str(output))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/(args.phase+'_commands.json')).write_text(json.dumps(commands, indent=2)+'\n')
    for item in commands:
        print(shlex.join(item['command']), flush=True)
        if args.execute:
            subprocess.run(item['command'], check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
