"""Run locking and exactly-once automatic evaluation after a completed budget."""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from utils.config import PROJECT_ROOT, load_config, prepare_run_artifacts
from utils.checkpoint import resolve_eval_checkpoint


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def run_training_pipeline(config, launch):
    train = config['train']
    run = Path(train['log_dir'])
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **kwargs):
            write_status(run / 'pipeline_status.json', dict(stage=stage, time=datetime.datetime.now().astimezone().isoformat(), **kwargs))
        previous = {}
        same_inputs = False
        if (run / 'status.json').exists() and (run / 'resolved_config.py').exists():
            previous = json.loads((run / 'status.json').read_text())
            archived = load_config(run / 'resolved_config.py')
            comparable = lambda values: {key: value for key, value in values.items() if key not in ('_config_path', '_config_dir')}
            same_inputs = comparable(archived) == comparable(config)
        prepare_run_artifacts(config)
        try:
            already_complete = (same_inputs and previous.get('status') == 'complete'
                                and previous.get('step') == int(train['total_steps'])
                                and (Path(train['saveroot']) / 'latest.pth').is_file())
            if not already_complete:
                status('training')
                launch(config)
            progress = json.loads((run / 'status.json').read_text())
            if progress.get('status') != 'complete':
                status('stopped', step=progress.get('step'))
                return
            if int(progress['step']) != int(train['total_steps']):
                raise RuntimeError('Completed status did not reach the configured training budget')
            auto_test = bool(config.get('test', {}).get('after_training', train.get('auto_test', config.get('dsapt', {}).get('auto_test', False))))
            if auto_test:
                if not (Path(train['saveroot']) / 'latest.pth').is_file():
                    raise FileNotFoundError('Final resumable checkpoint is missing')
                summary = (Path(config.get('test', {}).get('output_dir') or run / 'evaluation') / 'summary.json'
                           if train.get('engine') == 'dsapt' else Path(config['test']['metrics_output']))
                completed = run / 'automatic_test_complete.json'
                checkpoint = (config.get('test', {}).get('checkpoint') or str(Path(train['saveroot']) / 'latest.pth')
                              if train.get('engine') == 'dsapt' else resolve_eval_checkpoint(config, 'test'))
                if not checkpoint:
                    raise ValueError('Automatic evaluation requires trained weights; test.checkpoint=None is only for explicit smoke tests.')
                identity = dict(step=progress['step'], config=str(run / 'config.py'),
                                config_sha256=file_sha256(run / 'resolved_config.py'),
                                checkpoint_sha256=file_sha256(checkpoint))
                previous_test = json.loads(completed.read_text()) if completed.exists() else None
                if not (previous_test == identity and summary.exists()):
                    status('testing', step=progress['step'])
                    environment = os.environ.copy()
                    visible = environment.get('CUDA_VISIBLE_DEVICES')
                    if visible:
                        environment['CUDA_VISIBLE_DEVICES'] = visible.split(',')[0]
                    else:
                        environment['CUDA_VISIBLE_DEVICES'] = str(config['runtime'].get('distributed', {}).get('start_gpu', 0))
                    command = [sys.executable, '-u', str(PROJECT_ROOT / 'test.py'), '--config', str(run / 'resolved_config.py')]
                    with (run / 'test.stdout').open('a') as stdout, (run / 'test.stderr').open('a') as stderr:
                        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stdout, stderr=stderr, check=True)
                    if not summary.is_file():
                        raise RuntimeError(f'Evaluation exited without its required summary: {summary}')
                    write_status(completed, identity)
            status('complete', step=progress['step'])
        except BaseException as error:
            status('failed', error=repr(error))
            raise
