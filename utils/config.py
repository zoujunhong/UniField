from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    spec = importlib.util.spec_from_file_location(f"_whole_field_config_{config_path.stem}", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"Config file must export CONFIG: {config_path}")
    config = copy.deepcopy(module.CONFIG)
    if not isinstance(config, dict):
        raise TypeError(f"CONFIG must be a dict: {config_path}")
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def load_referenced_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return load_config(resolve_path(config[key], config.get("_config_dir")))


def _apply_dataset_kwargs(dataset_config: dict[str, Any], values: dict[str, Any]) -> None:
    if not values:
        return
    if not isinstance(values, dict):
        raise TypeError("dataset_kwargs must be a dict.")

    names = dataset_config.get("name")
    values = copy.deepcopy(values)
    selected_domains = values.pop("selected_domains", None)
    if selected_domains is not None:
        dataset_config["selected_domains"] = selected_domains
    if isinstance(names, str):
        kwargs = dict(dataset_config.get("kwargs", {}))
        common_values = values.pop("common_kwargs", {})
        if common_values:
            if not isinstance(common_values, dict):
                raise TypeError("dataset_kwargs['common_kwargs'] must be a dict.")
            kwargs.update(common_values)
        nested_values = values.pop("kwargs", {})
        if nested_values:
            if not isinstance(nested_values, dict):
                raise TypeError("dataset_kwargs['kwargs'] must be a dict.")
            if names in nested_values and isinstance(nested_values[names], dict):
                kwargs.update(nested_values[names])
            else:
                kwargs.update(nested_values)
        dataset_values = values.pop(names, {})
        if dataset_values:
            if not isinstance(dataset_values, dict):
                raise TypeError(f"dataset_kwargs[{names!r}] must be a dict.")
            kwargs.update(dataset_values)
        kwargs.update(values)
        dataset_config["kwargs"] = kwargs
        return

    if isinstance(names, (list, tuple)):
        dataset_names = set(names)
        common_patch = {}
        specific_patch = {}
        explicit_common = values.pop("common_kwargs", {})
        if explicit_common:
            if not isinstance(explicit_common, dict):
                raise TypeError("dataset_kwargs['common_kwargs'] must be a dict.")
            common_patch.update(explicit_common)
        explicit_kwargs = values.pop("kwargs", {})
        if explicit_kwargs:
            if not isinstance(explicit_kwargs, dict):
                raise TypeError("dataset_kwargs['kwargs'] must be a dict.")
            values.update(explicit_kwargs)
        for key, value in values.items():
            if key in dataset_names and isinstance(value, dict):
                specific_patch[key] = value
            else:
                common_patch[key] = value

        if common_patch:
            common_kwargs = dict(dataset_config.get("common_kwargs", {}))
            common_kwargs.update(common_patch)
            dataset_config["common_kwargs"] = common_kwargs
        if specific_patch:
            kwargs = copy.deepcopy(dataset_config.get("kwargs", {}))
            for name, patch in specific_patch.items():
                dataset_kwargs = dict(kwargs.get(name, {}))
                dataset_kwargs.update(patch)
                kwargs[name] = dataset_kwargs
            dataset_config["kwargs"] = kwargs
        return

    raise TypeError("dataset config 'name' must be a string or a list of strings.")


def dataset_config_for_phase(
    config: dict[str, Any],
    phase: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_config = copy.deepcopy(config["dataset"])
    top_level_values = config.get("dataset_kwargs", {})
    if top_level_values:
        if not isinstance(top_level_values, dict):
            raise TypeError("Top-level dataset_kwargs must be a dict.")
        if phase in top_level_values:
            _apply_dataset_kwargs(dataset_config, top_level_values[phase])
        elif {"train", "validation", "test", "visualization"}.isdisjoint(top_level_values):
            _apply_dataset_kwargs(dataset_config, top_level_values)

    phase_config = config.get(phase, {})
    if phase_config:
        if not isinstance(phase_config, dict):
            raise TypeError(f"Config section {phase!r} must be a dict.")
        _apply_dataset_kwargs(dataset_config, phase_config.get("dataset_kwargs", {}))
    if extra_kwargs:
        _apply_dataset_kwargs(dataset_config, extra_kwargs)
    return dataset_config


def apply_runtime_seed(config: dict[str, Any], seed: int | None = None) -> int:
    """Set a runtime seed and expand `{seed}` in run-scoped output/checkpoint paths."""

    runtime = config.setdefault("runtime", {})
    resolved_seed = int(runtime.get("seed", 200099) if seed is None else seed)
    runtime["seed"] = resolved_seed
    for section_name in ("train", "test", "visualization"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in ("saveroot", "checkpoint", "output_dir", "metrics_output"):
            value = section.get(key)
            if isinstance(value, str) and "{seed}" in value:
                section[key] = value.format(seed=resolved_seed)
    return resolved_seed


def require_keys(config: dict[str, Any], keys: tuple[str, ...] | list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")


def _resolve_dataset_option_path(config: dict) -> str:
    dataset_name = config.get("dataset")
    dataset_option = config.get("dataset_option")
    if not dataset_name or not dataset_option:
        raise KeyError("Experiment config must define dataset and dataset_option.")
    return str(resolve_path(f"config/dataset/{dataset_name}/{dataset_option}.py", config.get("_config_dir")))


def _resolve_model_size_path(config: dict) -> str:
    model_name = config.get("model")
    model_size = config.get("model_size")
    if not model_name or not model_size:
        raise KeyError("Experiment config must define model and model_size.")
    return str(resolve_path(f"config/model/{model_name}/{model_size}.py", config.get("_config_dir")))


def _task_items(task_config: dict) -> list[dict]:
    items = []
    for key in ("losses", "eval_fields"):
        values = task_config.get(key, [])
        if values:
            items.extend(values)
    return items


def _max_pred_slice_end(items: list[dict], domains: set[str]) -> int:
    ends = []
    for item in items:
        if item.get("domain") in domains and "pred_slice" in item:
            ends.append(int(item["pred_slice"][1]))
    return max(ends) if ends else 0


def _check_task_model_compatibility(config: dict) -> None:
    task_config = config.get("task", {})
    model_config = config.get("model", {})
    items = _task_items(task_config)
    coefficient_domains = {"coefficients", "coefficient_integral_consistency"}
    needs_coefficient_head = any(item.get("domain") in coefficient_domains for item in items)
    coefficient_head_kwargs = model_config.get("coefficient_head_kwargs")

    if needs_coefficient_head:
        if not task_config.get("coefficient_target_list"):
            raise ValueError("Task uses coefficient losses/eval fields but coefficient_target_list is empty.")
        if not coefficient_head_kwargs:
            raise ValueError(
                "Task uses coefficient losses/eval fields but model has no coefficient_head_kwargs. "
                "Use a coefficient model size such as MediumCoeff or remove coefficient items from the task."
            )
    elif "coefficient_head_kwargs" in model_config:
        model_config.pop("coefficient_head_kwargs", None)
        print(
            "Config check: removed coefficient_head_kwargs because task has no coefficient losses/eval fields.",
            flush=True,
        )

    native_model_kwargs = model_config.get("native_model_kwargs", model_config.get("model_kwargs", {}))
    query_out_channels = int(
        model_config.get("decoder_kwargs", {}).get(
            "out_channels",
            native_model_kwargs.get("output_channels", model_config.get("output_channels", 0)),
        )
    )
    query_required = _max_pred_slice_end(items, {"surface", "volume"})
    if query_out_channels and query_required > query_out_channels:
        raise ValueError(
            f"Task query pred_slice requires {query_required} output channels, "
            f"but model decoder out_channels={query_out_channels}."
        )
    if needs_coefficient_head:
        coeff_out_channels = int(coefficient_head_kwargs.get("out_channels", 0))
        coeff_required = _max_pred_slice_end(items, coefficient_domains)
        if coeff_out_channels and coeff_required > coeff_out_channels:
            raise ValueError(
                f"Task coefficient pred_slice requires {coeff_required} output channels, "
                f"but coefficient head out_channels={coeff_out_channels}."
            )


def load_main_config(path: str) -> dict:
    config = load_config(path)
    if not isinstance(config.get("dataset"), dict):
        config["_dataset_name"] = config.get("dataset")
        config["_dataset_option"] = config.get("dataset_option")
        config["dataset"] = load_config(_resolve_dataset_option_path(config))
    if not isinstance(config.get("model"), dict):
        config["_model_name"] = config.get("model")
        config["_model_size"] = config.get("model_size")
        config["model"] = load_config(_resolve_model_size_path(config))
    _check_task_model_compatibility(config)
    return config



def configure_training_run(config: dict[str, Any], *, for_training: bool = True) -> dict[str, Any]:
    """Enforce the single config entry point and derive canonical run paths."""
    import re

    disabled = config.get('migration', {}).get('training_disabled_reason')
    if for_training and disabled:
        raise ValueError(f'Historical run cannot be resumed by the current training engine: {disabled}')
    rebuild = config.get('migration', {}).get('require_cache_rebuild')
    if not for_training and rebuild:
        raise ValueError(
            f'Historical evaluation requires rebuilding its retired cache before use: {rebuild}. '
            'The compact pool is not a substitute for the original sampling protocol.')
    if not for_training and config.get('migration', {}).get('evaluation_requires_legacy_source'):
        raise ValueError('This historical checkpoint requires its archived exact model/evaluation source; the current model configuration cannot reproduce it.')
    source = Path(config['_config_path']).resolve()
    if for_training and source.is_relative_to(PROJECT_ROOT) and 'temp' in source.relative_to(PROJECT_ROOT).parts:
        raise ValueError('Training configs belong in config/experiment or log/<experiment>/<variant>, never temp/.')
    experiment = config.setdefault('experiment', {})
    name = str(experiment.get('name', source.stem))
    if not re.fullmatch(r'[A-Za-z0-9+]+_[A-Za-z0-9]+_[A-Za-z0-9]+_Surface[A-Za-z0-9]+_Volume[A-Za-z0-9]+', name):
        raise ValueError('experiment.name must follow Dataset_Model_Size_SurfaceFields_VolumeFields.')
    variant = str(experiment.get('variant', 'default'))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*', variant):
        raise ValueError('experiment.variant must be a safe directory name such as Loss_RelL2_GPU_3.')
    experiment.update(name=name, variant=variant)
    train = config['train']
    if int(train.get('total_steps', 0)) <= 0:
        raise ValueError('train.total_steps must be positive; epoch-based training budgets are no longer supported.')
    for obsolete in ('total_epoch', 'save_step', 'resume_epoch', 'resume_step'):
        if train.get(obsolete, 0):
            raise ValueError(f'train.{obsolete} is obsolete; use total_steps and resume_checkpoint with full state.')
        train.pop(obsolete, None)
    validation = train.setdefault('validation', {})
    validation.setdefault('enabled', True)
    # Resolve after variant overrides, so changing the budget also updates the default.
    if validation.get('interval') is None:
        validation['interval'] = max(10000, (int(train['total_steps']) + 9) // 10)
    validation.setdefault('query_points', 10000)
    validation.setdefault('batch_size', 1)
    validation.setdefault('workers', 2)
    if int(validation['interval']) <= 0:
        raise ValueError('train.validation.interval must be positive optimizer steps.')
    if int(validation['query_points']) != 10000:
        raise ValueError('Validation uses one group of 10000 query points per supervised domain.')
    train.setdefault('checkpoint_interval_steps', int(validation['interval']))
    if int(train['checkpoint_interval_steps']) <= 0:
        raise ValueError('train.checkpoint_interval_steps must be positive.')
    run_dir = PROJECT_ROOT / 'log' / name / variant
    saveroot = Path.home() / 'zoujunhong/model/savemodel/field' / name / variant
    train['saveroot'] = str(saveroot)
    train['log_dir'] = str(run_dir)
    train['log_path'] = str(run_dir / 'train.log')
    if train.get('engine') == 'dsapt':
        from decimal import Decimal
        recipe = config['dsapt']
        for key, source_key in (('total_steps', 'total_steps'), ('optimizer', 'optimizer'),
                                ('peak_lr', 'lr'), ('warmup_steps', 'warmup_iters'),
                                ('weight_decay', 'weight_decay'), ('clip_grad_norm', 'clip_grad')):
            if source_key in train:
                recipe[key] = train[source_key]
        if 'lr' in train and 'final_lr_ratio' in train:
            recipe['end_lr'] = float(Decimal(str(train['lr'])) * Decimal(str(train['final_lr_ratio'])))
        recipe['seed'] = int(config['runtime'].get('seed', recipe.get('seed', 200099)))
        recipe['world_size'] = int(config['runtime'].get('distributed', {}).get('gpu_num', 1))
        recipe['global_batch'] = recipe['world_size'] * int(train.get('batch_size', 1))
        if int(train.get('batch_size', 1)) != 1:
            raise ValueError('DSAPT currently supports one geometry per GPU; set train.batch_size=1.')
        field_settings = {}
        for item in config['task'].get('losses', []):
            if item['domain'] in ('surface', 'volume'):
                key = item['domain'] + '_' + item['field'].lower()
                field_settings[key] = dict(loss_type=item.get('loss_type', 'mse'), weight=float(item.get('weight', 1.0)))
        recipe['field_loss_config'] = field_settings
        loss_types = {value['loss_type'] for value in field_settings.values()}
        if len(loss_types) == 1:
            recipe['loss_type'] = next(iter(loss_types))
    test = config.setdefault('test', {})
    if not test.get('metrics_output'):
        test['metrics_output'] = str(run_dir / 'evaluation/summary.json')
    return config


def prepare_run_artifacts(config: dict[str, Any]) -> None:
    """Back up the executable run config, concise change note and resolved inputs."""
    import hashlib
    import json
    import os
    import pprint

    experiment = config['experiment']
    source = Path(config['_config_path'])
    base = PROJECT_ROOT / 'config/experiment' / f"{experiment['name']}.py"
    if source == base:
        config_text = (
            'from copy import deepcopy\n'
            f"from config.experiment.{experiment['name']} import CONFIG as BASE_CONFIG\n\n"
            'CONFIG = deepcopy(BASE_CONFIG)\n'
        )
    else:
        config_text = source.read_text(encoding='utf-8')
    description = str(experiment.get('description', '')).strip()
    if not description:
        description = '无；直接使用默认配置。' if experiment['variant'] == 'default' else f"调整：{experiment['variant']}。"
    resolved = copy.deepcopy(config)
    for key in ('_config_path', '_config_dir'):
        resolved.pop(key, None)
    resolved_text = '# Fully resolved inputs archived at launch.\nCONFIG = ' + pprint.pformat(resolved, sort_dicts=False, width=110) + '\n'
    manifest = {
        'experiment': experiment['name'], 'variant': experiment['variant'],
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'base': str(base),
    }
    if base.is_file():
        manifest['base_sha256'] = hashlib.sha256(base.read_bytes()).hexdigest()
    files = {'config.py': config_text, 'README.md': description + '\n',
             'resolved_config.py': resolved_text, 'config_manifest.json': json.dumps(manifest, indent=2) + '\n'}
    for destination in (Path(config['train']['log_dir']), Path(config['train']['saveroot'])):
        destination.mkdir(parents=True, exist_ok=True)
        previous = {name: (destination / name).read_text(encoding='utf-8')
                    for name in files if (destination / name).is_file()}
        if previous and any(previous[name] != files[name] for name in previous):
            # Older runs may have config/README but no resolved snapshot yet.
            fingerprint = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()[:16]
            archive = destination / 'config_history' / fingerprint
            archive.mkdir(parents=True, exist_ok=True)
            for name, content in previous.items():
                (archive / name).write_text(content, encoding='utf-8')
        for name, content in files.items():
            if previous.get(name) == content:
                continue
            temporary = destination / f'.{name}.{os.getpid()}.tmp'
            with temporary.open('w', encoding='utf-8') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination / name)
