"""Spatial GAN changes architecture only; preserve every original training setting."""
import copy
from pathlib import Path

import pytest
import yaml


CONFIGS = Path(__file__).resolve().parents[1] / 'z-image/configs'
BASE = 'zimage_2step_1024_dlr1e-4'
VARIANTS = [('', [1.0, 1.0]), ('_stepw0.25', [0.25, 1.0])]
QWEN_TEXT_REPO = 'jiachengcui888/zimage-ref-qwen-text'
GLOBAL_ENCODERS = ['dinov3_l', 'siglip2', 'aimv2_huge']


def _read(variant, spatial):
    suffix = 'spatial_patchgan_h20_fa3' if spatial else 'qwen_h20_fa3'
    with (CONFIGS / f'{BASE}{variant}_{suffix}.yaml').open() as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def _unchanged_settings(config):
    value = copy.deepcopy(config)
    # Exactly these environment/experiment paths and newly authorized architecture
    # knobs may differ. In particular, do not discard all zimage/memory/gan blocks.
    for key in ('exp_name', 'output_dir', 'reference_root', 'spatial_gan'):
        value.pop(key, None)
    value['zimage'].pop('checkpoint_group_size', None)
    return value


def _differences(expected, actual, path=''):
    """Show precise nested keys if a future edit accidentally changes training behavior."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            child = f'{path}.{key}' if path else key
            if key not in expected or key not in actual:
                result.append(child)
            else:
                result.extend(_differences(expected[key], actual[key], child))
        return result
    return [] if expected == actual else [path]


@pytest.mark.parametrize('variant,step_weights', VARIANTS)
def test_only_paths_checkpoint_group_and_new_spatial_block_change(variant, step_weights):
    original, spatial = _read(variant, False), _read(variant, True)
    expected, actual = _unchanged_settings(original), _unchanged_settings(spatial)
    assert actual == expected, f'Unauthorized setting changes: {_differences(expected, actual)}'
    assert spatial['zimage']['checkpoint_group_size'] == 2
    assert spatial['spatial_gan']['enabled'] is True
    assert spatial['exp_name'] != original['exp_name'] and 'spatial-patchgan' in spatial['exp_name']
    for key in ('output_dir', 'reference_root'):
        assert isinstance(spatial[key], str) and Path(spatial[key]).is_absolute()
    assert Path(spatial['reference_root']).name == 'reference_store_qwen'
    assert spatial['qwen_text_repo'] == original['qwen_text_repo'] == QWEN_TEXT_REPO
    assert spatial['encoders'] == original['encoders'] == GLOBAL_ENCODERS
    assert spatial['student']['step_weights'] == original['student']['step_weights'] == step_weights


@pytest.mark.parametrize('variant,_weights', VARIANTS)
def test_lr_gan_window_batch_and_resume_contract_remain_pinned(variant, _weights):
    """Name the user-visible invariants, even if both compared files are edited together."""
    for spatial in (False, True):
        config = _read(variant, spatial)
        assert config['lr'] == pytest.approx(2.83e-6)
        assert config['lr_schedule'] == {'name': 'cosine', 'warmup_steps': 100,
                                         'min_lr_ratio': 0.1, 'total_steps': None}
        assert config['gan']['lr'] == pytest.approx(1.0e-4)
        assert config['gan']['weight'] == 1.5 and config['gan']['adaptive'] is True
        assert config['gan']['g_start_step'] == 50
        assert config['gan']['betas'] == [0.0, 0.99]
        assert config['active_global_batch'] == config['window']['stride'] == 32
        assert config['window']['size'] == 128
        assert config['memory']['micro_batch'] == 2
        assert config['memory']['shard'] == 'fsdp' and config['memory']['grad_reduce'] == 'none'
        assert config['memory']['optimizer'] == 'adamw8bit'
        assert config['zimage']['attention_backend'] == '_flash_3_varlen_hub'
        assert config['zimage']['feature_pool'] == 'auto' and config['zimage']['feature_quantize'] is True
        assert config['joint'] is True and config['img_size'] == 1024
        assert config['steps'] == config['save_freq'] == 500
        assert config['save_resume'] is True and config['resume_every'] == 50
        assert config['save_cache'] is False and config['save_optimizer'] is False


def test_spatial_a_b_still_differ_only_in_first_step_weight_and_experiment_name():
    a, b = _read('', True), _read('_stepw0.25', True)
    assert a['student']['step_weights'] == [1.0, 1.0]
    assert b['student']['step_weights'] == [0.25, 1.0]
    b['exp_name'] = a['exp_name']
    b['student']['step_weights'] = a['student']['step_weights']
    assert b == a
