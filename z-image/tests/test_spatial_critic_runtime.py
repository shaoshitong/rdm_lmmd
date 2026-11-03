"""CPU mixed-runtime tests using the real legacy critic and tiny spatial maps."""
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch
from torch.nn import functional as F

# Avoid optional package __init__ imports; all tested implementation files are
# loaded unchanged under a private package, including the real legacy builder.
ROOT = Path(__file__).resolve().parents[2]
for name, directory in [('runtime_fixture', ROOT / 'rdm'),
                        ('runtime_fixture.sw_lmmd', ROOT / 'rdm/sw_lmmd'),
                        ('runtime_fixture.utils', ROOT / 'rdm/utils')]:
    package = types.ModuleType(name); package.__path__ = [str(directory)]
    sys.modules[name] = package
module = importlib.import_module('runtime_fixture.sw_lmmd.spatial_critic_runtime')
GANConfig = importlib.import_module('runtime_fixture.sw_lmmd.config').GANConfig
SpatialGANRuntime = module.SpatialGANRuntime
SpatialBranchSpec = importlib.import_module('runtime_fixture.sw_lmmd.spatial_adversarial').SpatialBranchSpec


class FakeStore:
    num_rows = 12
    dims = {'dinov3_l': 5, 'siglip2': 7, 'aimv2_huge': 6}

    def __init__(self):
        generator = torch.Generator().manual_seed(77)
        self.text = F.normalize(torch.randn(12, 2560, generator=generator), dim=-1)
        self.images = {name: torch.randn(12, dim, generator=generator) for name, dim in self.dims.items()}

    def dim(self, name): return self.dims[name]
    def reference_image_features(self, name, rows): return self.images[name][rows]
    def text_features(self, rows): return self.text[rows]


class SpatialRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(123)
        self.store = FakeStore()
        self.base_names = ['dinov3_l', 'siglip2', 'aimv2_huge']
        self.cfg = GANConfig(seed=11, hidden=16, depth=1, stats_rows=8, spectral_norm=True)
        self.specs = {'dinov3_l': SpatialBranchSpec(3, 8, 1),
                      'qwen3vl_4b': SpatialBranchSpec(4, 16, 2)}

    def runtime(self, micro, *, seed=19):
        return SpatialGANRuntime(self.cfg, self.store, self.base_names, 'cpu', torch.float64,
            critic_micro_batch=micro, step_seed=seed, spatial_branch_specs=self.specs,
            spatial_options={'heads': 2})

    def inputs(self, requires_grad=False):
        fake = {'dinov3_l': torch.randn(5, 3, 8, 8, dtype=torch.float64, requires_grad=requires_grad),
                'qwen3vl_4b': torch.randn(5, 4, 16, 16, dtype=torch.float64, requires_grad=requires_grad)}
        real = {name: torch.randn_like(value) for name, value in fake.items()}
        fake_g = {'aimv2_huge': torch.randn(5, 11, dtype=torch.float64, requires_grad=requires_grad)}
        real_g = {'aimv2_huge': torch.randn(5, 11, dtype=torch.float64)}
        tau = self.store.text[:5].double().detach().requires_grad_(requires_grad)
        return fake, real, fake_g, real_g, tau

    def test_aim_initialization_stats_and_rng_are_legacy_identical(self):
        before = torch.get_rng_state().clone(); runtime = self.runtime(2)
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        cfg = copy.deepcopy(self.cfg); cfg.seed = 19
        old = module.build_critic(cfg, self.store, self.base_names, device='cpu', dtype=torch.float64)
        actual = runtime.critic['global']
        self.assertEqual(actual.names, ['aimv2_huge'])
        for name, value in actual.heads['aimv2_huge'].state_dict().items():
            torch.testing.assert_close(value, old.critic.heads['aimv2_huge'].state_dict()[name], atol=0, rtol=0)
        for name, value in actual.image_norm['aimv2_huge'].state_dict().items():
            torch.testing.assert_close(value, old.critic.image_norm['aimv2_huge'].state_dict()[name], atol=0, rtol=0)
        for name, value in actual.text_norm.state_dict().items():
            torch.testing.assert_close(value, old.critic.text_norm.state_dict()[name], atol=0, rtol=0)

    def test_one_vs_uneven_chunks_parameter_update_and_sn_buffers_match(self):
        whole, chunks = self.runtime(5), self.runtime(2)
        inputs = self.inputs()
        full_logs, chunk_logs = whole.update(*inputs), chunks.update(*inputs)
        self.assertEqual(whole.updates, 1); self.assertEqual(chunks.updates, 1)
        for name, value in whole.critic.state_dict().items():
            torch.testing.assert_close(value, chunks.critic.state_dict()[name], atol=2e-9, rtol=1e-8)
        for name in full_logs:
            if isinstance(full_logs[name], bool): self.assertEqual(full_logs[name], chunk_logs[name])
            else: self.assertAlmostEqual(full_logs[name], chunk_logs[name], places=10)
        self.assertAlmostEqual(full_logs['gan/d_loss'], sum(full_logs[f'gan/{n}/d_loss']
                                                         for n in whole.names) / 3, places=12)

    def test_spectral_norm_training_forward_occurs_exactly_once_per_update(self):
        runtime = self.runtime(2); counts = {}; handles = []
        for name, child in runtime.critic.named_modules():
            if child.__class__.__name__ == '_SpectralNorm':
                counts[name] = 0
                def count(layer, args, key=name):
                    if layer.training: counts[key] += 1
                handles.append(child.register_forward_pre_hook(count))
        runtime.update(*self.inputs())
        self.assertTrue(counts); self.assertEqual(set(counts.values()), {1})
        fake, _, fake_g, _, tau = self.inputs(requires_grad=True)
        runtime.generator_loss(fake, fake_g, tau)[0].backward()
        self.assertEqual(set(counts.values()), {1})
        for handle in handles: handle.remove()

    def test_g_uses_updated_disc_caption_concat_weight_one_third_and_image_prefix_only(self):
        runtime = self.runtime(2); runtime.update(*self.inputs())
        fake, _, fake_g, _, tau = self.inputs(requires_grad=True)
        before = {n: b.clone() for n, b in runtime.critic.named_buffers()}
        captured = []
        handle = runtime.critic['spatial'].heads['dinov3_l'].fusion.register_forward_pre_hook(
            lambda layer, args: captured.append(args[0].detach()))
        loss, logs = runtime.generator_loss(fake, fake_g, tau); handle.remove()
        self.assertAlmostEqual(loss.item(), sum(logs[f'gan/{n}/g_loss'] for n in runtime.names) / 3, places=12)
        self.assertEqual(captured[0].shape, (2, 64, 16 + 2560))
        torch.testing.assert_close(captured[0][..., 16:], tau[:2, None].expand(-1, 64, -1))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in runtime.critic.parameters()))
        for value in [*fake.values(), fake_g['aimv2_huge']]:
            self.assertTrue(torch.isfinite(value.grad).all()); self.assertGreater(value.grad.abs().sum().item(), 0)
        torch.testing.assert_close(fake_g['aimv2_huge'].grad[:, 6:], torch.zeros(5, 5, dtype=torch.float64))
        self.assertIsNone(tau.grad)
        for name, value in runtime.critic.named_buffers():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_d_detaches_all_inputs_and_warmup_uses_existing_cfg(self):
        runtime = self.runtime(2); values = self.inputs(requires_grad=True)
        runtime.update(*values)
        for tensor in [*values[0].values(), *values[2].values(), values[4]]:
            self.assertIsNone(tensor.grad)
        self.assertFalse(runtime.generator_active(49)); self.assertTrue(runtime.generator_active(50))
        self.assertEqual(runtime.optimizer.param_groups[0]['lr'], self.cfg.lr)
        self.assertEqual(runtime.optimizer.param_groups[0]['betas'], self.cfg.betas)

    def test_checkpoint_restores_optimizer_and_allows_different_microbatch(self):
        first, resumed = self.runtime(2), self.runtime(1)
        first.update(*self.inputs()); resumed.load_state_dict(copy.deepcopy(first.state_dict()))
        values = self.inputs(); first.update(*values); resumed.update(*values)
        self.assertEqual(resumed.updates, 2)
        for name, value in first.critic.state_dict().items():
            torch.testing.assert_close(value, resumed.critic.state_dict()[name], atol=2e-9, rtol=1e-8)
        wrong_seed = self.runtime(1, seed=20)
        with self.assertRaises(ValueError): wrong_seed.load_state_dict(first.state_dict())

    def test_r1_and_unsupported_global_critic_are_explicitly_rejected(self):
        self.cfg.r1_gamma = 1.0
        with self.assertRaisesRegex(ValueError, 'R1'): self.runtime(2)
        self.cfg.r1_gamma = 0.0; self.cfg.critic = 'block'
        with self.assertRaisesRegex(ValueError, 'row'): self.runtime(2)


if __name__ == '__main__':
    unittest.main()
