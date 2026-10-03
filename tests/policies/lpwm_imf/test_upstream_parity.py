"""Opt-in exact parity against a local checkout of the pinned LPWM source."""

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

import lerobot.policies.lpwm_imf._vendor as vendor
from lerobot.policies.lpwm_imf.configuration_lpwm_imf import LPWMIMFConfig
from lerobot.policies.lpwm_imf.modeling_lpwm_imf import LPWMVisualEncoder


def test_pinned_upstream_encoder_ast_and_numeric_parity(monkeypatch):
    checkout = os.environ.get("LPWM_UPSTREAM_DIR")
    if not checkout:
        pytest.skip("Set LPWM_UPSTREAM_DIR to a trusted local checkout of the pinned LPWM source.")
    upstream = Path(checkout)
    vendored = Path(vendor.__file__).parent

    def definitions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }

    for local, remote in (
        ("particle_encoder.py", "modules/modules.py"),
        ("vision_encoder.py", "modules/vision_modules.py"),
        ("utils.py", "utils/util_func.py"),
    ):
        original = definitions(upstream / remote)
        selected = definitions(vendored / local)
        assert selected
        for name, definition in selected.items():
            assert original[name] == definition, f"Upstream mismatch in {local}:{name}"

    # Preserve the upstream module names only for this test. The full original
    # util_func imports plotting packages; the AST-identical helper subset above
    # avoids requiring matplotlib just to run the visual encoder.
    for name in ("modules", "utils"):
        package = types.ModuleType(name)
        package.__path__ = [str(upstream / name)]
        monkeypatch.setitem(sys.modules, name, package)
    for name, path in (
        ("modules.vision_modules", upstream / "modules/vision_modules.py"),
        ("utils.util_func", vendored / "utils.py"),
        ("modules.modules", upstream / "modules/modules.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)

    config = LPWMIMFConfig(device="cpu")
    reference = sys.modules["modules.modules"].DLPEncoder(**config.encoder_kwargs()).eval()
    local = LPWMVisualEncoder(config).eval()
    local.load_lpwm_encoder(reference.state_dict())
    images = torch.rand(1, 2, 3, 128, 128)
    with torch.no_grad():
        expected = reference(images, deterministic=True)
        actual = local(images)
    for name, key in (
        ("position", "z"),
        ("scale", "z_scale"),
        ("depth", "z_depth"),
        ("presence", "obj_on"),
        ("features", "z_features"),
        ("background", "z_bg_features"),
    ):
        torch.testing.assert_close(getattr(actual, name).squeeze(2), expected[key], rtol=0, atol=0)
