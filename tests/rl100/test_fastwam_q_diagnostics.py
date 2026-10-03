import torch

from RL.fastwam_q.diagnostics import LayerTrace, norm_statistics, parameter_report


def test_layer_trace_does_not_change_gradients():
    network = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 2))
    inputs = torch.randn(3, 4, requires_grad=True)
    network(inputs).square().sum().backward()
    reference = [parameter.grad.clone() for parameter in network.parameters()]
    network.zero_grad(set_to_none=True)
    trace = LayerTrace(network)
    network(inputs).square().sum().backward()
    report = trace.report()
    trace.close()
    for expected, parameter in zip(reference, network.parameters(), strict=True):
        torch.testing.assert_close(expected, parameter.grad)
    assert "0.output.gradient" in report
    assert parameter_report(network)["global_grad_l2"] > 0


def test_layernorm_trace_detects_small_variance():
    module = torch.nn.LayerNorm(4)
    stats = norm_statistics(torch.ones(2, 4), module)
    assert float(stats["variance_min"]) == 0
    torch.testing.assert_close(stats["inverse_scale_max"], torch.tensor(module.eps).rsqrt())


def test_decoder_training_uses_math_attention_only(monkeypatch):
    from RL.fastwam_q import model
    from RL.fastwam_q.config import FastWAMQConfig

    original = model.sdpa_kernel
    calls = []

    def record(backend):
        calls.append(backend)
        return original(backend)

    monkeypatch.setattr(model, "sdpa_kernel", record)
    config = FastWAMQConfig(dim_model=16, n_heads=4, dim_feedforward=32)
    layer = model.DecoderLayer(config)
    x = torch.randn(2, 4, 16, requires_grad=True)
    context = torch.randn(2, 6, 16, requires_grad=True)
    position = torch.randn(1, 4, 16)
    layer(x, context, position, None, None).square().mean().backward()
    assert calls == [model.SDPBackend.MATH]
    assert torch.isfinite(x.grad).all() and torch.isfinite(context.grad).all()
    layer.eval()(x, context, position, None, None)
    assert len(calls) == 1


def test_qk_norm_bounds_projected_scores_and_preserves_padding(monkeypatch):
    import torch.nn.functional as functional

    from RL.fastwam_q.config import FastWAMQConfig
    from RL.fastwam_q.model import QKNormMultiheadAttention

    attention = QKNormMultiheadAttention(FastWAMQConfig(dim_model=16, n_heads=4, qk_norm=True))
    with torch.no_grad():
        attention.in_proj_weight[:32].mul_(1000)
    original = functional.scaled_dot_product_attention
    captured = []

    def record(query, key, value, **kwargs):
        captured.append((query.detach(), key.detach()))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(functional, "scaled_dot_product_attention", record)
    query = torch.randn(2, 3, 16, requires_grad=True)
    context = torch.randn(2, 5, 16, requires_grad=True)
    padding = torch.tensor([[False, False, False, False, True], [False, False, False, False, True]])
    output, _ = attention(query, context, context, key_padding_mask=padding)
    output.square().mean().backward()
    q, k = captured[0]
    torch.testing.assert_close(q.square().mean(-1), torch.ones_like(q[..., 0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k.square().mean(-1), torch.ones_like(k[..., 0]), atol=1e-5, rtol=1e-5)
    scores = q @ k.transpose(-1, -2) / attention.head_dim**0.5
    assert float(scores.abs().max()) <= attention.head_dim**0.5 + 1e-5
    assert torch.isfinite(query.grad).all() and torch.isfinite(context.grad).all()
    torch.testing.assert_close(context.grad[:, -1], torch.zeros_like(context.grad[:, -1]))


def test_qk_norm_preserves_attention_checkpoint_parameters():
    from RL.fastwam_q.config import FastWAMQConfig
    from RL.fastwam_q.model import DecoderLayer

    config = FastWAMQConfig(dim_model=16, n_heads=4, dim_feedforward=32)
    torch.manual_seed(42)
    original = DecoderLayer(config)
    config.qk_norm = True
    torch.manual_seed(42)
    normalized = DecoderLayer(config)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, normalized.state_dict()[name])
    normalized.load_state_dict(original.state_dict(), strict=True)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in normalized.parameters())
