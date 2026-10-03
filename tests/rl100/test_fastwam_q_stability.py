import torch

from RL.fastwam_q.stability import CASES, GradientGuard, gradient_norm, learning_rates


def test_guard_rejects_before_update():
    g = GradientGuard()
    assert g.check(1.0, 2.0) is None
    assert g.check(float("inf"), 2.0) == "nonfinite_loss_or_gradient"
    assert g.check(1e6, 2.0) == "gradient_at_or_above_1e6"
    for _ in range(19):
        assert g.check(2000, 2.0) is None
    assert g.check(2000, 2.0) == "gradient_above_1e3_for_20_consecutive_updates"
    assert g.check(1, 2.0) is None and g.consecutive == 0


def test_warmup_and_factorial():
    c = CASES["both_lr_div10_warmup"]
    assert learning_rates(c, 300) == (3e-5, 9e-6)
    assert learning_rates(c, 600) == (3e-5, 9e-6)
    assert abs(learning_rates(c, 1)[0] - 1e-7) < 1e-15
    assert (
        len(
            {
                (CASES[n].head_lr, CASES[n].dino_lr)
                for n in ("baseline", "head_lr_div10", "dino_lr_div10", "both_lr_div10")
            }
        )
        == 4
    )


def test_norm_and_clipping():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.0, 4.0])
    assert float(gradient_norm([p])) == 5
    torch.nn.utils.clip_grad_norm_([p], 2)
    torch.testing.assert_close(gradient_norm([p]), torch.tensor(2.0))
