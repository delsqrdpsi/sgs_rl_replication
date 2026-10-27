import torch

from sgs_rod.sampler import SGSCfg, SGSSampler, UniformSampler


def test_weights_follow_paper_kernel():
    s = SGSSampler(3)
    s.record(torch.tensor([0] * 10 + [1] * 10 + [2] * 10),
             torch.tensor([False] * 10 + [True, False] * 5 + [True] * 10))
    c = SGSCfg()
    p = torch.tensor([0.0, 0.5, 1.0])
    w = (p + c.epsilon) ** 0.5 * (1 - p + c.epsilon) ** 0.5
    expected = torch.softmax(torch.log(w + c.epsilon) / 2, 0)
    assert torch.allclose(s.probabilities(), expected)
    assert s.probabilities().argmax() == 1  # p = 0.5 is favoured


def test_untried_states_start_uniform():
    s = SGSSampler(5)
    assert torch.allclose(s.probabilities(), torch.full((5,), 0.2))


def test_history_keeps_latest_outcomes_with_duplicates():
    s = SGSSampler(2, SGSCfg(history=4))
    s.record(torch.tensor([0, 0, 0]), torch.tensor([True, True, True]))
    s.record(torch.tensor([0, 1, 0, 0]), torch.tensor([False, True, False, False]))
    # state 0: T T T F F F -> latest four are T F F F
    assert s.count[0] == 4 and s.success_rate()[0] == 0.25
    # more outcomes in one batch than the history holds: only the last four survive
    s.record(torch.tensor([1] * 6), torch.tensor([True, True, False, False, False, False]))
    assert s.success_rate()[1] == 0.0


def test_uniform_ignores_outcomes():
    s = UniformSampler(4)
    s.record(torch.tensor([0, 1]), torch.tensor([True, False]))
    assert torch.allclose(s.probabilities(), torch.full((4,), 0.25))
