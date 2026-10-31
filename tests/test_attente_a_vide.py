"""L'attente sur une file vide s'étale : des workers démarrés ensemble ne sondent pas
dans la même seconde, et le rythme moyen reste celui de `_POLL_S`."""
from statistics import mean

from oto_runner import worker


def test_l_attente_reste_autour_de_15_s_et_varie():
    tirages = [worker._attente_a_vide() for _ in range(2000)]
    assert min(tirages) >= worker._POLL_S * 0.8
    assert max(tirages) <= worker._POLL_S * 1.2
    assert abs(mean(tirages) - worker._POLL_S) < 0.3
    assert len({round(t, 3) for t in tirages}) > 100


def test_poll_a_zero_reste_a_zero(monkeypatch):
    monkeypatch.setattr(worker, "_POLL_S", 0)
    assert worker._attente_a_vide() == 0
