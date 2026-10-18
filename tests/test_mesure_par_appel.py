"""Chaque appel au modèle payé par une clé PARTAGÉE se mesure au serveur AVANT la
suite du déroulé (18/10/2026) — et un rapport refusé arrête le déroulé.

Sans ça, la mesure n'avait lieu qu'à la conclusion : un worker tué, un bail mort ou une
conclusion perdue, et tout ce qui avait été dépensé sur notre clé n'était inscrit nulle
part. Ici, au pire, l'appel en vol.
"""
from __future__ import annotations

import pytest

from oto_runner import agent_runtime, worker
from oto_runner.agent_runtime import AgentSpec
from oto_runner.backend import BackendError
from oto_runner.llm_types import Turn

from tests.test_agent_runtime import FauxProvider, FauxTransport, _turn

SPEC = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=3)
USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0,
         "cache_read_input_tokens": 100}


def _avec_usage(turn: Turn) -> Turn:
    return Turn(text=turn.text, tool_calls=turn.tool_calls, stop_reason=turn.stop_reason,
                raw_content=turn.raw_content, usage=dict(USAGE), model="claude-sonnet-5")


def test_chaque_appel_est_mesure_AVANT_ses_outils():
    ordre = []

    class T(FauxTransport):
        def call(self, name, arguments):
            ordre.append(("outil", name))
            return super().call(name, arguments)

    p = FauxProvider([_avec_usage(_turn(calls=[("data_rows", {})])),
                      _avec_usage(_turn(text="fini"))])
    agent_runtime.run(SPEC, T(), p, prompt="go",
                      on_usage=lambda tour, u, m: ordre.append(("mesure", tour, m)))
    assert ordre == [("mesure", 0, "claude-sonnet-5"), ("outil", "data_rows"),
                     ("mesure", 1, "claude-sonnet-5")]


def test_un_rapport_refuse_ARRETE_le_deroule_avec_son_usage():
    t = FauxTransport()
    p = FauxProvider([_avec_usage(_turn(calls=[("data_rows", {})])),
                      _avec_usage(_turn(text="jamais atteint"))])

    def refus(tour, u, m):
        raise RuntimeError("mesure refusée")
    with pytest.raises(RuntimeError) as e:
        agent_runtime.run(SPEC, t, p, prompt="go", on_usage=refus)
    assert t.appels == [], "aucun outil après un tour non inscrit"
    assert p.file, "aucun appel au modèle de plus"
    # Le tour refusé fait partie de l'usage accroché à l'exception : la conclusion le
    # rapporte, et le serveur l'inscrit en reste.
    assert e.value.usage_partiel["input_tokens"] == 10
    assert e.value.usage_partiel["output_tokens"] == 5
    assert e.value.modele_partiel == "claude-sonnet-5"


class _File:
    def __init__(self, erreurs=()):
        self.erreurs, self.vu = list(erreurs), []

    def meter(self, job_id, tour, usage):
        self.vu.append((job_id, tour, usage))
        if self.erreurs:
            raise self.erreurs.pop(0)
        return {"ok": True}


def test_sans_cle_partagee_rien_n_est_mesure():
    assert worker._mesureur(_File(), {"id": 9}) is None


def test_le_rapport_porte_les_postes_sous_les_noms_du_serveur(monkeypatch):
    f = _File()
    worker._mesureur(f, {"id": 9, "meter_usage": True})(2, USAGE, "claude-sonnet-5")
    assert f.vu == [(9, 2, {"model": "claude-sonnet-5", "usage_input": 10,
                            "usage_output": 5, "usage_cache_write": 0,
                            "usage_cache_read": 100})]


def test_une_panne_de_transport_se_rejoue(monkeypatch):
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)
    f = _File([BackendError("502", status=502)])
    worker._mesureur(f, {"id": 9, "meter_usage": True})(0, USAGE, "m")
    assert len(f.vu) == 2


@pytest.mark.parametrize("erreurs", [
    [BackendError("refus", status=422, code="usage_unmeasured")],
    [BackendError("502", status=502)] * 3,
])
def test_un_refus_ou_trois_pannes_levent(monkeypatch, erreurs):
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)
    f = _File(list(erreurs))
    with pytest.raises(RuntimeError):
        worker._mesureur(f, {"id": 9, "meter_usage": True})(0, USAGE, "m")


def test_la_promesse_ne_part_que_sur_demande(monkeypatch):
    monkeypatch.delenv(worker._ENV_MESURE, raising=False)
    assert worker._mesure_par_appel() is False
    monkeypatch.setenv(worker._ENV_MESURE, "1")
    assert worker._mesure_par_appel() is True
