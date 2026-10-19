"""Chaque appel au modèle payé par une clé PARTAGÉE se RÉSERVE au budget du serveur
avant de partir (19/10/2026) et se RÈGLE avant la suite du déroulé (18/10/2026) — un
refus de l'un ou de l'autre arrête le déroulé.

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
    # Le tour refusé fait partie de l'usage accroché à l'exception ; le serveur, lui,
    # le règle au montant réservé à la conclusion.
    assert e.value.usage_partiel["input_tokens"] == 10
    assert e.value.usage_partiel["output_tokens"] == 5
    assert e.value.modele_partiel == "claude-sonnet-5"


class _Plafonne(FauxProvider):
    """Un fournisseur qui dit son plafond de sortie, comme `agent_llm`."""

    @staticmethod
    def plafond_de_sortie(max_output_tokens, effort=None):
        return max_output_tokens or 8192


def test_chaque_appel_est_reserve_AVANT_de_partir_puis_regle():
    ordre = []
    p = _Plafonne([_avec_usage(_turn(calls=[("data_rows", {})])),
                   _avec_usage(_turn(text="fini"))])
    agent_runtime.run(SPEC, FauxTransport(), p, prompt="go",
                      on_reserve=lambda tour, n: ordre.append(("reserve", tour, n)),
                      on_usage=lambda tour, u, m: ordre.append(("regle", tour)))
    assert ordre == [("reserve", 0, 8192), ("regle", 0), ("reserve", 1, 8192),
                     ("regle", 1)]


def test_la_reservation_porte_le_plafond_du_travail():
    vu = []
    spec = AgentSpec(system="s", tools=frozenset(), max_steps=1, max_output_tokens=2000)
    agent_runtime.run(spec, FauxTransport(), _Plafonne([_avec_usage(_turn(text="fini"))]),
                      prompt="go", on_reserve=lambda tour, n: vu.append(n))
    assert vu == [2000]


def test_une_reservation_refusee_n_envoie_PAS_l_appel():
    p = _Plafonne([_avec_usage(_turn(text="jamais envoyé"))])

    def refus(tour, n):
        raise RuntimeError("crédits insuffisants")
    with pytest.raises(RuntimeError):
        agent_runtime.run(SPEC, FauxTransport(), p, prompt="go", on_reserve=refus)
    assert len(p.file) == 1, "le modèle n'a pas été appelé"


def test_le_plafond_de_sortie_est_celui_qui_part_au_fournisseur(monkeypatch):
    from oto_runner import agent_llm
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS", raising=False)
    assert agent_llm.plafond_de_sortie(None) == agent_llm.max_tokens()
    assert agent_llm.plafond_de_sortie(3000) == 3000


class _File:
    def __init__(self, erreurs=()):
        self.erreurs, self.vu = list(erreurs), []

    def meter(self, job_id, tour, usage):
        self.vu.append((job_id, tour, usage))
        if self.erreurs:
            raise self.erreurs.pop(0)
        return {"ok": True}

    def reserve(self, job_id, tour, max_tokens):
        self.vu.append((job_id, tour, max_tokens))
        if self.erreurs:
            raise self.erreurs.pop(0)
        return {"ok": True, "held": True}


def test_sans_cle_partagee_rien_n_est_reserve_ni_regle():
    assert worker._mesureur(_File(), {"id": 9}) is None
    assert worker._reservateur(_File(), {"id": 9}) is None


def test_la_reservation_part_avec_le_plafond(monkeypatch):
    f = _File()
    worker._reservateur(f, {"id": 9, "meter_usage": True})(3, 8192)
    assert f.vu == [(9, 3, 8192)]


@pytest.mark.parametrize("erreurs, leve", [
    ([BackendError("502", status=502)], False),
    ([BackendError("refus", status=402, code="insufficient_credits")], True),
    ([BackendError("502", status=502)] * 3, True),
])
def test_la_reservation_se_rejoue_sur_panne_et_leve_sur_refus(monkeypatch, erreurs, leve):
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)
    reserver = worker._reservateur(_File(list(erreurs)), {"id": 9, "meter_usage": True})
    if leve:
        with pytest.raises(RuntimeError):
            reserver(0, 100)
    else:
        reserver(0, 100)


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


@pytest.mark.parametrize("reponse", [
    {"ok": True, "held": True, "reserved": 8192},     # solde nul ou négatif : réservé quand même
    {"ok": True, "held": False},                      # hors barème, ou registre indisponible
])
def test_en_report_le_backend_repond_OK_et_le_tour_part(reponse):
    """En `report` (le défaut du serveur), le backend ne refuse JAMAIS une réservation :
    il répond OK, réservée ou non. Le worker ne lit rien d'autre que le refus : le tour
    part, puis se règle."""
    class _FileReport(_File):
        def reserve(self, job_id, tour, max_tokens):
            self.vu.append(("reserve", tour))
            return dict(reponse)

    f = _FileReport()
    job = {"id": 9, "meter_usage": True}
    p = _Plafonne([_avec_usage(_turn(text="fini"))])
    res = agent_runtime.run(SPEC, FauxTransport(), p, prompt="go",
                            on_reserve=worker._reservateur(f, job),
                            on_usage=worker._mesureur(f, job))
    assert p.file == [], "l'appel au modèle est parti"
    assert res.stopped == "end_turn"
    assert [v[0] if isinstance(v[0], str) else "meter" for v in f.vu] == ["reserve", "meter"]
