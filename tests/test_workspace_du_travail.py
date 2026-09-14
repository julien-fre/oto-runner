"""Le workspace d'une clé d'ORGANISATION Anthropic : du travail à l'en-tête, et nulle part ailleurs.

Une clé Anthropic créée pour toute l'organisation, et non dans un workspace, fait refuser
chaque requête qui ne nomme pas le workspace à facturer (« This API key is not scoped to a
workspace, so this request must include the anthropic-workspace-id header »). Le backend
(oto-backend, 14/09/2026) le remet au claim, à côté de la clé (`model_workspace`).

Ces bancs tiennent le trajet : le travail → la boucle → le provider (seulement s'il est
posé) → l'en-tête `anthropic-workspace-id` ; et l'absence : ni dans le tour, ni dans les
événements du journal, ni sur la voie Chat Completions, qui le refuse en le nommant.
"""
from __future__ import annotations

import json

import pytest

from oto_runner import agent_llm, agent_llm_openai, agent_runtime
from oto_runner.agent_runtime import AgentSpec
from oto_runner.llm_types import LlmUnavailable
from tests.test_agent_runtime import FauxProvider, FauxTransport, _turn
from tests.test_cle_de_modele_du_travail import _vu_par_la_boucle
from tests.test_modele_du_travail import _FauxSdk

WS = "wrkspc_01banc"


def _job(**extra):
    base = {"id": 31, "kind": "start", "delegated_token": "oto_delegue",
            "payload": {"project_id": 3, "org_id": 42, "procedure": "veille",
                        "input": "Lis la procédure `veille` et applique-la."}}
    base.update(extra)
    return base


def test_le_workspace_du_travail_arrive_a_la_boucle_avec_la_cle(monkeypatch):
    vu = _vu_par_la_boucle(monkeypatch, _job(model_key="sk-de-l-org", model_workspace=WS))
    assert (vu["api_key"], vu["workspace"]) == ("sk-de-l-org", WS)


def test_sans_workspace_remis_la_boucle_n_en_recoit_aucun(monkeypatch):
    """`None`, pas chaîne vide : aucun en-tête ne doit partir."""
    assert _vu_par_la_boucle(monkeypatch, _job(model_key="sk-de-l-org"))["workspace"] is None


class _Mouchard(FauxProvider):
    def __init__(self, tours):
        super().__init__(tours)
        self.appels: list = []

    def complete(self, **kw):
        self.appels.append(kw)
        return super().complete(**kw)


def test_la_boucle_ne_transmet_le_workspace_que_s_il_est_pose():
    spec = AgentSpec(system="s", tools=frozenset(), max_steps=2)
    avec = _Mouchard([_turn(text="fini")])
    agent_runtime.run(spec, FauxTransport(), avec, prompt="go", api_key="k", workspace=WS)
    assert avec.appels[0]["workspace"] == WS
    sans = _Mouchard([_turn(text="fini")])
    agent_runtime.run(spec, FauxTransport(), sans, prompt="go", api_key="k")
    assert "workspace" not in sans.appels[0], (
        "un provider qui ne connaît pas le workspace ne doit jamais le recevoir")


def test_le_workspace_n_entre_dans_aucun_evenement_du_tour():
    evenements: list = []
    agent_runtime.run(AgentSpec(system="s", tools=frozenset(), max_steps=2), FauxTransport(),
                      FauxProvider([_turn(text="fini")]), prompt="go", api_key="k", workspace=WS,
                      on_event=lambda ev, champs: evenements.append((ev, champs)))
    assert evenements and WS not in json.dumps(evenements, default=str)


@pytest.fixture
def sdk(monkeypatch):
    vu: dict = {}

    class _Sdk(_FauxSdk):
        def Anthropic(self, api_key=None, **kw):  # noqa: N802 — le nom du SDK
            vu["client"] = {"api_key": api_key, **kw}
            return self

    monkeypatch.setattr(agent_llm, "_sdk", lambda: _Sdk(vu))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-plateforme")
    return vu


def test_anthropic_envoie_le_workspace_en_en_tete(sdk):
    turn = agent_llm.complete(system="s", messages=[], tools=[], api_key="sk-de-l-org",
                              workspace=WS)
    assert sdk["client"]["default_headers"] == {"anthropic-workspace-id": WS}
    assert WS not in repr(turn), "le tour ne porte pas le workspace"


def test_anthropic_sans_workspace_n_envoie_aucun_en_tete(sdk):
    agent_llm.complete(system="s", messages=[], tools=[], api_key="sk-de-l-org")
    assert "default_headers" not in sdk["client"], "la requête d'avant, inchangée"


def test_la_voie_chat_completions_refuse_un_workspace_en_le_nommant(monkeypatch):
    """Un workspace n'a de sens que pour une clé Anthropic : le recevoir ici est un travail
    mal routé. Rien ne part au fournisseur."""
    postes: list = []
    monkeypatch.setattr(agent_llm_openai.requests, "post",
                        lambda *a, **k: postes.append(k) or pytest.fail("rien ne doit partir"))
    with pytest.raises(LlmUnavailable) as e:
        agent_llm_openai.complete(system="s", messages=[], tools=[], api_key="k", workspace=WS)
    assert "model_workspace" in str(e.value) and postes == []
