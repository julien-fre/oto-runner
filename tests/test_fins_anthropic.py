"""Les fins d'un tour Anthropic : deux seulement concluent, les autres se nomment.

Relevé le 14/09/2026 en préparant le plafond de complétion porté par le travail : la
règle posée sur la voie Chat Completions par l'audit du 13/09/2026
(`test_fins_et_arguments.py`) n'existait pas ici. Un `stop_reason: max_tokens` — sortie
coupée — se lisait comme une fin : sans appel, le travail concluait `done` sur une
réponse tronquée ; avec un appel, la boucle exécutait un appel coupé.

Ce que ces bancs tiennent :
- `max_tokens`, `model_context_window_exceeded`, `pause_turn`, `stop_sequence`, une fin
  absente, et `tool_use` sans aucun appel sont des fins ANORMALES : le tour les nomme
  avec leur motif brut et garde ses usages ;
- une fin anormale arrête la boucle AVANT tout appel ;
- `end_turn`, et `tool_use` porteur d'un appel, restent des fins normales.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from oto_runner import agent_llm as A
from oto_runner import agent_runtime
from oto_runner.agent_runtime import AgentSpec
from tests.test_fins_et_arguments import _Outils

SPEC = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=3)


def _texte(texte):
    return SimpleNamespace(type="text", text=texte)


def _appel(nom="data_rows", arguments=None):
    return SimpleNamespace(type="tool_use", id="t1", name=nom,
                           input={"limit": 3} if arguments is None else arguments)


class _Sdk:
    """Le SDK Anthropic, qui rend dans l'ordre les réponses qu'on lui donne."""

    def __init__(self, *reponses):
        self.file = list(reponses)
        self.messages = self

    def Anthropic(self, api_key=None):  # noqa: N802 — le nom du SDK
        return self

    def create(self, **_kwargs):
        stop, contenu = self.file.pop(0)
        return SimpleNamespace(stop_reason=stop, content=contenu, model="claude-sonnet-5",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


@pytest.fixture(autouse=True)
def _cle(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-plateforme")
    monkeypatch.delenv("OTO_RUNNER_EFFORT", raising=False)


def _servir(monkeypatch, *reponses):
    monkeypatch.setattr(A, "_sdk", lambda: _Sdk(*reponses))


@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded",
                                  "pause_turn", "stop_sequence", None])
def test_une_fin_anormale_est_nommee_avec_son_motif_brut(monkeypatch, stop):
    _servir(monkeypatch, (stop, [_texte("la fiche est pr")]))
    turn = A.complete(system="s", messages=[], tools=[])
    assert turn.stop_reason == "fin_anormale"
    assert turn.defaut == {"forme": "fin_anormale", "stop_reason": stop}
    assert turn.usage.get("output_tokens") == 20, "les usages déclarés sont conservés"


def test_tool_use_sans_aucun_appel_ne_fabrique_pas_une_reussite(monkeypatch):
    _servir(monkeypatch, ("tool_use", [_texte("c'est écrit")]))
    turn = A.complete(system="s", messages=[], tools=[])
    assert turn.stop_reason == "fin_anormale" and turn.defaut["stop_reason"] == "tool_use"


@pytest.mark.parametrize("stop,contenu", [("end_turn", [_texte("fiche écrite")]),
                                          ("tool_use", [_appel()])])
def test_les_fins_normales_restent_normales(monkeypatch, stop, contenu):
    """Garde négative : prouvée par son vert seul, sans chute."""
    _servir(monkeypatch, (stop, contenu))
    turn = A.complete(system="s", messages=[], tools=[])
    assert turn.stop_reason == stop and turn.defaut is None


def test_un_appel_porte_par_une_sortie_coupee_n_est_pas_execute(monkeypatch):
    # Le second tour n'est servi qu'au code qui NE s'arrête pas : sans lui, ce banc
    # tombait sur la doublure vide, pas sur l'appel exécuté.
    _servir(monkeypatch, ("max_tokens", [_appel()]), ("end_turn", [_texte("fini")]))
    outils = _Outils()
    res = agent_runtime.run(SPEC, outils, A, prompt="go", api_key="k")
    assert outils.appels == [], "un appel porté par une sortie coupée n'est pas exécuté"
    assert res.stopped == "fin_anormale"
    assert (res.defaut or {}).get("stop_reason") == "max_tokens"
