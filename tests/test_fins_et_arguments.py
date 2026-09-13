"""Deux formes qu'un fournisseur Chat Completions peut rendre, et ce qu'elles ne font plus.

Audit du 13/09/2026 sur le code servi (6b78cd7) :
1. des arguments d'outil qui ne sont pas un objet JSON devenaient `{}`, et l'outil
   S'EXÉCUTAIT — un outil à paramètres facultatifs agissait ;
2. `finish_reason: length` sans appel se lisait `end_turn` : le travail concluait
   `done` sur une réponse tronquée.

Ce que ces bancs tiennent :
- l'appel aux arguments invalides n'est pas transporté, l'original est journalisé,
  le modèle lit une erreur de forme — et un vrai `{}` reste un appel valide ;
- une fin anormale (`length`, `model_length`, `error`, absente, inconnue, ou
  `tool_calls` sans aucun appel) arrête la boucle AVANT tout appel, garde son motif
  brut et les usages déclarés, et conclut le travail en échec nommé ;
- `stop`, et `tool_calls` porteur d'un appel, restent des fins normales.

Ces bancs tiennent sur le code d'avant le lot : leur rouge y est celui du défaut.
"""
from __future__ import annotations

import json

import pytest

from oto_runner import agent_llm_openai as P
from oto_runner import agent_runtime
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentSpec
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

SPEC = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=3)


class _Resp:
    def __init__(self, payload):
        self._payload, self.status_code, self.text = payload, 200, json.dumps(payload)

    def json(self):
        return self._payload


def _reponse(message, finish):
    choix = {"message": message}
    if finish is not None:
        choix["finish_reason"] = finish
    return {"choices": [choix], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def _texte(contenu):
    return {"role": "assistant", "content": contenu}


def _appel(arguments, nom="data_rows"):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": nom, "arguments": arguments}}]}


def _servir(monkeypatch, *reponses):
    file = list(reponses)
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp(file.pop(0)))


class _Outils:
    """Un outil AUTORISÉ dont tous les paramètres sont facultatifs : `{}` le fait agir."""

    def __init__(self):
        self.appels: list = []

    def schemas(self, names):
        return [{"name": n, "description": "",
                 "input_schema": {"type": "object",
                                  "properties": {"limit": {"type": "integer"}}}}
                for n in sorted(names)]

    def call(self, name, arguments):
        self.appels.append((name, arguments))
        return "3 lignes écrites", False


# ── 1. Les arguments ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("brut", ["{pas du json", "[1, 2]", "null", ""])
def test_des_arguments_qui_ne_sont_pas_un_objet_gardent_leur_original(monkeypatch, brut):
    _servir(monkeypatch, _reponse(_appel(brut), "tool_calls"))
    (appel,) = P.complete(system="s", messages=[], tools=[], api_key="k").tool_calls
    assert getattr(appel, "arguments_invalides", "non porté") == brut


def test_des_arguments_absents_ne_deviennent_pas_un_objet_vide(monkeypatch):
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "data_rows"}}]}
    _servir(monkeypatch, _reponse(msg, "tool_calls"))
    (appel,) = P.complete(system="s", messages=[], tools=[], api_key="k").tool_calls
    assert getattr(appel, "arguments_invalides", "non porté") == "null"


def test_un_vrai_objet_vide_reste_un_appel_valide(monkeypatch):
    """Garde négative : prouvée par son vert seul, sans chute."""
    _servir(monkeypatch, _reponse(_appel("{}"), "tool_calls"))
    (appel,) = P.complete(system="s", messages=[], tools=[], api_key="k").tool_calls
    assert appel.arguments == {} and getattr(appel, "arguments_invalides", None) is None


def test_un_appel_aux_arguments_invalides_n_est_jamais_transporte(monkeypatch):
    _servir(monkeypatch, _reponse(_appel("{pas du json"), "tool_calls"),
            _reponse(_texte("je renonce"), "stop"))
    outils, evenements = _Outils(), []
    agent_runtime.run(SPEC, outils, P, prompt="go", api_key="k",
                      on_event=lambda ev, champs: evenements.append((ev, champs)))
    assert outils.appels == [], "zéro exécution : l'outil à paramètres facultatifs n'a pas agi"
    (outil,) = [c for ev, c in evenements if ev == "outil"]
    assert outil["ok"] is False and "NON exécuté" in outil["texte"]
    assert outil.get("arguments_invalides") == "{pas du json", "l'original est journalisé"


# ── 2. Les fins ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("finish", ["length", "model_length", "error", None, "inattendue"])
def test_une_fin_anormale_est_nommee_avec_son_motif_brut(monkeypatch, finish):
    _servir(monkeypatch, _reponse(_texte("la fiche est pr"), finish))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.stop_reason == "fin_anormale"
    assert turn.defaut == {"forme": "fin_anormale", "finish_reason": finish}
    assert turn.usage.get("output_tokens") == 20, "les usages déclarés sont conservés"


def test_tool_calls_sans_aucun_appel_ne_fabrique_pas_une_reussite(monkeypatch):
    _servir(monkeypatch, _reponse(_texte("c'est écrit"), "tool_calls"))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.stop_reason == "fin_anormale" and turn.defaut["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("finish,message", [("stop", _texte("fiche écrite")),
                                            ("tool_calls", _appel('{"limit": 3}'))])
def test_les_fins_normales_restent_normales(monkeypatch, finish, message):
    """Garde négative : prouvée par son vert seul, sans chute."""
    _servir(monkeypatch, _reponse(message, finish))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.stop_reason == "end_turn" and turn.defaut is None


def test_une_fin_anormale_arrete_la_boucle_avant_tout_appel(monkeypatch):
    # Le second tour n'est servi qu'au code qui NE s'arrête pas : sans lui, ce banc
    # tombait sur la doublure vide, pas sur l'appel exécuté.
    _servir(monkeypatch, _reponse(_appel('{"limit": 3}'), "length"),
            _reponse(_texte("fini"), "stop"))
    outils = _Outils()
    res = agent_runtime.run(SPEC, outils, P, prompt="go", api_key="k")
    assert outils.appels == [], "un appel porté par une sortie coupée n'est pas exécuté"
    assert res.stopped == "fin_anormale" and (res.defaut or {}).get("finish_reason") == "length"


# ── 3. Jusqu'à la conclusion du worker ───────────────────────────────────────

def _travail_servi(monkeypatch, *reponses):
    """Un travail ENTIER : le vrai transport Chat Completions, la vraie boucle, le worker."""
    _servir(monkeypatch, *reponses)
    mcps: list = []

    class McpEspion(FauxMcp):
        def __init__(self, **kw):
            super().__init__(**kw)
            mcps.append(self)

    class File(FauxBackend):
        conclusion = None

        def complete(self, job_id, ok, error=None, run_id=None, result=None):
            self.conclusion = {"ok": ok, "error": error, "result": result}
            return super().complete(job_id, ok, error=error, run_id=run_id, result=result)

    monkeypatch.setattr(W, "McpSession", McpEspion)
    b = File()
    job = _job("start")
    job["model_key"] = "k"
    W._un_travail(b, job, P)
    (mcp,) = mcps
    return b, [a for nom, a in mcp.appels if nom == "run_finish"]


def test_une_sortie_tronquee_conclut_le_travail_en_echec_nomme(monkeypatch):
    b, clotures = _travail_servi(monkeypatch, _reponse(_texte("la fiche est pr"), "length"))
    assert [c["outcome"] for c in clotures] == ["failed"], "aucune réussite sur une sortie coupée"
    assert clotures[0]["note"] == "fin_anormale (length)"
    assert b.conclusion["ok"] is False and b.conclusion["error"] == "fin_anormale (length)"
    assert b.conclusion["result"]["stopped"] == "fin_anormale"
    assert b.conclusion["result"].get("defaut") == {"forme": "fin_anormale", "finish_reason": "length"}
    assert b.conclusion["result"]["usage_output"] == 20


def test_une_fin_normale_conclut_toujours_done(monkeypatch):
    """Garde négative : prouvée par son vert seul, sans chute."""
    b, clotures = _travail_servi(monkeypatch, _reponse(_texte("fiche écrite"), "stop"))
    assert [c["outcome"] for c in clotures] == ["done"] and b.conclusion["ok"] is True
