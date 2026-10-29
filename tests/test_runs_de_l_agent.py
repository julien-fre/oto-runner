"""Un run que l'agent ouvre et oublie de clore est fermé par la conclusion du travail.

Vécu le 29/10/2026 (voie abonnement) : la procédure fait ouvrir à
l'agent son propre run (`run_start`, étape 1) et le clore à la fin (`run_finish`,
étape 10). L'agent a terminé son tour après l'étape 9b sans l'étape 10. Le worker a
clos son run technique `done`, et le run de l'agent est resté « en cours, sans fin »
jusqu'à l'alerte des agents, une heure plus tard. Ce que ces bancs tiennent :

1. le suivi lit les deux formes du journal — `outil` (boucle ordinaire) et `claude`
   (flux du CLI), noms préfixés compris ; un `run_finish` refusé ne clôt rien ;
2. la conclusion d'un travail qui a conclu ferme le run oublié, `partial` avec sa
   note, AVANT le run technique ; un run que l'agent a clos lui-même n'est pas
   refermé ;
3. un travail mort ferme aussi le run oublié, en `failed`.
"""
from __future__ import annotations

import json
import time
import types

from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult
from oto_runner.runs_de_l_agent import RunsDeLAgent
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

_PROVIDER = types.SimpleNamespace(__name__="agent_llm_openai", model=lambda: "mistral-large-2512")


def _outil(suivi, nom, arguments, texte, ok=True):
    suivi.evenement("outil", {"nom": nom, "arguments": arguments, "ok": ok, "texte": texte})


def _cli(suivi, ident, nom, entree, contenu, is_error=False):
    suivi.evenement("claude", {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": ident, "name": nom, "input": entree}]}})
    suivi.evenement("claude", {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": ident, "is_error": is_error,
         "content": contenu}]}})


# ── 1. le suivi ──────────────────────────────────────────────────────────────

def test_la_boucle_ordinaire_ouvre_et_clot():
    s = RunsDeLAgent()
    _outil(s, "run_start", {"label": "x"}, json.dumps({"run_id": "r-A", "label": "x"}))
    _outil(s, "run_start", {"label": "y"}, json.dumps({"run_id": "r-B"}))
    _outil(s, "run_finish", {"run_id": "r-A", "outcome": "done"}, '{"ok": true}')
    assert s.ouverts() == ["r-B"]


def test_le_flux_du_cli_avec_noms_prefixes():
    s = RunsDeLAgent()
    _cli(s, "t1", "mcp__oto__run_start", {"label": "post-call"},
         [{"type": "text", "text": json.dumps({"run_id": "r-CLI", "guide": "g"})}])
    assert s.ouverts() == ["r-CLI"]
    _cli(s, "t2", "mcp__oto__run_finish", {"run_id": "r-CLI", "outcome": "done"},
         '{"ok": true, "run_id": "r-CLI"}')
    assert s.ouverts() == []


def test_un_run_finish_refuse_ne_clot_rien():
    s = RunsDeLAgent()
    _cli(s, "t1", "mcp__oto__run_start", {}, '{"run_id": "r-X"}')
    _cli(s, "t2", "mcp__oto__run_finish", {"run_id": "r-X", "outcome": "partial"},
         "note is required for partial", is_error=True)
    assert s.ouverts() == ["r-X"]


def test_un_run_start_en_echec_ou_sans_run_id_n_ouvre_rien():
    s = RunsDeLAgent()
    _outil(s, "run_start", {}, '{"run_id": "r-KO"}', ok=False)
    _outil(s, "run_start", {}, "réponse dégradée")
    _outil(s, "data_rows", {}, '{"run_id": "r-AUTRE"}')
    assert s.ouverts() == []


def test_un_evenement_mal_forme_ne_leve_pas():
    s = RunsDeLAgent()
    s.evenement("claude", {"type": "user", "message": {"content": "pas une liste"}})
    s.evenement("outil", {})
    assert s.ouverts() == []


# ── 2 et 3. la conclusion ────────────────────────────────────────────────────

def _travail(monkeypatch, *, ferme_le_sien: bool, meurt: bool = False):
    mcps = []

    class McpEspion(FauxMcp):
        def __init__(self, **kw):
            super().__init__(**kw)
            mcps.append(self)

    def faux_run(spec, transport, provider, on_event=None, **_):
        # Ce que la boucle écrit au journal quand l'agent appelle ses outils.
        on_event("outil", {"nom": "run_start", "arguments": {"label": "post-call"},
                           "ok": True, "texte": json.dumps({"run_id": "r-AGENT"})})
        if ferme_le_sien:
            on_event("outil", {"nom": "run_finish", "ok": True, "texte": "{}",
                               "arguments": {"run_id": "r-AGENT", "outcome": "done"}})
        if meurt:
            raise TimeoutError("ReadTimeout du fournisseur")
        return AgentResult(reply="fini", stopped="end_turn")

    monkeypatch.setattr(W, "McpSession", McpEspion)
    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    W._un_travail(FauxBackend(), _job("start"), _PROVIDER)
    (mcp,) = mcps
    return [(a["run_id"], a["outcome"], a.get("note")) for n, a in mcp.appels
            if n == "run_finish"]


def test_un_travail_conclu_ferme_le_run_oublie_avant_le_sien(monkeypatch):
    clotures = _travail(monkeypatch, ferme_le_sien=False)
    assert [(r, o) for r, o, _ in clotures] == [("r-AGENT", "partial"), ("r-NEUF", "done")]
    note = clotures[0][2]
    assert "without calling run_finish" in note and "job outcome: done" in note


def test_un_run_clos_par_l_agent_n_est_pas_referme(monkeypatch):
    clotures = _travail(monkeypatch, ferme_le_sien=True)
    assert [(r, o) for r, o, _ in clotures] == [("r-NEUF", "done")]


def test_un_travail_mort_ferme_aussi_le_run_oublie(monkeypatch):
    clotures = _travail(monkeypatch, ferme_le_sien=False, meurt=True)
    assert [(r, o) for r, o, _ in clotures] == [("r-AGENT", "failed"), ("r-NEUF", "failed")]


def test_la_note_dit_qui_a_arrete(monkeypatch):
    clotures = _travail(monkeypatch, ferme_le_sien=False, meurt=True)
    assert "the job failed before the agent called run_finish" in clotures[0][2]


# ── la voie de l'incident : abonnement, flux du CLI lu par `lire_flux` ───────

def test_la_voie_abonnement_ferme_le_run_oublie(monkeypatch):
    from oto_runner import agent_abonnement as A

    mcps = []

    class McpEspion(FauxMcp):
        def __init__(self, **kw):
            super().__init__(**kw)
            mcps.append(self)

    flux = [
        {"type": "system", "subtype": "init", "apiKeySource": "none", "model": "claude-sonnet-5"},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "mcp__oto__run_start",
             "input": {"label": "post-call 2026-10-29T13:00 Paris"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": False,
             "content": [{"type": "text", "text": '{"run_id": "r-CLI", "guide": "g"}'}]}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Rien à faire."}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "Rien à faire.",
         "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]

    class _P:
        __name__ = "agent_abonnement"
        ONE_SHOT = True
        SANDBOX = True

        @staticmethod
        def depot():
            return "claude_subscription"

        @staticmethod
        def model():
            return "sub:sonnet"

        @staticmethod
        def run_once(**kw):
            return A.lire_flux(flux, on_event=kw.get("on_event"), apposer=kw.get("apposer"))

    monkeypatch.setattr(W, "McpSession", McpEspion)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    job = {"id": 7, "kind": "start", "attempts": 1, "delegated_token": "oto_delegue",
           "sandbox_id": "u42", "payload": {"input": "vas-y", "model_family": "claude_subscription"}}
    W._un_travail(FauxBackend(), job, _P)
    (mcp,) = mcps
    clotures = [(a["run_id"], a["outcome"]) for n, a in mcp.appels if n == "run_finish"]
    assert clotures == [("r-CLI", "partial"), ("r-NEUF", "done")]
