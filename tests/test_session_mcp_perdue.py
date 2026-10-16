"""La session MCP ne survit pas à un redéploiement — et un job muet n'est pas fini.

Vécu le 28/08 : le service MCP redémarre, la session du worker est invalidée,
et TOUS les appels d'outils suivants échouent (`-32600` « Session not found »).
L'agent lit ça comme une réponse : il l'annonce proprement et conclut. Le job
finit « done » sans une écriture — donc il n'est même pas rejoué, et la ligne
reste « à enrichir » sans que personne ne le sache. Deux fiches perdues en
silence ; sur un palier de 500 lignes, ça ne se verrait pas.

Deux crans, ici : la session se ROUVRE (une fois par appel, jamais en boucle)
et l'appel est rejoué ; et si rien n'a pu être écrit alors que le transport a
lâché, le job ÉCHOUE — le backend le rejoue. Sur cette mission il n'existe
aucune issue légitime « conclu, rien écrit ».
"""
from __future__ import annotations

import pytest

from oto_runner import agent_runtime
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult, AgentSpec, AgentStep
from oto_runner.mcp import McpSession
from tests.test_worker_reprise import FauxBackend, FauxMcp

_CLAIM = "demo-connecteur_data_claim_next"
_WRITE = "demo-connecteur_data_write"
_PERDUE = {"jsonrpc": "2.0", "id": 1,
           "error": {"code": -32600, "message": "Session not found"}}


def _session(monkeypatch, reponses_call, initialize=True):
    """Une session sans réseau. `reponses_call` est consommée un appel à la
    fois ; `initialize` à False rend un serveur qui ne redonne plus de session."""
    etat = {"initialize": 0, "call": 0}

    def _post(self, corps, avec_entetes=False):
        methode = corps.get("method")
        if methode == "initialize":
            etat["initialize"] += 1
            muet = not initialize and etat["initialize"] > 1
            entetes = {} if muet else {"mcp-session-id": f"s{etat['initialize']}"}
            return (entetes, {}) if avec_entetes else {}
        if methode == "tools/list":
            return {"result": {"tools": [
                {"name": _WRITE, "inputSchema": {"type": "object",
                                                 "properties": {"row": {}}}}]}}
        if methode == "tools/call":
            etat["call"] += 1
            i = min(etat["call"], len(reponses_call)) - 1
            return reponses_call[i]
        return {}

    monkeypatch.setattr(McpSession, "_post", _post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return McpSession(url="http://x", token="t"), etat


def test_une_session_perdue_se_rouvre_et_lappel_est_rejoue(monkeypatch, caplog):
    """Le redéploiement est un non-événement : on rouvre, on rejoue une fois.
    Un `-32600` n'a jamais été exécuté — le rejeu ne peut pas doubler une
    écriture."""
    import logging
    caplog.set_level(logging.INFO, logger="oto_runner")
    ok = {"result": {"content": [{"type": "text", "text": "ligne écrite"}]}}
    s, etat = _session(monkeypatch, [_PERDUE, ok])
    assert s.call(_WRITE, {"row": 1}) == ("ligne écrite", False, ())
    assert etat["initialize"] == 2, "une réouverture, pas plus"
    assert etat["call"] == 2, "l'appel est rejoué UNE fois"
    assert "session MCP rouverte" in caplog.text


def test_une_reouverture_impossible_leve(monkeypatch):
    """Rendre l'erreur au modèle le ferait conclure « done » sans écriture :
    on lève, le job échoue, le backend le rejoue (`max_attempts`)."""
    s, etat = _session(monkeypatch, [_PERDUE], initialize=False)
    with pytest.raises(RuntimeError, match="session id"):
        s.call(_WRITE, {"row": 1})
    assert etat["initialize"] == 4, "1 à l'ouverture + 3 essais de réouverture"


def test_une_session_toujours_refusee_leve_sans_boucler(monkeypatch):
    """UNE réouverture par appel : si le serveur refuse encore, on échoue net —
    une boucle de réouvertures brûlerait le bail sans rien produire."""
    s, etat = _session(monkeypatch, [_PERDUE])
    with pytest.raises(RuntimeError, match="reste refusé"):
        s.call(_WRITE, {"row": 1})
    assert etat["call"] == 2 and etat["initialize"] == 2


def test_un_mcp_session_id_refuse_se_lit_hors_json(monkeypatch):
    """Le refus n'arrive pas toujours en JSON-RPC : un corps nu « Missing
    session ID » remonte en `_brut` — même signature, même traitement."""
    ok = {"result": {"content": [{"type": "text", "text": "ok"}]}}
    s, etat = _session(monkeypatch, [{"_brut": "Bad Request: Missing session ID"},
                                     ok])
    assert s.call(_WRITE, {"row": 1}) == ("ok", False, ())
    assert etat["initialize"] == 2


def test_une_erreur_metier_ne_rouvre_rien(monkeypatch):
    """Une erreur d'OUTIL est une réponse que le modèle lit pour se corriger :
    elle ne touche pas à la session."""
    metier = {"result": {"isError": True,
                         "content": [{"type": "text", "text": "champ inconnu"}]}}
    s, etat = _session(monkeypatch, [metier])
    assert s.call(_WRITE, {"row": 1}) == ("champ inconnu", True, ())
    assert etat["initialize"] == 1 and etat["call"] == 1


# ── la garde côté worker : « conclu, rien écrit » n'est pas une issue ────────
class TransportMort:
    def schemas(self, names):
        return [{"name": n, "description": "", "input_schema": {"type": "object"}}
                for n in sorted(names)]

    def call(self, name, arguments):
        raise RuntimeError("session MCP rouverte mais data_write reste refusé")


def test_la_boucle_distingue_la_panne_de_transport_de_lerreur_doutil():
    """Le modèle lit les deux de la même façon (il n'y a rien d'autre à lui
    dire) — mais le pas, lui, garde la différence."""
    from tests.test_agent_runtime import FauxProvider, FauxTransport, _turn

    spec = AgentSpec(system="s", tools=frozenset({_WRITE}), max_steps=3)
    p = FauxProvider([_turn(calls=[(_WRITE, {})]), _turn(text="je m'arrête")])
    pas = agent_runtime.run(spec, TransportMort(), p, prompt="go").steps[0]
    assert pas.ok is False and pas.transport_ko is True

    p = FauxProvider([_turn(calls=[(_WRITE, {})]), _turn(text="je corrige")])
    pas = agent_runtime.run(spec, FauxTransport({_WRITE: ("champ inconnu", True, ())}),
                            p, prompt="go").steps[0]
    assert pas.ok is False and pas.transport_ko is False, \
        "une erreur métier reste une réponse, pas une panne"


def _job():
    return {"id": 7, "kind": "start", "delegated_token": "oto_delegue", "run_id": None,
            "payload": {"procedure": "demo", "namespace": "lignes-demo",
                        "tools": [_CLAIM, _WRITE], "input": "Vas-y.",
                        "max_steps": 3}}


class BackendQuiRetientLerreur(FauxBackend):
    erreur = None

    def complete(self, job_id, ok, error=None, run_id=None, result=None):
        self.erreur = error
        return super().complete(job_id, ok, error, run_id, result)


def _conclure(monkeypatch, etapes):
    def faux_run(spec, transport, provider, prompt=None, history=None,
                 on_turn=None, **_):
        return AgentResult(reply="Je n'ai pas pu écrire : la session a expiré.",
                           stopped="end_turn", steps=etapes)

    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    monkeypatch.setattr(W, "McpSession", FauxMcp)
    b = BackendQuiRetientLerreur()
    W._traiter(b, _job(), provider=None)
    return b




def test_une_ecriture_reussie_conclut_le_job_malgre_une_panne(monkeypatch):
    """La garde ne vise QUE le job qui n'a rien produit : un transport qui
    lâche après l'écriture n'annule pas le travail fait."""
    b = _conclure(monkeypatch, [
        AgentStep(tool=_CLAIM, ok=True, duration_ms=1),
        AgentStep(tool=_WRITE, ok=True, duration_ms=1),
        AgentStep(tool="serper_search", ok=False, duration_ms=1,
                  error="Session not found", transport_ko=True)])
    assert ("complete", True, "r-NEUF") in b.appels and b.erreur is None


