"""Une conclusion que la file ne prend pas ne fait pas rejouer un travail qui a conclu.

Banc du 13/09/2026 : `complete(ok=true)` refusé en 400 `result_too_large` levait
une BackendError que `_un_travail` prenait pour une MORT. `en_echec` re-clôturait
alors le run `done` en `failed` et rendait le travail en échec : re-filé, rejoué.
Ce que ces bancs tiennent :

1. le run n'est clos qu'UNE fois, avec sa vraie issue, et un travail qui a conclu
   n'est jamais rendu en échec pour autant ;
2. `result_too_large` — le code, pas tout 400 : renvoyée une fois, réduite à ses
   valeurs scalaires, compteurs d'usage compris, avec le refus borné ; un 400 de
   contrat garde son diagnostic et ne se présente pas comme une taille ;
3. réponse perdue (transport, 5xx) : la même conclusion, renvoyée telle quelle,
   bornée ; un 404 ensuite ne prouve pas qu'elle a été prise, et se dit comme tel ;
4. travail qui n'est plus à ce worker (404) : rien à rejouer ;
5. chaque refus s'écrit au journal, et ce qui reste non rendu se dit en erreur.

Ces bancs tiennent sur le code d'avant le lot : leur rouge y est celui du défaut.
"""
from __future__ import annotations

import json
import logging
import time
import types

import pytest

from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult
from oto_runner.backend import Backend, BackendError
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

_PROVIDER = types.SimpleNamespace(__name__="agent_llm_openai", model=lambda: "mistral-large-2512")


class _File(FauxBackend):
    """La file qui oppose, dans l'ordre, ces refus, puis accepte."""

    def __init__(self, *refus):
        super().__init__()
        self.refus, self.conclusions = list(refus), []

    def complete(self, job_id, ok, error=None, run_id=None, result=None):
        self.conclusions.append({"ok": ok, "error": error, "run_id": run_id, "result": result})
        if self.refus:
            raise self.refus.pop(0)
        return {"ok": True, "status": "done" if ok else "pending"}


def _refus(status, texte, code=None):
    e = BackendError(f"/api/me/runner/jobs → {status} : {texte}", status=status)
    e.code = code   # posé à la main : la doublure tient aussi sur le code d'avant
    return e


TROP_GROS = lambda: _refus(400, "result_too_large", code="result_too_large")  # noqa: E731


def _brancher(monkeypatch, stopped):
    mcps, pauses = [], []

    class McpEspion(FauxMcp):
        def __init__(self, **kw):
            super().__init__(**kw)
            mcps.append(self)

    def faux_run(spec, transport, provider, **_):
        return AgentResult(reply="fini", stopped=stopped,
                           usage={"input_tokens": 700, "output_tokens": 30,
                                  "cache_read_input_tokens": 1200})

    monkeypatch.setattr(W, "McpSession", McpEspion)
    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    # `time.sleep` lui-même : la doublure doit tenir sur le code d'AVANT la correction,
    # sinon le rouge qu'elle montre est le sien.
    monkeypatch.setattr(time, "sleep", pauses.append)
    return mcps, pauses


def _conclure(monkeypatch, file, stopped="end_turn"):
    """Un travail ENTIER par le worker (`_un_travail`, là où la mort se rattrape)."""
    mcps, pauses = _brancher(monkeypatch, stopped)
    W._un_travail(file, _job("start"), _PROVIDER)
    (mcp,) = mcps
    return [a["outcome"] for n, a in mcp.appels if n == "run_finish"], pauses


# ── 1 et 2. la charge trop grosse, et elle seule ─────────────────────────────

def test_une_charge_trop_grosse_apres_succes_ne_rejoue_pas_le_travail(monkeypatch):
    file = _File(TROP_GROS())
    clotures, pauses = _conclure(monkeypatch, file)
    assert clotures == ["done"], "le run n'est clos qu'une fois, avec sa vraie issue"
    assert [c["ok"] for c in file.conclusions] == [True, True], \
        "jamais rendu en échec — donc jamais re-filé ni rejoué"
    premier, second = (c["result"] for c in file.conclusions)
    refus = second.pop("conclusion_refusee")
    assert refus["status"] == 400 and refus["code"] == "result_too_large"
    assert refus["octets"] == len(json.dumps(premier))
    assert second == {k: v for k, v in premier.items() if k != "tool_counts"}, \
        "les valeurs scalaires restent : une consommation connue ne devient pas zéro"
    assert second["usage_tokens"] == 730 and second["usage_cache_read"] == 1200
    assert pauses == []


def test_un_400_de_contrat_ne_se_presente_pas_comme_une_taille(monkeypatch, caplog):
    file = _File(_refus(400, "unknown_fields", code="unknown_fields"))
    with caplog.at_level(logging.ERROR, logger="oto_runner"):
        clotures, pauses = _conclure(monkeypatch, file)
    assert clotures == ["done"]
    assert len(file.conclusions) == 1, "un refus de contrat ne se rejoue pas"
    assert "NON rendue" in caplog.text and "unknown_fields" in caplog.text


def test_un_echec_nomme_dont_la_charge_est_trop_grosse_reste_un_echec_clos_une_fois(monkeypatch):
    file = _File(TROP_GROS())
    clotures, _ = _conclure(monkeypatch, file, stopped="appel_mal_encode")
    assert clotures == ["failed"]
    assert [c["ok"] for c in file.conclusions] == [False, False]
    assert file.conclusions[1]["error"] == file.conclusions[0]["error"]


def test_le_client_rest_porte_le_code_que_le_serveur_nomme(monkeypatch):
    class _Reponse:
        status_code, text = 400, "{}"
        content = b"{}"

        def json(self):
            return {"error": "result_too_large", "detail": "result > 4 Ko — un résumé, pas un contenu"}

    monkeypatch.setattr("oto_runner.backend.post_with_deadline", lambda url, **kw: _Reponse())
    with pytest.raises(BackendError) as e:
        # `_post`, la couture où le code se lit — pas un verbe de file, dont la forme
        # dépend du contrat de tentative (#196), étranger à ce lot.
        Backend(base="http://x", token="t")._post("/api/me/runner/jobs",
                                                   {"op": "complete", "job_id": 7, "ok": True})
    assert e.value.status == 400 and getattr(e.value, "code", None) == "result_too_large"


# ── 3 et 4. la réponse perdue, le travail perdu ──────────────────────────────

def test_une_reponse_perdue_se_renvoie_telle_quelle_et_bornee(monkeypatch):
    file = _File(_refus(None, "réseau : ReadTimeout"), _refus(502, "Bad Gateway"))
    clotures, pauses = _conclure(monkeypatch, file)
    assert clotures == ["done"]
    assert len(file.conclusions) == 3 and pauses == [2, 4]
    assert all(c == file.conclusions[0] for c in file.conclusions), "la même conclusion"


def test_reseau_puis_5xx_puis_taille_envoie_bien_le_resume(monkeypatch, caplog):
    file = _File(_refus(None, "réseau : ReadTimeout"), _refus(502, "Bad Gateway"), TROP_GROS())
    with caplog.at_level(logging.ERROR, logger="oto_runner"):
        clotures, pauses = _conclure(monkeypatch, file)
    assert clotures == ["done"] and pauses == [2, 4]
    assert len(file.conclusions) == 4, "le résumé préparé part — il n'est pas perdu en fin de boucle"
    assert "conclusion_refusee" in file.conclusions[-1]["result"]
    assert "NON rendue" not in caplog.text


def test_une_reponse_perdue_trois_fois_se_dit_sans_rejouer_le_travail(monkeypatch, caplog):
    file = _File(*[_refus(None, "réseau : ReadTimeout")] * 3)
    with caplog.at_level(logging.ERROR, logger="oto_runner"):
        clotures, pauses = _conclure(monkeypatch, file)
    assert clotures == ["done"]
    assert len(file.conclusions) == 3 and pauses == [2, 4]
    assert all(c["ok"] for c in file.conclusions)
    assert "NON rendue" in caplog.text


def test_un_404_apres_un_renvoi_ne_prouve_pas_que_la_conclusion_a_ete_prise(monkeypatch, caplog):
    file = _File(_refus(None, "réseau : ReadTimeout"), _refus(404, "job_not_found", code="job_not_found"))
    with caplog.at_level(logging.ERROR, logger="oto_runner"):
        clotures, pauses = _conclure(monkeypatch, file)
    assert clotures == ["done"] and len(file.conclusions) == 2 and pauses == [2]
    assert "NON rendue" in caplog.text and "PU être prise, rien ne le prouve" in caplog.text


def test_un_travail_qui_n_est_plus_a_ce_worker_ne_se_rejoue_pas(monkeypatch):
    file = _File(_refus(404, "job_not_found", code="job_not_found"))
    clotures, pauses = _conclure(monkeypatch, file)
    assert clotures == ["done"]
    assert len(file.conclusions) == 1 and pauses == []


# ── 5. le journal ────────────────────────────────────────────────────────────

def test_chaque_refus_s_ecrit_au_journal(monkeypatch):
    evenements: list = []
    journal_ = types.SimpleNamespace(ecrire=lambda ev, **c: evenements.append((ev, c)),
                                     evenement=None)
    _brancher(monkeypatch, "end_turn")
    file = _File(_refus(None, "réseau : ReadTimeout"), TROP_GROS())
    W._traiter(file, _job("start"), _PROVIDER, journal_=journal_)
    refus = [c for ev, c in evenements if ev == "conclusion_refusee"]
    assert [(r["essai"], r["status"], r["code"]) for r in refus] == \
        [(1, None, None), (2, 400, "result_too_large")]
    noms = [ev for ev, _ in evenements]
    assert noms.index("resultat") < noms.index("conclusion_refusee"), \
        "la preuve du résultat est écrite AVANT que la file ne la refuse"
