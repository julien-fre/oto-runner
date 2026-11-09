"""Le bac à sable du backend est plein : le runner attend, le modèle ne paie pas.

Vu le 09/11/2026 : 66 refus 429 `sandbox_busy` en prod dans la journée. Le backend
n'accepte qu'une exécution de recette-code par compte (et 2 en tout sur la box) ; un
agent hébergé dont plusieurs travaux tournent sous la même identité se voyait refuser
sa recette, lisait le refus et réessayait au tour suivant — un tour de modèle payé par
tentative. Le runner attend désormais le `Retry-After` et rejoue lui-même, borné.
"""
from __future__ import annotations

import pytest

from oto_runner import mcp as M
from oto_runner.mcp import McpSession

_OUTIL = "oto_recipe"
_OCCUPE = {"jsonrpc": "2.0", "id": 1, "error": {
    "code": -32602,
    "message": "sandbox_busy: this account already has 1 execution running in the "
               "sandbox; retry in 10 s, once it has finished.",
    "data": {"code": "sandbox_busy", "retry_after_seconds": 10}}}
_OCCUPE_TEXTE = {"jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": [
    {"type": "text", "text": "sandbox_busy: the sandbox is running its maximum of 2 "
                             "simultaneous executions; retry in 10 s."}]}}
_OK = {"jsonrpc": "2.0", "id": 1, "result": {
    "content": [{"type": "text", "text": '{"ok": true}'}],
    "structuredContent": {"ok": True}}}


def _session(monkeypatch, reponses):
    etat = {"call": 0, "pauses": []}

    def _post(self, corps, avec_entetes=False):
        methode = corps.get("method")
        if methode == "initialize":
            return ({"mcp-session-id": "s1"}, {}) if avec_entetes else {}
        if methode == "tools/list":
            return {"result": {"tools": [{"name": _OUTIL, "inputSchema": {
                "type": "object", "properties": {"op": {}}}}]}}
        if methode == "tools/call":
            etat["call"] += 1
            return reponses[min(etat["call"], len(reponses)) - 1]
        return {}

    monkeypatch.setattr(McpSession, "_post", _post)
    monkeypatch.setattr(M.time, "sleep", lambda s: etat["pauses"].append(s))
    return McpSession(url="http://x", token="t"), etat


def test_un_refus_passe_attend_le_retry_after_puis_rejoue(monkeypatch):
    s, etat = _session(monkeypatch, [_OCCUPE, _OCCUPE, _OK])
    texte, is_error, _ = s.call(_OUTIL, {"op": "run"})
    assert not is_error and '"ok": true' in texte
    assert etat["call"] == 3 and etat["pauses"] == [10, 10]


def test_le_refus_en_texte_de_resultat_est_reconnu(monkeypatch):
    s, etat = _session(monkeypatch, [_OCCUPE_TEXTE, _OK])
    _, is_error, _ = s.call(_OUTIL, {"op": "run"})
    assert not is_error and etat["call"] == 2


def test_l_attente_est_bornee_puis_le_refus_remonte(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_SANDBOX_WAIT_S", "25")
    s, etat = _session(monkeypatch, [_OCCUPE])
    texte, is_error, _ = s.call(_OUTIL, {"op": "run"})
    assert is_error and "sandbox_busy" in texte
    assert sum(etat["pauses"]) == 25 and etat["pauses"] == [10, 10, 5]


def test_zero_coupe_l_attente(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_SANDBOX_WAIT_S", "0")
    s, etat = _session(monkeypatch, [_OCCUPE, _OK])
    _, is_error, _ = s.call(_OUTIL, {"op": "run"})
    assert is_error and etat["call"] == 1 and etat["pauses"] == []


def test_le_delai_servi_est_borne(monkeypatch):
    trop = {"error": {"data": {"code": "sandbox_busy", "retry_after_seconds": 3600}}}
    assert M._bac_occupe(trop) == 60
    nul = {"error": {"data": {"code": "sandbox_busy", "retry_after_seconds": 0}}}
    assert M._bac_occupe(nul) == 10


def test_une_autre_erreur_n_attend_pas(monkeypatch):
    autre = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "not_found"}}
    s, etat = _session(monkeypatch, [autre, _OK])
    _, is_error, _ = s.call(_OUTIL, {"op": "run"})
    assert is_error and etat["call"] == 1 and etat["pauses"] == []


def test_une_valeur_illisible_leve(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_SANDBOX_WAIT_S", "trois minutes")
    with pytest.raises(ValueError):
        M.attente_bac_max()
