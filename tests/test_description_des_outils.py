"""La description d'un outil servie au modèle : bornée outil par outil par le réglage du travail, toute coupe DITE.

⚠️ Ce fichier existe parce que le runner coupait EN SILENCE chaque description à 1 024
caractères, une borne reprise du prototype et jamais mesurée. Pour `data_write` (7 865
caractères), l'agent recevait l'exemple `"comment": "@keep"` et jamais la règle « seul dans
la couche », ni `@empty`, ni l'écriture par `id` : 14 cases « @keep — … » sur le lot 01 du
vivier, le 11/09/2026. Servies ensuite entières (8 192), les 13 descriptions de la passe A
pesaient 34,4 k caractères et doublaient son coût. D'où la borne par outil, DÉCLARÉE par le
passage (jamais l'environnement) : `data_write` entière, les autres à 1 024 quand elle se tait.
"""
from __future__ import annotations

import json

import pytest
import requests
from requests.utils import get_encoding_from_headers

import oto_runner.mcp as M
from oto_runner import agent_runtime
from oto_runner.agent_runtime import AgentSpec
from oto_runner.declaration import load_spec, payload
from oto_runner.llm_types import Turn

_INIT = b'data: {"jsonrpc":"2.0","id":1,"result":{}}'


def _reponse(corps: bytes, entetes=None):
    r = requests.Response()
    r.status_code = 200
    r._content = corps
    r.headers["Content-Type"] = "text/event-stream"
    for k, v in (entetes or {}).items():
        r.headers[k] = v
    r.encoding = get_encoding_from_headers(r.headers)
    return r


def _servies(monkeypatch, descriptions, noms, reglage=None):
    """Une session dont le `tools/list` rend ces descriptions ; ce que `schemas` en sert."""
    outils = [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": {}}}
              for n, d in descriptions.items()]
    liste = ("data: " + json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"tools": outils}})).encode()

    def _post(url, **kw):
        if (kw.get("json") or {}).get("method") in ("initialize", "notifications/initialized"):
            return _reponse(_INIT, {"mcp-session-id": "s1"})
        return _reponse(liste)

    monkeypatch.setattr(M, "post_with_deadline", _post)
    s = M.McpSession(url="http://x", token="t", descriptions=reglage)
    return {x["name"]: x["description"] for x in s.schemas(frozenset(noms))}, s


def test_par_defaut_data_write_arrive_entiere(monkeypatch):
    servies, s = _servies(monkeypatch, {"data_write": "x" * 7865}, {"data_write"})
    assert len(servies["data_write"]) == 7865
    assert s.descriptions_servies == [{"outil": "data_write", "longueur": 7865, "servie": 7865}]


def test_par_defaut_les_autres_sont_coupees_a_1024_et_relevees(monkeypatch):
    servies, s = _servies(monkeypatch, {"oto_procedure": "y" * 20000, "fr_search": "z" * 1894},
                          {"oto_procedure", "fr_search"})
    assert len(servies["oto_procedure"]) == 1024 and len(servies["fr_search"]) == 1024
    assert {"outil": "oto_procedure", "longueur": 20000, "servie": 1024} in s.descriptions_servies


def test_une_description_courte_arrive_entiere(monkeypatch):
    servies, _ = _servies(monkeypatch, {"fr_bilans": "b" * 196}, {"fr_bilans"})
    assert len(servies["fr_bilans"]) == 196


def test_la_borne_se_declare(monkeypatch):
    servies, _ = _servies(monkeypatch, {"data_write": "x" * 7865, "oto_procedure": "y" * 20000},
                          {"data_write", "oto_procedure"}, reglage={"defaut": 8192, "entieres": []})
    assert len(servies["data_write"]) == 7865 and len(servies["oto_procedure"]) == 8192


def test_les_outils_entiers_se_declarent(monkeypatch):
    servies, _ = _servies(monkeypatch, {"data_write": "x" * 7865, "oto_procedure": "y" * 20000},
                          {"data_write", "oto_procedure"}, reglage={"entieres": ["oto_procedure"]})
    assert len(servies["oto_procedure"]) == 20000 and len(servies["data_write"]) == 1024


@pytest.mark.parametrize("brut", [{"defaut": 0}, {"defaut": "abc"}, {"defaut": True},
                                  {"entieres": "data_write"}, {"borne": 1024}, "1024"])
def test_un_reglage_invalide_leve(brut):
    with pytest.raises(ValueError):
        M.McpSession(url="http://x", token="t", descriptions=brut)


def test_l_environnement_ne_regle_rien(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_TOOL_DESC_MAX", "8192")
    servies, _ = _servies(monkeypatch, {"oto_procedure": "y" * 20000}, {"oto_procedure"})
    assert len(servies["oto_procedure"]) == 1024


def test_seuls_les_outils_permis_sont_releves(monkeypatch):
    _, s = _servies(monkeypatch, {"data_write": "x" * 10, "fr_search": "z" * 10}, {"data_write"})
    assert [d["outil"] for d in s.descriptions_servies] == ["data_write"]


def _declaration(tmp_path, reglage=""):
    y = tmp_path / "f.yaml"
    y.write_text("procedure: p\nnamespace: t\ntools: [oto_procedure]\ninput: 'go'\n" + reglage)
    return str(y)


def test_la_declaration_porte_le_reglage_jusqu_au_travail(tmp_path):
    p = payload(load_spec(_declaration(
        tmp_path, "descriptions_outils: {defaut: 2048, entieres: [data_write, data_rows]}\n")))
    assert p["descriptions_outils"] == {"defaut": 2048, "entieres": ["data_write", "data_rows"]}


def test_sans_reglage_le_travail_reste_identique(tmp_path):
    assert "descriptions_outils" not in payload(load_spec(_declaration(tmp_path)))


def test_un_reglage_faux_se_voit_a_la_lecture_de_la_declaration(tmp_path):
    with pytest.raises(ValueError):
        load_spec(_declaration(tmp_path, "descriptions_outils: {defaut: 0}\n"))


class _Transport:
    descriptions_servies = [{"outil": "data_rows", "longueur": 100, "servie": 100},
                            {"outil": "oto_procedure", "longueur": 20000, "servie": 1024}]

    def schemas(self, names):
        return [{"name": n, "description": "", "input_schema": {"type": "object"}}
                for n in sorted(names)]

    def call(self, name, arguments):
        return ("ok", False, ())


class _Provider:
    def complete(self, **kwargs):
        return Turn(text="fini", tool_calls=(), stop_reason="end_turn",
                    raw_content=[{"type": "text", "text": "fini"}])

    def user_message(self, text):
        return {"role": "user", "content": text}

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.raw_content}

    def tool_messages(self, results):
        return []

    def format_tools(self, schemas):
        return list(schemas)


def test_le_journal_du_travail_dit_la_coupe():
    evenements = []
    agent_runtime.run(AgentSpec(system="s", tools=frozenset({"data_rows", "oto_procedure"})),
                      _Transport(), _Provider(), prompt="go",
                      on_event=lambda ev, champs: evenements.append((ev, champs)))
    d = [c for ev, c in evenements if ev == "descriptions_outils"]
    assert d and d[0]["coupees"] == ["oto_procedure"] and len(d[0]["outils"]) == 2
