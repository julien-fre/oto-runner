"""La voie OpenRouter de l'adaptateur Chat Completions — sans réseau.

Ce qui se casse en silence si on ne le fige pas : l'hôte qui ne nomme pas son dépôt
(le worker sonderait une famille que personne ne demande), les hôtes épinglés qui ne
partent pas (le tour irait chez un hôte fp4 plus cher que la ligne du barème), un
raisonnement envoyé sous le nom que l'hôte ignore, et un cache absent qui ferait
refuser le règlement du tour.
"""
from __future__ import annotations

import json

import pytest

from oto_runner import agent_llm_openai as P
from oto_runner import worker as W

_OR = "https://openrouter.ai/api/v1"


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self._payload


def _corps(monkeypatch, base=_OR, **kw):
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", base)
    vu = {}

    def post(url, json=None, **_):
        vu.update(json)
        vu["_url"] = url
        return _Resp({"choices": [{"message": {"role": "assistant", "content": "ok"},
                                   "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    monkeypatch.setattr(P.requests, "post", post)
    turn = P.complete(system="s", messages=[], tools=[], api_key="k", **kw)
    return vu, turn


def test_l_hote_openrouter_nomme_son_depot(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", _OR)
    assert P.depot() == "openrouter"


def test_les_hotes_epingles_partent_seuls_et_sans_collecte(monkeypatch):
    vu, _ = _corps(monkeypatch, providers=("deepinfra/fp8", "deepseek"))
    assert vu["_url"] == _OR + "/chat/completions"
    assert vu["provider"] == {"order": ["deepinfra/fp8", "deepseek"],
                              "only": ["deepinfra/fp8", "deepseek"],
                              "allow_fallbacks": True, "data_collection": "deny",
                              "require_parameters": True}
    assert vu["usage"] == {"include": True}


def test_sans_hotes_epingles_la_collecte_reste_refusee(monkeypatch):
    vu, _ = _corps(monkeypatch)
    assert vu["provider"] == {"data_collection": "deny", "require_parameters": True}


def test_le_raisonnement_part_sous_reasoning(monkeypatch):
    vu, _ = _corps(monkeypatch, effort="medium", max_output_tokens=16000)
    assert vu["reasoning"] == {"effort": "medium"} and "reasoning_effort" not in vu
    vu, _ = _corps(monkeypatch, effort="none")
    assert vu["reasoning"] == {"enabled": False}


def test_des_hotes_epingles_hors_openrouter_sont_refuses(monkeypatch):
    with pytest.raises(P.LlmUnavailable, match="OpenRouter"):
        _corps(monkeypatch, base="https://api.mistral.ai/v1", providers=("deepseek",))


def test_une_autre_voie_ne_recoit_aucun_routage(monkeypatch):
    vu, _ = _corps(monkeypatch, base="https://api.mistral.ai/v1", effort="high",
                   max_output_tokens=16000)
    assert "provider" not in vu and "usage" not in vu
    assert vu["reasoning_effort"] == "high"


def test_un_cache_absent_se_lit_zero_et_le_tour_se_regle(monkeypatch):
    """Sans `cached_tokens`, le non-caché resterait inconnu : le backend refuserait le
    règlement et le travail s'arrêterait. Zéro = toute l'entrée au plein tarif."""
    _, turn = _corps(monkeypatch)
    assert turn.usage["cache_read_input_tokens"] == 0
    assert turn.usage["input_tokens"] == 100 and turn.usage["output_tokens"] == 20


def test_le_travail_porte_ses_hotes_jusqu_a_la_spec():
    p = {"model": "deepseek/deepseek-v4.1-flash-1017", "model_family": "openrouter",
         "providers": ["deepinfra/fp8", " ", 3, "deepseek"], "instruction": "x"}
    spec = W._spec_du_job({"id": 1, "payload": p, "procedure": "p"})
    assert spec.providers == ("deepinfra/fp8", "deepseek")
