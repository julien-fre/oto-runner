"""La voie OpenRouter de l'adaptateur Chat Completions — sans réseau.

Ce qui se casse en silence si on ne le fige pas : l'hôte qui ne nomme pas son dépôt
(le worker sonderait une famille que personne ne demande), les hôtes épinglés qui ne
partent pas (le tour irait chez un hôte fp4, plus cher), un raisonnement envoyé sous le
nom que l'hôte ignore, un cache absent qui ferait refuser le règlement du tour, et le coût
rendu par OpenRouter qui n'arriverait pas au backend (le tour s'y facture sur lui).
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
    # aucun hôte épinglé ne le déclare : avec `require_parameters`, il les écarterait
    assert "prompt_cache_key" not in vu


def test_sans_hotes_epingles_la_collecte_reste_refusee(monkeypatch):
    vu, _ = _corps(monkeypatch)
    assert vu["provider"] == {"data_collection": "deny", "require_parameters": True}


def test_le_raisonnement_part_sous_reasoning(monkeypatch):
    vu, _ = _corps(monkeypatch, effort="medium", max_output_tokens=16000)
    assert vu["reasoning"] == {"effort": "medium"} and "reasoning_effort" not in vu
    vu, _ = _corps(monkeypatch, effort="none")
    assert vu["reasoning"] == {"effort": "none"}


def test_des_hotes_epingles_hors_openrouter_sont_refuses(monkeypatch):
    with pytest.raises(P.LlmUnavailable, match="OpenRouter"):
        _corps(monkeypatch, base="https://api.mistral.ai/v1", providers=("deepseek",))


def test_une_autre_voie_ne_recoit_aucun_routage(monkeypatch):
    vu, _ = _corps(monkeypatch, base="https://api.mistral.ai/v1", effort="high",
                   max_output_tokens=16000)
    assert "provider" not in vu and "prompt_cache_key" in vu
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


def test_le_tour_openrouter_se_regle_avec_son_entree_et_sa_sortie(monkeypatch):
    """Sur une clé partagée, chaque tour se règle (`op=meter`) : sans entrée, sans sortie
    ou sans coût, le backend refuse (`usage_unmeasured`) et le travail s'arrête. L'usage que
    rend OpenRouter, cache et coût compris, arrive entier au règlement."""
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", _OR)
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp({
        "model": "deepseek/deepseek-v4.1-flash-1017",
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "provider": "StepFun",
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                  "prompt_tokens_details": {"cached_tokens": 700}, "cost": 0.0001}}))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")

    class _File:
        vu = []

        def meter(self, job_id, tour, charge):
            self.vu.append(charge)
            return {"ok": True}
    f = _File()
    mesurer = W._mesureur(f, {"id": 9, "meter_usage": True})
    mesurer(0, dict(turn.usage), turn.model)
    (charge,) = f.vu
    assert charge == {"model": "deepseek/deepseek-v4.1-flash-1017", "usage_input": 300,
                      "usage_output": 50, "usage_cache_write": None,
                      "usage_cache_read": 700, "usage_cost_usd": 0.0001,
                      "usage_host": "StepFun"}


def test_sans_cout_rendu_aucun_cout_n_est_invente(monkeypatch):
    """Un fournisseur qui ne rend pas de coût (Mistral, Scaleway) : rien n'est posé, et le
    règlement part comme avant, à l'octet."""
    _, turn = _corps(monkeypatch)
    assert "cost_usd" not in turn.usage

    class _File:
        vu = []

        def meter(self, job_id, tour, charge):
            self.vu.append(charge)
    f = _File()
    W._mesureur(f, {"id": 9, "meter_usage": True})(0, dict(turn.usage), "m")
    assert "usage_cost_usd" not in f.vu[0] and "usage_host" not in f.vu[0]
    for brut in (True, "0.1", None):
        assert "cost_usd" not in P.usage_declare({"prompt_tokens": 1, "cost": brut})
