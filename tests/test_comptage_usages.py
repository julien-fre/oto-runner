"""Le comptage fidèle des usages : un usage que le fournisseur ne déclare pas n'est pas zéro.

Arbitrages du 13/09/2026 (référent plan) :
- compteurs par poste, avec la couverture des tours ; inconnu = `None` ;
- un zéro n'est posé que là où une PHRASE du contrat du fournisseur le pose (Mistral,
  cache du chat) — une annotation de schéma `default: 0` n'y suffit pas ;
- l'entrée TOTALE déclarée reste distincte du non-caché exact, et ne se publie jamais
  à sa place ;
- une borne de jetons DEMANDÉE qui ne peut plus être suivie arrête le déroulé avant le
  tour suivant, nommément ; des caches inconnus seuls ne l'aveuglent pas ; sans borne,
  le résultat partiel le dit ;
- un usage connu n'est jamais effacé par un poste inconnu voisin.

Ces bancs tiennent sur le code d'avant le lot — ils passent par ses chemins publics
(`complete`, la boucle, le worker) et ne supposent aucun nom neuf : leur rouge y est
celui du défaut, un zéro fabriqué.
"""
from __future__ import annotations

import logging
import types

import pytest

from oto_runner import agent_conversations as C
from oto_runner import agent_llm_openai as O
from oto_runner import agent_runtime, conclusion
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult, AgentSpec
from oto_runner.bilan import ecrire_bilan
from oto_runner.file_de_travail import SansFile
from oto_runner.llm_types import ToolCall, Turn
from tests.test_agent_runtime import FauxProvider, FauxTransport
from tests.test_bilan import BackendBilan
from tests.test_bilan import _spec as _spec_bilan
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

MISTRAL, SCALEWAY = "https://api.mistral.ai/v1", "https://api.scaleway.ai/v1"


def _tour(**usage):
    """Un tour qui APPELLE un outil : la boucle continue après lui."""
    return Turn(text="", tool_calls=(ToolCall(id="t0", name="data_rows", arguments={}),),
                stop_reason="tool_use", raw_content=[], usage=usage)


def _fin(**usage):
    return Turn(text="fini", tool_calls=(), stop_reason="end_turn", raw_content=[], usage=usage)


class _Resp:
    def __init__(self, payload):
        self._payload, self.status_code, self.text = payload, 200, ""

    def json(self):
        return self._payload


def _usage_servi(monkeypatch, base, usage):
    """L'usage d'un tour tel que le VRAI transport Chat Completions le rend."""
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", base)
    reponse = {"choices": [{"message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}]}
    if usage is not None:
        reponse["usage"] = usage
    monkeypatch.setattr(O.requests, "post", lambda *a, **k: _Resp(reponse))
    return O.complete(system="s", messages=[], tools=[], api_key="k").usage


# ── Le contrat de chaque fournisseur ─────────────────────────────────────────

@pytest.mark.parametrize("base,attendu", [
    (MISTRAL, {"input_total_tokens": 1000, "input_tokens": 1000,
               "cache_read_input_tokens": 0, "output_tokens": 30}),
    (SCALEWAY, {"input_total_tokens": 1000, "output_tokens": 30}),
])
def test_un_cache_absent_suit_le_contrat_du_fournisseur(monkeypatch, base, attendu):
    """Mistral : « If the API doesn't serve tokens from cache, `cached_tokens` is `0` or
    omitted » — un zéro documenté. Scaleway ne rapporte aucun cache sur le chat : le cache
    est inconnu, le non-caché avec lui. L'entrée totale reste connue dans les deux cas."""
    assert _usage_servi(monkeypatch, base, {"prompt_tokens": 1000, "completion_tokens": 30}) == attendu


def test_un_cache_declare_separe_le_non_cache_de_l_entree_totale(monkeypatch):
    assert _usage_servi(monkeypatch, SCALEWAY, {
        "prompt_tokens": 2812, "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 2688}}) == {
        "input_total_tokens": 2812, "input_tokens": 124,
        "cache_read_input_tokens": 2688, "output_tokens": 3}


@pytest.mark.parametrize("usage,attendu", [(None, {}), ({}, {}),
                                           ({"completion_tokens": 7}, {"output_tokens": 7})])
def test_rien_n_est_fabrique_quand_le_fournisseur_ne_declare_pas(monkeypatch, usage, attendu):
    assert _usage_servi(monkeypatch, SCALEWAY, usage) == attendu


def test_conversations_declare_une_entree_totale_et_un_non_cache_inconnu():
    res = C._parse({"outputs": [{"type": "message.output", "content": "ok"}],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 50}},
                   tools=(), demande="m")
    assert (res.usage.get("input_total_tokens"), res.usage.get("output_tokens")) == (500, 50)
    assert res.usage.get("input_tokens", "absent") is None, "le non-caché reste inconnu"


# ── La boucle et sa borne ────────────────────────────────────────────────────

def test_une_borne_demandee_qui_ne_peut_plus_etre_suivie_arrete_avant_le_tour_suivant():
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=10,
                     max_tokens=100_000)
    p = FauxProvider([_tour(input_tokens=100, output_tokens=10),
                      _tour(input_tokens=100),                       # sortie non déclarée
                      _fin(input_tokens=1, output_tokens=1)])
    transport, evenements = FauxTransport(), []
    res = agent_runtime.run(spec, transport, p, prompt="go",
                            on_event=lambda ev, champs: evenements.append((ev, champs)))
    assert res.stopped == "max_tokens_non_mesurable"
    assert len(p.file) == 1, "le tour suivant n'a pas été joué"
    assert len(transport.appels) == 1, "l'appel du tour aveugle n'est pas exécuté"
    (borne,) = [c for ev, c in evenements if ev == "borne_non_suivie"]
    assert borne["manque"] == ["sortie"] and borne["tour"] == 2
    assert res.usage["input_tokens"] == 200, "l'entrée connue n'est pas effacée"
    assert res.usage["output_tokens"] is None


def test_des_caches_inconnus_seuls_ne_rendent_pas_la_borne_aveugle():
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=10, max_tokens=1000)
    p = FauxProvider([_tour(input_total_tokens=600, output_tokens=100),   # 700
                      _tour(input_total_tokens=400, output_tokens=100),   # 1200 — dépasse
                      _fin(input_total_tokens=1, output_tokens=1)])
    res = agent_runtime.run(spec, FauxTransport(), p, prompt="go")
    assert res.stopped == "max_tokens", "l'entrée totale connue suit la borne, en la majorant"
    assert len(p.file) == 1


def test_sans_borne_demandee_le_deroule_continue_et_le_resultat_dit_le_manque():
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=10)
    p = FauxProvider([_tour(input_tokens=100), _fin(input_tokens=5, output_tokens=5)])
    res = agent_runtime.run(spec, FauxTransport(), p, prompt="go")
    assert res.stopped == "end_turn"
    assert res.usage["output_tokens"] is None
    resultat = conclusion.resultat_declare(res, "m")
    assert resultat["usage_tokens"] is None, "jamais un total présenté comme complet"
    assert resultat["usage_input"] == 105
    assert (resultat.get("usage_couverture") or {}).get("tours") == 2


# ── Les deux cas asymétriques, par le vrai transport et la vraie boucle ──────
#
# Une seule composante connue, et la somme n'est plus mesurable : la borne s'arrête
# dès que l'entrée (totale ou non cachée) OU la sortie manque.

def _servir_appel_puis_fin(monkeypatch, usage_du_premier_tour):
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", SCALEWAY)
    reponses = [
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "data_rows", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}], "usage": usage_du_premier_tour},
        {"choices": [{"message": {"role": "assistant", "content": "fini"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 50, "completion_tokens": 5}},
    ]
    monkeypatch.setattr(O.requests, "post", lambda *a, **k: _Resp(reponses.pop(0)))
    return reponses


ASYMETRIES = [
    pytest.param({"prompt_tokens": 100}, "sortie", "input_total_tokens", 100, 150, "output_tokens",
                 id="entree-totale-connue-sortie-absente"),
    pytest.param({"completion_tokens": 10}, "entrée", "output_tokens", 10, 15, "input_total_tokens",
                 id="entree-absente-sortie-connue"),
]
CHAMP = {"input_total_tokens": "usage_input_total", "output_tokens": "usage_output"}


@pytest.mark.parametrize("usage,manque,connu,valeur,somme,inconnu", ASYMETRIES)
def test_une_seule_composante_connue_arrete_la_borne_demandee(monkeypatch, usage, manque, connu,
                                                              valeur, somme, inconnu):
    reponses = _servir_appel_puis_fin(monkeypatch, usage)
    transport, evenements = FauxTransport(), []
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=5, max_tokens=100_000)
    res = agent_runtime.run(spec, transport, O, prompt="go", api_key="k",
                            on_event=lambda ev, champs: evenements.append((ev, champs)))
    assert res.stopped == "max_tokens_non_mesurable"
    assert transport.appels == [] and len(reponses) == 1, "ni l'appel du tour, ni le tour suivant"
    assert [c.get("manque") for ev, c in evenements if ev == "borne_non_suivie"] == [[manque]]
    assert res.usage.get(connu) == valeur, "la composante connue est conservée"
    assert res.usage.get(inconnu, "absent") is None


@pytest.mark.parametrize("usage,manque,connu,valeur,somme,inconnu", ASYMETRIES)
def test_sans_borne_la_composante_manquante_reste_inconnue(monkeypatch, usage, manque, connu,
                                                           valeur, somme, inconnu):
    reponses = _servir_appel_puis_fin(monkeypatch, usage)
    transport = FauxTransport()
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=5)
    res = agent_runtime.run(spec, transport, O, prompt="go", api_key="k")
    assert res.stopped == "end_turn" and len(transport.appels) == 1 and reponses == []
    assert res.usage.get(inconnu, "absent") is None, "la composante manquante n'est pas un zéro"
    assert res.usage.get(connu) == somme, "la composante connue, sommée sur les deux tours"
    resultat = conclusion.resultat_declare(res, "m")
    assert resultat["usage_tokens"] is None
    assert resultat.get(CHAMP[connu]) == somme


# ── Jusqu'au travail conclu, hébergé et direct ───────────────────────────────

def _travail(monkeypatch, file=None):
    """Un travail dont la boucle rend l'entrée TOTALE et la sortie, sans le non-caché."""
    def faux_run(spec, transport, provider, **_):
        res = AgentResult(reply="fini", stopped="end_turn",
                          usage={"input_tokens": None, "input_total_tokens": 900,
                                 "output_tokens": 40, "cache_creation_input_tokens": None,
                                 "cache_read_input_tokens": None})
        res.couverture = {"tours": 1}   # posé à part : la doublure tient sur le code d'avant
        return res

    monkeypatch.setattr(W, "McpSession", FauxMcp)
    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    b = FauxBackend()
    W._un_travail(b, _job("start"), types.SimpleNamespace(__name__="agent_llm_openai",
                                                           model=lambda: "m"), file=file)
    return b


def test_en_heberge_le_resultat_porte_l_inconnu_et_sa_couverture(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        b = _travail(monkeypatch)
    (resultat,) = [appel[1] for appel in b.appels if appel[0] == "complete_result"]
    assert resultat["usage_tokens"] is None, "le non-caché inconnu ne s'additionne pas à zéro"
    assert resultat.get("usage_input_total") == 900
    assert resultat.get("usage_couverture") == {"tours": 1}
    assert "jetons : inconnus" in caplog.text


def test_en_direct_le_travail_conclu_porte_le_meme_resultat(monkeypatch):
    file = SansFile()
    _travail(monkeypatch, file=file)
    (conclu,) = file.conclus.values()
    assert conclu["result"]["usage_tokens"] is None
    assert conclu["result"].get("usage_input_total") == 900


def test_une_charge_reduite_pour_sa_taille_garde_sa_couverture():
    """Le résumé d'une conclusion refusée pour sa taille gardait les seuls scalaires : la
    couverture tombait, et le résultat réduit se lisait « non attesté » à tort."""
    from oto_runner.backend import BackendError

    class _File:
        def __init__(self):
            self.envois: list = []

        def complete(self, job_id, ok, error=None, run_id=None, result=None):
            self.envois.append(result)
            if len(self.envois) == 1:
                refus = BackendError("/api/me/runner/jobs → 400 : result_too_large", status=400)
                refus.code = "result_too_large"
                raise refus
            return {"ok": True}

    couverture = {"tours": 2, "declares": {"output_tokens": 2}, "sommes": {"output_tokens": 40}}
    file = _File()
    conclusion.rendre(file, 7, ok=True, error=None, run_id="r",
                      result={"usage_tokens": None, "usage_input_total": 900,
                              "usage_couverture": couverture, "tool_counts": {"data_rows": 1}},
                      note=lambda *a, **k: None)
    assert len(file.envois) == 2
    assert file.envois[1].get("usage_couverture") == couverture, "la couverture atteste encore"
    assert "tool_counts" not in file.envois[1]


# ── Flotte et bilan ──────────────────────────────────────────────────────────

def test_un_budget_de_flotte_qui_ne_peut_plus_etre_suivi_arrete_l_enfilement():
    from tests.test_fleet import FauxBackend as FileDeFlotte
    from tests.test_fleet import _run, _spec
    b = FileDeFlotte(counts=[100, 100, 100, 0], usage_par_job=None)
    bilan = _run(_spec(budget_tokens=10_000, ramp_seconds=0), b)
    assert bilan.arret.startswith("budget non suivable"), bilan.arret
    assert getattr(bilan, "jobs_usage_inconnu", 0) >= 1


def test_le_bilan_ne_presente_jamais_un_total_incomplet():
    jobs = {1: {"status": "done", "result": {"usage_tokens": 4000}},
            2: {"status": "done", "result": {"usage_tokens": None}}}
    bilan = ecrire_bilan(_spec_bilan(), BackendBilan(restantes=28), jobs,
                         lignes_initiales=30, secondes=60)
    jetons = bilan["jetons"]
    assert jetons["total"] is None and jetons.get("connus") == 4000
    assert jetons.get("travaux_sans_usage") == 1 and jetons["par_job"] is None
