"""La reprise d'un tour coupé au plafond de complétion (`finish_reason: length`).

Vécu le 15/11/2026 (minimax-m3 via OpenRouter) : au 7e tour,
99 avis BODACC en contexte, le modèle a voulu tout classer dans UNE réponse — 41 251
caractères, 16 000 jetons, `length` après 163 s. La boucle s'arrêtait en
`fin_anormale` sur 6 pas des 90 permis, sans rien écrire.

Ce que ces bancs tiennent :
- une coupe se REPREND : le texte du tour se rejoue, ses appels COMPLETS s'exécutent, le
  modèle lit une consigne de reprise, et le déroulé continue jusqu'à sa vraie fin ;
- un appel à demi écrit quitte le fil rejoué, n'est jamais exécuté, et la consigne le dit ;
- la borne : au-delà de `REPRISES_SUR_COUPE` coupes CONSÉCUTIVES, c'est la `fin_anormale`
  d'avant ; un tour normal remet le compte à zéro ;
- une borne du déroulé (tours, jetons) atteinte juste après une coupe ne conclut pas à
  sa place : la fin reste `fin_anormale (length)` ;
- `stop` est inchangé ; hors OpenRouter (Scaleway, Mistral) et sur la voie Anthropic
  (`max_tokens`), la coupe reste terminale — les bancs de `test_fins_et_arguments.py`
  et `test_fins_anthropic.py` le tiennent sans changement.
"""
from __future__ import annotations

import copy
import json

import pytest

from oto_runner import agent_llm as A
from oto_runner import agent_llm_openai as P
from oto_runner import agent_runtime
from oto_runner.agent_runtime import REPRISES_SUR_COUPE, AgentSpec
from tests.test_fins_anthropic import _Sdk
from tests.test_fins_anthropic import _texte as _texte_anthropic
from tests.test_fins_et_arguments import _Outils, _Resp, _reponse, _texte, _travail_servi

SPEC = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=10)
_OR = "https://openrouter.ai/api/v1"


@pytest.fixture(autouse=True)
def _openrouter(monkeypatch):
    """La voie du cas vécu : la reprise n'existe que sur OpenRouter."""
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", _OR)


def _appels(*paires):
    """Un message assistant à plusieurs appels : `(id, arguments bruts)`."""
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": i, "type": "function",
                            "function": {"name": "data_rows", "arguments": a}}
                           for i, a in paires]}


def _servir(monkeypatch, *reponses):
    """Comme `test_fins_et_arguments._servir`, mais GARDE chaque corps envoyé : le fil
    rejoué au fournisseur se vérifie, pas seulement ce que la boucle conclut."""
    file, corps = list(reponses), []

    def post(*_a, json=None, **_k):  # noqa: A002 — le nom du paramètre de `requests`
        corps.append(copy.deepcopy(json))
        return _Resp(file.pop(0))

    monkeypatch.setattr(P.requests, "post", post)
    return corps


def _run(spec=SPEC, outils=None):
    evenements, fil = [], []
    res = agent_runtime.run(spec, outils or _Outils(), P, prompt="go", api_key="k",
                            on_event=lambda ev, champs: evenements.append((ev, champs)),
                            on_turn=lambda role, neutre, brut: fil.append((role, brut)))
    return res, evenements, fil


def _evs(evenements, nom):
    return [c for ev, c in evenements if ev == nom]


# ── 1. La coupe se reprend ───────────────────────────────────────────────────

def test_une_coupe_se_reprend_puis_le_deroule_conclut(monkeypatch):
    corps = _servir(monkeypatch, _reponse(_texte("Classement : 1. CS Pneus → A, 2. "), "length"),
                    _reponse(_texte("12 lignes écrites."), "stop"))
    res, evenements, fil = _run()
    assert res.stopped == "end_turn" and res.reply == "12 lignes écrites."
    assert res.defaut is None
    (reprise,) = _evs(evenements, "reprise_longueur")
    assert reprise["finish_reason"] == "length" and reprise["tour"] == 1
    assert reprise["reprise"] == 1 and reprise["reprises_max"] == REPRISES_SUR_COUPE
    assert reprise["appels_gardes"] == [] and reprise["appels_abandonnes"] == 0
    assert _evs(evenements, "fin_anormale") == []
    # Le fil du second tour : le texte coupé rejoué, puis la consigne, en dernier.
    second = corps[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "Classement : 1. CS Pneus → A, 2. "}
    assert second[-1]["role"] == "user" and second[-1]["content"] == reprise["consigne"]
    assert "cut off at the output limit" in reprise["consigne"]
    assert "smaller batches" in reprise["consigne"]
    # Le fil du backend porte les mêmes messages : une reprise du run relit ce fil.
    assert [r for r, _ in fil] == ["user", "assistant", "user", "assistant"]
    assert fil[2][1] == second[-1]


def test_le_texte_d_un_tour_coupe_n_est_jamais_la_reponse(monkeypatch):
    _servir(monkeypatch, _reponse(_texte("la fiche est pr"), "length"),
            _reponse(_texte(""), "stop"))
    res, _, _ = _run()
    assert res.reply == "" and res.stopped == "no_reply"


def test_un_appel_a_demi_ecrit_est_ecarte_et_les_complets_s_executent(monkeypatch):
    coupe = _appels(("c1", '{"limit": 3}'), ("c2", '{"limit": 4'))
    coupe["content"] = "J'écris le premier lot."
    corps = _servir(monkeypatch, _reponse(coupe, "length"),
                    _reponse(_texte("fini"), "stop"))
    outils = _Outils()
    res, evenements, _ = _run(outils=outils)
    assert outils.appels == [("data_rows", {"limit": 3})], "seul l'appel complet agit"
    assert res.stopped == "end_turn" and res.reply == "fini"
    (reprise,) = _evs(evenements, "reprise_longueur")
    assert reprise["appels_gardes"] == ["data_rows"] and reprise["appels_abandonnes"] == 1
    assert "1 tool call(s) cut mid-way were dropped" in reprise["consigne"]
    second = corps[1]["messages"]
    assistant, outil, consigne = second[-3:]
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["c1"], "l'appel coupé quitte le fil"
    assert assistant["content"] == "J'écris le premier lot."
    assert outil["role"] == "tool" and outil["tool_call_id"] == "c1"
    assert consigne["role"] == "user" and consigne["content"] == reprise["consigne"]
    # Le brut du tour, au journal, reste ENTIER : l'appel coupé s'y relit.
    (modele, _) = _evs(evenements, "modele")
    assert len(modele["brut"]["tool_calls"]) == 2


def test_un_tour_coupe_sans_texte_ni_appel_complet_ne_rejoue_aucun_message_vide(monkeypatch):
    corps = _servir(monkeypatch, _reponse(_appels(("c1", '{"rows": [{"siren"')), "length"),
                    _reponse(_texte("fini"), "stop"))
    outils = _Outils()
    res, _, _ = _run(outils=outils)
    assert outils.appels == [] and res.stopped == "end_turn"
    second = corps[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "user"]
    assert not any(m["role"] in ("assistant", "tool") for m in second)
    # Rien du tour coupé n'est dans le fil : la consigne ne dit pas « continue ».
    assert "nothing of it was kept" in second[-1]["content"]
    assert "Restart that step" in second[-1]["content"]
    assert "Continue from where you stopped" not in second[-1]["content"]


# ── 2. La borne ──────────────────────────────────────────────────────────────

def test_au_dela_de_la_borne_c_est_la_fin_anormale_d_avant(monkeypatch):
    coupe = _reponse(_texte("la fiche est pr"), "length")
    corps = _servir(monkeypatch, *[coupe] * (REPRISES_SUR_COUPE + 1),
                    _reponse(_texte("fini"), "stop"))
    res, evenements, _ = _run()
    assert len(corps) == REPRISES_SUR_COUPE + 1, "aucun tour après la borne"
    assert res.stopped == "fin_anormale" and res.reply == ""
    assert res.defaut == {"forme": "fin_anormale", "finish_reason": "length"}
    assert len(_evs(evenements, "reprise_longueur")) == REPRISES_SUR_COUPE
    (fin,) = _evs(evenements, "fin_anormale")
    assert fin["finish_reason"] == "length" and fin["tour"] == REPRISES_SUR_COUPE + 1


def test_un_tour_normal_remet_le_compte_des_reprises_a_zero(monkeypatch):
    coupe = _reponse(_texte("trop long"), "length")
    appel = _reponse(_appels(("c1", '{"limit": 3}')), "tool_calls")
    _servir(monkeypatch, *[coupe] * REPRISES_SUR_COUPE, appel,
            *[coupe] * REPRISES_SUR_COUPE, _reponse(_texte("fini"), "stop"))
    res, evenements, _ = _run()
    assert res.stopped == "end_turn" and res.reply == "fini"
    assert [c["reprise"] for c in _evs(evenements, "reprise_longueur")] == [1, 2, 1, 2]


def test_une_coupe_au_dernier_tour_permis_reste_une_fin_anormale(monkeypatch):
    """`max_steps` = 1 : deux tours. Le second, coupé, est repris — et il n'y a plus de
    tour pour le relever : la fin est la coupe, pas un `max_steps` ordinaire."""
    _servir(monkeypatch, _reponse(_appels(("c1", '{"limit": 3}')), "tool_calls"),
            _reponse(_texte("trop long"), "length"))
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=1)
    res, evenements, _ = _run(spec=spec)
    assert res.stopped == "fin_anormale"
    assert res.defaut == {"forme": "fin_anormale", "finish_reason": "length"}
    (fin,) = _evs(evenements, "fin_anormale")
    assert fin["borne"] == "max_steps"


def test_une_coupe_qui_franchit_la_borne_de_jetons_reste_une_fin_anormale(monkeypatch):
    corps = _servir(monkeypatch, _reponse(_texte("trop long"), "length"),
                    _reponse(_texte("fini"), "stop"))
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=10, max_tokens=50)
    res, evenements, _ = _run(spec=spec)
    assert len(corps) == 1
    assert res.stopped == "fin_anormale" and res.reply == ""
    assert res.defaut == {"forme": "fin_anormale", "finish_reason": "length"}
    assert _evs(evenements, "reprise_longueur") == []
    (fin,) = _evs(evenements, "fin_anormale")
    assert fin["borne"] == "max_tokens"


# ── 3. Ce qui ne change pas ──────────────────────────────────────────────────

def test_une_fin_stop_est_inchangee(monkeypatch):
    """Garde négative : prouvée par son vert seul, sans chute."""
    corps = _servir(monkeypatch, _reponse(_texte("fiche écrite"), "stop"))
    res, evenements, _ = _run()
    assert res.stopped == "end_turn" and res.reply == "fiche écrite"
    assert len(corps) == 1 and _evs(evenements, "reprise_longueur") == []


def test_le_tour_coupe_garde_son_nom_et_son_motif(monkeypatch):
    """Le contrat du `Turn` ne bouge pas : c'est la boucle qui reprend."""
    _servir(monkeypatch, _reponse(_texte("la fiche est pr"), "length"))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.stop_reason == "fin_anormale"
    assert turn.defaut == {"forme": "fin_anormale", "finish_reason": "length"}
    message, appels, abandonnes = P.reprise_sur_coupe(turn)
    assert message == {"role": "assistant", "content": "la fiche est pr"}
    assert appels == () and abandonnes == 0


def test_seule_la_coupe_au_plafond_se_reprend(monkeypatch):
    """`model_length` (la fenêtre de contexte, Mistral), `error`, une fin absente : rien
    ne dit qu'un tour de plus les relèverait."""
    for fin in ("model_length", "error", None, "inattendue"):
        _servir(monkeypatch, _reponse(_texte("x"), fin))
        turn = P.complete(system="s", messages=[], tools=[], api_key="k")
        assert P.reprise_sur_coupe(turn) is None, fin


@pytest.mark.parametrize("base", ["https://api.scaleway.ai/v1", "https://api.mistral.ai/v1"])
def test_hors_openrouter_la_coupe_reste_terminale(monkeypatch, base):
    """L'ordre `tool` → `user` qu'une reprise pose a été refusé par l'API Mistral :
    hors OpenRouter, le comportement d'avant, à l'octet."""
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", base)
    corps = _servir(monkeypatch, _reponse(_appels(("c1", '{"limit": 3}')), "length"),
                    _reponse(_texte("fini"), "stop"))
    outils = _Outils()
    res, evenements, _ = _run(outils=outils)
    assert len(corps) == 1 and outils.appels == []
    assert res.stopped == "fin_anormale" and _evs(evenements, "reprise_longueur") == []


def test_la_voie_anthropic_n_a_pas_de_reprise(monkeypatch):
    """`max_tokens` y reste terminal : le crochet n'existe que sur la voie Chat
    Completions, où le fil se rejoue sans blocs de pensée signés."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-plateforme")
    sdk = _Sdk(("max_tokens", [_texte_anthropic("la fiche est pr")]),
               ("end_turn", [_texte_anthropic("fini")]))
    monkeypatch.setattr(A, "_sdk", lambda: sdk)
    assert not hasattr(A, "reprise_sur_coupe")
    res = agent_runtime.run(SPEC, _Outils(), A, prompt="go", api_key="k")
    assert res.stopped == "fin_anormale" and len(sdk.file) == 1, "aucun second tour"


# ── 4. Jusqu'à la conclusion du worker ───────────────────────────────────────

def test_une_coupe_reprise_conclut_le_travail_done(monkeypatch):
    b, clotures = _travail_servi(monkeypatch, _reponse(_texte("la fiche est pr"), "length"),
                                 _reponse(_texte("fiche écrite"), "stop"))
    assert [c["outcome"] for c in clotures] == ["done"]
    assert b.conclusion["ok"] is True and b.conclusion["result"]["stopped"] == "end_turn"
    assert b.conclusion["result"]["usage_output"] == 40, "le tour coupé est compté"
    assert json.dumps(b.conclusion["result"]).count("fin_anormale") == 0
