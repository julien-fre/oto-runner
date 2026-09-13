"""Un appel d'outil rendu en TEXTE n'est ni une conclusion, ni un appel à exécuter.

⚠️ Vécu le 13/09/2026, job 17275 (et deux runs directs, 06/09 et 09/09) : le
fournisseur rend un message sans `tool_calls`, terminé en `stop`, dont le contenu
est une liste — une partie `reference` qui nomme l'outil, puis les arguments en
texte. La boucle le prenait pour la réponse finale : travail `done` en deux pas,
rien d'écrit, la ligne jamais servie par la passe, et aucun compteur ne le voyait.

Ce que ces bancs tiennent :
1. le critère est STRUCTUREL et entier — il attrape la forme réelle (fixture
   expurgée du vrai tour), et rien de ce qui lui ressemble sans l'être ;
2. le JSON du texte n'est JAMAIS exécuté ;
3. le tour rejeté est COMPTÉ avant l'arrêt, et journalisé ;
4. le travail ne finit pas `done` : run clos `failed` (ce qui libère la ligne),
   `complete(ok=False)` avec le motif nommé.
"""
from __future__ import annotations

import json
import pathlib

from oto_runner import agent_llm_openai as P
from oto_runner import agent_runtime, conclusion
from oto_runner.agent_runtime import AgentResult, AgentSpec
from oto_runner.llm_types import ToolCall, Turn
from tests.test_agent_llm_openai import _Resp, _reponse
from tests.test_agent_runtime import FauxProvider, FauxTransport

FIXTURE = json.loads((pathlib.Path(__file__).parent / "fixtures"
                      / "appel_mal_encode_17275.json").read_text())
OUTILS = P.format_tools([{"name": n, "input_schema": {"type": "object"}}
                         for n in ("serper_search", "data_write", "data_claim_next")])


def _tour(message, monkeypatch, outils=OUTILS, finish="stop", usage=None):
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp(
        _reponse(message, finish=finish, usage=usage)))
    return P.complete(system="s", messages=[], tools=outils, api_key="k")


def _contenu(*parties):
    return {"role": "assistant", "tool_calls": None, "content": list(parties)}


REF = {"type": "reference", "reference_ids": ["serper_search"]}
ARGS = {"type": "text", "text": '{"query": "MAISON EXEMPLE", "num": 10}'}
PROSE = {"type": "text", "text": "Je lance la recherche."}


# ── 1. le critère attrape la forme réelle ─────────────────────────────────────

def test_la_forme_reelle_du_job_17275_est_nommee(monkeypatch):
    turn = _tour(FIXTURE["message"], monkeypatch, usage=FIXTURE["usage"])
    assert turn.stop_reason == "appel_mal_encode"
    assert turn.defaut == {"forme": "reference+texte_json", "outil": "serper_search"}
    assert turn.tool_calls == (), "le JSON du texte n'est JAMAIS transformé en appel"


# ── témoins négatifs : ce qui lui ressemble sans l'être ──────────────────────

def test_un_texte_qui_contient_du_json_reste_une_conclusion(monkeypatch):
    """La classe que le critère ne doit PAS faucher : 2 545 tours sur 45 698
    balayés portaient un objet JSON dans leur texte, sans aucun défaut."""
    msg = {"role": "assistant", "content":
           'Écrit : {"passe": "F", "statut": "enrichi"}. Travail terminé.'}
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_un_contenu_en_chaine_n_est_jamais_lu_comme_un_appel(monkeypatch):
    msg = {"role": "assistant", "content": '{"query": "MAISON EXEMPLE"}'}
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_une_reference_vide_n_est_pas_un_appel(monkeypatch):
    """Vécu trois fois (runs 13016, 13527, un direct) : `reference_ids: []`."""
    msg = _contenu(PROSE, {"type": "reference", "reference_ids": []}, ARGS)
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_une_reference_a_un_outil_non_envoye_n_est_pas_un_appel(monkeypatch):
    msg = _contenu(PROSE, {"type": "reference", "reference_ids": ["email_send"]}, ARGS)
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_une_reference_a_plusieurs_ids_n_est_pas_un_appel(monkeypatch):
    msg = _contenu(PROSE, {"type": "reference",
                           "reference_ids": ["serper_search", "data_write"]}, ARGS)
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_une_reference_suivie_de_prose_n_est_pas_un_appel(monkeypatch):
    msg = _contenu(PROSE, REF, {"type": "text", "text": 'voir {"query": "x"} plus haut'})
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_une_reference_suivie_d_un_tableau_json_n_est_pas_un_appel(monkeypatch):
    msg = _contenu(PROSE, REF, {"type": "text", "text": '[{"query": "x"}]'})
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_une_reference_en_derniere_partie_n_est_pas_un_appel(monkeypatch):
    msg = _contenu(PROSE, ARGS, REF)
    assert _tour(msg, monkeypatch).stop_reason == "end_turn"


def test_un_appel_structure_present_l_emporte(monkeypatch):
    """Un vrai `tool_calls` : le tour s'exécute normalement, quel que soit le texte."""
    msg = _contenu(PROSE, REF, ARGS)
    msg["tool_calls"] = [{"id": "c1", "function": {"name": "serper_search",
                                                   "arguments": '{"query": "x"}'}}]
    turn = _tour(msg, monkeypatch, finish="tool_calls")
    assert turn.stop_reason == "end_turn" and turn.defaut is None
    assert [c.name for c in turn.tool_calls] == ["serper_search"]


# ── 2 et 3. la boucle : rien d'exécuté, tour compté, défaut journalisé ────────

def _mal_encode(usage):
    return Turn(text="Je lance la recherche.", tool_calls=(),
                stop_reason="appel_mal_encode", raw_content=[], usage=usage,
                defaut={"forme": "reference+texte_json", "outil": "data_rows"})


def test_la_boucle_s_arrete_sans_executer_et_compte_le_tour_rejete():
    t = FauxTransport()
    premier = Turn(text="", tool_calls=(ToolCall(id="t0", name="data_rows", arguments={}),),
                   stop_reason="end_turn", raw_content=[],
                   usage={"input_tokens": 100, "output_tokens": 10})
    p = FauxProvider([premier, _mal_encode({"input_tokens": 700, "output_tokens": 30}),
                      Turn(text="jamais atteint")])
    evenements = []
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=5)
    res = agent_runtime.run(spec, t, p, prompt="go",
                            on_event=lambda ev, champs: evenements.append((ev, champs)))
    assert t.appels == [("data_rows", {})], "seul le VRAI appel a atteint le transport"
    assert res.stopped == "appel_mal_encode" and res.reply == ""
    assert res.defaut == {"forme": "reference+texte_json", "outil": "data_rows"}
    assert res.usage["input_tokens"] == 800 and res.usage["output_tokens"] == 40, \
        "le tour rejeté a été facturé : il est compté avant l'arrêt"
    notes = [c for ev, c in evenements if ev == "appel_mal_encode"]
    assert notes == [{"forme": "reference+texte_json", "outil": "data_rows", "tour": 2}]
    assert p.file == [Turn(text="jamais atteint")], "aucun tour de plus après le défaut"


def test_le_defaut_l_emporte_sur_la_borne_de_jetons_qu_il_franchit():
    """Revue du 13/09/2026 : la borne `max_tokens` était testée AVANT le défaut.
    Un tour mal encodé qui franchissait la borne finissait `max_tokens` — conclu
    `done` par le worker : le faux `done` subsistait exactement là."""
    t = FauxTransport()
    p = FauxProvider([_mal_encode({"input_tokens": 700, "output_tokens": 30}),
                      Turn(text="jamais atteint")])
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=5, max_tokens=500)
    res = agent_runtime.run(spec, t, p, prompt="go")
    assert res.stopped == "appel_mal_encode", "le défaut est nommé, même au-delà de la borne"
    assert res.usage["input_tokens"] == 700 and res.usage["output_tokens"] == 30
    assert t.appels == [] and conclusion.echec_nomme(res) == "appel_outil_mal_encode (data_rows)"


def test_un_tour_bien_forme_qui_franchit_la_borne_reste_max_tokens():
    """Témoin : l'ordre ne change rien aux autres arrêts."""
    t = FauxTransport()
    trop = Turn(text="", tool_calls=(ToolCall(id="t0", name="data_rows", arguments={}),),
                stop_reason="end_turn", raw_content=[],
                usage={"input_tokens": 700, "output_tokens": 30})
    p = FauxProvider([trop, Turn(text="jamais atteint")])
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=5, max_tokens=500)
    res = agent_runtime.run(spec, t, p, prompt="go")
    assert res.stopped == "max_tokens" and conclusion.echec_nomme(res) is None


# ── 4. la conclusion : un échec nommé, jamais un `done` ───────────────────────

def test_le_motif_d_echec_nomme_l_outil():
    res = AgentResult(reply="", stopped="appel_mal_encode",
                      defaut={"forme": "reference+texte_json", "outil": "serper_search"})
    assert conclusion.echec_nomme(res) == "appel_outil_mal_encode (serper_search)"


def _travail_mal_encode(monkeypatch, file=None):
    """Un travail entier par le worker, dont la boucle rend l'arrêt `appel_mal_encode`."""
    import types

    from oto_runner import worker as W
    from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

    vus: list = []

    class McpEspion(FauxMcp):
        def __init__(self, **kw):
            super().__init__(**kw)
            vus.append(self)

    class BackendEspion(FauxBackend):
        conclusion = None

        def complete(self, job_id, ok, error=None, run_id=None, result=None):
            self.conclusion = {"ok": ok, "error": error, "run_id": run_id, "result": result}
            return super().complete(job_id, ok, error=error, run_id=run_id, result=result)

    def faux_run(spec, transport, provider, prompt=None, history=None,
                 on_turn=None, on_event=None, **_):
        return AgentResult(reply="", stopped="appel_mal_encode",
                           usage={"input_tokens": 700, "output_tokens": 30},
                           defaut={"forme": "reference+texte_json", "outil": "serper_search"})

    monkeypatch.setattr(W, "McpSession", McpEspion)
    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    b = BackendEspion()
    provider = types.SimpleNamespace(__name__="agent_llm_openai",
                                     model=lambda: "mistral-large-2512")
    W._un_travail(b, _job("start"), provider, file=file)
    return b, vus


def test_le_travail_se_conclut_en_echec_nomme_et_libere_sa_ligne(monkeypatch):
    b, (mcp,) = _travail_mal_encode(monkeypatch)
    nom, args = mcp.appels[-1]
    assert nom == "run_finish" and args["outcome"] == "failed", \
        "le run est clos en échec — c'est `run_finish` qui libère la ligne réservée"
    assert args["note"] == "appel_outil_mal_encode (serper_search)"
    assert b.conclusion["ok"] is False, "le travail ne finit PAS done"
    assert b.conclusion["error"] == "appel_outil_mal_encode (serper_search)"
    assert b.conclusion["run_id"] == "r-NEUF"
    assert b.conclusion["result"]["stopped"] == "appel_mal_encode"
    assert b.conclusion["result"]["usage_tokens"] == 730, \
        "l'usage du travail, tour rejeté compris, part avec l'échec"


def test_en_mode_direct_le_travail_est_compte_en_echec(monkeypatch):
    from oto_runner.file_de_travail import SansFile
    file = SansFile()
    _travail_mal_encode(monkeypatch, file=file)
    (conclu,) = file.conclus.values()
    assert conclu["status"] == "failed"
    assert conclu["error"] == "appel_outil_mal_encode (serper_search)"


def test_aucun_autre_arret_ne_devient_un_echec():
    """`max_steps`, `max_tokens`, `refusal`, `no_reply` gardent leur conclusion
    d'aujourd'hui : ce lot n'élargit rien."""
    for arret in ("end_turn", "max_steps", "max_tokens", "refusal", "no_reply"):
        assert conclusion.echec_nomme(AgentResult(reply="", stopped=arret)) is None
