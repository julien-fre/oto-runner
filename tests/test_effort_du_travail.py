"""L'effort de réflexion porté par le TRAVAIL, jusqu'au corps de la requête.

Le catalogue du backend (oto-backend, 14/09/2026) attache un effort au MODÈLE —
`mistral-medium-2604` tourne en `high` — et le sert dans la charge du travail,
repris sur un `continue`. Un réglage servi que le runner ne lit pas est plus
coûteux qu'un réglage absent : on croit tourner en `high`. D'où ces bancs, du
travail au corps HTTP, et la contrainte que Mistral impose, mesurée le même jour.

L'ordre figé : **le travail, puis l'hôte (`OTO_RUNNER_EFFORT`), puis le fournisseur.**
"""
from __future__ import annotations

import inspect

import pytest

from oto_runner import agent_llm as A
from oto_runner import agent_llm_openai as P
from oto_runner import agent_runtime, worker
from oto_runner.agent_runtime import AgentSpec
from oto_runner.llm_types import LlmUnavailable, Turn
from tests.test_agent_runtime import FauxProvider, FauxTransport, _turn
from tests.test_modele_du_travail import _FauxSdk


class _Resp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


@pytest.fixture
def corps(monkeypatch):
    """Ce que le fournisseur reçoit VRAIMENT — le dernier corps posté."""
    vu: dict = {}

    def _post(url, **kw):
        vu.clear()
        vu.update(kw.get("json") or {})
        return _Resp()

    monkeypatch.setattr(P.requests, "post", _post)
    monkeypatch.delenv("OTO_RUNNER_EFFORT", raising=False)
    monkeypatch.delenv("OTO_RUNNER_TEMPERATURE", raising=False)
    monkeypatch.delenv("OTO_RUNNER_PARALLEL_TOOLS", raising=False)
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS", raising=False)
    # Le plafond d'un tour qui raisonne est posé par défaut dans ces bancs : sans lui, un
    # effort de travail LÈVE (cf. la section du bas, qui le retire nommément).
    monkeypatch.setenv("OTO_RUNNER_MAX_TOKENS_EFFORT", "16000")
    return vu


def _job(**payload):
    base = {"tools": ["data_rows"], "max_steps": 5}
    base.update(payload)
    return {"id": 1, "payload": base}


# ── Du travail à la spec ─────────────────────────────────────────────────────

def test_l_effort_du_travail_arrive_dans_la_spec():
    assert worker._spec_du_job(_job(model="mistral-medium-2604", effort="high")).effort == "high"


def test_sans_effort_porte_la_spec_n_en_porte_aucun():
    """`None`, et pas le défaut de l'hôte recopié ici : c'est le provider qui décide
    du repli."""
    assert worker._spec_du_job(_job()).effort is None


def test_un_effort_vide_vaut_une_absence():
    assert worker._spec_du_job(_job(effort="   ")).effort is None


# ── De la spec au corps de la requête (Chat Completions) ─────────────────────

def test_l_effort_du_travail_part_en_reasoning_effort(corps):
    turn = P.complete(system="s", messages=[], tools=[], api_key="k", effort="high")
    assert corps["reasoning_effort"] == "high"
    assert turn.effort == "high", "le tour porte ce qui est PARTI"


def test_le_travail_PRIME_sur_l_hote(corps, monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "low")
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high")
    assert corps["reasoning_effort"] == "high", (
        "deux campagnes servies par le même worker n'en veulent pas le même")


def test_l_hote_sert_quand_le_travail_ne_dit_rien(corps, monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "low")
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert corps["reasoning_effort"] == "low" and turn.effort == "low"


def test_un_effort_a_temperature_ZERO_envoie_top_p_1(corps):
    """⚠️ LE banc du lot. Mesuré le 14/09/2026 chez Mistral, sur mistral-medium-2604 :
    `reasoning_effort: high` + `temperature: 0` → 400 `invalid_request_greedy_sampling`
    (« top_p must be 1 when using greedy sampling ») ; avec `top_p: 1` → 200. Les
    campagnes tournent à T = 0 : sans ce champ, chaque travail Medium échouerait."""
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high", temperature=0)
    assert corps["temperature"] == 0 and corps["top_p"] == 1


def test_hors_temperature_zero_top_p_n_est_pas_pose(corps):
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high", temperature=0.3)
    assert "top_p" not in corps
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high")
    assert "top_p" not in corps


def test_sans_effort_la_requete_est_inchangee_a_l_octet(corps):
    """Les travaux Large ne portent aucun effort : leur requête ne bouge pas, même à
    T = 0 — ni `reasoning_effort`, ni `top_p`."""
    turn = P.complete(system="s", messages=[], tools=[], api_key="k", temperature=0)
    assert set(corps) == {"model", "max_tokens", "messages", "prompt_cache_key", "temperature"}
    assert turn.effort is None


def test_de_bout_en_bout_un_travail_medium_part_avec_effort_et_top_p(corps):
    spec = worker._spec_du_job(_job(model="mistral-medium-2604", model_family="mistral",
                                    effort="high", temperature=0))
    P.complete(system="s", messages=[], tools=[], api_key="k", modele=spec.model,
               temperature=spec.temperature, effort=spec.effort)
    assert (corps["model"], corps["reasoning_effort"], corps["temperature"], corps["top_p"]) == (
        "mistral-medium-2604", "high", 0, 1)


# ── Anthropic : la même signature, et le travail prime aussi ─────────────────

def test_anthropic_envoie_l_effort_du_travail(monkeypatch):
    vu: dict = {}
    monkeypatch.setattr(A, "_sdk", lambda: _FauxSdk(vu))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-plateforme")
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "medium")
    turn = A.complete(system="s", messages=[], tools=[], effort="low")
    assert vu["output_config"] == {"effort": "low"} and turn.effort == "low"


def test_anthropic_sans_effort_du_travail_garde_celui_du_worker(monkeypatch):
    vu: dict = {}
    monkeypatch.setattr(A, "_sdk", lambda: _FauxSdk(vu))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-plateforme")
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "medium")
    A.complete(system="s", messages=[], tools=[])
    assert vu["output_config"] == {"effort": "medium"}, "le comportement d'avant, intact"


# ── La couture du milieu : la boucle transmet, le journal le dit ─────────────

def test_la_boucle_TRANSMET_l_effort_de_la_spec():
    """La doublure de provider avale n'importe quels arguments : on regarde ce que
    la boucle PASSE, pas ce qu'elle rend."""
    vus: dict = {}

    class Mouchard(FauxProvider):
        def complete(self, **kw):
            vus.update(kw)
            return super().complete(**kw)

    agent_runtime.run(AgentSpec(system="s", tools=frozenset(), max_steps=2, effort="high"),
                      FauxTransport(), Mouchard([_turn(text="fini")]), prompt="go")
    assert vus["effort"] == "high"


def test_le_journal_du_tour_dit_l_effort_parti():
    evenements: list = []
    p = FauxProvider([Turn(text="fini", raw_content=[], effort="high")])
    agent_runtime.run(AgentSpec(system="s", tools=frozenset(), max_steps=2), FauxTransport(), p,
                      prompt="go", on_event=lambda ev, champs: evenements.append((ev, champs)))
    (tour,) = [c for ev, c in evenements if ev == "modele"]
    assert tour["effort"] == "high"


# ── La voie qui ne sait pas l'envoyer ────────────────────────────────────────

class _VoieConversations:
    ONE_SHOT = True


def test_un_effort_sur_la_voie_conversations_est_refuse():
    with pytest.raises(worker.EffortNonServi) as e:
        worker._exiger_effort_servi({"effort": "high"}, _VoieConversations())
    assert "high" in str(e.value) and "Conversations" in str(e.value)


def test_sans_effort_la_voie_conversations_passe():
    worker._exiger_effort_servi({}, _VoieConversations())


def test_un_effort_sur_chat_completions_passe():
    worker._exiger_effort_servi({"effort": "high"}, P)


def test_le_refus_precede_session_et_run():
    """Un travail qu'on ne peut pas exécuter ne doit rien coûter : le refus se place
    avec la famille, avant le jeton délégué et la session MCP."""
    src = inspect.getsource(worker._traiter)
    assert src.index("_exiger_effort_servi(p, provider)") < src.index('job.get("delegated_token")')


# ── Le plafond de complétion d'un tour qui raisonne (OTO_RUNNER_MAX_TOKENS_EFFORT) ──

def test_un_effort_de_travail_prend_le_plafond_de_raisonnement(corps, monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_MAX_TOKENS", "8192")
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high", temperature=0)
    assert corps["max_tokens"] == 16000


def test_un_effort_de_travail_sans_plafond_de_raisonnement_LEVE(corps, monkeypatch):
    """⚠️ Pas de repli sur `OTO_RUNNER_MAX_TOKENS` : 8 192 couperait un tour Medium qui
    raisonne (6 964 jetons de complétion mesurés au banc), et la coupe se lirait
    `fin_anormale` sur la fiche plutôt que « worker mal réglé »."""
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS_EFFORT", raising=False)
    with pytest.raises(LlmUnavailable) as e:
        P.complete(system="s", messages=[], tools=[], api_key="k", effort="high")
    assert "OTO_RUNNER_MAX_TOKENS_EFFORT" in str(e.value)
    assert corps == {}, "rien n'est parti au fournisseur"


def test_sans_effort_le_plafond_et_la_requete_ne_bougent_pas(corps):
    """La variable posée sur l'hôte ne touche AUCUNE requête sans effort : les travaux
    Large restent identiques à l'octet, plafond compris."""
    P.complete(system="s", messages=[], tools=[], api_key="k", temperature=0)
    assert corps["max_tokens"] == P.DEFAULT_MAX_TOKENS == 8192
    assert set(corps) == {"model", "max_tokens", "messages", "prompt_cache_key", "temperature"}


def test_l_effort_de_l_HOTE_garde_le_plafond_ordinaire(corps, monkeypatch):
    """Un hôte qui pose `OTO_RUNNER_EFFORT` règle aussi `OTO_RUNNER_MAX_TOKENS` : la règle
    ne vaut que pour l'effort porté par le TRAVAIL, et ne casse pas une configuration
    d'hôte qui marchait."""
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "low")
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS_EFFORT", raising=False)
    P.complete(system="s", messages=[], tools=[], api_key="k")
    assert corps["reasoning_effort"] == "low" and corps["max_tokens"] == 8192


def test_anthropic_ne_lit_pas_le_plafond_de_raisonnement(monkeypatch):
    """`oto-runner-anthropic@1` charge `.env` puis `.env.anthropic` : la variable y est
    visible, et n'y change rien. Ce provider envoie toujours un effort ; son plafond
    reste `OTO_RUNNER_MAX_TOKENS`, et l'absence de la variable ne le fait pas lever."""
    vu: dict = {}
    monkeypatch.setattr(A, "_sdk", lambda: _FauxSdk(vu))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-plateforme")
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("OTO_RUNNER_MAX_TOKENS_EFFORT", "16000")
    A.complete(system="s", messages=[], tools=[], effort="high")
    assert vu["max_tokens"] == A.max_tokens() != 16000
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS_EFFORT", raising=False)
    A.complete(system="s", messages=[], tools=[], effort="high")
