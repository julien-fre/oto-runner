"""L'effort de réflexion et le plafond de complétion portés par le TRAVAIL, jusqu'au corps
de la requête.

Le catalogue du backend (oto-backend, 14/09/2026) attache au MODÈLE un effort —
`mistral-medium-2604` tourne en `high`, `claude-haiku-4-5` en `none` — et un plafond de
complétion (`max_output_tokens`), et les sert dans la charge du travail, repris sur un
`continue`. Un réglage servi que le runner ne lit pas est plus coûteux qu'un réglage
absent : on croit tourner en `high`. D'où ces bancs, du travail au corps HTTP, et la
contrainte que Mistral impose, mesurée le même jour.

L'ordre figé : **le travail, puis l'hôte (`OTO_RUNNER_EFFORT`, `OTO_RUNNER_MAX_TOKENS`),
puis le fournisseur.**
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
    for cle in ("OTO_RUNNER_EFFORT", "OTO_RUNNER_TEMPERATURE", "OTO_RUNNER_PARALLEL_TOOLS",
                "OTO_RUNNER_MAX_TOKENS", "OTO_RUNNER_MAX_TOKENS_EFFORT"):
        monkeypatch.delenv(cle, raising=False)
    return vu


@pytest.fixture
def sdk(monkeypatch):
    """Ce que le SDK Anthropic reçoit VRAIMENT."""
    vu: dict = {}
    monkeypatch.setattr(A, "_sdk", lambda: _FauxSdk(vu))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-plateforme")
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "medium")
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


def test_le_plafond_du_travail_arrive_dans_la_spec():
    assert worker._spec_du_job(_job(max_output_tokens=16000)).max_output_tokens == 16000
    assert worker._spec_du_job(_job()).max_output_tokens is None


# ── De la spec au corps de la requête (Chat Completions) ─────────────────────

def test_l_effort_du_travail_part_en_reasoning_effort(corps):
    turn = P.complete(system="s", messages=[], tools=[], api_key="k", effort="high",
                      max_output_tokens=16000)
    assert corps["reasoning_effort"] == "high"
    assert turn.effort == "high", "le tour porte ce qui est PARTI"


def test_le_travail_PRIME_sur_l_hote(corps, monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "low")
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high",
               max_output_tokens=16000)
    assert corps["reasoning_effort"] == "high", (
        "deux campagnes servies par le même worker n'en veulent pas le même")


def test_l_hote_sert_quand_le_travail_ne_dit_rien(corps, monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "low")
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert corps["reasoning_effort"] == "low" and turn.effort == "low"


def test_un_effort_a_temperature_ZERO_envoie_top_p_1(corps):
    """⚠️ Mesuré le 14/09/2026 chez Mistral, sur mistral-medium-2604 :
    `reasoning_effort: high` + `temperature: 0` → 400 `invalid_request_greedy_sampling`
    (« top_p must be 1 when using greedy sampling ») ; avec `top_p: 1` → 200. Les
    campagnes tournent à T = 0 : sans ce champ, chaque travail Medium échouerait."""
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high", temperature=0,
               max_output_tokens=16000)
    assert corps["temperature"] == 0 and corps["top_p"] == 1


def test_hors_temperature_zero_top_p_n_est_pas_pose(corps):
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high", temperature=0.3,
               max_output_tokens=16000)
    assert "top_p" not in corps
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="high",
               max_output_tokens=16000)
    assert "top_p" not in corps


def test_sans_effort_ni_plafond_la_requete_est_inchangee_a_l_octet(corps):
    """Un travail sans réglage porté : ni `reasoning_effort`, ni `top_p`, et le plafond
    de l'hôte."""
    turn = P.complete(system="s", messages=[], tools=[], api_key="k", temperature=0)
    assert set(corps) == {"model", "max_tokens", "messages", "prompt_cache_key", "temperature"}
    assert corps["max_tokens"] == P.DEFAULT_MAX_TOKENS == 8192
    assert turn.effort is None


def test_de_bout_en_bout_un_travail_medium_part_avec_effort_plafond_et_top_p(corps):
    spec = worker._spec_du_job(_job(model="mistral-medium-2604", model_family="mistral",
                                    effort="high", temperature=0, max_output_tokens=16000))
    P.complete(system="s", messages=[], tools=[], api_key="k", modele=spec.model,
               temperature=spec.temperature, effort=spec.effort,
               max_output_tokens=spec.max_output_tokens)
    assert (corps["model"], corps["reasoning_effort"], corps["temperature"], corps["top_p"],
            corps["max_tokens"]) == ("mistral-medium-2604", "high", 0, 1, 16000)


# ── Le plafond de complétion porté par le travail ────────────────────────────

def test_le_plafond_du_travail_PRIME_sur_celui_de_l_hote(corps, monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_MAX_TOKENS", "4096")
    P.complete(system="s", messages=[], tools=[], api_key="k", max_output_tokens=8192)
    assert corps["max_tokens"] == 8192


def test_un_effort_de_travail_sans_plafond_porte_LEVE(corps):
    """⚠️ Pas de repli sur le plafond de l'hôte : 8 192 couperait un tour Medium qui
    raisonne (6 964 jetons de complétion mesurés au banc), et la coupe se lirait
    `fin_anormale` sur la fiche plutôt que « catalogue incomplet »."""
    with pytest.raises(LlmUnavailable) as e:
        P.complete(system="s", messages=[], tools=[], api_key="k", effort="high")
    assert "max_output_tokens" in str(e.value)
    assert corps == {}, "rien n'est parti au fournisseur"


def test_la_surcharge_d_hote_retiree_ne_supplee_plus_rien(corps, monkeypatch):
    """`OTO_RUNNER_MAX_TOKENS_EFFORT` attendait que le catalogue porte le plafond : il le
    porte. Encore posée sur un hôte, elle n'a plus de lecteur."""
    monkeypatch.setenv("OTO_RUNNER_MAX_TOKENS_EFFORT", "16000")
    with pytest.raises(LlmUnavailable):
        P.complete(system="s", messages=[], tools=[], api_key="k", effort="high")
    assert not hasattr(P, "max_tokens_effort")


def test_un_effort_none_PART_chez_mistral_et_n_exige_aucun_plafond(corps):
    """À la différence de la voie Anthropic, `none` est une valeur de l'API Mistral
    (`high` | `none`) : il part tel quel. L'omettre laisserait le défaut du fournisseur,
    qui peut raisonner et le facturer. Il ne raisonne pas : aucun plafond porté n'est
    exigé."""
    P.complete(system="s", messages=[], tools=[], api_key="k", effort="none")
    assert corps["reasoning_effort"] == "none"
    assert corps["max_tokens"] == P.DEFAULT_MAX_TOKENS


def test_l_effort_de_l_HOTE_garde_le_plafond_de_l_hote(corps, monkeypatch):
    """Un hôte qui pose `OTO_RUNNER_EFFORT` règle aussi `OTO_RUNNER_MAX_TOKENS` : la levée
    ne vaut que pour l'effort porté par le TRAVAIL."""
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "low")
    P.complete(system="s", messages=[], tools=[], api_key="k")
    assert corps["reasoning_effort"] == "low" and corps["max_tokens"] == 8192


# ── Anthropic : la même signature, et le travail prime aussi ─────────────────

def test_anthropic_envoie_l_effort_du_travail(sdk):
    turn = A.complete(system="s", messages=[], tools=[], effort="low")
    assert sdk["output_config"] == {"effort": "low"} and turn.effort == "low"


def test_anthropic_sans_effort_du_travail_garde_celui_du_worker(sdk):
    A.complete(system="s", messages=[], tools=[])
    assert sdk["output_config"] == {"effort": "medium"}, "le comportement d'avant, intact"


def test_anthropic_effort_none_n_envoie_AUCUN_output_config(sdk):
    """14/09/2026 : Haiku 4.5 refuse `output_config.effort` (400). Le catalogue le déclare
    `none`, et l'effort du worker ne s'y substitue pas."""
    turn = A.complete(system="s", messages=[], tools=[], modele="claude-haiku-4-5",
                      effort="none")
    assert "output_config" not in sdk and turn.effort == "none"


def test_anthropic_prend_le_plafond_du_travail(sdk):
    A.complete(system="s", messages=[], tools=[], effort="high", max_output_tokens=32000)
    assert sdk["max_tokens"] == 32000


def test_anthropic_sans_plafond_porte_garde_celui_du_worker_sans_lever(sdk):
    """Ce provider envoie un effort à chaque tour et le catalogue ne déclare aucun plafond
    Claude : lever sur un effort sans plafond les ferait tous échouer."""
    A.complete(system="s", messages=[], tools=[], effort="high")
    assert sdk["max_tokens"] == A.max_tokens() == A.DEFAULT_MAX_TOKENS


# ── La couture du milieu : la boucle transmet, le journal le dit ─────────────

def test_la_boucle_TRANSMET_l_effort_et_le_plafond_de_la_spec():
    """La doublure de provider avale n'importe quels arguments : on regarde ce que
    la boucle PASSE, pas ce qu'elle rend."""
    vus: dict = {}

    class Mouchard(FauxProvider):
        def complete(self, **kw):
            vus.update(kw)
            return super().complete(**kw)

    agent_runtime.run(AgentSpec(system="s", tools=frozenset(), max_steps=2, effort="high",
                                max_output_tokens=16000),
                      FauxTransport(), Mouchard([_turn(text="fini")]), prompt="go")
    assert (vus["effort"], vus["max_output_tokens"]) == ("high", 16000)


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
