"""L'adaptateur OpenAI-compatible — le parsing et les formes, sans réseau.

Ce qui se casse en silence si on ne le fige pas : les `arguments` qui arrivent
en CHAÎNE JSON (pas en dict comme chez Anthropic), le mapping d'usage
prompt/completion → input/output, le `content_filter` traduit en refus terminal,
et les deux formes de fil (message assistant complet rejoué tel quel, un
message `role:tool` par résultat).
"""
from __future__ import annotations

import json

import pytest
import requests

from oto_runner import agent_llm_openai as P
from oto_runner import agent_runtime
from oto_runner.agent_runtime import AgentSpec
from oto_runner.llm_types import LlmUnavailable, Turn
from tests.test_agent_runtime import FauxTransport


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _reponse(message, finish="stop", usage=None):
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20}}




def test_des_arguments_malformes_ne_sont_pas_repares(monkeypatch):
    """Ce banc disait « l'outil recevra {} et rendra son erreur » : un outil à
    paramètres facultatifs, lui, AGISSAIT sur `{}` (audit du 13/09/2026). L'original
    fait foi, et l'appel n'est pas exécuté (cf. `test_fins_et_arguments`)."""
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "c1", "function": {"name": "data_rows",
                                                    "arguments": "{pas du json"}}]}
    monkeypatch.setattr(P.requests, "post",
                        lambda *a, **k: _Resp(_reponse(msg, finish="tool_calls")))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert getattr(turn.tool_calls[0], "arguments_invalides", None) == "{pas du json"


def test_content_filter_est_un_refus_terminal(monkeypatch):
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp(
        _reponse({"role": "assistant", "content": ""}, finish="content_filter")))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.stop_reason == "refusal" and not turn.wants_tools


def test_une_erreur_http_remonte_avec_le_dire_du_serveur(monkeypatch):
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp(
        {"message": "invalid model"}, status=400))
    with pytest.raises(RuntimeError) as e:
        P.complete(system="s", messages=[], tools=[], api_key="k")
    assert "invalid model" in str(e.value) and "400" in str(e.value)


def test_le_system_passe_en_premier_message(monkeypatch):
    vu = {}

    def _post(url, json=None, timeout=None, headers=None):
        vu.update(corps=json)
        return _Resp(_reponse({"role": "assistant", "content": "ok"}))
    monkeypatch.setattr(P.requests, "post", _post)
    P.complete(system="LE CADRE", messages=[{"role": "user", "content": "hi"}],
               tools=[], api_key="k")
    assert vu["corps"]["messages"][0] == {"role": "system", "content": "LE CADRE"}
    assert vu["corps"]["messages"][1]["role"] == "user"


def test_les_deux_formes_de_fil():
    msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]}
    turn = Turn(text="", raw_content=msg)
    assert P.assistant_message(turn) is msg, \
        "le message assistant se rejoue COMPLET (tool_calls intacts, corrélés par id)"
    outs = P.tool_messages([{"id": "c1", "text": "res1", "is_error": False},
                            {"id": "c2", "text": "res2", "is_error": True}])
    assert [o["role"] for o in outs] == ["tool", "tool"], "un message PAR résultat"
    assert outs[0]["tool_call_id"] == "c1" and outs[1]["content"] == "res2"


def test_format_tools_enveloppe_en_function():
    out = P.format_tools([{"name": "data_rows", "description": "lit",
                           "input_schema": {"type": "object"}}])
    assert out == [{"type": "function",
                    "function": {"name": "data_rows", "description": "lit",
                                 "parameters": {"type": "object"}}}]


def test_le_plafond_wall_clock_coupe_un_serveur_qui_goutte(monkeypatch):
    """Le read timeout d'urllib3 se réarme à chaque octet : un serveur qui
    goutte tient la connexion indéfiniment (vécu : 35 min, pile dans ssl.read).
    La deadline SIGALRM coupe pour de vrai et lève une erreur propre."""
    import time as _t

    from oto_runner.llm_types import LlmUnavailable

    monkeypatch.setattr(P, "_WALL_TIMEOUT_S", 1)
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _t.sleep(5))
    with pytest.raises(LlmUnavailable) as e:
        P.complete(system="s", messages=[], tools=[], api_key="k")
    assert "wall-clock" in str(e.value)


# ── La RETENTATIVE : un incident de transport ne tue plus un travail ────────

def _sans_attente(monkeypatch):
    """Les attentes de 5 s et 20 s, mesurées mais pas subies par le banc."""
    dormi: list = []
    monkeypatch.setattr(P.time, "sleep", lambda s: dormi.append(s))
    return dormi


def test_un_incident_de_transport_est_RETENTE_et_le_tour_aboutit(monkeypatch):
    """⚠️ Nuit du 06/09 : deux travaux morts sur un `ReadTimeout` isolé — l'un
    pendant l'envoi (10 s), l'autre après 300 s sans un octet. Aucune
    retentative : en flotte le job se rejouait plus tard, en direct il était
    PERDU. Un incident de transport n'est pas une réponse."""
    dormi = _sans_attente(monkeypatch)
    essais: list = []

    def post(*a, **k):
        essais.append(1)
        if len(essais) < 3:
            raise requests.exceptions.ReadTimeout(
                "HTTPSConnectionPool(host='api.mistral.ai', port=443): "
                "Read timed out. (read timeout=300)")
        return _Resp(_reponse({"role": "assistant", "content": "enfin"}))

    monkeypatch.setattr(P.requests, "post", post)
    evs: list = []
    turn = P.complete(system="s", messages=[], tools=[], api_key="k",
                      on_event=lambda ev, champs: evs.append((ev, champs)))
    assert turn.text == "enfin" and len(essais) == 3
    assert dormi == [5, 20], "5 s puis 20 s"
    assert [ev for ev, _ in evs] == ["systeme", "systeme"], (
        "chaque retentative est DITE au journal du travail")
    assert evs[0][1] == {"quoi": "retentative du tour de modèle", "essai": 1,
                         "essais": 3, "sur": evs[0][1]["sur"], "attente_s": 5}
    assert "ReadTimeout" in evs[0][1]["sur"] and "read timeout=300" in evs[0][1]["sur"]


def test_apres_le_dernier_essai_c_est_LlmUnavailable_jamais_une_exception_requests(monkeypatch):
    """La boucle a une classe pour « le substrat n'a pas répondu » ; le worker
    en fait un échec propre (run clos, ligne rendue). Une `ConnectionError` nue
    remonterait comme un mystère de plus."""
    dormi = _sans_attente(monkeypatch)
    essais: list = []

    def post(*a, **k):
        essais.append(1)
        raise requests.exceptions.ConnectionError("connexion refusée")

    monkeypatch.setattr(P.requests, "post", post)
    with pytest.raises(LlmUnavailable) as e:
        P.complete(system="s", messages=[], tools=[], api_key="k")
    assert len(essais) == 3 and dormi == [5, 20]
    assert "3 essais" in str(e.value) and "connexion refusée" in str(e.value)
    assert not isinstance(e.value, requests.exceptions.RequestException)


def test_un_429_ou_un_5xx_est_rejoue_un_4xx_est_une_reponse(monkeypatch):
    """429 et 5xx : le fournisseur est débordé, c'est passager. Un 4xx est une
    RÉPONSE — le rejouer rejouerait le même verdict, trois fois."""
    dormi = _sans_attente(monkeypatch)
    codes = [503, 429, 200]
    vus: list = []

    def post(*a, **k):
        code = codes[len(vus)]
        vus.append(code)
        return _Resp(_reponse({"role": "assistant", "content": "ok"}), status=code)

    monkeypatch.setattr(P.requests, "post", post)
    assert P.complete(system="s", messages=[], tools=[], api_key="k").text == "ok"
    assert vus == [503, 429, 200] and dormi == [5, 20]

    vus.clear(); dormi.clear(); codes[:] = [400, 400, 400]
    with pytest.raises(RuntimeError, match="400"):
        P.complete(system="s", messages=[], tools=[], api_key="k")
    assert vus == [400] and dormi == [], "un 4xx n'est jamais rejoué"


def test_un_statut_rejouable_au_DERNIER_essai_rend_le_dire_du_serveur(monkeypatch):
    """On ne remplace pas ce que le fournisseur explique par notre résumé : au
    dernier essai, la réponse remonte et `complete` lève avec son texte."""
    _sans_attente(monkeypatch)
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp(
        {"message": "service overloaded, retry later"}, status=503))
    with pytest.raises(RuntimeError) as e:
        P.complete(system="s", messages=[], tools=[], api_key="k")
    assert "503" in str(e.value) and "service overloaded" in str(e.value)
    assert not isinstance(e.value, LlmUnavailable)


def test_la_deadline_murale_n_est_PAS_rejouee(monkeypatch):
    """Elle a déjà attendu sept minutes sur un serveur qui gouttait : la rejouer
    paierait trois fois cette attente."""
    import time as _t

    essais: list = []
    monkeypatch.setattr(P, "_WALL_TIMEOUT_S", 1)

    def post(*a, **k):
        essais.append(1)
        _t.sleep(5)      # ⚠️ un vrai sommeil : c'est SIGALRM qui doit le couper

    monkeypatch.setattr(P.requests, "post", post)
    with pytest.raises(LlmUnavailable, match="wall-clock"):
        P.complete(system="s", messages=[], tools=[], api_key="k")
    assert essais == [1], "un seul essai — la deadline murale ne se rejoue pas"


def test_un_contenu_en_liste_de_blocs_est_normalise(monkeypatch):
    """Mistral rend parfois `content` en LISTE de blocs typés au lieu d'une
    chaîne (vécu, job 52 : AttributeError au .strip() — déterministe au rejeu
    tant que la réponse garde cette forme)."""
    import oto_runner.agent_llm_openai as A

    class _R:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": [
                        {"type": "text", "text": "première partie"},
                        {"type": "text", "text": "seconde"}]},
                     "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2}}

    monkeypatch.setattr(A, "_post_borne", lambda url, corps, entetes, **_: _R)
    t = A.complete(system="s", messages=[A.user_message("go")], tools=[],
                   api_key="k")
    assert t.text == "première partie\nseconde"


# ── LE CACHE : une clé à poser, un compteur à lire ──────────────────────────

def test_la_cle_de_cache_est_posee_sur_chaque_appel(monkeypatch):
    """⚠️ Sans elle, le fournisseur ne met RIEN en cache — mesuré le 01/09 :
    deux appels identiques, zéro jeton caché ; avec elle, 96 % dès le second.
    Un passage de 33 fiches a coûté 0,108 $ la ligne faute de ce paramètre."""
    vus = {}

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5}}

    def faux_post(url, corps, entetes, **_):
        vus.update(corps)
        return R()

    monkeypatch.setattr(P, "_post_borne", faux_post)
    monkeypatch.setattr(P, "resolve_key", lambda: "k")
    P.complete(system="s", messages=[], tools=None)
    assert vus.get("prompt_cache_key"), "aucune clé de cache dans la requête"


def test_les_jetons_servis_par_le_cache_ne_comptent_pas_comme_neufs(monkeypatch):
    """⚠️ `prompt_tokens` INCLUT ce que le cache a servi. Les porter tels quels
    ferait payer au plein tarif, dans nos relevés, ce qui est facturé 10 %."""
    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2812, "completion_tokens": 3,
                              "prompt_tokens_details": {"cached_tokens": 2688}}}

    monkeypatch.setattr(P, "_post_borne", lambda *a, **k: R())
    monkeypatch.setattr(P, "resolve_key", lambda: "k")
    t = P.complete(system="s", messages=[], tools=None)
    assert t.usage["input_tokens"] == 124, "le neuf, pas le total"
    assert t.usage["cache_read_input_tokens"] == 2688


# ── Ce que Scaleway facture et coupe : le raisonnement, et le plafond de complétion ──

def _corps_envoye(monkeypatch):
    vu = {}

    def post(url, json=None, **k):
        vu.update(json)
        return _Resp(_reponse({"role": "assistant", "content": "ok"}))

    monkeypatch.setattr(P.requests, "post", post)
    P.complete(system="s", messages=[], tools=[], api_key="k")
    return vu


def test_sans_OTO_RUNNER_EFFORT_rien_n_est_envoye_le_fournisseur_applique_son_defaut(monkeypatch):
    """Aucune valeur par défaut ici : absent = on n'envoie rien. La variable n'était
    lue que côté Anthropic, et Scaleway active le raisonnement par défaut — et le
    facture."""
    monkeypatch.delenv("OTO_RUNNER_EFFORT", raising=False)
    assert "reasoning_effort" not in _corps_envoye(monkeypatch)


def test_OTO_RUNNER_EFFORT_part_en_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_EFFORT", "low")
    assert _corps_envoye(monkeypatch)["reasoning_effort"] == "low"


def test_OTO_RUNNER_MAX_TOKENS_borne_la_completion(monkeypatch):
    """Sur Scaleway les jetons de raisonnement partagent ce plafond avec la fiche :
    8192 coupe la réponse (`finish_reason: length`). La variable existe et est LUE."""
    monkeypatch.delenv("OTO_RUNNER_MAX_TOKENS", raising=False)
    assert _corps_envoye(monkeypatch)["max_tokens"] == P.DEFAULT_MAX_TOKENS == 8192
    monkeypatch.setenv("OTO_RUNNER_MAX_TOKENS", "20000")
    assert _corps_envoye(monkeypatch)["max_tokens"] == 20000
    monkeypatch.setenv("OTO_RUNNER_MAX_TOKENS", "beaucoup")
    with pytest.raises(P.LlmUnavailable, match="OTO_RUNNER_MAX_TOKENS"):
        P.max_tokens()


# ── Le GROUPEMENT des appels d'outils : un par tour, ou tous à la fois ───────
#
# Mesuré dans la nuit du 06/09/2026 : Mistral Large 3 groupe jusqu'à 13 appels
# dans un même tour, puis écrit la fiche sans jamais reformuler une requête après
# un résultat décevant. Le mode séquentiel se teste — donc il se règle, et le
# réglage se relit dans le journal du passage.

def test_sans_reglage_le_corps_ne_porte_RIEN_le_fournisseur_groupe_comme_avant(monkeypatch):
    """Absent = le comportement actuel, à l'octet près : la clé n'apparaît pas."""
    monkeypatch.delenv("OTO_RUNNER_PARALLEL_TOOLS", raising=False)
    assert "parallel_tool_calls" not in _corps_envoye(monkeypatch)
    assert P.parallel_tools() is True
    monkeypatch.setenv("OTO_RUNNER_PARALLEL_TOOLS", "1")
    assert "parallel_tool_calls" not in _corps_envoye(monkeypatch)


def test_a_zero_le_corps_porte_parallel_tool_calls_false(monkeypatch):
    """`parallel_tool_calls` est le nom OpenAI-compatible, accepté par Mistral :
    un seul appel d'outil par tour, donc un résultat lu avant l'appel suivant."""
    monkeypatch.setenv("OTO_RUNNER_PARALLEL_TOOLS", "0")
    assert _corps_envoye(monkeypatch)["parallel_tool_calls"] is False


def test_une_valeur_illisible_LEVE_jamais_un_repli_silencieux(monkeypatch):
    """Un réglage qu'on croit posé et qui ne l'est pas ferait conclure un banc sur
    le comportement d'en face."""
    monkeypatch.setenv("OTO_RUNNER_PARALLEL_TOOLS", "false")
    with pytest.raises(ValueError, match="OTO_RUNNER_PARALLEL_TOOLS"):
        P.parallel_tools()
    with pytest.raises(ValueError, match="OTO_RUNNER_PARALLEL_TOOLS"):
        P.complete(system="s", messages=[], tools=[], api_key="k")


def test_le_reglage_est_DIT_au_journal_a_cote_de_max_tool_output(monkeypatch):
    """Un banc qui compare groupé et séquentiel ne vaut que si chaque passage dit
    sous quel réglage il a tourné — une fois, à l'ouverture du journal, comme le
    plafond de sortie d'outil."""
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp(
        _reponse({"role": "assistant", "content": "fini"})))
    spec = AgentSpec(system="le cadre", tools=frozenset({"data_rows"}),
                     max_steps=2, label="job:1")

    def systeme(valeur):
        monkeypatch.setenv("OTO_RUNNER_PARALLEL_TOOLS", valeur)
        evs: list = []
        agent_runtime.run(spec, FauxTransport(), P, prompt="vas-y", api_key="k",
                          on_event=lambda ev, champs: evs.append((ev, champs)))
        assert evs[0][0] == "systeme"
        return evs[0][1]

    sequentiel = systeme("0")
    assert sequentiel["parallel_tool_calls"] is False
    assert sequentiel["max_tool_output"] == agent_runtime.max_tool_output()
    assert systeme("1")["parallel_tool_calls"] is True
