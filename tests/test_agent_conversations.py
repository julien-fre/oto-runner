"""Le chemin one-shot Conversations — les outils tournent chez Mistral.

Ce que le banc fige : le PARSE des outputs (le bilan de surveillance en dérive —
usage, pas, tool_counts), les rejeux transitoires explicites (le SDK ne rejoue
pas, et on ne l'utilise pas : son timeout est troué — basesdk.py:227, 20 h de
gel sur la boucle locale), la conformité `store=False`, la RÉSOLUTION de
l'alias de modèle en version concrète, et le chemin worker complet (fil à deux
tours, bilan déclaré). L'essai réel des 20 fiches valide
contre le service ce que le banc ne peut pas savoir.
"""
from __future__ import annotations

import json

import pytest

import oto_runner.agent_conversations as C
from oto_runner import worker as W
from tests.test_worker_reprise import FauxBackend, _job


_REPONSE = {
    "outputs": [
        {"type": "tool.execution", "name": "data_claim_next"},
        {"type": "tool.execution", "name": "serper_search"},
        {"type": "tool.execution", "name": "serper_search"},
        {"type": "tool.execution", "name": "data_write"},
        {"type": "tool.execution", "name": "data_release"},
        {"type": "message.output", "content": [
            {"type": "text", "text": "Fiche enrichie, "},
            {"type": "text", "text": "ligne libérée."}]},
    ],
    "usage": {"prompt_tokens": 21000, "completion_tokens": 900, "total_tokens": 21900},
}


class _R:
    def __init__(self, status=200, corps=None, texte=""):
        self.status_code = status
        self._corps = corps
        self.text = texte

    def json(self):
        return self._corps


# Le catalogue tel que le fournisseur le rend RÉELLEMENT (relevé 27/08 sur
# /v1/models) : SYMÉTRIQUE — la version cite l'alias ET l'alias cite la
# version. C'est ce qui a fait échouer la première règle en production.
_MODELS = {"data": [
    {"id": "mistral-large-2512", "aliases": ["mistral-large-latest"]},
    {"id": "mistral-large-latest", "aliases": ["mistral-large-2512"]},
    {"id": "mistral-small-2506", "aliases": ["mistral-small-latest"]},
    {"id": "mistral-small-latest", "aliases": ["mistral-small-2506"]},
]}


class _FauxRequests:
    """Le `requests` du module, simulé. La résolution de version est le SEUL
    appel qui y passe — les conversations vont par `post_with_deadline`."""

    def __init__(self, reponse):
        self.reponse = reponse
        self.appels = []

    def get(self, url, **kw):
        self.appels.append((url, kw))
        if isinstance(self.reponse, Exception):
            raise self.reponse
        return self.reponse


@pytest.fixture(autouse=True)
def _catalogue(monkeypatch):
    """AUCUN test ne sort : `run_once` résout l'alias au début de chaque job.
    Le cache est un état de MODULE — il se vide entre deux tests, sinon le
    premier décide pour les suivants."""
    C._resolutions.clear()
    monkeypatch.setattr(C, "requests", _FauxRequests(_R(corps=_MODELS)))
    yield
    C._resolutions.clear()


def _env(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_OPENAI_API_KEY", "k")
    monkeypatch.setenv("OTO_RUNNER_CONNECTOR_ID", "conn-1")
    monkeypatch.delenv("OTO_RUNNER_MODEL", raising=False)   # DEFAULT_MODEL, l'alias




def test_store_false_est_toujours_envoye(monkeypatch):
    """`store=False` est une contrainte de CONFORMITÉ du contrat client —
    rien ne reste chez Mistral. Pas un défaut implicite : ENVOYÉ, toujours."""
    _env(monkeypatch)
    vu = {}
    monkeypatch.setattr(C, "post_with_deadline",
                        lambda url, **kw: vu.update(kw["json"]) or _R(corps=_REPONSE))
    C.run_once(instructions="p", inputs="i", tools=())
    assert vu["store"] is False
    assert vu["tools"][0]["connector_id"] == "conn-1"


def test_un_transitoire_est_rejoue_puis_passe(monkeypatch):
    """Le SDK ne rejoue pas ; nous si — explicitement, sur les seuls
    transitoires HTTP, jamais sur une DeadlineExceeded (re-payer un run
    entier n'est pas une politique de rejeu)."""
    _env(monkeypatch)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    essais = {"n": 0}

    def post(url, **kw):
        essais["n"] += 1
        return _R(status=502, texte="bad gateway") if essais["n"] < 3 \
            else _R(corps=_REPONSE)

    monkeypatch.setattr(C, "post_with_deadline", post)
    res = C.run_once(instructions="p", inputs="i", tools=())
    assert essais["n"] == 3 and res.reply


def test_un_400_echoue_net_sans_rejeu(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(C, "post_with_deadline",
                        lambda url, **kw: _R(status=400, texte="invalid connector"))
    with pytest.raises(RuntimeError, match="conversations → 400"):
        C.run_once(instructions="p", inputs="i", tools=())


def test_sans_connector_id_erreur_franche(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_OPENAI_API_KEY", "k")
    monkeypatch.delenv("OTO_RUNNER_CONNECTOR_ID", raising=False)
    with pytest.raises(C.LlmUnavailable, match="OTO_RUNNER_CONNECTOR_ID"):
        C.run_once(instructions="p", inputs="i", tools=())




def test_une_base_openai_avec_v1_ne_double_pas_le_chemin(monkeypatch):
    """L'env de la box porte la base openai-compat AVEC /v1 : la réutiliser
    donnait /v1/v1/conversations → 404 « no Route matched » (vécu, job 274)."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", "https://api.mistral.ai/v1")
    vu = {}
    monkeypatch.setattr(C, "post_with_deadline",
                        lambda url, **kw: vu.update(url=url) or _R(corps=_REPONSE))
    C.run_once(instructions="p", inputs="i", tools=())
    assert vu["url"] == "https://api.mistral.ai/v1/conversations"






_CALL_BIDON = {
    "type": "function.call", "object": "entry", "id": "e-9",
    "created_at": "2026-08-27T21:00:00Z", "completed_at": "2026-08-27T21:00:01Z",
    "agent_id": None, "model": "un-modele",
    "tool_call_id": "call-abc",
    "name": "--- **Bilan** — je poursuis, prochaine étape : vérifier.",
    "arguments": "{}",
}
_EXECUTION = {"type": "tool.execution", "object": "entry", "id": "e-8",
              "created_at": "2026-08-27T20:59:00Z", "model": "un-modele",
              "name": "demo_lookup", "arguments": "{}",
              "info": {"resultat": "trois lignes"}}
_CHAMPS_SERVEUR = ("id", "created_at", "completed_at", "agent_id", "model")


def _reponse_avec_appel_renvoye():
    return {"outputs": [dict(_EXECUTION), dict(_CALL_BIDON)],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}


def _suite(monkeypatch, reponses):
    """Un POST = une réponse de la liste ; les corps envoyés sont conservés."""
    corps = []

    def post(url, **kw):
        corps.append(kw["json"])
        return _R(corps=reponses[len(corps) - 1])

    monkeypatch.setattr(C, "post_with_deadline", post)
    return corps


def test_sans_la_variable_un_appel_renvoye_narrete_quun_seul_post(monkeypatch):
    """La relance est un cran qu'on arme : variable absente ⟹ le comportement
    d'avant à l'octet près — un POST, l'appel rendu compté non exécuté."""
    _env(monkeypatch)
    monkeypatch.delenv("OTO_RUNNER_RELANCES_MAX", raising=False)
    corps = _suite(monkeypatch, [_reponse_avec_appel_renvoye()])
    res = C.run_once(instructions="p", inputs="i", tools=("demo_lookup",))
    assert len(corps) == 1, "aucune relance"
    assert corps[0]["inputs"] == "i"
    assert [(s.tool, s.ok) for s in res.steps][-1][1] is False
    assert res.steps[-1].error == "function.call non exécuté"


def test_une_relance_rejoue_le_fil_avec_un_function_result(monkeypatch):
    """Le fil rejoué porte l'ordre initial, les outputs REÇUS tels quels, puis
    la réponse à l'appel rendu — c'est ce qui remet l'agent au travail au lieu
    de perdre la ligne pour le prix d'un run entier."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "2")
    fin = {"outputs": [{"type": "tool.execution", "name": "demo_write",
                        "arguments": "{}"},
                       {"type": "message.output", "content": "fiche écrite"}],
           "usage": {"prompt_tokens": 500, "completion_tokens": 50}}
    corps = _suite(monkeypatch, [_reponse_avec_appel_renvoye(), fin])
    res = C.run_once(instructions="la procédure", inputs="vas-y",
                     tools=("demo_lookup", "demo_write"))

    assert len(corps) == 2, "un second POST, pas un /restart (store=False)"
    assert corps[1]["store"] is False
    assert corps[1]["instructions"] == corps[0]["instructions"]
    assert corps[1]["tools"] == corps[0]["tools"]

    entrees = corps[1]["inputs"]
    assert entrees[0] == {"object": "entry", "type": "message.input",
                          "role": "user", "content": "vas-y"}
    assert entrees[1]["type"] == "function.call" and entrees[1]["name"] == "demo_lookup"
    assert entrees[2] == {"object": "entry", "type": "function.result",
                          "tool_call_id": entrees[1]["tool_call_id"],
                          "result": '{"resultat":"trois lignes"}'}, \
        "ce que l'outil a rendu repart : sans lui le fil refait le travail"
    assert entrees[3] == {"object": "entry", "type": "function.call",
                          "tool_call_id": "call-abc", "name": _CALL_BIDON["name"],
                          "arguments": "{}"}, "le fond de l'appel, sans les champs serveur"
    assert entrees[-1] == {"object": "entry", "type": "function.result",
                           "tool_call_id": "call-abc",
                           "result": C._CONSIGNE_APPEL_RENVOYE}

    assert (res.usage.get("input_total_tokens"), res.usage.get("output_tokens")) == (1500, 150), \
        "le prix payé est la somme des passes"
    assert res.usage.get("input_tokens", "absent") is None, \
        "Conversations ne déclare pas le cache : le non-caché reste inconnu"
    assert (getattr(res, "couverture", None) or {}).get("tours") == 2
    assert [(s.tool, s.ok) for s in res.steps] == [
        ("demo_lookup", True), (_CALL_BIDON["name"], False),
        ("demo_write", True)], "l'appel rendu reste visible au bilan"
    assert res.reply == "fiche écrite"
    assert res.raw_outputs == fin["outputs"], "les entrées brutes de la DERNIÈRE passe"


def test_le_maximum_borne_les_relances(monkeypatch):
    """Un fil qui rend un appel à chaque passe s'arrête au plafond : la relance
    est bornée, jamais une boucle qui re-paie le run indéfiniment."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "2")
    corps = _suite(monkeypatch, [_reponse_avec_appel_renvoye() for _ in range(3)])
    res = C.run_once(instructions="p", inputs="i", tools=("demo_lookup",))
    assert len(corps) == 3, "1 passe + 2 relances, puis stop"
    assert (res.usage.get("input_total_tokens"), res.usage.get("output_tokens")) == (3000, 300)
    assert len(res.steps) == 6
    assert corps[2]["inputs"][-1]["type"] == "function.result"


def test_une_relance_repond_a_chaque_appel_de_la_traine(monkeypatch):
    """Le fournisseur peut en rendre plusieurs d'affilée : chacun reçoit sa
    réponse, sur SON tool_call_id — un appel sans réponse bloque le fil."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    second = dict(_CALL_BIDON, tool_call_id="call-def", id="e-10")
    debut = {"outputs": [dict(_CALL_BIDON), second],
             "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    fin = {"outputs": [{"type": "message.output", "content": "ok"}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    corps = _suite(monkeypatch, [debut, fin])
    C.run_once(instructions="p", inputs="i", tools=())
    resultats = [e for e in corps[1]["inputs"] if e["type"] == "function.result"]
    assert [e["tool_call_id"] for e in resultats] == ["call-abc", "call-def"]


def test_un_appel_sans_tool_call_id_leve(monkeypatch):
    """Répondre à un appel exige son identifiant : une réponse qui n'en porte
    pas est mal formée — elle lève, elle ne se devine pas."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    muet = {"outputs": [{"type": "function.call", "name": "x", "arguments": "{}"}],
            "usage": {}}
    _suite(monkeypatch, [muet])
    with pytest.raises(RuntimeError, match="sans tool_call_id"):
        C.run_once(instructions="p", inputs="i", tools=())


def test_une_valeur_de_relance_non_entiere_leve(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "beaucoup")
    with pytest.raises(C.LlmUnavailable, match="OTO_RUNNER_RELANCES_MAX"):
        C.run_once(instructions="p", inputs="i", tools=())


def test_un_transitoire_est_rejoue_a_chaque_passe(monkeypatch):
    """Les rejeux HTTP valent pour la relance comme pour la première passe."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    fin = {"outputs": [{"type": "message.output", "content": "ok"}],
           "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    suite = [_R(corps=_reponse_avec_appel_renvoye()),
             _R(status=503, texte="unavailable"),
             _R(corps=fin)]
    monkeypatch.setattr(C, "post_with_deadline",
                        lambda url, **kw: suite.pop(0))
    res = C.run_once(instructions="p", inputs="i", tools=())
    assert suite == [] and res.reply == "ok"




def test_aucune_entree_renvoyee_ne_porte_de_champ_serveur(monkeypatch):
    """« Input entries send by the user can't specify ids » — 422 au premier cas
    réel : `id`, `created_at`, `completed_at`, `agent_id` et `model` sont posés
    par le serveur et refusés d'un client. Le `tool_call_id`, lui, n'en est pas
    un : c'est la clé que cite la réponse à l'appel."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    fin = {"outputs": [{"type": "message.output", "content": "ok"}],
           "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    corps = _suite(monkeypatch, [_reponse_avec_appel_renvoye(), fin])
    C.run_once(instructions="p", inputs="i", tools=())
    entrees = corps[1]["inputs"]
    for entree in entrees:
        assert not [c for c in _CHAMPS_SERVEUR if c in entree], entree
    appel = next(e for e in entrees if e.get("name") == _CALL_BIDON["name"])
    assert appel["tool_call_id"] == entrees[-1]["tool_call_id"] == "call-abc"


def test_une_entree_de_type_inconnu_leve(monkeypatch):
    """Renvoyer un type qu'on ne sait pas dépouiller, c'est un 422 opaque à
    l'arrivée : on lève ici, en nommant le type."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    exotique = {"outputs": [{"type": "tool.autre.chose", "id": "e-1"},
                            dict(_CALL_BIDON)],
                "usage": {}}
    _suite(monkeypatch, [exotique])
    with pytest.raises(RuntimeError, match="tool.autre.chose"):
        C.run_once(instructions="p", inputs="i", tools=())


def test_le_message_de_lordre_initial_ne_porte_que_son_fond(monkeypatch):
    """L'entrée initiale est FABRIQUÉE ici : elle passe par le même
    dépouillement que les autres — une seule règle, pas deux chemins."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    fin = {"outputs": [{"type": "message.output", "content": "ok"}], "usage": {}}
    corps = _suite(monkeypatch, [_reponse_avec_appel_renvoye(), fin])
    C.run_once(instructions="p", inputs="vas-y", tools=())
    assert corps[1]["inputs"][0] == {"object": "entry", "type": "message.input",
                                     "role": "user", "content": "vas-y"}


def test_une_execution_doutil_est_redite_en_paire_appel_resultat(monkeypatch):
    """Sondé le 27/08 : le service REFUSE un `tool.execution` en entrée (422
    « Input should be 'message.input' »), avec ou sans `info`, mais accepte un
    `function.call` suivi de son `function.result`. Le travail déjà payé reste
    donc dans le fil, sous la seule forme qui passe."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    debut = {"outputs": [dict(_EXECUTION),
                         dict(_EXECUTION, name="demo_write",
                              info={"ecrit": 1}, id="e-7"),
                         dict(_CALL_BIDON)],
             "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    fin = {"outputs": [{"type": "message.output", "content": "ok"}], "usage": {}}
    corps = _suite(monkeypatch, [debut, fin])
    C.run_once(instructions="p", inputs="vas-y", tools=())

    entrees = corps[1]["inputs"]
    assert [e["type"] for e in entrees] == [
        "message.input", "function.call", "function.result",
        "function.call", "function.result",
        "function.call", "function.result"]
    assert not [e for e in entrees if e["type"] == "tool.execution"], \
        "aucune exécution ne repart telle quelle"
    assert [e.get("name") for e in entrees if e["type"] == "function.call"] == [
        "demo_lookup", "demo_write", _CALL_BIDON["name"]]

    synthetises = [entrees[1]["tool_call_id"], entrees[3]["tool_call_id"]]
    assert all(len(t) == 9 and t.isalnum() for t in synthetises), synthetises
    assert len(set(synthetises)) == 2, "un identifiant par appel, jamais partagé"
    assert entrees[2]["tool_call_id"] == entrees[1]["tool_call_id"]
    assert entrees[4] == {"object": "entry", "type": "function.result",
                          "tool_call_id": entrees[3]["tool_call_id"],
                          "result": '{"ecrit":1}'}
    assert entrees[6] == {"object": "entry", "type": "function.result",
                          "tool_call_id": "call-abc",
                          "result": C._CONSIGNE_APPEL_RENVOYE}, \
        "l'appel RENDU garde son identifiant : c'est lui que cite sa réponse"


def test_une_execution_sans_info_le_dit(monkeypatch):
    """Le fournisseur ne conserve pas toujours ce que l'outil a rendu : on le
    DIT au modèle, plutôt que de lui présenter un résultat vide qu'il lira
    comme « l'outil n'a rien trouvé »."""
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    muette = dict(_EXECUTION)
    muette.pop("info")
    debut = {"outputs": [muette, dict(_CALL_BIDON)], "usage": {}}
    fin = {"outputs": [{"type": "message.output", "content": "ok"}], "usage": {}}
    corps = _suite(monkeypatch, [debut, fin])
    C.run_once(instructions="p", inputs="i", tools=())
    assert corps[1]["inputs"][2]["result"] == C._RESULTAT_NON_CONSERVE


def test_une_execution_sans_name_leve(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("OTO_RUNNER_RELANCES_MAX", "1")
    debut = {"outputs": [{"type": "tool.execution", "arguments": "{}"},
                         dict(_CALL_BIDON)], "usage": {}}
    _suite(monkeypatch, [debut])
    with pytest.raises(RuntimeError, match="tool.execution sans name"):
        C.run_once(instructions="p", inputs="i", tools=())
