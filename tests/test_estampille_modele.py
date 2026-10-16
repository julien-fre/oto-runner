"""L'estampille du modèle : ce qui a produit une ligne se retrouve à froid.

Le champ `model` du résultat d'un job était DÉCLARÉ, lu par le worker et compté
par le bilan — et **aucun transport ne le posait**. Trois consommateurs, zéro
producteur : `null` sur 100 % des jobs. Personne ne l'a vu tant que personne n'a
demandé « quelles lignes viennent de quel modèle ? » ; il a fallu recouper
l'horodatage du journal des écritures, ce qui ne marche plus dès que deux
modèles tournent le même jour.

⚠️ **Ce que ces tests gardent, c'est la CHAÎNE, pas le champ** — le défaut ne
vivait dans aucun maillon pris isolément, il vivait dans le fait que personne ne
suivait la valeur du fournisseur jusqu'au résultat servi.
"""
from __future__ import annotations

import json

import pytest

from oto_runner import agent_llm_openai as OA
from oto_runner import agent_runtime as R
from oto_runner.llm_types import ToolCall, Turn


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _reponse(model="mistral-large-2512", message=None):
    return {"model": model,
            "choices": [{"message": message or {"role": "assistant",
                                                "content": "fini"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def test_le_tour_porte_ce_que_le_fournisseur_dit_avoir_servi(monkeypatch):
    """C'est la version SERVIE qui compte, pas l'alias demandé : `…-latest` ne se
    date pas après coup."""
    monkeypatch.setenv("OTO_RUNNER_MODEL", "mistral-large-latest")
    monkeypatch.setattr(OA.requests, "post",
                        lambda *a, **k: _Resp(_reponse("mistral-large-2512")))
    turn = OA.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.model == "mistral-large-2512"


def test_sans_reponse_du_fournisseur_on_estampille_ce_qu_on_a_demande(monkeypatch):
    """Une estampille approchée vaut infiniment mieux qu'un `null` — un `null` ne
    se distingue PAS d'un job qui n'a jamais tourné."""
    monkeypatch.setenv("OTO_RUNNER_MODEL", "mistral-medium-2508")
    payload = _reponse()
    del payload["model"]
    monkeypatch.setattr(OA.requests, "post", lambda *a, **k: _Resp(payload))
    turn = OA.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.model == "mistral-medium-2508"


def test_un_refus_porte_l_estampille_lui_aussi(monkeypatch):
    """Un job refusé a COÛTÉ. Ne pas l'estamper laisse un trou pile là où on
    cherchera l'anomalie."""
    monkeypatch.setenv("OTO_RUNNER_MODEL", "m")
    payload = _reponse("mistral-large-2512")
    payload["choices"][0]["finish_reason"] = "content_filter"
    monkeypatch.setattr(OA.requests, "post", lambda *a, **k: _Resp(payload))
    turn = OA.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.stop_reason == "refusal" and turn.model == "mistral-large-2512"


class _FauxProvider:
    """Forme Anthropic minimale — la boucle ne connaît que ce protocole."""

    def __init__(self, tours):
        self.file = list(tours)

    def complete(self, **_kw):
        return self.file.pop(0)

    def user_message(self, text):
        return {"role": "user", "content": text}

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.raw_content}

    def tool_messages(self, results):
        return [{"role": "user", "content": r["text"]} for r in results]

    def format_tools(self, schemas):
        return list(schemas)


class _FauxTransport:
    @staticmethod
    def schemas(_tools):
        return [{"name": "data_rows"}]

    @staticmethod
    def call(name, _args):
        return f"{name}: ok", False, ()


_SPEC = R.AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=3)


def test_la_boucle_remonte_l_estampille_jusqu_au_resultat():
    """Le maillon qui manquait : le tour la portait, `AgentResult` la déclarait,
    et rien ne faisait le trajet."""
    p = _FauxProvider([Turn(text="fini", model="claude-sonnet-4-6")])
    res = R.run(_SPEC, _FauxTransport(), p, prompt="vas-y")
    assert res.model == "claude-sonnet-4-6"


def test_le_dernier_tour_fait_foi_si_l_alias_bascule_en_plein_vol():
    """Deux versions ont servi ; c'est la seconde qu'on retrouvera en base, et
    c'est donc celle qui doit être écrite."""
    # Le premier tour APPELLE, sinon la boucle conclut et le second ne sert
    # jamais — c'est le déroulé réel d'une bascule : elle arrive en cours de vol.
    p = _FauxProvider([
        Turn(text="", model="mistral-large-2512",
             tool_calls=(ToolCall(id="t0", name="data_rows", arguments={}),)),
        Turn(text="fini", model="mistral-large-2601")])
    res = R.run(_SPEC, _FauxTransport(), p, prompt="vas-y")
    assert res.model == "mistral-large-2601"


def test_un_transport_muet_ne_reintroduit_pas_un_null():
    """⚠️ Le repli du WORKER, pas du transport : c'est lui qui ferme la classe.
    Un transport futur qui oublierait l'estampille rendrait à nouveau `null`
    partout — exactement le défaut d'origine, sous un autre nom."""
    from oto_runner import worker as W

    class _P:
        @staticmethod
        def model():
            return "claude-opus-5"

    assert W._modele_courant(_P()) == "claude-opus-5"

    class _Muet:
        @staticmethod
        def model():
            raise RuntimeError("pas de modèle configuré")

    # Un relevé d'observabilité ne fait JAMAIS échouer un job déjà payé.
    assert W._modele_courant(_Muet()) == "inconnu"
