"""Le cache de prompt du provider Anthropic — la forme envoyée, sans réseau.

Ce qui se casse en SILENCE si on ne le fige pas : un point de cache qui
disparaît (le tour repaie la procédure au plein tarif, et rien ne le dit —
l'API ne renvoie ni erreur ni avertissement), un cinquième point qui fait
échouer la requête, un marqueur posé sur un bloc de pensée (refusé par l'API),
et un marqueur qui FUIT dans le fil apposé au backend — où il se rejouerait à
contretemps au rechargement.
"""
from __future__ import annotations

import copy

import pytest

from oto_runner import agent_llm as P


class _Usage:
    def __init__(self, **champs):
        self.__dict__.update(champs)


class _Reponse:
    def __init__(self, usage, content=(), stop_reason="end_turn"):
        self.usage = usage
        self.content = list(content)
        self.stop_reason = stop_reason


class _FauxClient:
    """Espionne les kwargs de `messages.create` — c'est la FORME qui est testée."""

    def __init__(self, reponse):
        self.messages = self
        self.reponse = reponse
        self.vu: list[dict] = []

    def create(self, **kwargs):
        self.vu.append(kwargs)
        return self.reponse


class _FauxSdk:
    def __init__(self, client):
        self._client = client

    def Anthropic(self, api_key=None):  # noqa: N802 — c'est le nom du SDK
        return self._client


def _client(monkeypatch, usage=None, content=(), stop="end_turn"):
    c = _FauxClient(_Reponse(usage or _Usage(input_tokens=10, output_tokens=2),
                             content, stop))
    monkeypatch.setattr(P, "_sdk", lambda: _FauxSdk(c))
    return c


OUTILS = [{"name": "data_claim_next", "description": "", "input_schema": {}},
          {"name": "data_write", "description": "", "input_schema": {}}]


def _points(kwargs) -> int:
    """Combien de `cache_control` la requête porte, tous étages confondus."""
    n = 0
    for bloc in kwargs.get("system") or []:
        if isinstance(bloc, dict) and "cache_control" in bloc:
            n += 1
    for outil in kwargs.get("tools") or []:
        if "cache_control" in outil:
            n += 1
    for message in kwargs.get("messages") or []:
        contenu = message.get("content")
        if isinstance(contenu, list):
            n += sum(1 for b in contenu
                     if isinstance(b, dict) and "cache_control" in b)
    return n


def test_les_trois_points_de_cache_sont_poses_et_pas_un_de_plus(monkeypatch):
    c = _client(monkeypatch)
    P.complete(system="la procédure", messages=[P.user_message("go")],
               tools=OUTILS, api_key="k")
    kwargs = c.vu[0]
    assert _points(kwargs) == 3, "système + outils + queue du fil"
    assert _points(kwargs) <= 4, "la limite DURE de l'API est 4 points"
    assert kwargs["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in kwargs["tools"][0], \
        "seul le DERNIER outil porte le point — le cache est un préfixe"
    assert kwargs["messages"][-1]["content"][-1]["cache_control"] \
        == {"type": "ephemeral"}


def test_un_systeme_chaine_devient_une_liste_de_blocs_marquee_au_dernier(monkeypatch):
    c = _client(monkeypatch)
    P.complete(system="la procédure", messages=[P.user_message("go")],
               tools=[], api_key="k")
    systeme = c.vu[0]["system"]
    assert systeme == [{"type": "text", "text": "la procédure",
                        "cache_control": {"type": "ephemeral"}}]


def test_un_systeme_vide_ne_prend_pas_un_point_de_cache(monkeypatch):
    c = _client(monkeypatch)
    P.complete(system="", messages=[P.user_message("go")], tools=[], api_key="k")
    assert c.vu[0]["system"] == "", "un bloc texte vide n'est pas cachable"
    assert _points(c.vu[0]) == 1, "seule la queue du fil reste marquée"


def test_le_point_mobile_se_pose_sur_le_dernier_bloc_du_dernier_message(monkeypatch):
    c = _client(monkeypatch)
    fil = [P.user_message("go"),
           {"role": "assistant", "content": [{"type": "text", "text": "je cherche"}]}]
    fil += P.tool_messages([{"id": "t0", "text": "ligne 1", "is_error": False},
                            {"id": "t1", "text": "ligne 2", "is_error": False}])
    P.complete(system="s", messages=fil, tools=[], api_key="k")
    envoye = c.vu[0]["messages"]
    assert "cache_control" in envoye[-1]["content"][-1]
    assert "cache_control" not in envoye[-1]["content"][0], \
        "UN seul point par tour : le dernier bloc couvre tout ce qui précède"
    assert _points(c.vu[0]) == 2


def test_le_marqueur_ne_fuit_jamais_dans_le_fil_appose_au_backend(monkeypatch):
    """Le fil de la boucle est la matière de `provider_raw` : on ne le mute pas."""
    c = _client(monkeypatch)
    fil = [P.user_message("go")]
    avant = copy.deepcopy(fil)
    P.complete(system="s", messages=fil, tools=list(OUTILS), api_key="k")
    assert fil == avant, "le fil d'appel reste intact — copie de surface"
    assert "cache_control" not in OUTILS[-1], "les schémas du transport aussi"


def test_un_bloc_de_pensee_ne_porte_jamais_le_point(monkeypatch):
    """L'API refuse `cache_control` sur un bloc de pensée : on remonte au dernier
    bloc marquable, sans rien inventer."""
    c = _client(monkeypatch)
    fil = [{"role": "assistant",
            "content": [{"type": "text", "text": "voilà"},
                        {"type": "thinking", "thinking": "…"}]}]
    P.complete(system="s", messages=fil, tools=[], api_key="k")
    blocs = c.vu[0]["messages"][-1]["content"]
    assert "cache_control" not in blocs[-1]
    assert blocs[0]["cache_control"] == {"type": "ephemeral"}


def test_un_tour_tout_en_pensee_ne_pose_aucun_point_dans_le_fil(monkeypatch):
    c = _client(monkeypatch)
    fil = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "…"}]}]
    P.complete(system="s", messages=fil, tools=[], api_key="k")
    assert _points(c.vu[0]) == 1, "seul le système reste marqué"


def test_l_usage_du_cache_remonte_tel_que_l_api_le_rend(monkeypatch):
    c = _client(monkeypatch, usage=_Usage(input_tokens=120, output_tokens=40,
                                          cache_creation_input_tokens=900,
                                          cache_read_input_tokens=9800))
    turn = P.complete(system="s", messages=[P.user_message("go")], tools=[],
                      api_key="k")
    assert turn.usage == {"input_tokens": 120, "output_tokens": 40,
                          "cache_creation_input_tokens": 900,
                          "cache_read_input_tokens": 9800}


def test_sans_cache_dans_la_reponse_les_postes_restent_inconnus(monkeypatch):
    """Le contrat Anthropic (lu le 13/09/2026) rend TOUJOURS les postes de cache,
    nullables : « 0 » veut dire pas de cache, et rien ne dit ce que vaut leur absence.
    Ce banc écrivait « l'API ne rend simplement pas les champs » et en faisait des
    zéros — des zéros fabriqués. Absents, ils restent inconnus (cf. `comptage`)."""
    _client(monkeypatch, usage=_Usage(input_tokens=300, output_tokens=12))
    turn = P.complete(system="s", messages=[P.user_message("go")], tools=[],
                      api_key="k")
    assert turn.usage == {"input_tokens": 300, "output_tokens": 12}


@pytest.mark.parametrize("valeur,attendu", [(0, 0), (None, "inconnu")])
def test_un_poste_de_cache_a_zero_vaut_zero_et_null_reste_inconnu(monkeypatch, valeur, attendu):
    _client(monkeypatch, usage=_Usage(input_tokens=1, output_tokens=1,
                                      cache_creation_input_tokens=valeur,
                                      cache_read_input_tokens=valeur))
    turn = P.complete(system="s", messages=[], tools=[], api_key="k")
    assert turn.usage.get("cache_read_input_tokens", "inconnu") == attendu
    assert turn.usage.get("cache_creation_input_tokens", "inconnu") == attendu
