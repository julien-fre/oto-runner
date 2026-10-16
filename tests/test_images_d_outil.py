"""Une image rendue par un outil ATTEINT le modèle (#32) — ou le modèle lit
qu'elle ne l'a pas atteint. Jamais d'image perdue en silence.

Le bout en bout part du résultat MCP tel que `web_image` le rend (une ligne de
texte, puis un bloc image) et s'arrête à la requête envoyée au modèle.
"""
from __future__ import annotations

import base64

import pytest

from oto_runner import agent_abonnement, agent_llm, agent_llm_openai, agent_runtime
from oto_runner.agent_runtime import AgentSpec
from oto_runner.contenu_outil import (DEROULE_MAX_IMAGES, BlocNonGere, Image,
                                      admettre, lire)
from oto_runner.mcp import McpSession
from oto_runner.relais_mcp import Relais

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64).decode()
LIGNE = "Image from https://exemple.test/photo.png — png, 72 bytes."
RESULTAT_WEB_IMAGE = {"content": [{"type": "text", "text": LIGNE},
                                  {"type": "image", "data": PNG, "mimeType": "image/png"}]}


def _session(monkeypatch, resultat):
    def _post(self, corps, avec_entetes=False):
        methode = corps.get("method")
        if methode == "initialize":
            return ({"mcp-session-id": "s1"}, {}) if avec_entetes else {}
        if methode == "tools/list":
            return {"result": {"tools": [{"name": "web_image", "inputSchema": {
                "type": "object", "properties": {"url": {}}}}]}}
        if methode == "tools/call":
            return {"result": resultat}
        return {}

    monkeypatch.setattr(McpSession, "_post", _post)
    return McpSession(url="http://x", token="t")


# ── la lecture du résultat MCP ───────────────────────────────────────────────

def test_la_session_rend_le_texte_ET_l_image(monkeypatch):
    texte, erreur, images = _session(monkeypatch, RESULTAT_WEB_IMAGE).call(
        "web_image", {"url": "https://exemple.test/photo.png"})
    assert (texte, erreur) == (LIGNE, False)
    assert images == (Image("image/png", PNG),), \
        "l'image s'arrêtait au premier bloc texte (#32)"


def test_un_bloc_non_gere_leve_une_erreur_nommee():
    with pytest.raises(BlocNonGere, match="audio"):
        lire([{"type": "text", "text": "x"}, {"type": "audio", "data": "AA", "mimeType": "audio/wav"}])
    with pytest.raises(BlocNonGere, match="image/svg"):
        lire([{"type": "image", "data": "AA", "mimeType": "image/svg+xml"}])


def test_un_bloc_non_gere_arrive_au_modele_comme_une_erreur(monkeypatch):
    s = _session(monkeypatch, {"content": [{"type": "resource_link", "uri": "x://y"}]})
    spec = AgentSpec(system="s", tools=frozenset({"web_image"}))
    texte, erreur, _, images = agent_runtime.execute_tool(
        spec, s, agent_runtime.ToolCall(id="t0", name="web_image", arguments={}))
    assert erreur and "resource_link" in texte and images == ()


# ── chemin Anthropic : de bout en bout, jusqu'à la requête ───────────────────

class _Bloc:
    def __init__(self, **champs):
        self.__dict__.update(champs)

    def model_dump(self, mode=None):
        return dict(self.__dict__)


class _Reponse:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason = content, stop_reason
        self.usage = _Bloc(input_tokens=10, output_tokens=2)
        self.model = "claude-test"


class _Client:
    def __init__(self, reponses):
        self.messages, self.reponses, self.vu = self, list(reponses), []

    def create(self, **kwargs):
        self.vu.append(kwargs)
        return self.reponses.pop(0)


def test_l_image_de_web_image_atteint_la_requete_anthropic(monkeypatch):
    client = _Client([
        _Reponse([_Bloc(type="tool_use", id="tu1", name="web_image",
                        input={"url": "https://exemple.test/photo.png"})], "tool_use"),
        _Reponse([_Bloc(type="text", text="une photo")], "end_turn")])
    monkeypatch.setattr(agent_llm, "_sdk", lambda: _Bloc(Anthropic=lambda api_key=None: client))
    fil = []
    res = agent_runtime.run(
        AgentSpec(system="s", tools=frozenset({"web_image"})),
        _session(monkeypatch, RESULTAT_WEB_IMAGE), agent_llm, prompt="regarde",
        api_key="k", on_turn=lambda role, neutre, brut: fil.append((role, brut)))
    assert res.reply == "une photo"

    resultat = client.vu[1]["messages"][-1]["content"][0]
    assert resultat["type"] == "tool_result" and resultat["tool_use_id"] == "tu1"
    assert resultat["content"][0] == {"type": "text", "text": LIGNE}
    assert resultat["content"][1]["source"] == {
        "type": "base64", "media_type": "image/png", "data": PNG}

    persiste = [brut for role, brut in fil if role == "tool"][0]
    assert PNG not in str(persiste), "le backend plafonne un tour à 256 000 caractères"
    assert "non conservée dans le fil persisté" in persiste["content"][0]["content"]


def test_au_dela_du_plafond_du_deroule_l_image_est_dite_non_transmise():
    images = tuple(Image("image/png", PNG) for _ in range(3))
    admises, refus = admettre(images, DEROULE_MAX_IMAGES - 2, 0)
    assert len(admises) == 2 and len(refus) == 1 and "NON transmise" in refus[0]


# ── chemin OpenAI-compatible : le modèle lit qu'il ne reçoit pas l'image ─────

def test_la_voie_openai_dit_l_image_qu_elle_ne_transmet_pas():
    out = agent_llm_openai.tool_messages([{"id": "c1", "text": LIGNE, "is_error": False,
                                          "images": (Image("image/png", PNG),)}])
    assert out[0]["content"].startswith(LIGNE)
    assert "1 image(s)" in out[0]["content"] and PNG not in out[0]["content"]


# ── voie abonnement : le relais transmet, le fil persisté substitue ──────────

def test_le_relais_rend_l_image_au_cli(monkeypatch):
    relais = Relais(_session(monkeypatch, RESULTAT_WEB_IMAGE), frozenset({"web_image"}))
    rep = relais.repondre({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "web_image", "arguments": {}}})
    assert rep["result"]["content"] == [
        {"type": "text", "text": LIGNE},
        {"type": "image", "data": PNG, "mimeType": "image/png"}]


def test_le_fil_de_l_abonnement_ne_garde_pas_l_image():
    message = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
        {"type": "text", "text": LIGNE},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}]}
    contenu = agent_abonnement._message_borne(message)["content"][0]["content"]
    assert contenu[0]["text"] == LIGNE
    assert contenu[1]["type"] == "text" and "image/png" in contenu[1]["text"]
    assert PNG not in str(contenu)
