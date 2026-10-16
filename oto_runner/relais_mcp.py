"""Le RELAIS MCP d'un sandbox : Claude Code parle stdio, le relais parle à oto.

Pourquoi il existe : sur la voie `claude-subscription` (`agent_abonnement`), la
boucle d'outils tourne dans le CLI Claude Code, dans le sandbox de la personne — pas
dans le worker. Brancher le CLI directement sur `mcp.oto.cx` lui laisserait deux
gestes que le runner ne confie jamais au modèle : poser les jetons de contexte
(`_org`, `_project`, `_run_id` — mesuré le 08/09/2026 : un `_org` inventé quinze
fois) et respecter l'allowlist du travail. Le relais est une `McpSession`, la
même que la boucle du worker : il pose le contexte, retire ces jetons des schémas
servis, et refuse tout outil hors de l'allowlist.

Il tourne DANS le sandbox (sous l'utilisateur de la personne), lancé par le CLI via
`--mcp-config`. Il ne voit que ce que le travail lui remet par l'environnement :
le jeton DÉLÉGUÉ du travail (borné au bail), jamais le secret du worker.

Environnement : `OTO_MCP_URL`, `OTO_TOKEN`, `OTO_ORG`, `OTO_PROJECT`, `OTO_RUN_ID`,
`OTO_TOOLS` (noms séparés par des virgules).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from .contenu_outil import Image
from .mcp import McpSession

_VERSION_PAR_DEFAUT = "2025-06-18"


def _entier(nom: str) -> Optional[int]:
    v = (os.environ.get(nom) or "").strip()
    return int(v) if v else None


def _session() -> McpSession:
    return McpSession(url=os.environ["OTO_MCP_URL"], token=os.environ["OTO_TOKEN"],
                      project=_entier("OTO_PROJECT"), org=_entier("OTO_ORG"),
                      run_id=(os.environ.get("OTO_RUN_ID") or "").strip() or None)


def _allowlist() -> frozenset:
    return frozenset(n.strip() for n in (os.environ.get("OTO_TOOLS") or "").split(",")
                     if n.strip())


class Relais:
    """Répond aux messages JSON-RPC du CLI. Une instance = une session de travail."""

    def __init__(self, session: McpSession, outils: frozenset):
        self.session = session
        self.outils = outils

    def repondre(self, message: dict) -> Optional[dict]:
        methode = message.get("method")
        if "id" not in message:          # une notification : rien à rendre
            return None
        if methode == "initialize":
            version = (message.get("params") or {}).get("protocolVersion")
            return self._ok(message, {
                "protocolVersion": version or _VERSION_PAR_DEFAUT,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "oto", "version": "relais-sandbox"}})
        if methode == "ping":
            return self._ok(message, {})
        if methode == "tools/list":
            return self._ok(message, {"tools": [
                {"name": s["name"], "description": s["description"],
                 "inputSchema": s["input_schema"]}
                for s in self.session.schemas(self.outils)]})
        if methode == "tools/call":
            p = message.get("params") or {}
            nom = p.get("name") or ""
            if nom not in self.outils:
                return self._ok(message, self._contenu(
                    f"`{nom}` n'est pas autorisé pour ce travail.", erreur=True))
            texte, erreur, images = self.session.call(nom, p.get("arguments") or {})
            return self._ok(message, self._contenu(texte, erreur=erreur, images=images))
        return {"jsonrpc": "2.0", "id": message["id"],
                "error": {"code": -32601, "message": f"méthode inconnue : {methode}"}}

    @staticmethod
    def _ok(message: dict, resultat: dict) -> dict:
        return {"jsonrpc": "2.0", "id": message["id"], "result": resultat}

    @staticmethod
    def _contenu(texte: str, erreur: bool, images: tuple[Image, ...] = ()) -> dict:
        """Le résultat rendu au CLI : le texte, puis les images en blocs MCP —
        le CLI les fait voir au modèle (#32 : elles s'arrêtaient au relais)."""
        return {"content": [{"type": "text", "text": texte}, *(
                    {"type": "image", "data": i.data, "mimeType": i.media_type}
                    for i in images)],
                "isError": erreur}


def main() -> None:
    relais = Relais(_session(), _allowlist())
    for ligne in sys.stdin:
        if not ligne.strip():
            continue
        reponse = relais.repondre(json.loads(ligne))
        if reponse is not None:
            sys.stdout.write(json.dumps(reponse, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
