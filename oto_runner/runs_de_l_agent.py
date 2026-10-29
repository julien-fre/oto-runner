"""Les runs que l'AGENT ouvre lui-même — et qu'il oublie parfois de clore.

Le worker ouvre un run TECHNIQUE par travail et le clôt toujours (`conclusion`).
Mais une procédure dit souvent à l'agent d'ouvrir le SIEN (`run_start(label=…,
guide=…)`, étape 1) et de le clore à la fin (`run_finish`, dernière étape). Un
agent qui termine son tour sans cette dernière étape laisse ce run ouvert POUR
TOUJOURS : rien d'autre ne le ferme. Vécu le 29/10/2026 sur la voie abonnement :
le run de l'agent est resté « en cours, sans fin », l'alerte des agents l'a
signalé une heure plus tard, alors que le travail avait conclu `done` en deux
minutes et que les cinq runs suivants avaient tourné normalement.

Ce module SUIT ces runs dans le journal du travail — les deux formes qu'il
reçoit : l'événement `outil` de la boucle ordinaire, et l'événement `claude` du
flux du CLI (voies abonnement et ferme). Il ne juge rien de ce que l'agent a
fait : il retient quels runs l'agent a ouverts et lesquels il a clos, pour que
la conclusion ferme les autres (`conclusion.clore_ceux_de_l_agent`).

⚠️ Un `run_finish` REFUSÉ (`partial` sans note, par exemple) ne clôt rien : seul
un appel rendu sans erreur retire le run de la liste.
"""
from __future__ import annotations

import json
import re
from typing import Optional

_RUN_ID = re.compile(r'"run_id"\s*:\s*"([^"]+)"')


def _est(nom: Optional[str], verbe: str) -> bool:
    # ⚠️ Le connecteur MCP peut PRÉFIXER les noms (`mcp__oto__run_start`,
    # `<connecteur>_run_start`) : l'appartenance se teste par SUFFIXE.
    return bool(nom) and (nom == verbe or nom.endswith("_" + verbe))


def _texte(contenu) -> str:
    """Le contenu d'un `tool_result` du CLI : une chaîne, ou des blocs `text`."""
    if isinstance(contenu, str):
        return contenu
    if isinstance(contenu, list):
        return "\n".join(b.get("text") or "" for b in contenu
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _run_id_rendu(texte: str) -> Optional[str]:
    try:
        d = json.loads(texte)
        if isinstance(d, dict) and isinstance(d.get("run_id"), str):
            return d["run_id"]
    except (ValueError, TypeError):
        pass
    m = _RUN_ID.search(texte or "")
    return m.group(1) if m else None


def _run_id_demande(arguments) -> Optional[str]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if isinstance(arguments, dict) and isinstance(arguments.get("run_id"), str):
        return arguments["run_id"]
    return None


class RunsDeLAgent:
    """Les runs ouverts par l'agent et pas encore clos par lui, dans l'ordre d'ouverture."""

    def __init__(self) -> None:
        self._ouverts: dict = {}     # run_id → None (un dict garde l'ordre)
        self._en_vol: dict = {}      # tool_use_id → (nom, arguments) — flux du CLI

    def ouverts(self) -> list:
        return list(self._ouverts)

    def evenement(self, ev: str, champs: dict) -> None:
        """La forme `on_event(ev, champs)` du journal. Ne lève jamais : suivre ces
        runs est un service rendu, pas une condition du travail."""
        try:
            if ev == "outil":
                self._issue(champs.get("nom"), champs.get("arguments"),
                            ok=bool(champs.get("ok")), texte=champs.get("texte") or "")
            elif ev == "claude":
                self._flux(champs)
        except Exception:  # noqa: BLE001 — cf. docstring
            pass

    def _flux(self, ev: dict) -> None:
        message = ev.get("message") or {}
        if ev.get("type") == "assistant":
            for bloc in message.get("content") or []:
                if isinstance(bloc, dict) and bloc.get("type") == "tool_use":
                    self._en_vol[bloc.get("id")] = (bloc.get("name"), bloc.get("input"))
        elif ev.get("type") == "user":
            for bloc in message.get("content") or []:
                if isinstance(bloc, dict) and bloc.get("type") == "tool_result":
                    nom, arguments = self._en_vol.pop(bloc.get("tool_use_id"), (None, None))
                    self._issue(nom, arguments, ok=not bloc.get("is_error"),
                                texte=_texte(bloc.get("content")))

    def _issue(self, nom, arguments, ok: bool, texte: str) -> None:
        if not ok:
            return
        if _est(nom, "run_start"):
            run_id = _run_id_rendu(texte)
            if run_id:
                self._ouverts[run_id] = None
        elif _est(nom, "run_finish"):
            self._ouverts.pop(_run_id_demande(arguments), None)
