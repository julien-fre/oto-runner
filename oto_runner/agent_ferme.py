"""La voie `claude-farm` : un travail de la famille `anthropic`, exécuté par Claude Code
dans la FERME, sur la clé d'API de l'organisation.

Le pendant de `agent_abonnement` (la voie `claude-subscription`), avec ce qui change
quand c'est une CLÉ qui paie et non une session :

- **Le sandbox est celui de l'ORG**, pas d'une personne : `o<empreinte de l'org>`, créé
  au premier run (l'agent de la ferme le fait sans effet s'il existe). Il ne détient
  rien — aucune session, aucune clé.
- **La clé part avec CHAQUE run** (`api_key`, et `workspace` pour une clé
  d'organisation Anthropic) : la ferme l'écrit pour ce run seul et l'efface à la fin.
  Aucune copie ne vit sur la box ; la rotation dans le coffre vaut au run suivant.
- **Le CLI doit annoncer `apiKeySource: ANTHROPIC_API_KEY`** : une session qui paierait
  à la place de la clé arrête le run avant son premier tour.
- **Pas de verrou par sandbox** côté ferme (aucun rafraîchissement OAuth à protéger) :
  plusieurs travaux d'une même org tournent en parallèle, dans la limite des places de
  la box. Une box pleine (429) se réessaie ici, bail prolongé, avant d'échouer.
- Aucun forfait à rapporter : `abonnement` reste vide.

Ce qui ne change pas : le flux lu (`agent_abonnement.lire_flux`), ses limites en vol
(`max_tokens`, `max_seconds`) et celles que la ferme tient (`max_turns` = les tours
déclarés sur l'agent ; le budget reste celui de la ferme), le fil tenu en direct, le
relais MCP dans le sandbox.

Opt-in par processus : `OTO_RUNNER_PROVIDER=claude-farm`, dépôt `anthropic`. Prévu pour
un worker « clés clients seules » (`OTO_RUNNER_ORG_KEYS_ONLY=1`) : la plateforme ne
paie aucun run. Environnement : `OTO_FERME_URL`, `OTO_FERME_TOKEN`, `OTO_MCP_URL`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Callable, Optional

import requests

from . import agent_abonnement
from .agent_runtime import AgentResult

logger = logging.getLogger("oto_runner")

ONE_SHOT = True
SANDBOX = True
MOTEUR = "claude_code_ferme"   # ce que le résultat dit de ce qui a tourné
#: Le moteur que le worker DÉCLARE au claim (`engine`) : seul un worker qui le déclare
#: réserve les travaux d'une org routée vers la ferme (option d'org `claude_farm`,
#: oto-backend). Sans lui, ces travaux attendraient ce worker-ci sans qu'il les voie.
ENGINE = "farm"
EFFORT_SERVI = True       # le CLI prend `--effort` (cf. `worker._exiger_effort_servi`)
ORG_DU_TRAVAIL = True     # le sandbox est celui de l'org DU TRAVAIL (`worker._contexte_du_sandbox`)
#: Ne démarre qu'en worker « clés clients seules » (`worker._verifier_cle_au_demarrage`) :
#: sans ce mode, le backend lui servirait les agents posés SANS modèle (qu'aucune clé ne
#: paie ici) et ceux des orgs sans clé déposée, au lieu de les arrêter à la réservation.
CLES_CLIENTS_EXIGEES = True
WORKSPACE_SERVI = True    # le workspace d'une clé d'organisation part avec le run
FAMILLE = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5"
SOURCE_ATTENDUE = "ANTHROPIC_API_KEY"

_ENV_AGENT, _ENV_JETON = agent_abonnement._ENV_AGENT, agent_abonnement._ENV_JETON
FermePleine = agent_abonnement.FermePleine
# La file des places de la box : la même pour les deux voies, qui partagent ses places.
RESERVE_SA_PLACE = agent_abonnement.RESERVE_SA_PLACE
PlacesNonServies = agent_abonnement.PlacesNonServies
reserver_place, rendre_place = agent_abonnement.reserver_place, agent_abonnement.rendre_place
tenir_place, lacher_place = agent_abonnement.tenir_place, agent_abonnement.lacher_place
_PLEINE_ESSAIS = agent_abonnement._PLEINE_ESSAIS

_sandboxes_creees: set = set()


class LlmUnavailable(RuntimeError):
    pass


def model() -> str:
    return os.environ.get("OTO_RUNNER_MODEL") or DEFAULT_MODEL


def depot() -> str:
    return FAMILLE


def resolve_key() -> str:
    """Aucune clé à ce worker : il ne paie rien. Ce qu'il lui faut, c'est l'agent de la
    ferme. La clé est celle de l'org, remise avec chaque travail (`run_once`)."""
    for nom in (_ENV_AGENT, _ENV_JETON):
        if not (os.environ.get(nom) or "").strip():
            raise LlmUnavailable(f"{nom} absent de l'environnement du worker")
    return ""


def sandbox_de_l_org(org_id) -> str:
    """Le sandbox d'une org : un nom stable, qui ne dit pas laquelle. Préfixe `o` (celui
    d'une personne est `u`, côté backend) : les deux ne se rencontrent jamais."""
    return "o" + hashlib.sha256(f"org:{org_id}".encode()).hexdigest()[:20]


def _url(chemin: str) -> str:
    return f"{os.environ[_ENV_AGENT].rstrip('/')}/api/sandboxes/{chemin}"


def _entetes(**extra) -> dict:
    return {"Authorization": f"Bearer {os.environ[_ENV_JETON]}", **extra}


def _assurer_sandbox(slug: str) -> None:
    """Crée le sandbox de l'org s'il n'existe pas — une fois par processus : la ferme
    répond sans rien faire à un sandbox qui existe déjà."""
    if slug in _sandboxes_creees:
        return
    r = requests.put(_url(slug), headers=_entetes(), timeout=(10, 60))
    if r.status_code != 200:
        raise RuntimeError(f"agent de la ferme : création du sandbox → {r.status_code} {r.text[:300]}")
    _sandboxes_creees.add(slug)


def run_once(*, instructions: str, inputs: str, tools, api_key: Optional[str] = None,
             modele: Optional[str] = None, on_event=None, sandbox: Optional[str] = None,
             mcp=None, prolonger: Optional[Callable[[], None]] = None,
             apposer: Optional[Callable[[str, dict, dict], None]] = None,
             max_tokens: Optional[int] = None, max_seconds: Optional[int] = None,
             effort: Optional[str] = None, workspace: Optional[str] = None,
             org: Optional[int] = None, max_turns: Optional[int] = None,
             horloge=time.monotonic, attendre=time.sleep) -> AgentResult:
    """UN run complet dans le sandbox de l'org, sur la clé de l'org, → AgentResult.

    `org` : l'org DU TRAVAIL (celle que la réservation rend), jamais celle de la session
    MCP — une charge sans org ferait partager un seul sandbox à toutes."""
    if not api_key:
        # La plateforme ne paie aucun run : sans la clé de l'org, rien ne part.
        raise RuntimeError("travail sans clé d'organisation : la voie ferme ne tourne que "
                           "sur la clé que l'org a déposée")
    if mcp is None:
        raise RuntimeError("travail sans session MCP : le relais n'aurait aucun contexte")
    resolve_key()
    if sandbox is None and org is None:
        raise RuntimeError("travail sans org : aucun sandbox d'org où le lancer")
    slug = sandbox or sandbox_de_l_org(org)
    _assurer_sandbox(slug)
    corps = {
        "prompt": inputs,
        "model": modele or model(),
        "system": instructions,
        "auth": "key",
        "api_key": api_key,
        **({"workspace": workspace} if workspace else {}),
        **({"effort": effort} if effort else {}),
        "mcp": {"url": mcp.url, "token": mcp.token, "org": mcp.org, "project": mcp.project,
                "run_id": mcp.run_id, "tools": sorted(tools or ())},
        **({"max_seconds": int(max_seconds)} if max_seconds else {}),
        **agent_abonnement.champ_des_tours(max_turns),
    }
    lecture = (min(agent_abonnement._LECTURE_S, max_seconds) if max_seconds
               else agent_abonnement._LECTURE_S)
    with agent_abonnement.poster_le_run(
            _url(f"{slug}/runs"), corps, _entetes(Accept="application/x-ndjson"), lecture,
            prolonger=prolonger, on_event=on_event, attendre=attendre,
            place=agent_abonnement.prendre_place()) as r:
        if r.status_code != 200:
            raise RuntimeError(f"agent de la ferme : {r.status_code} {r.text[:300]}")
        echeance = agent_abonnement.echeance_du_run(max_seconds, horloge)
        lignes = (json.loads(l) for l in r.iter_lines(decode_unicode=True) if l)
        res = agent_abonnement.lire_flux(
            agent_abonnement._jusqu_a(lignes, echeance, horloge), on_event=on_event,
            prolonger=prolonger, apposer=apposer, horloge=horloge,
            max_tokens=max_tokens, echeance=echeance, source_attendue=SOURCE_ATTENDUE)
    # Une clé ne porte pas de forfait : rien à rapporter comme un abonnement.
    res.abonnement = None
    return res
