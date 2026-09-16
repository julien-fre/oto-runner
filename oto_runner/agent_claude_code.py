"""Le moteur CLAUDE CODE : la boucle de Claude Code (Agent SDK), servie par le worker.

Pourquoi il existe : un run hébergé devait pouvoir faire ce que fait une session
`claude -p` lancée sur GitHub Actions contre la face MCP — un runner client en est
la référence. La boucle maison (`agent_runtime`) n'a ni délégation à des
sous-agents, ni compaction (elle TRONQUE à 60 messages), ni tours au-delà de 64 :
un sourcing qui pagine une centaine de candidats y perd la procédure en route ou
s'arrête `blocked`. Ce moteur délègue la boucle entière à Claude Code, qui a les
trois.

Ce qui ne change PAS, et c'est la raison de la forme :

- **Les outils passent par la session MCP du travail** (`McpSession`), exposée à
  Claude Code comme un serveur MCP EN PROCESSUS. Claude Code ne parle jamais au
  backend directement : les jetons de contexte (`_org`, `_project`, `_run_id`)
  restent posés par le worker, les paramètres de cycle de vie restent retirés, la
  session perdue reste rouverte. Un Claude Code branché en HTTP sur `/mcp` aurait
  tout perdu d'un coup, sans rien casser de visible.
- **L'allowlist est le catalogue** : le serveur en processus n'expose QUE les
  outils du travail. Un sous-agent hérite de ce serveur ; il ne peut donc pas voir
  plus que son parent — la garantie ne dépend d'aucun réglage de permission.
- **Aucun outil intégré** hors la délégation à un sous-agent (`Agent`) : ni
  shell, ni fichiers, ni web. Aucun réglage disque lu (`setting_sources=[]`), aucun
  autre serveur MCP (`strict_mcp_config`), un répertoire de configuration NEUF par
  travail — rien ne passe d'une org à l'autre.
- **La clé est celle du travail** (dépôt `anthropic`), sinon celle du worker ;
  jamais un abonnement. Les secrets du worker sont effacés de l'environnement
  hérité par le sous-processus.

Ce qui change : c'est un chemin ONE-SHOT (comme Conversations) — pas de tours
apposés au fil, le verbatim va au journal du travail. Les appels d'outils sont
SÉRIALISÉS (la session MCP n'est pas réentrante, et sa deadline SIGALRM exige le
thread principal) : des sous-agents parallèles attendent leur tour d'outil.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shutil
import tempfile
import time
from typing import Callable, Optional

from . import agent_llm
from .agent_runtime import AgentResult, AgentStep, _cap, max_tool_output
from .comptage import Compteur
from .deadline import DeadlineExceeded
from .llm_types import EFFORT_SANS_RAISONNEMENT, LlmUnavailable

logger = logging.getLogger("oto_runner")

ONE_SHOT = True
#: `run_once` reçoit la session MCP, le cadre et le workspace du travail.
OUTILS_LOCAUX = True
#: L'effort de réflexion du travail est servi (`ClaudeAgentOptions.effort`).
EFFORT_SERVI = True

SERVEUR = "oto"
PREFIXE = f"mcp__{SERVEUR}__"
#: Le seul outil intégré offert : la délégation à un sous-agent. Le CLI 2.1.273 le
#: sert sous le nom `Task` quand on demande `Agent` ; les deux sont autorisés.
OUTILS_INTEGRES = ("Agent",)
_NOMS_DELEGATION = ("Agent", "Task")

#: Plafond de tours du fil principal. La boucle maison plafonne à 64 et la
#: plateforme accepte davantage ; ici le travail est servi jusqu'à ce plafond, et
#: une valeur au-delà est DITE au journal plutôt que rabotée en silence.
PLAFOND_TOURS = 400

_ENV_WALL = "OTO_RUNNER_CLAUDE_CODE_WALL_S"
#: Sous la patience de systemd à l'arrêt (16 min, `docs/deploiement-et-arret.md`) : au-delà,
#: un déploiement tuerait un déroulé qui avait le droit de finir. La relever exige de
#: relever `TimeoutStopSec` avec elle.
_WALL_DEFAUT_S = 900
_BATTEMENT_S = 60

#: Ce que le sous-processus hérite du worker et ne doit JAMAIS lire : il n'a pas
#: d'outil pour le faire, mais un secret absent ne se divulgue pas.
_SECRETS_DU_WORKER = ("OTO_WORKER_SECRET", "OTO_TOKEN", "OTO_RUNNER_OPENAI_API_KEY",
                      "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


def model() -> str:
    return agent_llm.model()


def depot() -> str:
    return "anthropic"


def resolve_key() -> str:
    return agent_llm.resolve_key()


def _sdk():
    try:
        import claude_agent_sdk  # noqa: PLC0415 — import gardé : sans la lib, pas de moteur
    except ImportError as e:
        raise LlmUnavailable(
            "claude-agent-sdk absent : installe l'extra `claude-code` "
            "(`pip install oto-runner[claude-code]`)") from e
    return claude_agent_sdk


def wall_s() -> int:
    brut = os.environ.get(_ENV_WALL, "").strip()
    if not brut:
        return _WALL_DEFAUT_S
    if not brut.isdigit() or int(brut) < 1:
        raise LlmUnavailable(f"{_ENV_WALL} = {brut!r} : un entier ≥ 1 est attendu")
    return int(brut)


@dataclasses.dataclass
class _Etat:
    steps: list = dataclasses.field(default_factory=list)
    panne: Optional[str] = None


def _outils(sdk, mcp, noms: frozenset, etat: _Etat, note) -> list:
    """L'allowlist du travail, en outils du serveur MCP en processus."""
    limite = max_tool_output()
    schemas = mcp.schemas(noms)
    manquants = sorted(noms - {s["name"] for s in schemas})
    if manquants:
        note("outils_absents", noms=manquants)
    return [_un_outil(sdk, mcp, s, limite, etat, note) for s in schemas]


def _un_outil(sdk, mcp, schema: dict, limite: int, etat: _Etat, note):
    nom = schema["name"]

    async def appeler(args):
        debut = time.monotonic()
        try:
            # Synchrone, sur le thread de la boucle : la deadline de la session est un
            # SIGALRM, et la session n'est pas réentrante — les appels s'enchaînent.
            texte, erreur = mcp.call(nom, dict(args or {}))
        except Exception as e:  # noqa: BLE001 — un transport mort arrête le TRAVAIL
            # Rendre l'exception au modèle la ferait lire comme une réponse métier :
            # il l'annoncerait et conclurait « done » sans écriture. Le travail échoue.
            etat.panne = f"{nom} : {type(e).__name__}: {e}"
            etat.steps.append(AgentStep(tool=nom, ok=False,
                                        duration_ms=int((time.monotonic() - debut) * 1000),
                                        error=str(e)[:300], transport_ko=True))
            return {"content": [{"type": "text",
                                 "text": "Transport indisponible : le travail s'arrête."}],
                    "is_error": True}
        lu, coupe = _cap(texte, limite)
        note("outil", nom=nom, erreur=erreur, sortie=texte, coupee=coupe)
        etat.steps.append(AgentStep(tool=nom, ok=not erreur,
                                    duration_ms=int((time.monotonic() - debut) * 1000),
                                    error=(texte[:300] if erreur else None)))
        return {"content": [{"type": "text", "text": lu}], "is_error": bool(erreur)}

    return sdk.tool(nom, schema.get("description") or "", schema["input_schema"])(appeler)


def _environnement(cle: str, workspace: Optional[str], spec, dossier: str) -> dict:
    env = {s: "" for s in _SECRETS_DU_WORKER}
    env.update({
        "ANTHROPIC_API_KEY": cle,
        "CLAUDE_CONFIG_DIR": os.path.join(dossier, "config"),
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # Notre plafond en caractères est celui qui mord, et il le DIT au modèle
        # (`_cap`). Celui du CLI est en jetons ; un jeton ne fait jamais moins d'un
        # caractère, donc le même nombre le laisse toujours au-dessus.
        "MAX_MCP_OUTPUT_TOKENS": str(max_tool_output()),
    })
    if workspace:
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"anthropic-workspace-id: {workspace}"
    if spec is not None and spec.max_output_tokens:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(spec.max_output_tokens))
    return env


def _options(sdk, *, instructions: str, serveur, noms: frozenset, spec, modele,
             env: dict, dossier: str, note):
    demande = spec.max_steps if spec is not None else PLAFOND_TOURS
    tours = max(1, min(int(demande), PLAFOND_TOURS))
    if tours != demande:
        note("plafond_tours", demande=demande, servi=tours, plafond=PLAFOND_TOURS)
    effort = (spec.effort if spec is not None else None) or agent_llm.effort_hote()
    reglages = {
        "system_prompt": {"type": "preset", "preset": "claude_code",
                          "append": instructions},
        "mcp_servers": {SERVEUR: serveur},
        "strict_mcp_config": True,
        "tools": list(OUTILS_INTEGRES),
        "allowed_tools": ([PREFIXE + n for n in sorted(noms)]
                          + list(OUTILS_INTEGRES) + ["Task"]),
        "permission_mode": "dontAsk",
        "setting_sources": [],
        "max_turns": tours,
        "model": modele,
        "cwd": dossier,
        "env": env,
    }
    if effort and effort != EFFORT_SANS_RAISONNEMENT:
        reglages["effort"] = effort
    if spec is not None and spec.temperature is not None:
        # Claude Code n'expose pas de température : la servir en silence ferait croire
        # que deux passages comparés l'étaient à réglage égal.
        note("temperature_non_servie", temperature=spec.temperature)
        logger.warning("température %s déclarée et non servie par Claude Code",
                       spec.temperature)
    note("claude_code", **{k: v for k, v in reglages.items()
                           if k not in ("env", "mcp_servers")},
         env=sorted(k for k in env if env[k]))
    return sdk.ClaudeAgentOptions(**reglages)


_CLE_REFUSEE = (401, 403)

_ARRETS = {"success": "end_turn", "error_max_turns": "max_steps",
           "error_max_budget_usd": "max_tokens"}


def _usage_du_resultat(resultat) -> dict:
    """L'usage de la SESSION, sous-agents compris : `model_usage` par modèle quand il
    est rendu (il couvre les sous-agents), sinon `usage`."""
    par_modele = getattr(resultat, "model_usage", None) or {}
    if par_modele:
        def somme(cle):
            valeurs = [m.get(cle) for m in par_modele.values() if isinstance(m, dict)]
            return None if any(v is None for v in valeurs) else sum(int(v) for v in valeurs)
        entree, sortie = somme("inputTokens"), somme("outputTokens")
        lus, ecrits = somme("cacheReadInputTokens"), somme("cacheCreationInputTokens")
    else:
        u = getattr(resultat, "usage", None) or {}
        entree, sortie = u.get("input_tokens"), u.get("output_tokens")
        lus, ecrits = u.get("cache_read_input_tokens"), u.get("cache_creation_input_tokens")
    usage = {"input_tokens": entree, "output_tokens": sortie,
             "cache_read_input_tokens": lus, "cache_creation_input_tokens": ecrits}
    if None not in (entree, lus, ecrits):
        usage["input_total_tokens"] = entree + lus + ecrits
    return {k: v for k, v in usage.items() if v is not None}


def _jetons_du_message(message) -> int:
    u = getattr(message, "usage", None) or {}
    return sum(int(u.get(k) or 0) for k in
               ("input_tokens", "output_tokens", "cache_creation_input_tokens"))


async def _derouler(sdk, prompt: str, options, *, etat: _Etat, spec, note,
                    battre: Callable[[], None]):
    resultat, texte, servi, arret = None, "", None, None
    vus, budget = set(), 0
    plafond = spec.max_tokens if spec is not None else None
    flux = sdk.query(prompt=prompt, options=options).__aiter__()
    try:
        while True:
            try:
                message = await flux.__anext__()
            except StopAsyncIteration:
                break
            except Exception as e:
                # Un résultat en erreur (plafond de tours, clé refusée…) fait sortir le CLI
                # en code 1, et le SDK lève `ResultError` APRÈS avoir rendu le résultat.
                # Le résultat est déjà lu : c'est lui qui conclut, pas l'exception.
                if resultat is not None and type(e).__name__ == "ResultError":
                    break
                raise
            genre = type(message).__name__
            note("message", genre=genre, contenu=message)
            battre()
            if genre == "SystemMessage" and getattr(message, "subtype", "") == "api_retry":
                statut = (getattr(message, "data", None) or {}).get("error_status")
                if statut in _CLE_REFUSEE:
                    # Le CLI rejoue une clé refusée dix fois, en minutes d'attente croissante :
                    # une clé révoquée n'est pas un transitoire.
                    raise RuntimeError(f"clé de modèle refusée par Anthropic ({statut}) — "
                                       "vérifie la clé déposée par l'org (ou celle du worker)")
            if genre == "AssistantMessage":
                principal = getattr(message, "parent_tool_use_id", None) is None
                for bloc in getattr(message, "content", None) or ():
                    nom_bloc = type(bloc).__name__
                    if nom_bloc == "ToolUseBlock" and getattr(bloc, "name", "") in _NOMS_DELEGATION:
                        # Pas un pas : `tool_counts` compte des appels RÉUSSIS, et une
                        # délégation n'a pas encore d'issue quand elle part. Elle se dit
                        # au journal ; ses appels d'outils, eux, sont comptés un à un.
                        note("delegation", parent_principal=principal,
                             entree=getattr(bloc, "input", None))
                    elif nom_bloc == "TextBlock" and principal and getattr(bloc, "text", ""):
                        texte = bloc.text
                if principal and getattr(message, "model", None):
                    servi = message.model
                # Un message d'API arrive découpé en plusieurs messages (un par bloc), qui
                # portent tous son usage : il ne se compte qu'une fois.
                identifiant = getattr(message, "message_id", None)
                if plafond and (identifiant is None or identifiant not in vus):
                    vus.add(identifiant)
                    budget += _jetons_du_message(message)
                    if budget > plafond:
                        arret = "max_tokens"
                        note("budget_depasse", jetons=budget, plafond=plafond)
                        break
            elif genre == "ResultMessage":
                resultat = message
            if etat.panne:
                break
    finally:
        # Sortir tôt (budget, panne, clé refusée) doit arrêter le sous-processus, pas
        # le laisser dépenser derrière un travail déjà conclu.
        fermer = getattr(flux, "aclose", None)
        if fermer is not None:
            await fermer()
    return resultat, texte, servi, arret


def run_once(*, instructions: str, inputs: str, tools, api_key: Optional[str] = None,
             modele: Optional[str] = None, on_event=None, mcp=None, spec=None,
             workspace: Optional[str] = None,
             heartbeat: Optional[Callable[[], None]] = None) -> AgentResult:
    """UN déroulé Claude Code complet → AgentResult.

    Lève : `LlmUnavailable` (SDK ou clé absents), `DeadlineExceeded` (au-delà de
    `OTO_RUNNER_CLAUDE_CODE_WALL_S`), `RuntimeError` (transport MCP mort, ou
    Claude Code sorti sans résultat) — le retry de job décide."""
    if mcp is None:
        raise LlmUnavailable("le moteur Claude Code exige la session MCP du travail")
    sdk = _sdk()
    cle = api_key or resolve_key()
    nom = modele or model()
    noms = frozenset(tools or ())

    def note(ev: str, **champs) -> None:
        if on_event:
            on_event(ev, champs)

    dernier = [0.0]

    def battre() -> None:
        if heartbeat is None or time.monotonic() - dernier[0] < _BATTEMENT_S:
            return
        dernier[0] = time.monotonic()
        heartbeat()

    etat = _Etat()
    dossier = tempfile.mkdtemp(prefix="oto-claude-code-")
    try:
        serveur = sdk.create_sdk_mcp_server(SERVEUR, tools=_outils(sdk, mcp, noms, etat, note))
        options = _options(sdk, instructions=instructions, serveur=serveur, noms=noms,
                           spec=spec, modele=nom,
                           env=_environnement(cle, workspace, spec, dossier),
                           dossier=dossier, note=note)
        limite = wall_s()
        try:
            resultat, texte, servi, arret = asyncio.run(asyncio.wait_for(
                _derouler(sdk, inputs, options, etat=etat, spec=spec, note=note,
                          battre=battre), timeout=limite))
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded(
                f"déroulé Claude Code > {limite}s wall-clock ({_ENV_WALL})") from e
    finally:
        shutil.rmtree(dossier, ignore_errors=True)

    if etat.panne:
        raise RuntimeError(f"transport MCP mort pendant le déroulé — {etat.panne}")
    compte = Compteur()
    defaut = None
    if resultat is not None:
        compte.ajouter(_usage_du_resultat(resultat))
        note("resultat_claude_code", sous_type=resultat.subtype,
             tours=resultat.num_turns, cout_usd=resultat.total_cost_usd,
             duree_ms=resultat.duration_ms, refus_permission=resultat.permission_denials,
             # `model` du bilan ne nomme que le fil principal : les sous-agents peuvent
             # en servir d'autres, et c'est ici qu'ils se lisent.
             modeles=sorted((getattr(resultat, "model_usage", None) or {}).keys()))
    if arret is None:
        if resultat is None:
            raise RuntimeError("Claude Code s'est terminé sans message de résultat")
        if getattr(resultat, "stop_reason", None) == "refusal":
            arret = "refusal"
        else:
            arret = _ARRETS.get(resultat.subtype)
            if arret is None or (arret == "end_turn" and resultat.is_error):
                arret = "fin_anormale"
                defaut = {"forme": "fin_anormale", "finish_reason": resultat.subtype}
    reponse = (getattr(resultat, "result", None) or texte or "").strip()
    return AgentResult(reply=reponse, steps=list(etat.steps), stopped=arret,
                       usage=compte.usage(), couverture=compte.couverture(),
                       model=servi or nom, defaut=defaut)
