"""Client LLM « chat + tool calling » — le seam provider du worker.

Transplanté du prototype `server-agent` d'oto-backend (25/07), avec UNE
simplification structurelle : ici la clé vient de l'ENVIRONNEMENT, point. Le
worker est un client pur (ADR 0064-D1) — il ne connaît ni coffre, ni cascade,
ni org : l'org qui paie se choisit en amont, en posant la clé dans l'env de
l'unit du worker (V1 : un worker = une org). La cascade de credentials reste
côté backend, où elle a toujours vécu.

Le reste du contrat est inchangé, et c'est lui qui compte :
- `complete()` rend un `Turn` : texte visible + `tool_calls` demandés +
  `raw_content` — les blocs BRUTS de la réponse, à réinjecter INCHANGÉS dans le
  fil comme tour assistant (les blocs de pensée se rejouent verbatim, les
  reconstruire casse le tour suivant). C'est exactement ce que le fil du
  backend stocke en `provider_raw` (R1) : ce module et la table parlent la
  même langue.
- `stop_reason` se lit AVANT le contenu : un `refusal` revient en HTTP 200 avec
  un contenu vide — c'est un tour terminal, pas un crash.
- La profondeur se règle par `output_config.effort`, jamais par temperature.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from .comptage import POSTES
from .llm_types import EFFORT_SANS_RAISONNEMENT, LlmUnavailable, ToolCall, Turn

# Sonnet par défaut — divergence ASSUMÉE avec le prototype (Opus) : un run hébergé
# tourne sans humain qui regarde le compteur, et la campagne réelle a montré qu'un
# modèle de milieu de gamme exécute correctement une procédure cadrée. Opus se
# choisit par flotte ou par env quand le jugement le justifie.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 8192

_ENV_KEY = "ANTHROPIC_API_KEY"


def model() -> str:
    return os.environ.get("OTO_RUNNER_MODEL") or DEFAULT_MODEL


def effort_hote() -> str:
    return os.environ.get("OTO_RUNNER_EFFORT") or DEFAULT_EFFORT


def max_tokens() -> int:
    try:
        return int(os.environ.get("OTO_RUNNER_MAX_TOKENS", "") or DEFAULT_MAX_TOKENS)
    except ValueError:
        return DEFAULT_MAX_TOKENS


def depot() -> str:
    """Le nom du dépôt de clé que ce provider sait consommer (voir `worker`).

    L'org qui a déposé sa clé Anthropic la voit servie avec chaque travail que
    ses agents réservent ; sans dépôt, le worker tourne sur la clé de la
    plateforme, prise dans son environnement.
    """
    return "anthropic"


def resolve_key() -> str:
    """La clé, de l'env, ou une erreur ACTIONNABLE — jamais un repli muet."""
    key = os.environ.get(_ENV_KEY, "").strip()
    if not key:
        raise LlmUnavailable(
            f"{_ENV_KEY} absente de l'environnement du worker — c'est elle qui "
            "décide quelle org paie ses tours (V1 : un worker = une org)")
    return key


def _sdk():
    try:
        import anthropic  # noqa: PLC0415 — import guardé : sans la lib, pas de worker
        return anthropic
    except Exception:  # noqa: BLE001
        return None


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("type") or "")
    return getattr(block, "type", "") or ""


def _block_to_dict(block: Any) -> dict:
    """Bloc de réponse → dict JSON-sérialisable, réinjectable TEL QUEL (et stockable
    tel quel en `provider_raw` du fil). On ne reconstruit jamais un bloc de pensée :
    on le transporte."""
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if v is not None}
    dump = getattr(block, "model_dump", None)
    if dump is not None:
        try:
            return {k: v for k, v in dump(mode="json").items() if v is not None}
        except Exception:  # noqa: BLE001
            pass
    return {"type": _block_type(block) or "text",
            "text": str(getattr(block, "text", "") or "")}


# ── La FORME du fil, confinée ici (la boucle ne connaît que ces 4 hooks) ─────
def user_message(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_message(turn: Turn) -> dict:
    """Le tour assistant à réinjecter — blocs BRUTS, jamais reconstruits."""
    return {"role": "assistant", "content": turn.raw_content}


def tool_messages(results: list[dict]) -> list[dict]:
    """Les résultats d'outils d'UN tour → messages de fil. Anthropic : TOUS dans
    UN message user (les scinder apprend au modèle à cesser de paralléliser).
    `results` = [{id, text, is_error}]."""
    return [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": r["id"],
         "content": r["text"], "is_error": r["is_error"]} for r in results]}]


def format_tools(schemas: list[dict]) -> list[dict]:
    """Schémas NEUTRES ({name, description, input_schema}) → format Anthropic
    (identique, c'est lui qui a fixé la forme neutre)."""
    return list(schemas)


# ── Cache de prompt : un tour ne repaie plus la procédure ────────────────────
# Le modèle est SANS ÉTAT : la boucle lui renvoie tout à chaque tour — procédure,
# schémas d'outils, et tous les résultats d'outils déjà lus. Sans point de cache,
# chaque tour repaie ce préfixe au plein tarif (mesuré : ~180 k jetons d'entrée
# par fiche, contre ~24 k sur le chemin Conversations).
#
# Le cache d'Anthropic est un PRÉFIXE, et l'ordre de rendu est FIXE :
# `tools` → `system` → `messages`. Un point posé plus loin couvre donc tout ce
# qui précède, et une LECTURE se facture ~0,1× le prix d'entrée (l'écriture
# 1,25× en TTL 5 min, le défaut). On en pose TROIS — la limite dure est 4 :
#   - le dernier bloc de `system` : la procédure, figée pour tout le job ;
#   - la dernière définition de `tools` : figée pour tout le job (et rendue
#     AVANT `system`, donc ce point-là couvre le seul catalogue) ;
#   - le dernier bloc du DERNIER message : le point MOBILE, celui qui fait que
#     le tour N+1 relit en cache tout ce que le tour N a écrit dans le fil,
#     résultats d'outils compris.
# Le marqueur mobile se DÉPLACE à chaque tour, et ce déplacement n'invalide
# rien : un bloc marqué au tour N reste un point de lecture au tour N+1.
#
# ⚠️ Trois limites documentées, assumées telles quelles :
#   - un préfixe sous le MINIMUM du modèle n'est pas caché, SANS erreur ni
#     avertissement (512 jetons sur Opus 5 / Fable 5, 1024 sur Sonnet 5 — notre
#     défaut —, 2048 sur Opus 4.7, 4096 sur Opus 4.6 et Haiku 4.5) ; un run court
#     ne verra donc rien, et c'est normal ;
#   - un bloc de PENSÉE ne se marque pas : on remonte au dernier bloc marquable ;
#   - le point mobile ne remonte que 20 blocs en arrière pour retrouver l'entrée
#     précédente. Un tour qui appose plus de 20 blocs d'un coup (≥ 20 appels
#     d'outils parallèles) la pousse hors de la fenêtre et réécrit tout le fil.
CACHE_CONTROL = {"type": "ephemeral"}
_NON_CACHABLES = ("thinking", "redacted_thinking")


def _marque(bloc: dict) -> dict:
    return {**bloc, "cache_control": dict(CACHE_CONTROL)}


def systeme_cache(system):
    """`system` → blocs dont le DERNIER porte le point de cache.

    Une chaîne devient UN bloc texte (l'API n'accepte `cache_control` que sur un
    bloc). Un système vide n'est pas marqué : un bloc texte vide n'est pas
    cachable, et le marquer serait un point de cache perdu."""
    if isinstance(system, str):
        if not system.strip():
            return system
        blocs = [{"type": "text", "text": system}]
    elif isinstance(system, list):
        blocs = [dict(b) if isinstance(b, dict) else b for b in system]
        if not blocs or not isinstance(blocs[-1], dict):
            return system
    else:
        return system
    blocs[-1] = _marque(blocs[-1])
    return blocs


def outils_cache(tools: list[dict]) -> list[dict]:
    """La DERNIÈRE définition d'outil porte le point de cache — le catalogue est
    figé pour tout le job, et il se rend avant tout le reste."""
    if not tools:
        return tools
    return [*tools[:-1], _marque(tools[-1])]


def fil_cache(messages: list) -> list:
    """Le dernier bloc MARQUABLE du dernier message porte le point mobile.

    Copie de surface, jamais une mutation : le fil que la boucle appose au
    backend garde des `provider_raw` SANS marqueur. Un marqueur figé dans le fil
    stocké se rejouerait à contretemps au rechargement, et ferait un point de
    cache mort au milieu d'un fil rechargé."""
    if not messages or not isinstance(messages[-1], dict):
        return messages
    dernier = messages[-1]
    contenu = dernier.get("content")
    if isinstance(contenu, str):
        if not contenu.strip():
            return messages
        blocs: list = [{"type": "text", "text": contenu}]
    elif isinstance(contenu, list):
        blocs = [dict(b) if isinstance(b, dict) else b for b in contenu]
    else:
        return messages
    i = len(blocs) - 1
    while i >= 0 and (not isinstance(blocs[i], dict)
                      or _block_type(blocs[i]) in _NON_CACHABLES):
        i -= 1
    if i < 0:          # un tour tout en pensée : rien à marquer ici
        return messages
    blocs[i] = _marque(blocs[i])
    return [*messages[:-1], {**dernier, "content": blocs}]


def complete(*, system: str, messages: list, tools: list[dict],
             api_key: Optional[str] = None,
             temperature: Optional[float] = None,
             modele: Optional[str] = None,
             effort: Optional[str] = None,
             max_output_tokens: Optional[int] = None,
             on_event: Optional[Callable[[str, dict], None]] = None) -> Turn:
    """UN tour de modèle — synchrone : le worker est un process dédié, pas un
    serveur mono-loop, il a le droit d'attendre.

    `messages` = le fil au format provider (les `provider_raw` du fil backend,
    rejoués dans l'ordre) ; `tools` = `[{name, description, input_schema}]`.
    Toute erreur réseau/quota remonte à la boucle, qui décide (backoff du job).

    `on_event(type, champs)` : le journal du travail — le seam en sert UN seul
    contrat aux deux adaptateurs. Ici il ne reçoit rien : le SDK Anthropic
    retente lui-même les incidents de transport, et ce qu'il retente ne remonte
    pas jusqu'ici. Côté OpenAI-compatible, la retentative est à nous et se dit.

    ⚠️ `modele` (et non `model`) : le paramètre porterait sinon le nom de la
    fonction du module qu'il remplace, et la masquerait dans tout le corps.
    `None` = le modèle du worker (`OTO_RUNNER_MODEL`), inchangé.
    """
    if temperature is not None:
        # REFUS FRANC, et c'est délibéré. Ce provider règle la profondeur par
        # `output_config.effort` (cf. l'en-tête du module) ; la température n'y a
        # pas sa place. Les deux autres conduites étaient pires :
        #   — la poser quand même : le fournisseur rend un 400 opaque, et qui a
        #     déclaré la campagne cherche la faute dans sa procédure ;
        #   — l'ignorer en silence : un réglage offert qui ne fait rien est plus
        #     coûteux qu'un réglage absent, parce qu'on l'ajuste pendant des
        #     heures en croyant mesurer quelque chose.
        # Le travail échoue avec CE texte, lisible depuis la fiche du passage.
        raise ValueError(
            f"la campagne déclare temperature={temperature}, mais le provider "
            "anthropic règle la profondeur par `output_config.effort` et "
            "n'accepte pas de température. Retire-la de la campagne, ou sers-la "
            "par un worker OTO_RUNNER_PROVIDER=openai.")
    anthropic = _sdk()
    if anthropic is None:
        raise LlmUnavailable("le paquet `anthropic` n'est pas installé")
    client = anthropic.Anthropic(api_key=api_key or resolve_key())
    nom = modele or model()
    effort_retenu = effort or effort_hote()
    kwargs: dict = {
        "model": nom,
        # Le plafond que porte le TRAVAIL (le catalogue du backend le déclare par modèle),
        # à défaut celui du worker. ⚠️ Pas de levée sur un effort sans plafond, à la
        # différence de la voie Chat Completions : ce provider envoie un effort à chaque
        # tour et le catalogue ne déclare aucun plafond Claude — lever les ferait tous
        # échouer.
        "max_tokens": max_output_tokens or max_tokens(),
        "system": systeme_cache(system),
        "messages": fil_cache(messages),
    }
    # L'effort du TRAVAIL, à défaut celui du worker (env). Il est fixe sur tout le
    # déroulé — c'est ce que le cache demande : le faire varier d'un tour à l'autre
    # invaliderait le fil. ⚠️ `none` n'envoie RIEN : Haiku 4.5 refuse
    # `output_config.effort` (400, mesuré le 14/09/2026), et le catalogue le déclare ainsi.
    if effort_retenu != EFFORT_SANS_RAISONNEMENT:
        kwargs["output_config"] = {"effort": effort_retenu}
    if tools:
        kwargs["tools"] = outils_cache(tools)
    resp = client.messages.create(**kwargs)

    stop = getattr(resp, "stop_reason", None)
    # ⚠️ `input_tokens` n'est QUE le reste NON caché — le volume d'entrée réel vaut
    # input + cache_creation + cache_read. Un run qui cache bien affiche un
    # `input_tokens` minuscule : c'est la somme qui se lit, pas le champ seul.
    # Un poste n'est posé que s'il est DÉCLARÉ (cf. `comptage`). Contrat Anthropic,
    # lu le 13/09/2026 : les postes de cache sont toujours rendus et NULLABLES ;
    # « 0 » veut dire pas de cache, `null` n'a aucun sens documenté — il reste inconnu.
    u = getattr(resp, "usage", None)
    usage = {poste: int(valeur) for poste in POSTES
             if (valeur := getattr(u, poste, None)) is not None}

    # La version SERVIE si l'API la rend, à défaut celle qu'on a demandée : une
    # estampille approchée vaut infiniment mieux qu'un `null`, qui ne se distingue
    # pas d'un job qui n'a jamais tourné.
    servi = getattr(resp, "model", None) or nom

    raw = [_block_to_dict(b) for b in getattr(resp, "content", []) or []]
    if stop == "refusal":
        return Turn(text="", tool_calls=(), stop_reason="refusal",
                    raw_content=raw, usage=usage, model=servi, effort=effort_retenu)

    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in getattr(resp, "content", []) or []:
        kind = _block_type(block)
        if kind == "text":
            texts.append(getattr(block, "text", "") or "")
        elif kind == "tool_use":
            raw_args = getattr(block, "input", None)
            calls.append(ToolCall(
                id=getattr(block, "id", "") or "",
                name=getattr(block, "name", "") or "",
                arguments=raw_args if isinstance(raw_args, dict) else {}))
    # ⚠️ Deux fins seulement CONCLUENT un tour : `end_turn`, et `tool_use` quand un appel
    # est là — la règle de la voie Chat Completions (audit du 13/09/2026), que celle-ci
    # n'appliquait pas. `max_tokens` (sortie coupée) se lisait comme une fin : le travail
    # concluait `done` sur une réponse tronquée, ou exécutait un appel coupé. Énumération
    # Anthropic lue le 14/09/2026 : end_turn | max_tokens | stop_sequence | tool_use |
    # pause_turn | refusal (plus haut) | model_context_window_exceeded. Ce provider
    # n'envoie ni séquence d'arrêt ni outil serveur, et une fin absente n'est pas une fin.
    anormale = stop != "end_turn" and not (stop == "tool_use" and calls)
    return Turn(text="\n".join(t for t in texts if t.strip()).strip(),
                tool_calls=tuple(calls),
                stop_reason="fin_anormale" if anormale else stop,
                raw_content=raw, usage=usage, model=servi, effort=effort_retenu,
                defaut={"forme": "fin_anormale", "stop_reason": stop} if anormale else None)
