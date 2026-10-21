"""La voie `claude-subscription` : le travail tourne sur l'ABONNEMENT Claude de son porteur.

Famille `claude_subscription` (oto-backend#1043, modèles `sub:*`). Le travail ne
s'exécute pas ici : il s'exécute dans le SANDBOX de la personne — un utilisateur Unix
de la box de la ferme, où elle a connecté elle-même son abonnement au CLI Claude
Code officiel (dépôt `otomata-tech/claude-sandbox-manager`). Ce module demande le run à
l'AGENT de la box, qui le lance dans le sandbox et en renvoie le flux.

Ce qui ne passe JAMAIS par ici : l'identifiant de l'abonnement. Il vit dans le
HOME du sandbox ; le worker ne reçoit du backend qu'un `sandbox_id`, et ne remet à
l'agent que ce que le travail porte déjà — la consigne, l'allowlist, le jeton
DÉLÉGUÉ (borné au bail) pour le relais MCP (`relais_mcp`).

Chemin ONE-SHOT : la boucle d'outils tourne dans le CLI, le worker reçoit le
résultat. Le bail se PROLONGE pendant le run, à chaque événement du flux, au plus
une fois par `_PROLONGER_S` — un déroulé plus long que son bail libérerait la
personne pendant que le CLI tourne encore (deux exécutions sur une session).

Ce que le worker rapporte au backend en plus : `abonnement` — l'état du forfait
tel que le fournisseur l'annonce (`rate_limit_event`), et `deconnecte` quand le
sandbox n'est plus connecté. C'est ce qui met la personne en attente au seuil, avant
le refus.

Environnement : `OTO_FERME_URL` (l'agent, sur le réseau privé de la box),
`OTO_FERME_TOKEN` (son jeton), `OTO_MCP_URL`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Optional

import requests

from .agent_runtime import AgentResult, AgentStep
from .contenu_outil import substitut

logger = logging.getLogger("oto_runner")

ONE_SHOT = True     # le worker choisit le chemin là-dessus
PAYE_PAR = "abonnement"   # ce que le résultat dit de qui a payé (`worker._paye_par`)
MOTEUR = "claude_code_abonnement"   # ce que le résultat dit de ce qui a tourné
SANDBOX = True          # … et remet à ce provider le contexte du sandbox (cf. worker._traiter)

FAMILLE = "claude_subscription"
DEFAULT_MODEL = "sub:sonnet"
_PREFIXE_MODELE = "sub:"
_PREFIXE_OUTIL = "mcp__oto__"     # le nom que le CLI donne aux outils du serveur `oto`
_ENV_AGENT = "OTO_FERME_URL"
_ENV_JETON = "OTO_FERME_TOKEN"
_PROLONGER_S = 60
_LECTURE_S = 600    # silence toléré entre deux événements (un outil lent)
# Le fil borne un tour à 256 000 caractères (oto-backend, `run_thread`) : une sortie d'outil
# se coupe AVANT, bloc par bloc, et la coupe se DIT dans le texte gardé.
_SORTIE_MAX = 50_000

# Les fins du CLI (`result.subtype`) → le vocabulaire de la boucle (`AgentResult.stopped`).
_FINS = {"success": "end_turn"}
# Les BORNES de la ferme (`--max-turns`, `--max-budget-usd`) : le CLI les rend avec
# `is_error: true`, mais une borne atteinte a fait son travail — elle conclut `blocked`,
# jamais un échec rejoué, qui repaierait le même plafond à chaque tentative.
_BORNES = {"error_max_turns": "max_steps", "error_max_budget_usd": "max_budget"}
#: Les tours que la ferme accepte (`max_turns`, borné 1–1000 de son côté).
_TOURS_MIN, _TOURS_MAX = 1, 1000


class LlmUnavailable(RuntimeError):
    pass


class FermePleine(RuntimeError):
    """La ferme tient déjà toutes ses places, essai après essai."""


#: Une box pleine (429) : réessayée, bail prolongé entre deux essais, puis l'échec nommé.
#: Commun aux deux voies de la ferme : les runs par clé et par abonnement se partagent
#: les mêmes places, et un travail d'abonnement ne doit pas perdre une tentative parce
#: que des runs par clé occupent la box.
_PLEINE_ESSAIS, _PLEINE_ATTENTE_S = 4, 20


def poster_le_run(url: str, corps: dict, entetes: dict, lecture: int,
                  prolonger=None, on_event=None, attendre=time.sleep):
    """`POST …/runs` en flux — la réponse OUVERTE (à fermer par `with`), une fois qu'elle
    n'est plus un 429."""
    for essai in range(_PLEINE_ESSAIS):
        r = requests.post(url, json=corps, headers=entetes, stream=True,
                          timeout=(10, lecture))
        if r.status_code != 429:
            return r
        r.close()
        if on_event:
            on_event("ferme_pleine", {"essai": essai + 1})
        if prolonger:
            prolonger()
        if essai + 1 < _PLEINE_ESSAIS:
            attendre(_PLEINE_ATTENTE_S * (essai + 1))
    raise FermePleine(f"la ferme est restée pleine sur {_PLEINE_ESSAIS} essais — "
                      "le travail repassera à sa prochaine tentative")


def model() -> str:
    return os.environ.get("OTO_RUNNER_MODEL") or DEFAULT_MODEL


def depot() -> str:
    return FAMILLE


def resolve_key() -> str:
    """Aucune clé : ce worker ne paie rien. Ce qu'il lui faut, c'est l'agent de la box."""
    for nom in (_ENV_AGENT, _ENV_JETON):
        if not (os.environ.get(nom) or "").strip():
            raise LlmUnavailable(f"{nom} absent de l'environnement du worker")
    return ""


def rapport_du_forfait(forfait: Optional[dict]) -> Optional[dict]:
    """Le `rate_limit_info` du CLI tel que le backend le lit (`_abonnement.noter_rapport`).

    `fenetre` / `echeance` : la fenêtre que le fournisseur NOMME (`rateLimitType`) et son
    échéance (`resetsAt`). Sur un refus, c'est elle qui a mordu — sans elles, le backend
    attendait l'échéance la plus lointaine de toutes les fenêtres (sept jours pour un
    refus sur cinq heures)."""
    if not forfait:
        return None
    rapport = {"etat": forfait.get("status"), "fenetres": forfait.get("unifiedWindows")}
    if forfait.get("rateLimitType"):
        rapport["fenetre"] = forfait["rateLimitType"]
    if isinstance(forfait.get("resetsAt"), (int, float)):
        rapport["echeance"] = forfait["resetsAt"]
    return rapport


def _modele_du_cli(modele: str) -> str:
    """`sub:opus` → `opus` : le CLI connaît les alias, le catalogue les préfixe."""
    if not modele.startswith(_PREFIXE_MODELE):
        raise ValueError(f"modèle `{modele}` hors de la famille {FAMILLE} (préfixe `{_PREFIXE_MODELE}`)")
    return modele[len(_PREFIXE_MODELE):]


def _nom_outil(nom: str) -> str:
    return nom[len(_PREFIXE_OUTIL):] if nom.startswith(_PREFIXE_OUTIL) else nom


def _borner(texte: str) -> str:
    if len(texte) <= _SORTIE_MAX:
        return texte
    return texte[:_SORTIE_MAX] + f"\n[… coupé pour le fil : {len(texte)} caractères au total]"


def _message_borne(message: dict) -> dict:
    """Le message tel que le CLI l'a rendu, sorties d'outils bornées pour le fil.

    ⚠️ Une image d'un résultat d'outil y devient un SUBSTITUT texte : le backend
    plafonne un tour à 256 000 caractères, qu'une seule image dépasse."""
    blocs = []
    for bloc in message.get("content") or []:
        if isinstance(bloc, dict) and bloc.get("type") == "tool_result":
            contenu = bloc.get("content")
            if isinstance(contenu, str):
                bloc = {**bloc, "content": _borner(contenu)}
            elif isinstance(contenu, list):
                bloc = {**bloc, "content": [_bloc_borne(c) for c in contenu]}
        blocs.append(bloc)
    return {**message, "content": blocs}


def _bloc_borne(bloc):
    if isinstance(bloc, dict) and bloc.get("type") == "image":
        source = bloc.get("source") or {}
        return substitut(source.get("media_type") or "?", len(source.get("data") or ""))
    if isinstance(bloc, dict) and "text" in bloc:
        return {**bloc, "text": _borner(bloc["text"])}
    return bloc


class _Compteur:
    """Ce que le flux a déjà coûté, EN VOL — pour arrêter un run à sa borne de jetons.

    ⚠️ Le CLI répète l'usage d'un message sur CHAQUE bloc qu'il émet (texte, pensée,
    appel) : on garde, par identifiant de message, le maximum de chaque poste — une
    somme naïve compterait deux à trois fois le même message.

    ⚠️ La sortie d'un message, en vol, est PARTIELLE (le flux annonce 1 ou 3 jetons pour
    un message qui en fera des centaines) : ce compte en est un MINORANT. L'entrée non
    cachée et l'écriture de cache, elles, sont exactes dès le premier bloc. La borne
    compte comme la boucle ordinaire (`comptage._pour_la_borne`) : entrée non cachée +
    sortie + écriture de cache, jamais la lecture de cache. Le compte qui fait foi est
    celui du `result` final ; celui-ci ne sert qu'à arrêter."""

    def __init__(self):
        self.messages: dict = {}

    def ajouter(self, message: dict) -> None:
        usage = message.get("usage") or {}
        cle = message.get("id") or f"sans-id-{len(self.messages)}"
        vu = self.messages.setdefault(cle, {})
        for poste in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                      "cache_read_input_tokens"):
            if isinstance(usage.get(poste), int):
                vu[poste] = max(vu.get(poste, 0), usage[poste])

    def borne(self) -> int:
        return sum(m.get("input_tokens", 0) + m.get("output_tokens", 0)
                   + m.get("cache_creation_input_tokens", 0) for m in self.messages.values())

    def usage_connu(self) -> dict:
        """L'usage d'un run ARRÊTÉ en vol : l'entrée et les caches sont exacts, la
        sortie ne l'est pas — elle reste NON DÉCLARÉE (`None`), jamais un minorant
        publié comme un compte (cf. `comptage`)."""
        somme = lambda p: sum(m.get(p, 0) for m in self.messages.values())  # noqa: E731
        return {"input_tokens": somme("input_tokens"), "output_tokens": None,
                "cache_creation_input_tokens": somme("cache_creation_input_tokens"),
                "cache_read_input_tokens": somme("cache_read_input_tokens")}


def _usage_du_resultat(resultat: dict) -> tuple[dict, Optional[dict]]:
    """L'usage d'un run conclu, et son détail PAR MODÈLE.

    `result.modelUsage` porte chaque modèle qui a servi, sous-agents compris ; `usage`
    ne porte que le fil principal. Quand le premier est là, il fait foi pour les postes
    (leur somme) et se rapporte en détail. Sinon, `usage` comme avant."""
    par_modele = resultat.get("modelUsage")
    if not isinstance(par_modele, dict) or not par_modele:
        return resultat.get("usage") or {}, None
    total = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0}
    champs = {"input_tokens": "inputTokens", "output_tokens": "outputTokens",
              "cache_creation_input_tokens": "cacheCreationInputTokens",
              "cache_read_input_tokens": "cacheReadInputTokens"}
    detail = {}
    for nom, u in par_modele.items():
        u = u or {}
        for poste, champ in champs.items():
            total[poste] += int(u.get(champ) or 0)
        # Deux clés brutes du même modèle (fenêtres de contexte différentes) s'ADDITIONNENT.
        vu = detail.setdefault(u.get("canonicalModel") or nom, {
            "entree": 0, "sortie": 0, "cache_lu": 0, "cache_ecrit": 0, "cout_usd": None})
        for cle, champ in (("entree", "inputTokens"), ("sortie", "outputTokens"),
                           ("cache_lu", "cacheReadInputTokens"),
                           ("cache_ecrit", "cacheCreationInputTokens")):
            vu[cle] += int(u.get(champ) or 0)
        # Le coût que le CLI calcule au TARIF PUBLIC (`costBasis: list`) — une estimation
        # lisible, pas une facture.
        if isinstance(u.get("costUSD"), (int, float)):
            vu["cout_usd"] = round((vu["cout_usd"] or 0) + float(u["costUSD"]), 4)
    return total, detail


def lire_flux(evenements, on_event=None, prolonger: Optional[Callable[[], None]] = None,
              apposer: Optional[Callable[[str, dict, dict], None]] = None,
              horloge=time.monotonic, max_tokens: Optional[int] = None,
              echeance: Optional[float] = None, source_attendue: str = "none") -> AgentResult:
    """Le flux `stream-json` du CLI (plus le résumé de l'agent) → `AgentResult`.

    `apposer(role, neutre, brut)` : le fil du run, tenu EN DIRECT — chaque message du CLI y
    part à son arrivée, au format de la boucle ordinaire (`assistant` : texte + appels ;
    `tool` : issue de chaque appel). Le brut est le message Anthropic que le CLI a rendu.

    `max_tokens` / `echeance` (instant `horloge()`) : les limites du run déclarées sur
    l'agent. Atteinte, la lecture S'ARRÊTE et rend `stopped=max_tokens|max_seconds` —
    l'appelant ferme alors le flux, et l'agent de la ferme arrête l'unité du run.
    L'événement `ferme_arret` (posé par le transport quand le flux se tait au-delà de
    l'échéance) vaut la même chose.

    Séparé du transport pour se tester sur un flux enregistré."""
    compteur = _Compteur()
    arret: Optional[str] = None
    steps: list[AgentStep] = []
    en_vol: dict = {}          # tool_use_id → (nom, instant du départ)
    init, resultat, forfait, resume = {}, None, None, {}
    dernier = horloge()
    for ev in evenements:
        maintenant = horloge()      # UNE lecture par événement
        if on_event:
            on_event("claude", ev)
        if prolonger and maintenant - dernier >= _PROLONGER_S:
            prolonger()
            dernier = maintenant
        t = ev.get("type")
        if t == "ferme_arret":
            arret = ev.get("raison") or "max_seconds"
            break
        if echeance is not None and maintenant >= echeance and t not in ("result", "ferme_resume"):
            arret = "max_seconds"
            break
        if t == "system" and ev.get("subtype") == "init":
            init = ev
            # Vérifié DÈS l'annonce, pas à la fin : un run arrêté à sa borne ne passe
            # jamais par la fin, et il doit être refusé avant le premier tour payé.
            source = ev.get("apiKeySource")
            if source != source_attendue:
                # Le CLI paie avec autre chose que ce qui a été servi : une clé dans le
                # sandbox d'un abonnement, ou une session là où on a remis une clé.
                raise RuntimeError(
                    f"le run n'a pas tourné sur ce qui devait le payer "
                    f"(apiKeySource={source!r}, attendu {source_attendue!r})")
        elif t == "rate_limit_event":
            forfait = ev.get("rate_limit_info")
        elif t == "assistant":
            message = ev.get("message") or {}
            textes, appels = [], []
            for bloc in message.get("content") or []:
                if bloc.get("type") == "tool_use":
                    nom = _nom_outil(bloc.get("name") or "?")
                    en_vol[bloc.get("id")] = (nom, maintenant)
                    appels.append({"name": nom})
                elif bloc.get("type") == "text":
                    textes.append(bloc.get("text") or "")
            if apposer:
                apposer("assistant", {"text": "".join(textes), "tool_calls": appels}, message)
            compteur.ajouter(message)
            if max_tokens is not None and compteur.borne() >= max_tokens:
                arret = "max_tokens"
                break
        elif t == "user":
            message = ev.get("message") or {}
            issues = []
            for bloc in message.get("content") or []:
                if isinstance(bloc, dict) and bloc.get("type") == "tool_result":
                    nom, depart = en_vol.pop(bloc.get("tool_use_id"), ("?", maintenant))
                    duree = int((maintenant - depart) * 1000)
                    steps.append(AgentStep(tool=nom, ok=not bloc.get("is_error"),
                                           duration_ms=duree))
                    issues.append({"name": nom, "ok": not bloc.get("is_error"),
                                   "duration_ms": duree})
            if apposer and issues:
                apposer("tool", {"tool_calls": issues}, _message_borne(message))
        elif t == "result":
            resultat = ev
        elif t == "ferme_resume":
            resume = ev
    if arret is not None:
        if on_event:
            on_event("borne_atteinte", {"borne": arret, "max_tokens": max_tokens,
                                        "jetons_bornes": compteur.borne(),
                                        "sortie": "minorant"})
        # Pas une exception : une borne DÉCLARÉE a fait son travail. Le travail conclut
        # `blocked`, comme à `max_tokens` sur la boucle ordinaire.
        return AgentResult(reply="", steps=steps, stopped=arret,
                           usage=compteur.usage_connu(), model=init.get("model"),
                           abonnement=rapport_du_forfait(forfait))
    if resume.get("connecte") is False:
        # Pas une exception : c'est une CONCLUSION que le backend doit lire
        # (`deconnecte` → la personne passe `needs_login`, ses travaux attendent).
        return AgentResult(reply="", stopped="fin_anormale",
                           defaut={"finish_reason": "sandbox_deconnecte"},
                           abonnement={"deconnecte": True})
    if resultat is None and echeance is not None and horloge() >= echeance:
        # La ferme a arrêté l'unité à SA durée (le filet) avant qu'un événement ne nous
        # rende la main : sans résultat, mais au-delà de l'échéance — la borne, pas une
        # panne. Sans ça, le travail échouait et son rejeu retombait sur la même borne.
        if on_event:
            on_event("borne_atteinte", {"borne": "max_seconds", "par": "ferme",
                                        "erreur": resume.get("erreur")})
        return AgentResult(reply="", steps=steps, stopped="max_seconds",
                           usage=compteur.usage_connu(), model=init.get("model"),
                           abonnement=rapport_du_forfait(forfait))
    if resultat is None:
        raise RuntimeError(f"le CLI n'a rendu aucun résultat ({resume.get('erreur') or 'flux coupé'})")
    if init.get("apiKeySource") != source_attendue:
        raise RuntimeError(f"le run n'a pas tourné sur ce qui devait le payer "
                           f"(apiKeySource={init.get('apiKeySource')!r}, "
                           f"attendu {source_attendue!r})")
    usage, par_modele = _usage_du_resultat(resultat)
    borne = _BORNES.get(resultat.get("subtype"))
    if borne is not None:
        if on_event:
            on_event("borne_atteinte", {"borne": borne, "par": "ferme",
                                        "fin": resultat.get("subtype")})
        return AgentResult(reply=resultat.get("result") or "", steps=steps, stopped=borne,
                           usage=usage, par_modele=par_modele, model=init.get("model"),
                           abonnement=rapport_du_forfait(forfait))
    erreur_du_cli = bool(resultat.get("is_error"))
    return AgentResult(
        reply=resultat.get("result") or "",
        steps=steps,
        # Une erreur du CLI (fournisseur, plafond atteint en vol) est une fin ANORMALE,
        # nommée comme telle : conclue `done`, elle cacherait un travail non fait.
        stopped=("fin_anormale" if erreur_du_cli
                 else _FINS.get(resultat.get("subtype"), resultat.get("subtype") or "no_reply")),
        defaut=({"finish_reason": resultat.get("subtype") or "erreur_du_cli"}
                if erreur_du_cli else None),
        usage=usage,
        par_modele=par_modele,
        model=init.get("model"),
        abonnement=rapport_du_forfait(forfait))


def run_once(*, instructions: str, inputs: str, tools, api_key: Optional[str] = None,
             modele: Optional[str] = None, on_event=None, sandbox: Optional[str] = None,
             mcp=None, prolonger: Optional[Callable[[], None]] = None,
             apposer: Optional[Callable[[str, dict, dict], None]] = None,
             max_tokens: Optional[int] = None, max_seconds: Optional[int] = None,
             max_turns: Optional[int] = None,
             horloge=time.monotonic, attendre=time.sleep) -> AgentResult:
    """UN run complet dans le sandbox `sandbox`, outils compris (côté CLI), → AgentResult.

    `max_tokens` / `max_seconds` : les limites déclarées sur l'agent, tenues ICI, sur le
    flux (`lire_flux`). Quitter le `with` ferme la connexion ; l'agent de la ferme arrête
    alors l'unité du run (son `finally`). `max_seconds` part aussi dans le corps : une
    ferme qui sait le lire pose la même échéance à l'unité — les deux se recouvrent,
    aucune n'est de trop."""
    if api_key:
        raise RuntimeError("un travail d'abonnement ne porte jamais de clé de modèle")
    if not sandbox:
        raise RuntimeError("travail d'abonnement sans `sandbox_id` : aucun sandbox où le lancer")
    if mcp is None:
        raise RuntimeError("travail d'abonnement sans session MCP : le relais n'aurait aucun contexte")
    resolve_key()
    corps = {
        "prompt": inputs,
        "model": _modele_du_cli(modele or model()),
        "system": instructions,
        "mcp": {"url": mcp.url, "token": mcp.token, "org": mcp.org, "project": mcp.project,
                "run_id": mcp.run_id, "tools": sorted(tools or ())},
        **({"max_seconds": int(max_seconds)} if max_seconds else {}),
        **champ_des_tours(max_turns),
    }
    # Le silence toléré entre deux événements, jamais au-delà de l'échéance : sans ça, un
    # outil qui tourne sans rien dire ferait survivre le run à sa limite.
    lecture = min(_LECTURE_S, max_seconds) if max_seconds else _LECTURE_S
    url = f"{os.environ[_ENV_AGENT].rstrip('/')}/api/sandboxes/{sandbox}/runs"
    entetes = {"Authorization": f"Bearer {os.environ[_ENV_JETON]}",
               "Accept": "application/x-ndjson"}
    with poster_le_run(url, corps, entetes, lecture, prolonger=prolonger,
                       on_event=on_event, attendre=attendre) as r:
        if r.status_code != 200:
            raise RuntimeError(f"agent de la ferme : {r.status_code} {r.text[:300]}")
        echeance = echeance_du_run(max_seconds, horloge)
        lignes = (json.loads(l) for l in r.iter_lines(decode_unicode=True) if l)
        return lire_flux(_jusqu_a(lignes, echeance, horloge), on_event=on_event,
                         prolonger=prolonger, apposer=apposer, horloge=horloge,
                         max_tokens=max_tokens, echeance=echeance)


#: La SONDE du forfait (`verifier_forfait`) : un tour, sans outil, sur le plus petit modèle.
VERIFIE_LE_FORFAIT = True   # le worker sonde avant de lancer (cf. worker._sonder_le_forfait)
_SONDE_MODELE = "haiku"
_SONDE_S = 60


def verifier_forfait(sandbox: Optional[str], *, attendre=time.sleep) -> Optional[dict]:
    """Ce que le fournisseur dit du forfait MAINTENANT, avant de lancer un travail.

    Un tour minuscule dans le sandbox du porteur (`prompt` d'un mot, sans outil ni
    relais MCP, `max_turns` 1), dont on ne lit que le `rate_limit_event`. Il coûte
    quelques jetons du forfait ; il évite qu'une pause posée par un rapport ancien —
    ou faux — fige le forfait jusqu'à sa réinitialisation, puisque sans travail servi
    aucun rapport ne venait la corriger.

    Rend le rapport (`rapport_du_forfait`), ou `None` quand le flux n'en porte pas :
    alors on ne sait rien, et le travail part comme avant. ⚠️ Un refus ne se DÉDUIT
    jamais d'un message d'erreur — seul l'état que le fournisseur annonce compte."""
    if not sandbox:
        return None
    resolve_key()
    corps = {"prompt": "ok", "model": _SONDE_MODELE, "max_seconds": _SONDE_S,
             **champ_des_tours(1)}
    url = f"{os.environ[_ENV_AGENT].rstrip('/')}/api/sandboxes/{sandbox}/runs"
    entetes = {"Authorization": f"Bearer {os.environ[_ENV_JETON]}",
               "Accept": "application/x-ndjson"}
    forfait = None
    with poster_le_run(url, corps, entetes, _SONDE_S, attendre=attendre) as r:
        if r.status_code != 200:
            raise RuntimeError(f"agent de la ferme (sonde) : {r.status_code} {r.text[:300]}")
        for ligne in r.iter_lines(decode_unicode=True):
            if not ligne:
                continue
            ev = json.loads(ligne)
            if ev.get("type") == "rate_limit_event":
                forfait = ev.get("rate_limit_info")
    return rapport_du_forfait(forfait)


def champ_des_tours(max_turns: Optional[int]) -> dict:
    """`max_turns` pour la ferme, s'il est déclaré — ramené dans ce qu'elle accepte.
    Absent : rien, la ferme garde son défaut."""
    if not max_turns:
        return {}
    return {"max_turns": max(_TOURS_MIN, min(int(max_turns), _TOURS_MAX))}


def echeance_du_run(max_seconds: Optional[int], horloge) -> Optional[float]:
    """L'échéance du run, posée quand la ferme l'a ACCEPTÉ (réponse 200) : les
    attentes d'une box pleine (429) ne se prennent pas sur la durée du run — la ferme
    ne lance l'unité, et sa propre échéance, qu'à ce moment-là."""
    return horloge() + max_seconds if max_seconds else None


def _jusqu_a(lignes, echeance: Optional[float], horloge):
    """Le flux, tel quel — sauf un silence qui dépasse l'échéance : il devient
    `ferme_arret`, une borne atteinte, plutôt qu'une panne de transport. Avant
    l'échéance, un silence reste la panne qu'il était."""
    try:
        yield from lignes
    except requests.exceptions.RequestException:
        if echeance is None or horloge() < echeance:
            raise
        yield {"type": "ferme_arret", "raison": "max_seconds"}
