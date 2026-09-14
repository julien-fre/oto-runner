"""Le chemin ONE-SHOT : l'API Conversations de Mistral, outils exécutés CHEZ EUX.

Pourquoi il existe : l'API stateless re-facture tout l'historique à chaque tour
de boucle — 58 % du coût d'un job mesuré au vol v86, la 5e fiche d'un fil coûte
6,6× la première. Conversations orchestre les appels d'outils CÔTÉ MISTRAL (le
connecteur MCP oto y est déclaré) dans UN appel facturé une fois : ~26 k jetons
par fiche mesurés sur 194 fiches réelles, plat au rang. Décision Alexis du
19/08 : la campagne bascule dessus avant tout lancement.

Ce que ce chemin CHANGE : la boucle ne tourne plus dans le worker — il ne voit
pas les tours, il reçoit le RÉSULTAT (les `outputs` : entrées typées, dont les
exécutions d'outils une à une). La surveillance est PRÉSERVÉE au grain job :
usage, motif d'arrêt, pas, `tool_counts` sont dérivés des outputs — ce qui se
PERD est le verbatim des tours intermédiaires (`store=False` : rien ne reste
chez Mistral — c'est une contrainte de CONFORMITÉ du contrat client, pas un
réglage de confort).

⚠️ En requests NU + deadline SIGALRM — JAMAIS le SDK mistralai : son
`timeout_ms` devient un timeout httpx dont le `read` se réarme à chaque octet
reçu (basesdk.py:227) — 20 heures de gel silencieux sur la boucle locale le
18/08. Le SDK ne rejoue pas non plus les transitoires sur cet appel : les
rejeux sont ICI, explicites.

## L'appel RENVOYÉ au client, et la relance du fil

En mode connecteur, aucun appel d'outil n'a à revenir au client : le
fournisseur les exécute. Un `function.call` en fin d'`outputs` est donc un fil
INTERROMPU — le fournisseur ne connaît pas l'outil demandé et rend la main. Sans
relance, le job conclut sans avoir rien écrit, tout en ayant payé le run entier.

`OTO_RUNNER_RELANCES_MAX` (entier, défaut 0 = jamais) autorise N relances. Une
relance est un NOUVEAU POST `/v1/conversations` : `store=False` interdit
`/restart`, qui repart d'une entrée STOCKÉE chez le fournisseur — rien n'est
stocké, le fil se rejoue donc entier par ses `inputs`. La référence de l'API
(docs.mistral.ai/api — `ConversationRequest.inputs`, type `ConversationInputs`)
accepte au choix une chaîne ou une liste d'entrées `InputEntries`, union de
`MessageInputEntry`, `MessageOutputEntry`, `FunctionResultEntry`,
`FunctionCallEntry`, `ToolExecutionEntry` et `AgentHandoffEntry` — les quatre
derniers types étant précisément ceux que rendent les `outputs`. Ils y
RETOURNENT donc, suivis d'une `FunctionResultEntry` (`tool_call_id` + `result`,
tous deux requis) par appel renvoyé.

⚠️ Mais PAS tels quels, et pas tous : les modèles du client Python décrivent
une union `InputEntries` PLUS LARGE que ce que le service accepte réellement.
Deux refus successifs en production, puis une SONDE de l'API depuis la box
(requêtes minuscules, `store=False`, le 27/08) — c'est ce relevé empirique qui
fait foi, pas les modèles :

- `[message.input]` → 200 ;
- `[message.input, message.output, message.input]` → 200 ;
- `[message.input, function.call, function.result]` → 200 — un `name` inconnu
  du service (`demo_lookup`) passe, avec un `tool_call_id` de 9 alphanumériques ;
- `[message.input, tool.execution(name, arguments, info), message.input]` →
  **422** « Input should be 'message.input' » ; **idem sans `info`** ;
- `[message.input, function.result]` sans son `function.call` → **400**.

D'où les deux règles de construction du fil rejoué. **(1)** Une entrée sortante
porte des champs POSÉS PAR LE SERVEUR — `id`, `created_at`, `completed_at`,
`agent_id`, `model` — refusés en entrée (422 « Input entries send by the user
can't specify ids ») : chaque entrée est dépouillée sur la liste blanche de son
type (`_CHAMPS_ENTREE`). `tool_call_id` n'en est pas un champ serveur : c'est la
clé que cite la réponse, elle reste. **(2)** Un `tool.execution` ne se renvoie
JAMAIS : il se CONVERTIT en la paire que le service accepte — un
`function.call` portant son nom et ses arguments, sous un `tool_call_id`
synthétisé, suivi du `function.result` qui porte son `info`. Le travail déjà
fait reste donc dans le fil, sous la seule forme qui passe.

## La version RÉSOLUE du modèle

Le modèle est nommé par un ALIAS (`mistral-large-latest`) : le fournisseur le
fait pointer ailleurs quand il le décide, sans préavis. Une anomalie de
campagne ne se date alors PAS — on ignore quel modèle a réellement tourné, et
sur quels jobs. `modele_resolu()` interroge `/v1/models` et rend la version
concrète derrière l'alias ; `run_once` la pose sur chaque `AgentResult`, et le
worker la déclare au job. C'est un relevé d'observabilité : il ne fait jamais
échouer un run que la campagne a payé.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import string
import time
from typing import Optional

import requests

from .agent_runtime import AgentResult, AgentStep
from .comptage import Compteur
from .deadline import DeadlineExceeded, post_with_deadline  # noqa: F401 — DeadlineExceeded
# fait partie du contrat d'erreur de run_once (remonte au job, jamais rejouée ici).

logger = logging.getLogger("oto_runner")

ONE_SHOT = True                      # le worker choisit le chemin là-dessus

_ENV_KEY = "OTO_RUNNER_OPENAI_API_KEY"   # même clé que le mode openai (La Plateforme)
_ENV_BASE = "OTO_RUNNER_OPENAI_BASE"
_ENV_RELANCES = "OTO_RUNNER_RELANCES_MAX"
_DEFAULT_BASE = "https://api.mistral.ai"
DEFAULT_MODEL = "mistral-large-latest"

# Un one-shot porte un run ENTIER (les tours d'outils compris) : la deadline est
# large — mais TOUJOURS plus courte que le bail one-shot du worker (1800 s), sinon
# un pair re-claimerait un job dont la conversation tourne encore.
_WALL_S = 900

# Les champs de FOND acceptés en entrée, par type (référence API — modèles
# `InputEntries` du SDK). Tout le reste est posé par le serveur (`id`,
# `created_at`, `completed_at`, `agent_id`, `model`) et fait échouer le POST.
_CHAMPS_ENTREE = {
    "message.input": ("object", "type", "role", "content", "prefix"),
    "message.output": ("object", "type", "role", "content"),
    "function.call": ("object", "type", "tool_call_id", "name", "arguments"),
    "function.result": ("object", "type", "tool_call_id", "result"),
}

_ALPHABET_APPEL = string.ascii_lowercase + string.digits
_LONGUEUR_APPEL = 9                  # la forme qu'accepte le service, sondée
_RESULTAT_NON_CONSERVE = "(résultat non conservé par le fournisseur)"

_CONSIGNE_APPEL_RENVOYE = (
    "Cet outil n'existe pas. Les outils sont exécutés par le connecteur : "
    "appelle-les par leur nom exact, sans rédiger dans le nom de l'appel. "
    "Reprends là où tu en étais.")


class LlmUnavailable(RuntimeError):
    pass


def model() -> str:
    return os.environ.get("OTO_RUNNER_MODEL") or DEFAULT_MODEL


_DEPOTS_PAR_HOTE = {"api.mistral.ai": "mistral"}


def depot() -> str:
    """Cf. `agent_llm_openai.depot` : l'hôte configuré décide, pas le module."""
    from urllib.parse import urlparse
    return _DEPOTS_PAR_HOTE.get(urlparse(base_api()).netloc, "")


def resolve_key() -> str:
    key = os.environ.get(_ENV_KEY, "").strip()
    if not key:
        raise LlmUnavailable(f"{_ENV_KEY} absente de l'environnement du worker")
    return key


def connector_id() -> str:
    cid = os.environ.get("OTO_RUNNER_CONNECTOR_ID", "").strip()
    if not cid:
        raise LlmUnavailable(
            "OTO_RUNNER_CONNECTOR_ID absent : le mode conversations exécute les "
            "outils chez Mistral via un connecteur MCP déclaré côté compte — sans "
            "son id, aucun outil. (Le connecteur porte son propre secret vers "
            "mcp.oto.cx ; le worker n'en voit jamais.)")
    return cid


def relances_max() -> int:
    """Combien de fois relancer un fil qui rend un appel au client. 0 = aucune."""
    brut = os.environ.get(_ENV_RELANCES, "").strip()
    if not brut:
        return 0
    if not brut.isdigit():
        raise LlmUnavailable(
            f"{_ENV_RELANCES} = {brut!r} : un entier positif ou nul est attendu")
    return int(brut)


_TRANSITOIRE = (429, 500, 502, 503, 504)

# La résolution d'alias est mise en cache AVEC UNE DURÉE : une bascule doit se
# voir dans l'heure de vol, pas au prochain redémarrage du worker.
_TTL_RESOLUTION_S = 600
_resolutions: dict = {}      # nom d'alias → (instant monotone, version concrète)


def base_api() -> str:
    """La base du service, SANS `/v1`.

    La base openai-compat de l'env de la box le porte déjà : la réutiliser
    telle quelle donnait `/v1/v1/conversations` → 404 « no Route matched »
    (vécu au premier appel réel de l'essai)."""
    base = (os.environ.get(_ENV_BASE) or _DEFAULT_BASE).rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def _lister_modeles() -> list:
    """Le catalogue du fournisseur — `data`, une entrée par modèle."""
    r = requests.get(f"{base_api()}/v1/models",
                     headers={"Authorization": f"Bearer {resolve_key()}"},
                     timeout=(10, 30))
    if r.status_code >= 400:
        raise RuntimeError(f"models → {r.status_code} : {r.text[:200]}")
    return (r.json() or {}).get("data") or []


def _rang(mid: str) -> int:
    """Le millésime d'un id (`mistral-large-2512` → 2512), -1 s'il n'en a pas."""
    m = re.search(r"-(\d{4})$", mid)
    return int(m.group(1)) if m else -1


def _version_concrete(nom: str, data: list) -> Optional[str]:
    """`mistral-large-latest` → `mistral-large-2512`.

    ⚠️ Le catalogue réel est SYMÉTRIQUE — relevé le 27/08 sur `/v1/models` :
    `{"id": "mistral-large-2512", "aliases": ["mistral-large-latest"]}` ET
    `{"id": "mistral-large-latest", "aliases": ["mistral-large-2512"]}`. Une
    règle qui déduit « ce nom flotte » du seul fait qu'il est CITÉ en alias
    fait donc flotter la version concrète elle-même, et ne laisse aucun
    candidat (vécu en prod : « introuvable au catalogue (56 modèles) »). Ce qui
    distingue les deux formes n'est pas la citation, c'est le SUFFIXE `-latest`.

    Candidat = un `id` ≠ `nom` qui cite `nom` en alias, ou que `nom` cite en
    alias ; les `-latest` sont écartés. Aucun candidat et `nom` ne finit pas par
    `-latest` ⟹ `nom` est déjà concret. Plusieurs candidats (un même alias a
    connu plusieurs millésimes) ⟹ le millésime le plus grand, journalisé."""
    cites = {a for e in data
             if (e or {}).get("id") == nom
             for a in ((e or {}).get("aliases") or [])}
    candidats = sorted({
        eid for e in data
        if (eid := (e or {}).get("id"))
        and eid != nom and not eid.endswith("-latest")
        and (nom in ((e or {}).get("aliases") or []) or eid in cites)})
    if not candidats:
        return None if nom.endswith("-latest") else nom
    if len(candidats) > 1:
        logger.info("alias %s : %s millésimes candidats (%s) — retenu le plus "
                    "récent", nom, len(candidats), ", ".join(candidats))
    return max(candidats, key=lambda m: (_rang(m), m))


def modele_resolu(nom: str) -> Optional[str]:
    """La version concrète derrière `nom`, ou None si elle ne s'établit pas.

    SEULE tolérance de ce chemin : une panne réseau (ou un catalogue muet) rend
    None, journalisé en warning — un relevé d'observabilité ne fait jamais
    échouer un job que la campagne a déjà payé. L'échec n'est PAS mis en cache :
    le job suivant retente."""
    fige = _resolutions.get(nom)
    if fige is not None and time.monotonic() - fige[0] < _TTL_RESOLUTION_S:
        return fige[1]
    try:
        data = _lister_modeles()
    except Exception as e:  # noqa: BLE001 — cf. docstring
        logger.warning("version de %s non résolue (%s) : le job ne la portera pas",
                       nom, e)
        return None
    resolu = _version_concrete(nom, data)
    if resolu is None:
        logger.warning("version de %s introuvable au catalogue (%s modèles) : "
                       "le job ne la portera pas", nom, len(data))
        return None
    _resolutions[nom] = (time.monotonic(), resolu)
    return resolu


def run_once(*, instructions: str, inputs: str, tools,
             api_key: Optional[str] = None, modele: Optional[str] = None,
             on_event=None) -> AgentResult:
    """UNE conversation complète (outils compris, côté Mistral) → AgentResult.

    `on_event(type, champs)` : le journal du travail — `conversation` (la requête
    ENTIÈRE moins la clé), `reponse` (les `outputs` bruts, entiers, avec usage et
    modèle), `relance` (l'appel renvoyé et le fil rejoué). Sur ce chemin c'est la
    seule trace des tours : `store=False`, rien ne reste chez le fournisseur.

    Rejeux : 2, espacés, sur les seuls TRANSITOIRES HTTP — un DeadlineExceeded
    (>15 min murales) remonte tel quel : le retry de JOB décide, re-payer
    aveuglément un run entier n'est pas une politique de rejeu.

    Un fil qui rend un `function.call` au client est relancé au plus
    `OTO_RUNNER_RELANCES_MAX` fois ; les passes forment UN bilan (pas
    concaténés, jetons additionnés).

    Le bilan porte la version CONCRÈTE que l'alias résolvait au moment de
    l'appel : sans elle, une bascule d'alias ne se date pas après coup."""
    # `modele` = celui que l'agent DÉCLARE (oto-backend #939) ; à défaut celui du
    # worker. Il traverse `modele_resolu` comme l'autre : c'est l'alias servi qui
    # est relevé, et un alias déclaré par un agent en a autant besoin.
    nom = modele or model()
    corps = {
        "model": nom,
        "inputs": inputs,
        "instructions": instructions,
        "tools": [{"type": "connector", "connector_id": connector_id(),
                   "tool_configuration": {"include": list(tools or ())}}],
        "store": False,
        "stream": False,
    }
    entetes = {"Authorization": f"Bearer {api_key or resolve_key()}",
               "Content-Type": "application/json"}
    url = f"{base_api()}/v1/conversations"
    maxi = relances_max()
    # Relevé AVANT le premier POST : c'est la version en vigueur au moment de
    # l'appel qu'on enregistre, pas celle d'après un run d'un quart d'heure.
    resolu = modele_resolu(nom)

    def note(ev: str, **champs) -> None:
        if on_event:
            on_event(ev, champs)

    cumul: Optional[AgentResult] = None
    for relance in range(maxi + 1):
        note("conversation", url=url, corps=corps, relance=relance,
             modele_resolu=resolu)
        d = _poster(url, corps, entetes)
        note("reponse", outputs=d.get("outputs"), usage=d.get("usage"),
             modele=d.get("model"), relance=relance)
        cumul = _cumuler(cumul, _parse(d, tools, demande=nom))
        renvoyes = _appels_renvoyes(d)
        if not renvoyes or relance == maxi:
            break
        logger.info("conversation relancée (%s/%s) : function.call renvoyé « %s »",
                    relance + 1, maxi, str(renvoyes[0].get("name") or "?")[:60])
        corps = dict(corps, inputs=_entrees_de_relance(inputs, d, renvoyes))
        note("relance", numero=relance + 1, renvoyes=renvoyes)
    # ⚠️ NON CHANGÉ, et la question mérite d'être posée ailleurs : quand la
    # version concrète ne se résout pas (catalogue muet, panne réseau), ce champ
    # tombe à `null` — décision explicite, tenue par un banc
    # (`test_un_job_survit_a_une_resolution_impossible`). Tant que le modèle
    # venait de l'environnement du worker, ce null coûtait peu : on savait
    # lequel c'était. Depuis que l'agent le DÉCLARE, il efface aussi la trace de
    # SON choix. Le repli naturel serait `resolu or nom` ; il renverse un
    # arbitrage tenu par un test, et ce lot n'est pas le bon endroit pour le
    # faire en passant.
    cumul.model = resolu
    return cumul


def _poster(url: str, corps: dict, entetes: dict) -> dict:
    """UN POST vers /v1/conversations, rejeux transitoires compris → son JSON."""
    derniere = None
    for essai in range(3):
        # read 660 s : une conversation exécute ses outils côté Mistral SANS
        # streamer — un silence de >5 min y est normal, pas une panne (vécu :
        # un run légitime coupé à read=300 dès la 3e fiche de campagne). La
        # deadline MURALE (900 s) reste le seul vrai couperet.
        r = post_with_deadline(url, json=corps, headers=entetes,
                               timeout=(10, 660), wall_s=_WALL_S)
        if r.status_code in _TRANSITOIRE:
            derniere = f"{r.status_code} : {r.text[:200]}"
            logger.warning("conversations %s (essai %s/3)", derniere, essai + 1)
            time.sleep(5 * (essai + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"conversations → {r.status_code} : {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"conversations → transitoire persistant ({derniere})")


def _appels_renvoyes(d: dict) -> list:
    """La TRAÎNE de `function.call` qui termine les outputs — les appels rendus
    au client. En mode connecteur aucun n'a lieu d'être, quel qu'en soit le nom :
    tous se répondent, aucun ne se joue."""
    traine = []
    for e in reversed(d.get("outputs") or []):
        if ((e or {}).get("type") or "") != "function.call":
            break
        traine.append(e)
    traine.reverse()
    return traine


def _entree_initiale(inputs) -> list:
    if isinstance(inputs, str):
        return [{"object": "entry", "type": "message.input",
                 "role": "user", "content": inputs}]
    return list(inputs)


def _depouiller(entree: dict) -> dict:
    """Une entrée du fil, réduite à ce que l'API accepte d'un client."""
    typ = (entree or {}).get("type") or ""
    champs = _CHAMPS_ENTREE.get(typ)
    if champs is None:
        raise RuntimeError(
            f"conversations → entrée de type {typ!r} inconnue : le fil ne peut "
            "pas être renvoyé sans savoir ce qui s'y envoie")
    return {c: entree[c] for c in champs if c in entree}


def _identifiant_dappel(deja: set) -> str:
    while True:
        tid = "".join(random.choices(_ALPHABET_APPEL, k=_LONGUEUR_APPEL))
        if tid not in deja:
            deja.add(tid)
            return tid


def _execution_en_paire(sortie: dict, deja: set) -> list:
    """Une exécution d'outil, dite dans la seule forme que le service accepte
    d'un client : l'appel, puis son résultat. Un `tool.execution` renvoyé tel
    quel est refusé (422), et le taire ferait refaire un travail déjà payé."""
    nom = sortie.get("name")
    if not nom:
        raise RuntimeError(
            "conversations → tool.execution sans name : l'exécution ne peut "
            "pas être redite au modèle")
    info = sortie.get("info")
    tid = _identifiant_dappel(deja)
    return [
        {"object": "entry", "type": "function.call", "tool_call_id": tid,
         "name": nom,
         # L'appel est de l'HISTOIRE : il a déjà eu lieu, ses arguments ne
         # rejouent rien. Ce qui compte est le résultat, juste dessous.
         "arguments": sortie.get("arguments") or "{}"},
        {"object": "entry", "type": "function.result", "tool_call_id": tid,
         "result": json.dumps(info, ensure_ascii=False, separators=(",", ":"))
         if info else _RESULTAT_NON_CONSERVE},
    ]


def _entrees_de_relance(inputs, d: dict, renvoyes: list) -> list:
    """Le fil rejoué : l'ordre initial, les outputs reçus — exécutions d'outils
    converties en paires appel/résultat —, puis la réponse à chaque appel rendu.
    Chaque entrée est dépouillée de ses champs serveur."""
    deja = {a.get("tool_call_id") for a in renvoyes}
    entrees = [_depouiller(e) for e in _entree_initiale(inputs)]
    for sortie in d.get("outputs") or []:
        if ((sortie or {}).get("type") or "") == "tool.execution":
            entrees.extend(_execution_en_paire(sortie, deja))
        else:
            entrees.append(_depouiller(sortie))
    for appel in renvoyes:
        tid = appel.get("tool_call_id")
        if not tid:
            raise RuntimeError(
                "conversations → function.call sans tool_call_id : la réponse "
                "ne permet pas de répondre à l'appel")
        entrees.append({"object": "entry", "type": "function.result",
                        "tool_call_id": tid,
                        "result": _CONSIGNE_APPEL_RENVOYE})
    return entrees


def _cumuler(cumul: Optional[AgentResult], passe: AgentResult) -> AgentResult:
    """Les passes d'un même job font UN bilan : le prix payé est leur somme, et
    le relevé des pas doit montrer l'appel rendu autant que ce qui l'a suivi."""
    if cumul is None:
        return passe
    # Les tours, déclarants et sommes s'ajoutent : un poste qu'une passe ne déclare
    # pas reste inconnu pour le travail, sans effacer ce que l'autre a déclaré.
    compte = Compteur.depuis(cumul.couverture).fusion(Compteur.depuis(passe.couverture))
    textes = [t for t in (cumul.reply, passe.reply) if t]
    return AgentResult(
        reply="\n".join(textes),
        steps=list(cumul.steps) + list(passe.steps),
        stopped=passe.stopped,
        usage=compte.usage(), couverture=compte.couverture(),
        raw_outputs=passe.raw_outputs,
        model=passe.model or cumul.model)


def _nom_outil(name: str, tools) -> str:
    """Le connecteur Mistral PRÉFIXE les noms d'outils de son propre nom
    (`oto-11aout_data_write`) : sans normalisation, les `tool_counts` du bilan
    ne matchent plus les noms de la plateforme — 13 jobs à « zéro data_write »
    alors que les fiches étaient écrites (vécu à l'essai des 20). On normalise
    par l'ALLOWLIST du job : exacte, courte, jamais une devinette de préfixe."""
    for t in tools or ():
        if name == t or name.endswith("_" + t):
            return t
    return name


def _parse(d: dict, tools=(), demande: Optional[str] = None) -> AgentResult:
    """Les `outputs` d'une conversation → le contrat AgentResult du worker.

    `demande` = le modèle DEMANDÉ pour ce travail — l'estampille de repli quand
    la réponse n'en porte pas. Passé en argument et non relu de l'environnement :
    depuis oto-backend#939 le modèle peut venir du travail, et `model()` rendrait
    alors celui du worker — une estampille fausse, ce qui est pire qu'aucune.

    Défensif sur la forme (chunks texte ou chaîne nue) — le banc fige ce qui est
    parsé, l'essai réel des 20 fiches valide contre le service."""
    steps: list[AgentStep] = []
    textes: list[str] = []
    for e in d.get("outputs") or []:
        typ = (e or {}).get("type") or ""
        if typ == "tool.execution":
            steps.append(AgentStep(tool=_nom_outil(e.get("name") or "?", tools),
                                   ok=True, duration_ms=0))
        elif typ.startswith("message"):
            contenu = e.get("content")
            if isinstance(contenu, list):
                contenu = "".join(c.get("text", "") for c in contenu
                                  if isinstance(c, dict))
            if contenu:
                textes.append(str(contenu))
        elif typ == "function.call":
            # Un connecteur bien configuré exécute côté serveur ; un function.call
            # NU qui remonte signifie « à toi de jouer » — le pas est compté non
            # exécuté, visible au bilan, et la relance y répond quand elle est armée.
            steps.append(AgentStep(tool=_nom_outil(e.get("name") or "?", tools),
                                   ok=False,
                                   duration_ms=0, error="function.call non exécuté"))
    # Un poste n'est posé que s'il est DÉCLARÉ (cf. `comptage`). Contrat Mistral
    # Conversations, lu le 13/09/2026 : aucun champ d'usage n'est obligatoire et aucun
    # ne dit ce que vaut son absence — absent, il reste inconnu. Aucun poste de cache
    # n'y est rendu : `prompt_tokens` est l'entrée TOTALE déclarée, et le non-caché
    # reste inconnu. `connector_tokens`/`connectors` (outils exécutés côté Mistral) ne
    # sont pas comptés : la couverture de ce chemin est partielle, et elle le dit.
    u = d.get("usage") or {}
    compte = Compteur()
    compte.ajouter({"input_total_tokens": u.get("prompt_tokens"),
                    "output_tokens": u.get("completion_tokens")})
    # La docstring de ce module PROMETTAIT déjà cette estampille — « le bilan
    # porte la version CONCRÈTE que l'alias résolvait au moment de l'appel » —
    # sans que personne ne la pose. Une promesse tenue par un commentaire est
    # exactement ce qui empêche de chercher le défaut.
    return AgentResult(reply="\n".join(textes).strip(), steps=steps,
                       stopped="end_turn", usage=compte.usage(),
                       couverture=compte.couverture(),
                       raw_outputs=d.get("outputs"),
                       model=d.get("model") or demande or model())
