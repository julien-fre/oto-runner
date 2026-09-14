"""Adaptateur OpenAI-compatible — Scaleway Generative APIs d'abord, tout endpoint
compatible ensuite.

Même contrat que l'adaptateur Anthropic (`complete` + les 4 hooks de forme de
fil), implémenté en `requests` NU — pas de SDK : l'API OpenAI-compatible est un
POST JSON, et une dépendance de moins est une dépendance de moins. La cible par
défaut est Scaleway (`api.scaleway.ai/v1`) : modèles ouverts, hébergement
France, prix plancher — le tool calling y est PROUVÉ par l'appel (14/08, quatre
modèles, arguments corrects du premier coup).

La forme de fil OpenAI, confinée ici :
- le tour assistant se rejoue comme LE MESSAGE COMPLET rendu par l'API
  (`content` + `tool_calls` intacts — reconstruire les tool_calls casserait la
  corrélation par id) ;
- chaque résultat d'outil est UN message `role:"tool"` séparé, corrélé par
  `tool_call_id` — l'inverse exact d'Anthropic (un seul message user), et c'est
  précisément pour ça que la boucle ne connaît AUCUNE de ces deux formes.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Callable, Optional

import requests

from .llm_types import EFFORT_SANS_RAISONNEMENT, LlmUnavailable, ToolCall, Turn

logger = logging.getLogger("oto_runner")

DEFAULT_BASE = "https://api.scaleway.ai/v1"
# gpt-oss-120b : le candidat du banc — 0,15/0,60 €/M, tool calling prouvé, et le
# même modèle que le spike Letta avait retenu. Surchargé par OTO_RUNNER_MODEL.
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_MAX_TOKENS = 8192
_TIMEOUT = (10, 300)
# ⚠️ Le read timeout d'urllib3 se RÉARME à chaque octet reçu : un serveur qui
# goutte tient la connexion indéfiniment (vécu : un tour de modèle figé 35 min,
# pile bloquée dans ssl.read). Le plafond wall-clock coupe pour de vrai.
_WALL_TIMEOUT_S = 420
# ── La RETENTATIVE d'un tour de modèle ───────────────────────────────────────
#
# ⚠️ Un incident de TRANSPORT n'est pas une réponse : il ne tue plus le travail.
# Nuit du 06/09/2026, deux travaux en mode direct morts sur un `ReadTimeout`
# isolé (l'un pendant l'envoi — 10 s —, l'autre après 300 s sans un octet, cinq
# minutes après le dernier appel d'outil). En flotte, le job se serait rejoué
# plus tard ; en direct, il était PERDU (« volume atteint »), avec son run
# ouvert et sa ligne verrouillée. Trois essais, 5 s puis 20 s.
_ESSAIS = 3
_ATTENTES_S = (5, 20)
# ⚠️ `ReadTimeout` et `ConnectTimeout` descendent de `Timeout` ; `ConnectTimeout`
# descend AUSSI de `ConnectionError`. Ces deux classes couvrent les trois cas.
_TRANSPORT = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
# Ce que rend un fournisseur DÉBORDÉ, par opposition à un appel mal formé : 429
# (quota) et les 5xx sont des états passagers. Un 4xx est une RÉPONSE — le
# rejouer rejouerait le même verdict, et le journal porterait trois fois la
# même erreur au lieu d'une.
_STATUTS_REJOUES = frozenset({429, 500, 502, 503, 504})


class _Deadline(Exception):
    pass


def _post_une_fois(url: str, corps: dict, entetes: dict):
    """UN POST, sous un VRAI plafond de durée (SIGALRM — le worker est mono-thread).

    ⚠️ L'alarme est armée ICI et désarmée ICI, à chaque essai : une attente entre
    deux essais ne doit jamais courir sous l'alarme du précédent, et le handler
    d'origine est rendu à chaque sortie."""
    def _coupe(signum, frame):
        raise _Deadline()

    ancien = signal.signal(signal.SIGALRM, _coupe)
    signal.alarm(_WALL_TIMEOUT_S)
    try:
        return requests.post(url, json=corps, timeout=_TIMEOUT, headers=entetes)
    except _Deadline:
        raise LlmUnavailable(
            f"tour de modèle > {_WALL_TIMEOUT_S}s (deadline wall-clock) — "
            "le serveur gouttait sans conclure") from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, ancien)


def _post_borne(url: str, corps: dict, entetes: dict,
                on_event: Optional[Callable[[str, dict], None]] = None):
    """Le POST au fournisseur, RETENTÉ sur incident de transport.

    Trois essais, 5 s puis 20 s d'attente, chacun sous son propre plafond mural.
    Chaque retentative est DITE au journal du travail (événement `systeme`) :
    sans elle, un tour qui a coûté trente secondes de plus n'a aucune explication
    après coup, et c'est le journal qui fait foi.

    Après le dernier essai : `LlmUnavailable` — jamais une exception `requests`
    nue. C'est la classe que la boucle a déjà pour « le substrat n'a pas
    répondu », et le worker en fait un échec propre (run clos, ligne rendue).

    Un statut rejouable encore présent au DERNIER essai rend la réponse telle
    quelle : c'est `complete` qui lève alors, avec le dire du serveur ENTIER —
    on ne remplace pas ce que le fournisseur explique par notre propre résumé.

    ⚠️ La deadline murale n'est PAS rejouée : elle a déjà attendu sept minutes
    sur un serveur qui gouttait, la rejouer paierait trois fois cette attente."""
    dernier: Optional[BaseException] = None
    motif = ""
    for essai in range(1, _ESSAIS + 1):
        try:
            r = _post_une_fois(url, corps, entetes)
        except _TRANSPORT as e:
            dernier, motif = e, f"{type(e).__name__} : {e}"
        else:
            if r.status_code not in _STATUTS_REJOUES or essai == _ESSAIS:
                return r
            dernier, motif = None, f"HTTP {r.status_code} : {r.text[:300]}"
        if essai == _ESSAIS:
            break
        attente = _ATTENTES_S[essai - 1]
        if on_event:
            on_event("systeme", {"quoi": "retentative du tour de modèle",
                                 "essai": essai, "essais": _ESSAIS,
                                 "sur": motif, "attente_s": attente})
        logger.warning("tour de modèle : %s — essai %s/%s dans %s s",
                       motif, essai + 1, _ESSAIS, attente)
        time.sleep(attente)
    raise LlmUnavailable(
        f"le fournisseur n'a pas répondu en {_ESSAIS} essais — dernier "
        f"incident : {motif}") from dernier


_ENV_KEY = "OTO_RUNNER_OPENAI_API_KEY"
_ENV_BASE = "OTO_RUNNER_OPENAI_BASE"


def model() -> str:
    return os.environ.get("OTO_RUNNER_MODEL") or DEFAULT_MODEL


def base_url() -> str:
    return (os.environ.get(_ENV_BASE) or DEFAULT_BASE).rstrip("/")


def max_tokens() -> int:
    """`OTO_RUNNER_MAX_TOKENS` — le plafond de COMPLÉTION d'un tour (défaut 8192).

    ⚠️ Sur Scaleway les jetons de RAISONNEMENT partagent ce plafond avec la
    réponse : une fiche fait 3–6 k, et 8192 coupe la réponse quand le modèle a
    raisonné avant (`finish_reason: length`, visible dans le journal du travail
    au `stop_reason` du tour). Le régler par flotte est une affaire d'env du
    worker, comme le modèle."""
    brut = os.environ.get("OTO_RUNNER_MAX_TOKENS", "").strip()
    if not brut:
        return DEFAULT_MAX_TOKENS
    if not brut.isdigit() or int(brut) < 1:
        raise LlmUnavailable(f"OTO_RUNNER_MAX_TOKENS = {brut!r} : un entier ≥ 1 est attendu")
    return int(brut)


def plafond_de_sortie(max_output_tokens: Optional[int], effort: Optional[str]) -> int:
    """Le plafond de COMPLÉTION d'un tour : celui que porte le TRAVAIL, à défaut celui de
    l'hôte (`max_tokens()`). Le catalogue du backend le déclare par modèle
    (`max_output_tokens`, 14/09/2026), comme il y déclare l'effort.

    ⚠️ Les jetons de raisonnement se comptent dans la complétion et partagent ce
    plafond avec la réponse. Mesuré au banc le 14/09/2026, `max_tokens` à
    16 000 : `mistral-medium-2604` en `high` monte à 6 964 jetons de complétion par
    tour (p90 3 968).

    Un effort de raisonnement porté par le travail SANS plafond porté LÈVE, plutôt que de
    retomber en silence sur celui de l'hôte, qui couperait la réponse : la coupe se lirait
    `fin_anormale` sur la fiche plutôt que « catalogue incomplet ». `none` ne raisonne pas.
    Un effort d'HÔTE (`OTO_RUNNER_EFFORT`) garde le plafond d'hôte : cet hôte règle les deux.

    Remplace `OTO_RUNNER_MAX_TOKENS_EFFORT`, la surcharge d'hôte qui attendait ce champ du
    catalogue ; posée sur un hôte, elle n'a plus aucun lecteur."""
    if max_output_tokens is not None:
        return max_output_tokens
    if effort and effort != EFFORT_SANS_RAISONNEMENT:
        raise LlmUnavailable(
            f"ce travail porte l'effort de réflexion `{effort}` sans plafond de complétion "
            "(`max_output_tokens`) : le raisonnement partage ce plafond avec la réponse, et "
            "celui de l'hôte la couperait. Le catalogue du backend le déclare par modèle "
            "(`runner_models`) — il n'y a pas de repli.")
    return max_tokens()


def temperature_hote() -> Optional[float]:
    """`OTO_RUNNER_TEMPERATURE` — le défaut de CET hôte, quand le travail n'en
    déclare aucune ; absent aussi = on n'envoie RIEN et le fournisseur applique
    le sien.

    ⚠️ Elle ne prime jamais sur ce que le passage déclare (`AgentSpec.temperature`,
    servi par la campagne depuis oto-backend v1.244.0). L'ordre est : le passage,
    puis l'hôte, puis le fournisseur — du plus proche du métier au plus lointain.
    C'est la raison du renommage : un `temperature()` nu se lisait comme LA
    température, alors qu'il n'en est que le dernier recours.

    ⚠️ Mesuré le 06/09/2026 : sans cette clé, Mistral échantillonne à son défaut,
    et deux passages de la MÊME procédure sur le MÊME banc de trois lignes vont
    de 11 à 18 sur 18. Une journée d'itérations a comparé des versions de texte
    dont l'écart était entièrement dans ce bruit. Poser `0` rend les passages
    comparables ; ne rien poser garde le comportement d'avant, celui de la
    production."""
    v = os.environ.get("OTO_RUNNER_TEMPERATURE", "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        raise ValueError(
            f"OTO_RUNNER_TEMPERATURE={v!r} : un nombre est attendu (par ex. 0)")


def effort_hote() -> Optional[str]:
    """`OTO_RUNNER_EFFORT` — le défaut de CET hôte, quand le travail n'en porte aucun ;
    envoyé en `reasoning_effort` (le nom OpenAI-compatible). Absent aussi = on
    n'envoie RIEN et le fournisseur applique son défaut. Renommé comme
    `temperature_hote` le 14/09/2026 : l'effort du TRAVAIL prime désormais sur lui. ⚠️ Scaleway active le raisonnement par défaut et le FACTURE : ne pas
    pouvoir le régler coûte. Aucune valeur par défaut ici — la variable n'était lue
    que côté Anthropic (`output_config.effort`), et personne ne le savait."""
    return os.environ.get("OTO_RUNNER_EFFORT", "").strip() or None


_ENV_PARALLEL = "OTO_RUNNER_PARALLEL_TOOLS"


def parallel_tools() -> bool:
    """`OTO_RUNNER_PARALLEL_TOOLS` — `1` (défaut, le comportement actuel) ou `0`
    pour UN SEUL appel d'outil par tour, envoyé en `parallel_tool_calls: false`
    (le nom OpenAI-compatible, accepté par Mistral).

    ⚠️ Mesure de la nuit du 06/09/2026 : Mistral Large 3 groupe jusqu'à **13
    appels d'outils dans un même tour**, puis écrit la fiche sans JAMAIS
    reformuler une requête après un résultat décevant — il a tout demandé avant
    d'avoir rien lu. Le mode séquentiel rend chaque résultat visible avant
    l'appel suivant ; il se TESTE, il n'est pas encore le défaut.

    ⚠️ Une valeur illisible LÈVE. Un réglage qu'on croit posé et qui ne l'est pas
    ferait conclure un banc sur le comportement d'en face."""
    brut = os.environ.get(_ENV_PARALLEL, "").strip()
    if not brut:
        return True
    if brut not in ("0", "1"):
        raise ValueError(
            f"{_ENV_PARALLEL} = {brut!r} : `1` (groupé, le défaut) ou `0` "
            "(séquentiel) est attendu")
    return brut == "1"


def reglages() -> dict:
    """Ce que ce provider a de RÉGLABLE et qui change le déroulé — lu par la
    boucle pour l'événement `systeme` du journal, à côté de `max_tool_output` :
    un passage se relit sans avoir à deviner sous quel réglage il a tourné."""
    return {"parallel_tool_calls": parallel_tools()}


# ⚠️ Ce provider parle à un hôte CONFIGURABLE (Scaleway par défaut, mais aussi
# La Plateforme). Le dépôt de clé se lit donc de la base URL, jamais du nom du
# module ni de la variable d'environnement : `OTO_RUNNER_OPENAI_API_KEY` sert
# les deux, et croire qu'une clé « openai » appartient à Mistral demanderait à
# une org la clé d'un fournisseur chez qui elle ne tourne pas. Un hôte absent de
# cette table n'a PAS de dépôt : la plateforme paie, et ça se dit ainsi.
_DEPOTS_PAR_HOTE = {"api.mistral.ai": "mistral"}


def depot() -> str:
    from urllib.parse import urlparse
    return _DEPOTS_PAR_HOTE.get(urlparse(base_url()).netloc, "")


# Les dépôts dont le contrat DOCUMENTÉ définit un `cached_tokens` absent comme zéro.
# Mistral (docs.mistral.ai, « Prompt caching », lu le 13/09/2026) : « If the API
# doesn't serve tokens from cache, `cached_tokens` is `0` or omitted. » Scaleway ne
# rapporte aucun cache sur le chat, et OpenAI a retiré sa garantie du zéro : chez eux
# l'absence reste INCONNUE. Le zéro est celui du contrat, jamais décidé ici.
_CACHE_ABSENT_VAUT_ZERO = frozenset({"mistral"})


def usage_declare(u: Optional[dict]) -> dict:
    """L'usage d'un tour tel que le fournisseur le DÉCLARE (cf. `comptage`) : un
    poste qu'il ne déclare pas n'est pas posé.

    ⚠️ `prompt_tokens` COMPTE les jetons servis par le cache (Mistral : « `prompt_tokens`
    contains all prompt tokens »). Les porter tels quels ferait payer au plein tarif,
    dans nos relevés, ce qui est facturé 10 %. Il part donc en `input_total_tokens`,
    et `input_tokens` — le non caché — n'est posé que si le cache lu est CONNU :
    cache inconnu, le non-caché l'est aussi (cf. `comptage`)."""
    if not u:
        return {}
    caches = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if caches is None and depot() in _CACHE_ABSENT_VAUT_ZERO:
        caches = 0
    usage: dict = {}
    if caches is not None:
        usage["cache_read_input_tokens"] = int(caches)
    if u.get("prompt_tokens") is not None:
        usage["input_total_tokens"] = int(u["prompt_tokens"])
        if caches is not None:
            usage["input_tokens"] = max(0, int(u["prompt_tokens"]) - int(caches))
    if u.get("completion_tokens") is not None:
        usage["output_tokens"] = int(u["completion_tokens"])
    return usage


def resolve_key() -> str:
    key = os.environ.get(_ENV_KEY, "").strip()
    if not key:
        raise LlmUnavailable(
            f"{_ENV_KEY} absente de l'environnement du worker — pour Scaleway, "
            "une clé IAM du projet qui paie (SCW secret key)")
    return key


# ── La FORME du fil, confinée ici ────────────────────────────────────────────
def user_message(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_message(turn: Turn) -> dict:
    """Le message assistant COMPLET rendu par l'API, rejoué tel quel — les
    `tool_calls` doivent revenir intacts pour que les messages `role:tool`
    suivants se corrèlent par id."""
    return turn.raw_content if isinstance(turn.raw_content, dict) else {
        "role": "assistant", "content": turn.text}


def tool_messages(results: list[dict]) -> list[dict]:
    """UN message `role:tool` PAR résultat, corrélé par `tool_call_id`.
    `results` = [{id, text, is_error}] — l'erreur est un contenu comme un autre,
    le modèle la lit pour se corriger."""
    return [{"role": "tool", "tool_call_id": r["id"], "content": r["text"]}
            for r in results]


def format_tools(schemas: list[dict]) -> list[dict]:
    """Schémas neutres → format OpenAI (`type:function`, `parameters`)."""
    return [{"type": "function",
             "function": {"name": s["name"],
                          "description": s.get("description", ""),
                          "parameters": s.get("input_schema")
                          or {"type": "object", "properties": {}}}}
            for s in schemas]


def complete(*, system: str, messages: list, tools: list[dict],
             api_key: Optional[str] = None,
             temperature: Optional[float] = None,
             modele: Optional[str] = None,
             effort: Optional[str] = None,
             max_output_tokens: Optional[int] = None,
             on_event: Optional[Callable[[str, dict], None]] = None,
             workspace: Optional[str] = None) -> Turn:
    """UN tour de modèle — synchrone, le worker a le droit d'attendre.

    Le `system` passe en premier message (la convention OpenAI) ; `messages` est
    le fil au format OpenAI (les `provider_raw` rejoués). Toute erreur HTTP
    remonte à la boucle avec le DIRE du serveur, entier — jamais avalée.

    `on_event(type, champs)` : le journal du travail, quand la boucle en tient
    un — il reçoit chaque retentative de transport (cf. `_post_borne`)."""
    if workspace:
        # Un workspace n'a de sens que pour une clé Anthropic : le recevoir ici, c'est un
        # travail mal routé. Refusé en le nommant plutôt qu'ignoré — un réglage qui ne
        # fait rien se croit appliqué.
        raise LlmUnavailable(
            "ce travail porte un workspace de clé Anthropic (`model_workspace`), et ce "
            "worker sert la voie Chat Completions, qui n'en a pas l'usage : il n'est pas "
            "exécuté.")
    nom = modele or model()
    corps = {
        "model": nom,
        # Le plafond que porte le TRAVAIL, à défaut celui de l'hôte ; un effort de
        # travail sans plafond lève (cf. `plafond_de_sortie`).
        "max_tokens": plafond_de_sortie(max_output_tokens, effort),
        "messages": [{"role": "system", "content": system}, *messages],
        # ⚠️ SANS cette cle, le fournisseur ne met rien en cache — mesure du
        # 01/09 : deux appels identiques, zero jeton mis en cache ; avec elle,
        # 96 % des le second appel. Elle est STABLE par procedure : ce qui vaut
        # d'etre garde est le prefixe partage — consigne et outils — pas le fil
        # d'une fiche. Un identifiant par fiche ne partagerait rien.
        "prompt_cache_key": os.environ.get("OTO_RUNNER_CACHE_KEY")
        or "oto-runner-procedure",
    }
    if tools:
        corps["tools"] = tools
    # Le travail d'abord, l'hôte à défaut, rien sinon — le même ordre que la
    # température. Calculé UNE fois : le tour le porte, le journal le lit.
    effort_retenu = effort or effort_hote()
    # ⚠️ `none` PART tel quel, à la différence de la voie Anthropic : c'est une valeur de
    # l'API (`mistral-medium-2604` n'accepte que `high` et `none`, 400 sur `medium`,
    # mesuré le 14/09/2026). L'omettre laisserait le défaut du fournisseur, qui peut
    # raisonner et le facturer.
    if effort_retenu:
        corps["reasoning_effort"] = effort_retenu
    # Le passage d'abord, l'hôte à défaut, rien sinon. Le calcul est fait UNE
    # fois : appeler deux fois relisait l'environnement entre le test et l'usage.
    retenue = temperature if temperature is not None else temperature_hote()
    if retenue is not None:
        corps["temperature"] = retenue
    if effort_retenu and retenue == 0:
        # ⚠️ Mistral REFUSE un effort en échantillonnage glouton sans `top_p: 1`.
        # Mesuré le 14/09/2026 sur mistral-medium-2604 : `reasoning_effort: high` et
        # `temperature: 0` → 400 `invalid_request_greedy_sampling` (« top_p must be 1
        # when using greedy sampling ») ; 200 avec `top_p: 1`. Sans effort, ou hors
        # T = 0, rien n'est ajouté : ces requêtes restent inchangées à l'octet.
        corps["top_p"] = 1
    if not parallel_tools():
        # ⚠️ Le serveur peut TOUT DE MÊME rendre plusieurs appels dans un tour :
        # la boucle les exécutera comme d'habitude. Aucune garde ici — ce serait
        # cacher que le fournisseur n'a pas respecté ce qu'on lui a demandé.
        corps["parallel_tool_calls"] = False
    r = _post_borne(base_url() + "/chat/completions", corps,
                    {"Authorization": f"Bearer {api_key or resolve_key()}"},
                    on_event=on_event)
    if r.status_code >= 400:
        try:
            detail = r.json().get("message") or r.json().get("error") or r.text
        except Exception:  # noqa: BLE001
            detail = r.text
        # Le DIRE du serveur, ENTIER : il finit dans le journal du travail, où
        # c'est la seule trace de ce qu'un fournisseur sans état a répondu.
        raise RuntimeError(f"chat/completions → {r.status_code} : {detail}")
    d = r.json()

    choix = (d.get("choices") or [{}])[0]
    msg = choix.get("message") or {}
    fin = choix.get("finish_reason")
    usage = usage_declare(d.get("usage"))

    # Ce que le fournisseur DIT avoir servi, à défaut ce qu'on a demandé.
    servi = d.get("model") or nom

    if fin == "content_filter":
        return Turn(text="", tool_calls=(), stop_reason="refusal",
                    raw_content=msg, usage=usage, model=servi, temperature=retenue, effort=effort_retenu)

    calls = []
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function") or {}
        brut = f.get("arguments")
        try:
            args = json.loads(brut) if isinstance(brut, str) else None
        except ValueError:
            args = None
        # ⚠️ Des arguments qui ne sont pas un OBJET JSON ne se réparent pas. Les
        # remplacer par `{}` faisait EXÉCUTER l'outil sans eux — un outil à paramètres
        # facultatifs agissait (audit du 13/09/2026). Un vrai `"{}"` reste valide ;
        # une chaîne illisible, une liste ou une absence ne le sont pas : l'appel
        # porte ce qui a été rendu, et la boucle refuse de l'exécuter.
        valide = isinstance(args, dict)
        calls.append(ToolCall(id=tc.get("id") or "", name=f.get("name") or "",
                              arguments=args if valide else {},
                              arguments_invalides=None if valide else (
                                  brut if isinstance(brut, str) else json.dumps(brut))))
    contenu = msg.get("content") or ""
    defaut = None if calls else appel_mal_encode(contenu, _noms_envoyes(tools))
    if isinstance(contenu, list):
        # Mistral rend parfois le contenu en LISTE de blocs typés au lieu
        # d'une chaîne (vécu, job 52 — AttributeError au .strip()).
        contenu = "\n".join(b.get("text", "") for b in contenu
                            if isinstance(b, dict) and b.get("type") == "text")
    # ⚠️ Deux fins seulement CONCLUENT un tour : `stop`, et `tool_calls` quand un appel
    # est là. Toute autre — `length`, `model_length` (sortie coupée), `error`, absente,
    # inconnue — est une fin ANORMALE : la réponse ou l'appel peut être incomplet. Lue
    # `end_turn`, elle concluait `done` un travail tronqué (audit du 13/09/2026). Et
    # `tool_calls` sans aucun appel ne fabrique pas une réussite. Énumérations lues le
    # 13/09/2026 : Mistral stop|length|model_length|error|tool_calls, Scaleway
    # stop|length|tool_calls, OpenAI y ajoute content_filter (refus, plus haut).
    anormale = fin != "stop" and not (fin == "tool_calls" and (calls or defaut))
    if anormale:
        defaut = {"forme": "fin_anormale", "finish_reason": fin}
    return Turn(text=contenu.strip(),
                tool_calls=tuple(calls),
                stop_reason=("fin_anormale" if anormale
                             else "appel_mal_encode" if defaut else "end_turn"),
                raw_content=msg, usage=usage, model=servi, temperature=retenue, effort=effort_retenu,
                defaut=defaut)


def _noms_envoyes(tools: list[dict]) -> frozenset:
    """Les noms des outils ENVOYÉS à ce tour (format OpenAI, cf. `format_tools`)."""
    return frozenset((t.get("function") or {}).get("name")
                     for t in (tools or ()) if isinstance(t, dict))


def appel_mal_encode(contenu, noms: frozenset) -> Optional[dict]:
    """Un appel d'outil que le fournisseur a rendu en TEXTE au lieu d'un `tool_calls`.

    ⚠️ Vécu : job 17275 (13/09/2026), et deux runs directs (06/09, 09/09). Le
    message n'a aucun `tool_calls` et se termine en `stop` ; son `content` est
    une liste où une partie `reference` nomme l'outil, suivie des arguments en
    texte. Lu comme une conclusion, le travail finissait `done` en deux pas, sans
    rien écrire, et sa ligne n'était jamais servie par la passe.

    Le critère est STRUCTUREL, et entier :
    - une partie `{"type": "reference"}` à EXACTEMENT un `reference_ids`,
      égal au nom d'un outil envoyé à ce tour ;
    - IMMÉDIATEMENT suivie d'une partie `{"type": "text"}` dont le texte ENTIER
      se lit en objet JSON.
    Rien d'autre ne le déclenche : un texte qui contient du JSON (2 545 tours sur
    45 698 balayés le 13/09), une `reference` vide, une chaîne. Le JSON n'est
    JAMAIS exécuté : le défaut est décrit, la boucle s'arrête, le travail échoue
    en le nommant."""
    if not isinstance(contenu, list):
        return None
    for partie, suivante in zip(contenu, contenu[1:]):
        if not (isinstance(partie, dict) and partie.get("type") == "reference"):
            continue
        ids = partie.get("reference_ids")
        if not (isinstance(ids, list) and len(ids) == 1 and ids[0] in noms):
            continue
        if not (isinstance(suivante, dict) and suivante.get("type") == "text"):
            continue
        try:
            arguments = json.loads((suivante.get("text") or "").strip())
        except ValueError:
            continue
        if isinstance(arguments, dict):
            return {"forme": "reference+texte_json", "outil": ids[0]}
    return None
