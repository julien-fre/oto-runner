"""La boucle d'agent du worker — transplantée du prototype, allégée par le design.

Le prototype (`server-agent`, oto-backend 25/07) tournait DANS le process du
backend : il devait ré-appliquer lui-même la rédaction de champs et surveiller
ses chemins de credential, parce que `Tool.run` court-circuitait le middleware.
**Ici le worker est un client MCP pur (ADR 0064-D1), et tout ça disparaît** :
chaque appel d'outil traverse la face MCP du backend, qui applique credential,
RBAC, activation, rédaction et journal — comme pour n'importe quel client. La
boucle n'a plus que trois responsabilités : le tour de modèle, l'allowlist,
les bornes.

Ce qui est conservé du prototype, à l'identique :
- l'allowlist FAIL-CLOSED (un outil hors liste revient au modèle en tour
  d'erreur — le modèle se corrige — jamais en exception qui tuerait le job) ;
- la troncature MARQUÉE d'une sortie d'outil (le modèle doit SAVOIR qu'il
  manque quelque chose, sinon il conclut sur un extrait en croyant tout voir) ;
- les résultats d'un tour rendus GROUPÉS au modèle — leur forme dans le fil
  appartient au provider (un message user chez Anthropic, N messages role:tool
  chez OpenAI) ;
- le texte intermédiaire gardé comme repli si le budget de tours s'épuise.

Divergence ASSUMÉE : le plafond de tours par défaut passe de 6 à 24. Le 6 du
prototype bornait un chat public à petites questions ; un run de procédure est
un travail (la veille LinkedIn réelle = 15 tours). Le plafond effectif se règle
par job, borné dur à 64.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .comptage import Compteur
from .llm_types import ToolCall, Turn  # noqa: F401 — le contrat du provider

# ── Le plafond de la sortie d'outil SERVIE AU MODÈLE, en caractères ─────────
#
# ⚠️ 12 000 caractères ont fait dérailler TOUS les passages de la nuit du
# 06/09/2026. La consigne métier rendue par `oto_procedure` fait 44 818
# caractères, le schéma du tableau 38 079 : le modèle n'en a lu que le premier
# tiers (~3 900 jetons d'entrée au tour suivant). Il a inventé des options
# (`active_maison_du_livre`, `liquidation_judiciaire`, `registre`) et une colonne
# (`dirigeants`), oublié `statut: enrichi` et `notes_verification`, relu la
# procédure trois fois pour tenter d'en voir plus, et conclu que « le schéma ne
# contient pas de colonne contacts » — elle existait, dans la partie coupée.
#
# 120 000 caractères ≈ 30 k jetons : une procédure de 45 k et un schéma de 40 k
# passent ENTIERS, et on reste très en dessous des fenêtres des modèles servis.
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 120_000
_ENV_MAX_TOOL_OUTPUT = "OTO_RUNNER_MAX_TOOL_OUTPUT"
# Signatures d'erreurs TRANSITOIRES d'outil : rejouées UNE fois, silencieusement
# (le modèle ne voit que la seconde réponse). La politique de reprise est de la
# MÉCANIQUE, pas de la consigne : en prose elle coûte des caractères (payés en
# écritures perdues, mesuré) et invite l'agent à contourner l'outil capricieux
# plutôt qu'à le retenter. Une erreur MÉTIER (not_found, 400) n'est jamais
# rejouée — elle est une réponse.
_TRANSIENT_RE = None  # compilé au premier usage (module importable sans re)
DEFAULT_MAX_STEPS = 24
HARD_MAX_STEPS = 64
MAX_HISTORY_MESSAGES = 60   # tours provider transportés au modèle (le fil complet
                            # reste au backend — ici on borne le COÛT d'un tour)


class ToolTransport(Protocol):
    """Ce que la boucle attend d'un transport d'outils : la face MCP du backend.

    `schemas(names)` → [{name, description, input_schema}] ; `call(name, args)` →
    (texte, is_error). Les gates et la rédaction sont CÔTÉ SERVEUR — le transport
    ne filtre rien, il transporte."""
    def schemas(self, names: frozenset) -> list[dict]: ...
    def call(self, name: str, arguments: dict) -> tuple[str, bool]: ...


@dataclass(frozen=True)
class AgentSpec:
    """Ce que l'agent est autorisé à être, pour UN run."""
    system: str
    tools: frozenset
    max_steps: int = DEFAULT_MAX_STEPS
    # ⚠️ Le plafond de JETONS d'un déroulé, appliqué PAR L'AGENT. `None` = pas de
    # borne (le comportement d'avant).
    #
    # Il existait jusqu'ici sur la flotte, et seul un ordonnanceur savait le lire
    # — donc personne, dès qu'un passage tourne sans lui. L'agent ne connaissait
    # qu'un plafond d'ÉTAPES, **qui ne dit rien de ce qu'une étape coûte** : une
    # ligne mesurée à 65 571 jetons le 01/09 tenait largement sous ses 40 pas.
    #
    # ⚠️ La borne est ici et pas ailleurs parce que c'est le SEUL endroit qui voit
    # le cumul en temps réel. Posée sur l'ordonnanceur, elle arrive après coup :
    # elle empêche le PROCHAIN travail, jamais celui qui dérive.
    max_tokens: Optional[int] = None
    # La température du déroulé, DÉCLARÉE par le passage. `None` = on n'envoie
    # rien et le fournisseur applique son défaut — le comportement d'avant.
    #
    # ⚠️ Elle vit ici, et pas dans l'environnement du worker, parce qu'elle est
    # un choix de PROCÉDURE et non d'hôte : deux campagnes servies par le même
    # worker n'en veulent pas la même. Mesuré le 06/09/2026 : sans valeur posée,
    # deux passages de la même procédure sur le même banc de trois lignes vont
    # de 11 à 18 sur 18 — un écart qui avale entièrement celui qu'on cherchait
    # à mesurer entre deux versions du texte.
    temperature: Optional[float] = None
    # L'effort de réflexion porté par le TRAVAIL — une propriété du modèle catalogué
    # côté backend (14/09/2026 : `mistral-medium-2604` part en `high`). `None` = rien
    # de plus qu'avant : l'hôte (`OTO_RUNNER_EFFORT`), puis le fournisseur.
    effort: Optional[str] = None
    # Le plafond de COMPLÉTION d'un tour porté par le TRAVAIL — propriété du modèle
    # catalogué côté backend, comme l'effort (14/09/2026 : 16 000 pour
    # `mistral-medium-2604`). `None` = celui de l'hôte (`OTO_RUNNER_MAX_TOKENS`).
    max_output_tokens: Optional[int] = None
    # Le MODÈLE de ce déroulé, déclaré par l'agent qui l'a demandé (oto-backend
    # #939 : le travail porte `payload.model`). `None` = celui du worker, pris
    # dans son environnement — le comportement d'avant, et celui de tout agent
    # déclaré sans modèle.
    #
    # ⚠️ Il vit ici, à côté de la température, pour la même raison : c'est un
    # choix de CE QU'ON DEMANDE, pas de l'hôte qui sert. Jusqu'ici il n'y avait
    # aucun endroit où le poser — `runner_fleets.model` était servi, validé à la
    # déclaration, et ignoré ; un champ qui PROMET une attribution qui n'arrive
    # pas coûte plus cher qu'un champ absent.
    #
    # ⚠️ Ce qu'il coûte, dit franchement : le cache de prompt est indexé par
    # modèle. Deux passages de modèles différents sur le même worker ne
    # partagent plus de préfixe — chacun garde le sien, aucun n'invalide
    # l'autre, mais le premier tour de chacun se paie plein tarif. Le pool reste
    # homogène par FAMILLE (le backend ne sert un travail qu'à un worker du
    # dépôt correspondant) ; c'est la variation À L'INTÉRIEUR d'une famille qui
    # segmente le cache.
    model: Optional[str] = None
    label: str = "run"


@dataclass
class AgentStep:
    tool: str
    ok: bool
    duration_ms: int
    error: Optional[str] = None
    # L'échec vient du TRANSPORT (session MCP perdue, réseau, protocole), pas de
    # l'outil : l'appel n'a jamais été exécuté. Un tel échec ne se lit pas comme
    # une réponse métier — un job qui en porte n'a pas « fait le travail ».
    transport_ko: bool = False

    def as_dict(self) -> dict:
        out = {"tool": self.tool, "ok": self.ok, "duration_ms": self.duration_ms}
        if self.error:
            out["error"] = self.error
        if self.transport_ko:
            out["transport_ko"] = True
        return out


@dataclass
class AgentResult:
    reply: str
    steps: list = field(default_factory=list)
    stopped: str = "end_turn"   # end_turn | max_steps | max_tokens | refusal | no_reply
    #                             | appel_mal_encode
    usage: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    raw_outputs: Optional[list] = None   # les entrées BRUTES du fournisseur,
    # quand il en rend (chemin Conversations) — de quoi diagnostiquer un run
    # dont le fil ne garde qu'une synthèse. Les autres providers laissent None.
    model: Optional[str] = None          # la version CONCRÈTE qui a tourné,
    # quand le provider sait la résoudre (chemin Conversations) : un alias
    # flottant ne se date pas après coup. Les autres providers laissent None.
    defaut: Optional[dict] = None        # le défaut de forme qui a ARRÊTÉ la
    # boucle (cf. `Turn.defaut`) — `None` quand elle s'est arrêtée autrement.
    couverture: Optional[dict] = None    # ce qui fonde `usage` (cf. `comptage`) :
    # les tours, et par poste combien l'ont déclaré et quoi. `None` = non compté.


# `on_turn(role, content_neutre, provider_raw)` : le point d'ancrage du FIL (R1).
# La boucle appose chaque tour au fil du backend PENDANT le run — c'est ce qui rend
# le worker jetable entre deux tours (un kill se répare par re-claim + rechargement).
# None = pas de persistance (tests, dry-run).
OnTurn = Callable[[str, dict, dict], None]
# `on_event(type, champs)` : le JOURNAL du travail — tout ce que la boucle voit,
# ENTIER : le prompt système, chaque tour du modèle (texte, appels avec leurs
# arguments complets, usage), chaque sortie d'outil telle que le transport l'a
# rendue (avant le plafond `_cap` que le modèle, lui, subit), et la fin. C'est ce
# qui manquait pour relire un passage après coup : le fil ne garde que la sortie
# tronquée, le résultat déclaré ne garde que des comptes. None = pas de journal.
OnEvent = Callable[[str, dict], None]


def max_tool_output() -> int:
    """Le plafond effectif : `OTO_RUNNER_MAX_TOOL_OUTPUT`, sinon le défaut.

    ⚠️ Une valeur illisible LÈVE. Un plafond qu'on croit posé et qui ne l'est
    pas coupe la consigne en silence — c'est exactement le défaut qu'on corrige,
    on ne va pas le réintroduire par un repli complaisant."""
    brut = os.environ.get(_ENV_MAX_TOOL_OUTPUT, "").strip()
    if not brut:
        return DEFAULT_MAX_TOOL_OUTPUT_CHARS
    if not brut.isdigit() or int(brut) < 1:
        raise ValueError(
            f"{_ENV_MAX_TOOL_OUTPUT} = {brut!r} : un entier ≥ 1 est attendu")
    return int(brut)


def _reglages_du_provider(provider) -> dict:
    """Les réglages PROPRES au provider qui changent le déroulé, pour le journal.

    La boucle n'en connaît aucun — elle demande, le provider répond ; celui qui
    n'a rien à déclarer n'ajoute rien à l'événement `systeme`. C'est le pendant
    de `max_tool_output` : sous quel réglage ce passage a tourné, dit UNE fois,
    à l'ouverture du journal. Sans ça, un banc qui compare deux réglages ne
    saurait pas, après coup, lequel a produit quel déroulé."""
    lire = getattr(provider, "reglages", None)
    return dict(lire()) if callable(lire) else {}


def _cap(text: str, limite: int) -> tuple[str, bool]:
    """(ce que le modèle lit, a-t-on coupé). La coupure est DITE, en français,
    avec ce qui manque.

    ⚠️ Un modèle qui ignore qu'il lit un extrait conclut SUR l'extrait en croyant
    tout voir : le 06/09, « le schéma ne contient pas de colonne contacts » sur
    un schéma coupé à son premier tiers. La phrase de fin nomme donc le nombre de
    caractères manquants et interdit explicitement la conclusion par absence."""
    if len(text) <= limite:
        return text, False
    return (text[:limite]
            + f"\n\n…[SORTIE TRONQUÉE : tu ne lis que les {limite} premiers "
              f"caractères sur {len(text)} — il en manque {len(text) - limite}. "
              "N'en conclus PAS que ce qui n'apparaît pas ici n'existe pas : "
              "affine ta requête (filtre, limite, section) pour lire la suite.]",
            True)


def _trim(messages: list) -> list:
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return list(messages)
    return list(messages[-MAX_HISTORY_MESSAGES:])


def execute_tool(spec: AgentSpec, transport: ToolTransport,
                 call: ToolCall) -> tuple[str, bool, bool]:
    """UN appel d'outil → (texte, is_error, panne de transport). Ne lève jamais :
    une erreur d'outil est un résultat que le modèle lit pour se corriger.
    Fail-closed sur l'allowlist AVANT tout transport.

    ⚠️ Le texte rendu est ENTIER, tel que le transport l'a rendu : c'est la boucle
    qui le plafonne pour le modèle (`_cap`), APRÈS l'avoir journalisé. Plafonner
    ici faisait disparaître la partie coupée pour tout le monde — le refus d'un
    schéma nomme la colonne et la raison, c'est exactement ce qu'on perdait.

    Le troisième terme sépare l'erreur MÉTIER (une réponse : not_found, 400) de
    la PANNE DE TRANSPORT — le transport a levé, l'appel n'a pas eu lieu. Le
    modèle lit les deux de la même façon (il n'y a rien d'autre à lui dire),
    mais le worker, lui, ne doit pas conclure « done » sur un travail que le
    transport a empêché."""
    if call.name not in spec.tools:
        return (f"Outil `{call.name}` indisponible pour ce run. "
                f"Outils autorisés : {', '.join(sorted(spec.tools)) or '(aucun)'}.",
                True, False)
    if call.arguments_invalides is not None:
        # ⚠️ Des arguments qui ne sont pas un objet JSON : l'appel n'est NI exécuté NI
        # réparé. Le modèle lit pourquoi, et renvoie l'appel s'il le veut — par la
        # boucle et ses bornes, jamais par un rejeu caché du transport.
        return (f"Appel de `{call.name}` NON exécuté : ses arguments ne sont pas un "
                f"objet JSON ({call.arguments_invalides[:200]!r}). Renvoie l'appel avec "
                "un objet JSON valide.", True, False)
    try:
        text, is_error = transport.call(call.name, call.arguments or {})
        if is_error and _est_transitoire(text):
            time.sleep(2)
            text, is_error = transport.call(call.name, call.arguments or {})
    except Exception as e:  # noqa: BLE001 — l'erreur de la cible EST un résultat
        return (f"Erreur de l'outil `{call.name}` : {e}", True, True)
    return (text, is_error, False)


def _est_transitoire(texte: str) -> bool:
    return bool(re.search(
        r"timeout|timed?\s*out|d\u00e9lai|429|too many requests|"
        r"50[234]|bad gateway|unavailable|connection|connexion",
        (texte or "")[:400], re.IGNORECASE))


def run(spec: AgentSpec, transport: ToolTransport, provider,
        prompt: Optional[str] = None,
        history: Optional[list] = None, on_turn: Optional[OnTurn] = None,
        api_key: Optional[str] = None,
        on_event: Optional[OnEvent] = None,
        workspace: Optional[str] = None) -> AgentResult:
    """La boucle, plus **ce qu'elle a déjà dépensé quand elle casse**.

    ⚠️ Les jetons d'un déroulé MORT (13/09/2026) : ils ont été dépensés chez le
    fournisseur quoi qu'il arrive ensuite, et ils mouraient avec la pile — le
    serveur comptait donc ZÉRO sur les déroulés qui partent en vrille,
    c'est-à-dire les plus chers, et qui recommencent à chaque tentative.

    L'accumulateur est créé ICI et passé à la boucle, qui le REMPLIT en place :
    c'est ce qui le rend lisible depuis l'extérieur quand une exception traverse.
    Accroché à l'exception plutôt que rendu — le contrat de `run` est de LEVER
    quand ça casse, et le changer ferait passer un échec pour un résultat auprès
    de tous ses appelants.
    """
    compte: dict = {"compteur": Compteur(), "pas": 0, "servi": None}
    try:
        return _run(spec, transport, provider, compte, prompt=prompt,
                    history=history, on_turn=on_turn, api_key=api_key,
                    on_event=on_event, workspace=workspace)
    except BaseException as e:
        # L'usage PAR POSTE tel que le comptage le rend (`None` = non déclaré sur au
        # moins un tour) et la couverture qui le fonde : un déroulé mort se lit comme
        # un déroulé conclu, sans zéro fabriqué (cf. `comptage`).
        e.usage_partiel = compte["compteur"].usage()               # type: ignore[attr-defined]
        e.couverture_partielle = compte["compteur"].couverture()   # type: ignore[attr-defined]
        e.modele_partiel = compte["servi"]                         # type: ignore[attr-defined]
        e.pas_partiels = compte["pas"]                             # type: ignore[attr-defined]
        raise


def _run(spec: AgentSpec, transport: ToolTransport, provider, compte: dict,
         prompt: Optional[str] = None,
         history: Optional[list] = None, on_turn: Optional[OnTurn] = None,
         api_key: Optional[str] = None,
         on_event: Optional[OnEvent] = None,
         workspace: Optional[str] = None) -> AgentResult:
    """La boucle : tours de modèle et d'outils jusqu'à conclusion ou plafond.

    `history` = les `provider_raw` du fil, rejoués dans l'ordre (continuation d'un
    run) ; `prompt` = le nouveau tour user (None = reprendre sans rien ajouter,
    ex. après un kill en plein tour). `on_turn` appose chaque tour au fil backend.

    `on_event(type, champs)` reçoit le journal ENTIER du déroulé : `systeme`,
    `historique` (le fil rechargé, tel que transporté), `utilisateur`, `modele`
    (un par tour, arguments d'appel complets), `outil` (un par appel, sortie
    complète), `fin`. Rien n'y est tronqué."""
    def note(ev: str, **champs) -> None:
        if on_event:
            on_event(ev, champs)

    messages = _trim(history or [])
    plafond = max(1, min(spec.max_steps, HARD_MAX_STEPS))
    # Lu UNE fois par déroulé, et DIT au journal : un passage se relit sans avoir
    # à deviner sous quel plafond de sortie d'outil il a tourné.
    limite_sortie = max_tool_output()
    note("systeme", texte=spec.system, outils=sorted(spec.tools),
         max_steps=plafond, max_tokens=spec.max_tokens, label=spec.label,
         max_tool_output=limite_sortie, **_reglages_du_provider(provider))
    if history:
        note("historique", messages=list(messages), total=len(history),
             transportes=len(messages))
    if prompt is not None:
        um = provider.user_message(prompt)
        messages.append(um)
        note("utilisateur", texte=prompt)
        if on_turn:
            on_turn("user", {"text": prompt}, um)

    schemas = provider.format_tools(transport.schemas(spec.tools))
    # Ce que le modèle lit de chaque outil : la longueur servie, et toute coupe DITE (11/09/2026 :
    # la coupe muette à 1 024 caractères masquait des règles de `data_write`).
    servies = getattr(transport, "descriptions_servies", None)
    if servies:
        note("descriptions_outils", outils=list(servies),
             coupees=[d["outil"] for d in servies if d["servie"] < d["longueur"]])
    steps: list[AgentStep] = []
    # ⚠️ Le compteur vient de l'APPELANT et se remplit en place : c'est ce qui le
    # rend lisible quand une exception traverse la boucle (cf. `run`).
    compteur = compte["compteur"]
    stopped = "end_turn"
    reply = ""
    servi: Optional[str] = None
    defaut: Optional[dict] = None
    n_tours = 0

    for _ in range(plafond + 1):
        # ⚠️ Le tour est CHRONOMÉTRÉ : sans ça, un journal ne dit pas si un tour a
        # pris deux secondes ou cinq minutes — et c'est la première question qu'on
        # se pose devant un travail mort sur un délai d'attente du fournisseur.
        debut_tour = time.monotonic()
        turn = provider.complete(system=spec.system, messages=messages,
                                 tools=schemas, api_key=api_key,
                                 temperature=spec.temperature,
                                 effort=spec.effort,
                                 max_output_tokens=spec.max_output_tokens,
                                 modele=spec.model,
                                 on_event=on_event,
                                 # Le workspace d'une clé d'organisation, comme la clé : un
                                 # satellite du credential, jamais journalisé. Transmis
                                 # seulement s'il est posé.
                                 **({"workspace": workspace} if workspace else {}))
        duree_tour_ms = int((time.monotonic() - debut_tour) * 1000)
        n_tours += 1
        non_mesures = compteur.ajouter(turn.usage)
        # Le DERNIER tour fait foi : un fournisseur qui bascule d'alias en cours
        # de déroulé a servi les deux, et c'est le second qu'on retrouvera.
        servi = turn.model or servi
        compte["servi"], compte["pas"] = servi, len(steps)
        note("modele", texte=turn.text, stop_reason=turn.stop_reason,
             appels=[{"id": c.id, "nom": c.name, "arguments": c.arguments}
                     for c in turn.tool_calls],
             usage=dict(turn.usage or {}), modele=turn.model,
             temperature=turn.temperature, effort=turn.effort,
             plafond=turn.plafond, top_p=turn.top_p,
             duree_ms=duree_tour_ms, brut=turn.raw_content)

        # ⚠️ Un appel d'outil rendu en TEXTE n'est pas une conclusion (job 17275) :
        # le prendre pour la réponse finale faisait un travail `done` sans rien
        # écrire. Il n'est pas exécuté non plus — le JSON d'un texte n'est pas un
        # appel. Le tour est déjà compté (usage ci-dessus) et journalisé (`modele`,
        # brut compris) ; la boucle s'arrête en le NOMMANT, et c'est à qui conclut
        # le travail d'en faire un échec.
        # ⚠️ AVANT la borne de jetons : un tour mal encodé qui franchit la borne
        # serait sinon un `max_tokens`, conclu `done` — le faux `done` subsistait
        # (relevé en revue le 13/09/2026).
        if turn.stop_reason == "appel_mal_encode":
            note("appel_mal_encode", **(turn.defaut or {}), tour=n_tours)
            stopped, reply, defaut = "appel_mal_encode", "", turn.defaut
            break
        # ⚠️ Une fin ANORMALE — sortie coupée, erreur du fournisseur, fin non déclarée :
        # la réponse ou l'appel de ce tour peut être incomplet. Rien n'est exécuté ni
        # conclu ; la boucle s'arrête en la NOMMANT, et qui conclut le travail en fait
        # un échec (audit du 13/09/2026 : `length` sans appel concluait `done`).
        if turn.stop_reason == "fin_anormale":
            note("fin_anormale", **(turn.defaut or {}), tour=n_tours)
            stopped, reply, defaut = "fin_anormale", "", turn.defaut
            break

        # ⚠️ La borne se vérifie APRÈS le tour, jamais avant : on ne connaît le
        # coût d'un tour qu'une fois qu'il a eu lieu. Elle empêche donc le tour
        # SUIVANT — c'est le plus tôt qu'on puisse s'arrêter, et ça borne la
        # dérive à un tour de dépassement au lieu d'un déroulé entier.
        #
        # ⚠️ Ce qui compte dans le total, c'est ce qui est FACTURÉ : l'entrée
        # non cachée, la sortie, et l'écriture de cache. Les jetons LUS en cache
        # coûtent une fraction et gonfleraient le compteur d'un facteur trois sur
        # un déroulé bien caché — une borne qui les compterait couperait des
        # passages économes en croyant les protéger.
        #
        # ⚠️ Une borne DEMANDÉE ne se suit que sur des tours mesurés. Un tour qui ne
        # déclare pas son entrée ou sa sortie l'aveugle : on s'arrête AVANT le tour
        # suivant, en le nommant, plutôt que de promettre un plafond qu'on ne mesure
        # plus (arbitrage du 13/09/2026). Des caches inconnus seuls ne l'aveuglent
        # pas. Sans borne demandée, le déroulé continue et la couverture dit le manque.
        if spec.max_tokens is not None and non_mesures:
            note("borne_non_suivie", borne="max_tokens", max_tokens=spec.max_tokens,
                 tour=n_tours, manque=non_mesures, jetons_bornes=compteur.borne)
            stopped = "max_tokens_non_mesurable"
            break
        if spec.max_tokens is not None and compteur.borne >= spec.max_tokens:
            stopped = "max_tokens"
            break

        if turn.stop_reason == "refusal":
            stopped, reply = "refusal", ""
            break

        assistant_raw = provider.assistant_message(turn)
        messages.append(assistant_raw)
        if on_turn:
            on_turn("assistant",
                    {"text": turn.text,
                     "tool_calls": [{"name": c.name} for c in turn.tool_calls]},
                    assistant_raw)

        if not turn.wants_tools:
            reply, stopped = turn.text, "end_turn"
            break

        results = []
        neutre = []
        for call in turn.tool_calls:
            started = time.monotonic()
            text, is_error, transport_ko = execute_tool(spec, transport, call)
            ms = int((time.monotonic() - started) * 1000)
            # Journalisé ENTIER, puis plafonné pour le modèle : le journal garde
            # ce que le transport a rendu, le modèle lit ce qu'il peut porter.
            pour_le_modele, tronque = _cap(text, limite_sortie)
            note("outil", id=call.id, nom=call.name, arguments=call.arguments,
                 arguments_invalides=call.arguments_invalides,
                 ok=not is_error, transport_ko=transport_ko, duree_ms=ms,
                 texte=text, tronque_pour_le_modele=tronque,
                 # Ce que le modèle a RÉELLEMENT reçu (marqueur de troncature
                 # compris) : la question « qu'a-t-il lu ? » se lit au journal.
                 servi_chars=len(pour_le_modele))
            steps.append(AgentStep(tool=call.name, ok=not is_error, duration_ms=ms,
                                   error=text[:200] if is_error else None,
                                   transport_ko=transport_ko))
            neutre.append({"name": call.name, "ok": not is_error, "duration_ms": ms})
            results.append({"id": call.id, "text": pour_le_modele,
                            "is_error": is_error})
        # La FORME des résultats dans le fil appartient au provider (un message
        # user chez Anthropic, N messages role:tool chez OpenAI) — la boucle ne
        # la connaît pas, elle appose ce qu'on lui rend.
        for tool_raw in provider.tool_messages(results):
            messages.append(tool_raw)
            if on_turn:
                on_turn("tool", {"tool_calls": neutre}, tool_raw)
        if turn.text:
            reply = turn.text

    else:
        stopped = "max_steps"

    if not reply and stopped == "end_turn":
        stopped = "no_reply"
    note("fin", stopped=stopped, reponse=reply, usage=compteur.usage(),
         couverture=compteur.couverture(), pas=len(steps), modele=servi)
    # ⚠️ L'estampille remonte du tour, pas de la configuration : c'est ce que le
    # fournisseur a SERVI. Elle était déclarée sur `AgentResult` depuis l'origine,
    # lue par le worker et comptée par le bilan — mais AUCUN transport ne la
    # posait. Trois consommateurs, zéro producteur : le champ valait `None` sur
    # 100 % des jobs, et « quelles lignes viennent de quel modèle » n'avait plus
    # de réponse (constaté au vol le 02/09, sur des passages réels).
    return AgentResult(reply=reply, steps=steps, stopped=stopped, usage=compteur.usage(),
                       couverture=compteur.couverture(),
                       messages=messages, model=servi, defaut=defaut)


def serialize(payload) -> str:
    """Payload structuré → texte pour le fil (utilitaire des transports)."""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(payload)
