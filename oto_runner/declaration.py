"""La DÉCLARATION d'une flotte — et le travail qu'elle fait enfiler.

Ce module porte ce que la flotte et le mode direct PARTAGENT : la spec (un YAML
par flotte, ou la flotte déclarée en base), et le payload du travail qu'elle
construit. Sorti de `fleet.py` le 06/09/2026 pour que le mode direct joue
EXACTEMENT le même travail que la flotte enfile — même payload, même instruction
interpolée — sans recopier une ligne : un banc qui mesurerait un autre travail
sous le même nom serait un instrument menteur de plus.

La déclaration est un YAML par flotte (cf. `docs/fleet-example.yaml`) — jamais
un secret dedans : le jeton et la clé de modèle viennent de l'environnement.
⚠️ Le worker est un pool HOMOGÈNE : son modèle vient de SON environnement
(`OTO_RUNNER_MODEL`), pas de la déclaration — un champ `model` dans le YAML
est logué puis ignoré, pour que la divergence soit VISIBLE, jamais silencieuse.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from dataclasses import dataclass, field, fields
from typing import Optional

import yaml

from .bilan import PERIODE_S as _BILAN_PERIODE_S
from . import descriptions as _descriptions_outils
from .agent_runtime import max_steps_declare

# Le même journal que l'ordonnanceur : une déclaration se lit au moment où la
# flotte se charge, et c'est là qu'on cherche son avertissement.
logger = logging.getLogger("oto_runner.fleet")

# ⚠️ Le message de lancement NOMME la file : un agent à qui on dit « la file de
# travail » sans la nommer DEVINE des noms de tableaux (vécu : entreprises,
# projet_220, data… tous inconnus, puis des SIREN hallucinés et une conclusion
# vide). Le harnais historique nommait le tableau dans sa conversation — le
# driver fait pareil, depuis la déclaration.
# ⚠️ **Il n'y a PAS d'instruction par défaut, et c'est délibéré.**
#
# Le worker est un client MCP : il exécute une instruction, il ne la compose pas.
# Il ne sait pas ce que l'instruction contient, ni ce que l'agent va faire — donc
# il ne peut pas en écrire une qui vaille.
#
# Il en existait une, en dur, sept lignes : « ta file est ce tableau, réserve
# chaque ligne, traite-les selon la procédure, puis conclus ». Deux défauts, et
# le second a coûté cher :
#
#   ① elle mettait du MÉTIER dans le worker — un tableau, des lignes, une
#     réservation — alors qu'il ne sait rien de tout ça ;
#   ② sa FORME enseignait un court-circuit. Réserve → traite → conclus est une
#     partition en trois temps où « chercher » n'apparaît nulle part, sinon caché
#     dans « selon la procédure ». Mesuré dans la nuit du 03 au 04/09 sur des
#     vagues réelles : **7 jobs sur 11 n'appelaient AUCUN outil** et écrivaient
#     quand même une fiche complète — le modèle RACONTAIT les appels au lieu de
#     les émettre, avec des dates et des dirigeants inventés, dans un compte rendu
#     parfaitement structuré. Avec une instruction qui dit d'où viennent les
#     données : 1 sur 9, puis 1 sur 20.
#
# ⚠️ Et la garde d'alors visait à côté : « n'invente jamais une ligne ni un
# identifiant » protège l'EXISTENCE d'une ligne, pas le CONTENU d'une fiche.
# Inventer un dirigeant ne violait aucune consigne.
#
# L'instruction vient donc de qui déclare le passage, et elle est dérivée de
# l'objet côté SERVEUR — là où l'on sait de quoi on parle. Ce que ce module fait
# d'elle : l'interpoler et la transmettre. Rien d'autre.

@dataclass(frozen=True)
class FleetSpec:
    procedure: str
    namespace: str
    tools: tuple
    # Le nom de la flotte : le TAG apposé à chaque job (`fleet`), par lequel on
    # retrouve les jobs d'une campagne — plus par « id ≥ N ». `load_spec` le
    # tire du nom du fichier de déclaration ; une spec construite en code le
    # DÉCLARE. Aucun repli sur le namespace : deux flottes peuvent drainer la
    # même file, et un tag deviné est un tag faux — pire qu'un tag absent.
    name: str
    # L'identifiant de la flotte DÉCLARÉE EN BASE. Absent ⟹ le driver la déclare
    # au démarrage et journalise l'identifiant obtenu ; le remettre dans la
    # déclaration fait REPRENDRE le même passage au lieu d'en ouvrir un second.
    # ⚠️ Il remplace le tag texte `payload["fleet"]` comme rattachement de
    # référence : un tag vit dans un JSON libre, un identifiant porte une clé
    # étrangère, se compte, et se refuse s'il désigne la flotte d'une autre org.
    fleet_id: Optional[int] = None
    filter: dict = field(default_factory=dict)   # ce qui est encore à traiter
    project: Optional[int] = None
    org: Optional[int] = None       # l'org de la MISSION (le namespace y vit)
    concurrency: int = 3
    ramp_seconds: int = 60
    volume: Optional[int] = None                 # None = épuisement de la file
    budget_tokens: Optional[int] = None
    max_steps: Optional[int] = None   # None = pas de borne en tours (comme une flotte hébergée)
    # L'instruction de départ, telle que le déclarant l'a écrite. OBLIGATOIRE :
    # un passage sans instruction est un défaut de ce qui l'a déclaré, pas
    # quelque chose que le worker complète de lui-même.
    input: str = ""
    # Les outils sans lesquels un job « done » est un job FAUX : leur PANNE
    # arrête la flotte (arrêt ANORMAL ⟹ relance auto quand ils reviennent).
    #
    # ⚠️ CE N'EST PAS une liste de droits. Ce que l'agent a le DROIT d'appeler se
    # gouverne en base, par org (activation et restriction de connecteur) — et
    # l'allowlist d'un run est `tools`, juste au-dessus. Faire de ce champ-ci une
    # seconde source de vérité pour « qui peut appeler quoi » créerait un doublon
    # dont l'un des deux finirait par mentir. Ici on ne dit pas ce qui est
    # PERMIS : on dit ce dont la panne rend le résultat FAUX.
    critical_tools: tuple = ()
    # La température du passage, DÉCLARÉE, jamais déduite de l'hôte : `0` est
    # la valeur qu'on pose pour rendre deux passages comparables, et elle
    # descend dans chaque travail (`payload`) comme dans la campagne déclarée.
    # Absente ⟹ l'hôte décide (`OTO_RUNNER_TEMPERATURE`), et une grille ne
    # peut plus dire d'où vient la valeur. Décision d'Alexis du 09/09/2026 :
    # « je ne veux pas poser ce paramètre en env, il doit être paramétrable ».
    temperature: Optional[float] = None
    # Le MODÈLE du passage — déclaré, et désormais SERVI : il descend dans chaque
    # travail (`payload["model"]`) et l'agent tourne dessus (oto-backend#939).
    #
    # ⚠️ Il était servi par le serveur, validé à la déclaration, et IGNORÉ ici —
    # `tests/test_champs_servis_et_lus.py` l'inscrivait en toutes lettres. Un
    # champ inerte est une dette ; un champ qui PROMET une attribution qui
    # n'arrive pas en est une plus chère, parce qu'on lit la promesse dans les
    # relevés. Absent ⟹ le modèle du worker, comme avant.
    #
    # ⚠️ Ce qu'il coûte : le cache de prompt est indexé par modèle. Deux
    # passages de modèles différents sur le même worker ne partagent plus de
    # préfixe. Le pool reste homogène par FAMILLE (le serveur ne sert un travail
    # qu'à un worker du bon dépôt) ; c'est la variation à l'intérieur d'une
    # famille qui segmente le cache.
    model: Optional[str] = None
    # La borne des descriptions d'outils servies au modèle, outil par outil, DÉCLARÉE comme la
    # température et pour la même raison : un choix de passage, pas d'hôte. Absente ⟹ les défauts de
    # `descriptions.py` (`data_write` entière, les autres à 1 024). Forme : {defaut: <entier>,
    # entieres: [<outil>, …]}, validée à la lecture de la déclaration.
    descriptions_outils: Optional[dict] = None
    # Le plafond de jetons D'UNE LIGNE, descendu dans CHAQUE travail enfilé —
    # donc appliqué par l'agent lui-même, quel que soit le chemin qui l'a mis en
    # file. Absent ⟹ aucune borne par ligne : 65 571 jetons sur une seule ligne,
    # mesurés le 01/09.
    #
    # ⚠️ Ce n'est PAS le « rendement » (jetons par écriture produite, jugé sur une
    # fenêtre glissante) que le README a décrit du 27/08 au 02/09 : ce
    # mécanisme-là a été conçu, documenté sous les noms `jetons_par_ecriture_max`
    # et `rendement_fenetre`, puis remplacé par cette borne simple — sans que la
    # doc suive. Aucun des deux noms n'a jamais existé dans le code. Qui écrivait
    # sa déclaration depuis le README repartait donc SANS borne, en croyant en
    # avoir une.
    max_tokens_per_row: Optional[int] = None
    # Combien d'échecs d'affilée arrêtent le passage. Absent ⟹ le défaut du
    # runner (`_MAX_FAILED_CONSECUTIFS`).
    #
    # ⚠️ Le serveur porte ce champ depuis l'origine, le VALIDE à la déclaration
    # (il doit valoir au moins 1) — et le runner l'IGNORAIT, appliquant sa
    # constante quoi qu'on déclare. **Une borne déclarée mais pas appliquée ne se
    # découvre que le jour où on comptait dessus** : elle ne fausse pas un relevé,
    # elle laisse tourner une campagne qu'on croyait bornée. Et la validation
    # côté serveur achevait de convaincre qu'elle était prise en compte.
    #
    # Mesuré le 03/09 avant de corriger : les 14 campagnes déclarées la laissaient
    # à `null`. **Personne ne s'était cru protégé** — c'est ce qui distingue ce
    # cas d'un incident.
    max_consecutive_failures: Optional[int] = None
    bilan_periode_s: int = _BILAN_PERIODE_S   # cadence du bilan intermédiaire
    source: str = ""              # la déclaration : le bilan JSON se pose à côté

    def __post_init__(self):
        if not (self.input or "").strip():
            # ⚠️ Le refus dit ce qui manque, PAS ce qu'il faudrait écrire : le
            # worker ne sait pas ce qu'une instruction doit contenir. Lister ici
            # « dis d'où viennent les données, nomme les outils comme source… »
            # remettrait du métier dans le transport — et ce métier-là est celui
            # d'UNE famille de passages (enrichir des fiches), pas de tous.
            raise ValueError(
                "instruction de départ absente : un passage ne démarre pas sans "
                "elle. Le worker exécute une instruction, il n'en compose pas.")
        if not self.name:
            raise ValueError(
                "nom de flotte vide : c'est le tag `fleet` de chaque job, ce "
                "par quoi on retrouve une campagne. Il vient du nom du fichier "
                "de déclaration (`campagne.yaml` ⟹ `campagne`) ou se déclare "
                "explicitement — il ne se devine pas.")


# Ce qu'une déclaration ne peut PAS porter : `name` vient du nom du fichier,
# `source` de son chemin, `fleet_id` est attribué par la base.
_NON_DECLARABLES = frozenset({"name", "source", "fleet_id"})

# ⚠️ DÉRIVÉ du dataclass, jamais réécrit à la main. La liste manuelle avait pris
# deux champs de retard — `critical_tools` et `max_tokens_per_row` — et
# l'avertissement criait donc sur des réglages qui MARCHENT. Un opérateur a failli
# retirer la ligne qui bornait sa dépense parce que le runner lui disait qu'elle
# était ignorée (02/09). **Un avertissement faux est pire que pas d'avertissement :
# il pousse au geste inverse du bon.** Un champ ajouté demain à `FleetSpec` est
# reconnu ici sans que personne y pense.
_CHAMPS = frozenset(f.name for f in fields(FleetSpec)) - _NON_DECLARABLES


# L'outil qui LIT une procédure. Depuis le 05/09 (dc78c10c) le worker n'injecte
# plus la procédure dans le prompt : si la déclaration en nomme une, c'est l'agent
# qui la lit, avec cet outil — et il ne peut appeler que ce que `tools` autorise.
OUTIL_PROCEDURE = "oto_procedure"


def verifier_outils(spec: FleetSpec) -> None:
    """La déclaration autorise-t-elle les outils qu'elle DEMANDE d'appeler ? Sinon
    refus franc, qui nomme le défaut — jamais une correction silencieuse.

    ⚠️ Trouvé le 06/09/2026 dans le journal d'un travail (événement #5 : « Outil
    `oto_procedure` indisponible pour ce run ») : l'instruction disait « Lis d'abord
    la procédure … avec `oto_procedure` », et `oto_procedure` n'était pas dans
    `tools`. L'allowlist est fail-closed, le worker n'injecte rien de métier :
    **l'agent n'a JAMAIS lu la consigne, dans aucune flotte de la campagne** — et
    chaque travail concluait « done ». Une déclaration qui demande de lire ce
    qu'elle n'autorise pas à lire ne part pas."""
    defauts = []
    if spec.procedure and OUTIL_PROCEDURE not in spec.tools:
        defauts.append(f"elle nomme une `procedure` ({spec.procedure!r}) et n'autorise pas "
                       f"`{OUTIL_PROCEDURE}`, l'outil qui la lit")
    if OUTIL_PROCEDURE in (spec.input or "") and OUTIL_PROCEDURE not in spec.tools:
        defauts.append(f"son instruction nomme `{OUTIL_PROCEDURE}` et `tools` ne l'autorise pas")
    if defauts:
        raise ValueError(
            "déclaration refusée : la déclaration demande de lire une procédure et "
            "n'autorise pas l'outil qui la lit — " + " ; ".join(defauts) + ". L'agent ne "
            "peut appeler que ce que `tools` autorise : il n'aurait jamais lu la consigne "
            f"(vécu du 04 au 06/09/2026). Ajouter `{OUTIL_PROCEDURE}` à `tools`.")


def _reglage_declare(brut) -> Optional[dict]:
    """Le réglage des descriptions tel que le passage le déclare, validé ; None s'il se tait."""
    if brut is None:
        return None
    r = _descriptions_outils.reglage(brut)
    return {"defaut": r["defaut"], "entieres": list(r["entieres"])}


def load_spec(path: str) -> FleetSpec:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    inconnus = sorted(set(raw) - _CHAMPS)
    if inconnus:
        # ⚠️ Dire ce qui EST reconnu à côté de ce qui ne l'est pas : sans le
        # voisinage, une faute de frappe (`max_token_per_row`) se lit comme une
        # fonctionnalité absente, et on cherche dans le code plutôt que dans le
        # fichier.
        logger.warning("déclaration : champs inconnus, ignorés : %s — les champs "
                       "reconnus sont : %s", ", ".join(inconnus),
                       ", ".join(sorted(_CHAMPS)))
    volume = raw.get("volume")
    if not isinstance(volume, int):
        volume = None                            # « épuisement » ou absent
    spec = FleetSpec(
        procedure=raw["procedure"],
        namespace=raw["namespace"],
        tools=tuple(raw.get("tools") or ()),
        filter=dict(raw.get("filter") or {}),
        project=raw.get("project"),
        org=raw.get("org"),
        concurrency=int(raw.get("concurrency") or 3),
        ramp_seconds=int(raw.get("ramp_seconds") or 60),
        volume=volume,
        budget_tokens=raw.get("budget_tokens"),
        max_steps=max_steps_declare(raw.get("max_steps")),
        max_tokens_per_row=raw.get("max_tokens_per_row"),
        input=raw.get("input") or "",
        critical_tools=tuple(raw.get("critical_tools") or ()),
        temperature=(float(raw["temperature"]) if raw.get("temperature") is not None else None),
        model=(str(raw["model"]).strip() or None) if raw.get("model") else None,
        descriptions_outils=_reglage_declare(raw.get("descriptions_outils")),
        bilan_periode_s=int(raw.get("bilan_periode_s") or _BILAN_PERIODE_S),
        source=path,
        name=os.path.splitext(os.path.basename(path))[0])
    verifier_outils(spec)
    return spec


def spec_depuis_flotte(f: dict) -> FleetSpec:
    """Une spec construite depuis la flotte DÉCLARÉE en base.

    C'est le pendant de `load_spec` : la même chose, lue là où le dashboard et
    les agents la lisent aussi. **Un passage piloté par sa configuration en base
    est le même objet pour tout le monde** — piloté par un fichier posé à côté de
    l'exécutable, il n'existe que pour qui a accès à la machine.

    ⚠️ Ce qui n'a PAS d'équivalent en base reste au défaut du runner :
    `ramp_seconds`, `critical_tools` et la cadence du bilan sont des réglages
    d'EXÉCUTION locale, pas de la configuration déclarée du passage. Les inventer
    en base pour « tout avoir au même endroit » mélangerait ce qu'un opérateur
    déclare et ce qu'une machine règle.

    **Le critère qui tient la frontière dans le temps : si ce réglage change,
    quelqu'un doit-il le savoir ?** La cadence d'un bilan, non. La montée en
    charge, non plus — *à condition que la borne de DÉPENSE soit déclarée*, sinon
    une machine mal réglée dépasserait sans que la configuration ait bougé. Elle
    l'est (`max_rows`, `max_tokens`, `max_tokens_per_row` vivent dans la flotte).

    ⚠️ Et `critical_tools` reste local parce qu'il désigne *ce dont la panne rend
    un résultat FAUX*, pas *ce que l'agent a le droit d'appeler* — cette
    seconde question a déjà son domicile en base (activation de connecteur par
    org), et deux domiciles pour une même règle finissent par diverger.
    """
    manquants = [c for c in ("id", "procedure") if not f.get(c)]
    if manquants:
        raise ValueError(
            f"flotte illisible — champs absents : {', '.join(manquants)}. "
            "Une flotte se déclare avant d'être pilotée.")
    spec = FleetSpec(
        procedure=f["procedure"],
        namespace=f.get("namespace") or "",
        tools=tuple(f.get("tools") or ()),
        filter=dict(f.get("row_filter") or {}),
        project=f.get("project_id"),
        org=f.get("org_id"),
        concurrency=int(f.get("workers") or 3),
        volume=f.get("max_rows"),
        budget_tokens=f.get("max_tokens"),
        max_steps=max_steps_declare(f.get("max_steps")),
        max_tokens_per_row=f.get("max_tokens_per_row"),
        max_consecutive_failures=f.get("max_consecutive_failures"),
        temperature=(float(f["temperature"]) if f.get("temperature") is not None else None),
        model=(str(f["model"]).strip() or None) if f.get("model") else None,
        input=f.get("input") or "",
        # La flotte EXISTE déjà : on la reprend, on n'en déclare pas une seconde.
        fleet_id=int(f["id"]),
        source=f"flotte #{f['id']}",
        name=f.get("label") or f"flotte-{f['id']}")
    verifier_outils(spec)
    return spec


def payload(spec: FleetSpec) -> dict:
    # Interpolation PRUDENTE (replace, jamais .format : un input custom peut
    # porter des accolades qui ne sont pas des placeholders).
    message = (spec.input
               .replace("{namespace}", spec.namespace)
               .replace("{filter}", json.dumps(spec.filter, ensure_ascii=False))
               # La date du jour, lue quand le travail se construit : l'agent n'en reçoit
               # aucune autre, et il recopiait celle des exemples de sa procédure.
               .replace("{date_du_jour}", date.today().strftime("%Y-%m-%d")))
    return {"procedure": spec.procedure, "tools": list(spec.tools),
            "project_id": spec.project, "org_id": spec.org,
            "namespace": spec.namespace,
            "fleet": spec.name,
            "max_steps": spec.max_steps,
            # ⚠️ La borne DESCEND avec le travail. Elle vivait sur la flotte, où
            # seul un ordonnanceur savait la lire — donc personne dès qu'un
            # passage tourne sans lui. Portée par le travail, elle s'applique
            # quel que soit le chemin qui l'a enfilé.
            "max_tokens": spec.max_tokens_per_row,
            # `is not None` : `temperature: 0` est une valeur, pas une absence.
            "temperature": spec.temperature,
            # Le modèle DESCEND avec le travail, comme la borne et la
            # température : c'est ce qui le rend réel. Seulement quand il est
            # déclaré — un travail sans lui reste identique, octet pour octet.
            # ⚠️ Pas de `model_family` ici : elle se déduit du catalogue, qui
            # vit côté serveur. Le worker n'en a pas besoin pour lui-même (il
            # sert ce qu'il réserve), et un runner qui la devinerait ferait
            # diverger deux définitions de la même chose.
            **({"model": spec.model} if spec.model else {}),
            # Seulement quand le passage le déclare : sans réglage, la session sert les
            # défauts de `descriptions.py`.
            **({"descriptions_outils": spec.descriptions_outils}
               if spec.descriptions_outils else {}),
            "input": message,
            "label": f"flotte {spec.namespace} — {spec.procedure}"}
