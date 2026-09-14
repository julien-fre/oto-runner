"""Le JOURNAL d'un travail — tout ce que le worker a vu, sur disque, au fil de l'eau.

Pourquoi il existe. Nuit du 05 au 06/09/2026, flotte `banc-v151-medium` : trois
lignes, « abouties 3/3 », et **7 écritures refusées sur 8**. Après coup, rien ne
permettait de dire ce que le modèle avait envoyé ni ce que le schéma lui avait
répondu : le bilan portait les motifs coupés à soixante caractères — juste avant
la colonne et la raison —, le fil du run porte des sorties d'outils déjà tronquées
pour le modèle, et la plateforme n'offre aucune lecture des jobs. *« Il faut que
le runner conserve tout. »* (Alexis, 06/09.)

Ce module conserve donc TOUT, par travail, dans un fichier JSONL : un événement
par ligne, horodaté, ÉCRIT À CHAQUE ÉVÉNEMENT (append, fichier refermé) — un
travail qui plante laisse sa trace jusqu'au plantage. Aucun texte n'y est
tronqué : la sortie d'outil journalisée est celle que le transport a rendue, pas
celle que le modèle a lue.

Emplacement : `<OTO_RUNNER_PASSAGES_DIR ou passages>/<flotte>/<job_id>.<tentative>.jsonl` — un
répertoire par flotte, nommé d'après le tag `fleet` du travail, c'est-à-dire le
nom de la déclaration (`<flotte>.yaml`), celui qui nomme déjà `<flotte>.bilan.json`
et `<flotte>.log`. Un travail sans flotte va dans `hors-flotte`. ⚠️ Le journal est
écrit LÀ OÙ LE WORKER TOURNE : quand les workers sont sur une autre machine que
l'ordonnanceur, c'est sur cette machine-là qu'il se lit.

⚠️ Ce fichier porte de la DONNÉE de la file de travail (lignes clientes, sorties
d'outils) : écrit en 0600, dans un répertoire 0700. Il ne porte JAMAIS un secret :
`debut` recopie le travail SANS son jeton délégué ni sa clé de modèle.

⚠️ Une écriture qui échoue LÈVE. Le journal est le contrat « conserver tout » : un
runner qui cesserait de conserver en silence violerait précisément ce qu'on lui
demande. Le répertoire est vérifié au démarrage du worker (`preparer`) pour qu'un
droit manquant se voie au boot, pas au premier travail.
"""
from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Optional

_ENV_DIR = "OTO_RUNNER_PASSAGES_DIR"
_DEFAUT = "passages"
_HORS_FLOTTE = "hors-flotte"
_SUR = re.compile(r"[^A-Za-z0-9._-]+")
# Les clés d'un travail qui portent un secret remis à la réservation. Tout ce qui
# finit par `_token`/`_key`/`secret` tombe aussi : un champ ajouté demain côté
# serveur sous cette forme ne doit pas finir sur disque par oubli d'ici.
# `model_workspace` (14/09/2026) n'est pas un secret, mais un satellite de la clé
# d'organisation remis avec elle : il ne va pas plus sur disque qu'elle, et son nom
# n'est pris par aucun suffixe.
_SECRETS = ("delegated_token", "model_key", "model_workspace")
_SUFFIXES_SECRETS = ("_token", "_key", "secret")


def dossier() -> str:
    """Le répertoire racine des journaux — la variable d'env, sinon `passages`
    (relatif au répertoire courant du processus, comme `<flotte>.bilan.json` est
    relatif à sa déclaration)."""
    return os.environ.get(_ENV_DIR) or _DEFAUT


def nom_de_flotte(label) -> str:
    """Le tag de flotte, rendu sûr pour un nom de répertoire — jamais un chemin."""
    propre = _SUR.sub("_", str(label or "")).strip("._")
    return propre or _HORS_FLOTTE


def chemin(label, job_id, tentative=None) -> str:
    """Où se trouve (ou se trouvera) le journal de la tentative `tentative` du
    travail `job_id` de la flotte `label`. La même fonction sert au worker, qui
    écrit, et à l'ordonnanceur, qui pointe : un seul endroit décide de la convention.

    ⚠️ Un fichier PAR TENTATIVE (oto#196). Une reprise garde le même identifiant de
    travail : nommé par le seul `job_id`, le journal de la seconde tentative
    s'ajoutait à celui de la première sur la même machine, et rien ne disait où
    l'une finissait et où l'autre commençait. `tentative` est le compteur que le
    serveur rend à la réservation (`attempts`). Sans tentative — le mode direct,
    qui ne reprend jamais un travail — le nom reste `<job_id>.jsonl`."""
    nom = f"{job_id}.jsonl" if tentative is None else f"{job_id}.{int(tentative)}.jsonl"
    return os.path.join(dossier(), nom_de_flotte(label), nom)


def preparer() -> str:
    """Crée la racine et vérifie qu'on peut y écrire ; rend son chemin absolu.
    Appelé au boot du worker : un journal impossible se dit AVANT le premier
    travail, pas en le faisant échouer."""
    racine = os.path.abspath(dossier())
    os.makedirs(racine, mode=0o700, exist_ok=True)
    if not os.access(racine, os.W_OK):
        raise PermissionError(
            f"journaux par travail : {racine} n'est pas inscriptible — "
            f"poser {_ENV_DIR} sur un répertoire où le worker peut écrire")
    return racine


class JournalIllisible(RuntimeError):
    """Le journal annoncé n'est pas là, ou pas relisible. Le message NOMME le
    chemin : c'est ce qu'on cherche quand on lit cette erreur."""


def relire(chemin: str) -> tuple[int, str]:
    """Le journal EST-il là ? Il existe, il n'est pas vide, sa dernière ligne se
    parse. Rend (nombre d'événements, type du dernier). Sinon LÈVE.

    ⚠️ Règle posée le 06/09/2026 : **l'annonce d'un journal vient APRÈS sa
    relecture sur disque.** Un ordonnanceur a écrit « journal complet :
    passages/…/12670.jsonl » pour un travail servi par un worker d'une autre
    version, sur une autre machine : aucun fichier n'existait. Un instrument qui
    dit « journal complet » sans fichier est pire que pas d'instrument — c'est
    exactement le défaut que ce journal existe pour éliminer."""
    if not os.path.isfile(chemin):
        raise JournalIllisible(f"aucun journal à {chemin}")
    with open(chemin, "rb") as f:
        lignes = [l for l in f.read().splitlines() if l.strip()]
    if not lignes:
        raise JournalIllisible(f"journal vide : {chemin}")
    try:
        dernier = json.loads(lignes[-1].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise JournalIllisible(f"dernière ligne illisible dans {chemin} : {e}") from e
    if not isinstance(dernier, dict) or "ev" not in dernier:
        raise JournalIllisible(f"dernière ligne sans événement dans {chemin}")
    return len(lignes), str(dernier["ev"])


def relu(chemin: str) -> str:
    """« <chemin> (n événements, dernier : <ev>) » — ce qu'on écrit au journal de
    flotte ou de worker APRÈS relecture. Lève comme `relire`."""
    n, dernier = relire(chemin)
    return f"{chemin} ({n} événement{'s' if n > 1 else ''}, dernier : {dernier})"


def _horodatage() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _sans_secrets(job: dict) -> dict:
    return {k: v for k, v in (job or {}).items()
            if k not in _SECRETS and not str(k).lower().endswith(_SUFFIXES_SECRETS)}


class Journal:
    """Le journal d'UN travail : un fichier JSONL, une ligne par événement.

    Chaque ligne : `{"t": <UTC ms>, "ev": <type>, ...champs}`. Les champs sont
    écrits tels quels (`ensure_ascii=False`, objets non sérialisables rendus par
    `str`) — jamais coupés."""

    def __init__(self, chemin: str):
        self.chemin = chemin
        os.makedirs(os.path.dirname(os.path.abspath(chemin)), mode=0o700,
                    exist_ok=True)
        # Créé en 0600 dès l'ouverture : la première ligne ne tombe pas dans un
        # fichier aux droits du parapluie (umask).
        os.close(os.open(chemin, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))

    def ecrire(self, ev: str, **champs) -> None:
        ligne = json.dumps({"t": _horodatage(), "ev": ev, **champs},
                           ensure_ascii=False, default=str)
        fd = os.open(self.chemin, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as f:
            f.write((ligne + "\n").encode("utf-8"))

    def evenement(self, ev: str, champs: dict) -> None:
        """La forme `on_event(ev, champs)` qu'attendent la boucle d'agent et le
        chemin Conversations."""
        self.ecrire(ev, **champs)


def du_travail(job: dict) -> Journal:
    """Le journal de la tentative réservée — nommé par sa flotte, son identifiant
    et le compteur de tentatives rendu par la réservation."""
    p = job.get("payload") or {}
    return Journal(chemin(p.get("fleet"), job.get("id"), job.get("attempts")))


def debut(journal: Journal, job: dict, provider) -> None:
    """L'événement d'ouverture : le travail tel que reçu (sans ses secrets), les
    outils AUTORISÉS, et le fournisseur qui va le servir. Le payload porte
    l'instruction de départ — le « message initial » avant toute interpolation."""
    p = job.get("payload") or {}
    journal.ecrire("debut", job=_sans_secrets(job),
                   outils_autorises=sorted(p.get("tools") or ()),
                   provider=getattr(provider, "__name__", None),
                   modele_demande=_modele(provider))


def ecart_instruction(catalogue: Optional[frozenset], autorises, instruction: str) -> dict:
    """Ce que l'instruction NOMME sans que `tools` l'autorise, et ce que `tools`
    autorise sans que le catalogue le connaisse — l'événement qui aurait tout dit
    dès le 04/09 : « avec `oto_procedure` » dans l'instruction, `oto_procedure`
    hors de la liste, et chaque travail « done » sans avoir lu la consigne.

    Le catalogue vient de la session MCP (`McpSession.catalogue`) : seul un nom
    que la plateforme SERT compte comme un outil nommé — sans lui, on ne devine
    pas (`a_enrichir`, `lot_test` ressemblent à des outils et n'en sont pas).
    Catalogue absent ⟹ `null` avec sa raison, jamais une liste vide qui
    rassurerait."""
    autorises = set(autorises or ())
    if catalogue is None:
        return {"nommes_hors_liste": None, "autorises_inconnus": None,
                "ecart_omis": "transport sans catalogue d'outils"}
    texte = instruction or ""
    nommes = sorted(t for t in catalogue if t not in autorises
                    and re.search(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", texte))
    return {"nommes_hors_liste": nommes,
            "autorises_inconnus": sorted(autorises - set(catalogue)),
            "catalogue": len(catalogue)}


def erreur(journal: Journal, e: BaseException) -> None:
    """Le plantage, avec sa pile ENTIÈRE — la CAUSE, écrite avant tout le reste.

    ⚠️ Ce n'est plus le dernier événement : un travail mort rend ensuite ce qu'il
    tient (run clos en `failed`, ligne libérée) et le dit en `resultat`, cf.
    `conclusion.py`. L'ordre est voulu — la cause d'abord, ce qu'on en a fait
    ensuite ; rien de ce qui suit ne doit masquer ce qui a tué le travail."""
    journal.ecrire("erreur", type=type(e).__name__, message=str(e),
                   traceback=traceback.format_exc())


def _modele(provider) -> Optional[str]:
    try:
        return provider.model()
    except Exception:  # noqa: BLE001 — un relevé d'observabilité ne bloque pas l'ouverture
        return None
