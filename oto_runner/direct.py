"""Le mode DIRECT : des boucles agentiques sur CE poste, sans file de travaux serveur.

« Soit prise par DB, soit direct. » (Alexis, 06/09/2026.) La flotte enfile des
travaux dans la file serveur, que des workers réservent ; le mode direct joue les
MÊMES travaux ici, tout de suite, sans file :

    python -m oto_runner.direct <flotte.yaml> [--lignes N] [--concurrence K]

N travaux successifs (défaut : le `volume` de la déclaration ; sans volume, jusqu'à
la file vide), K agents à la fois (défaut 1). Chaque travail = une boucle
agentique complète — la même instruction de départ que la flotte, les mêmes
outils, le modèle de CE processus (`OTO_RUNNER_MODEL`) — et son journal JSONL
complet sous `passages/<flotte>/direct-<horodatage>-<n>.jsonl`. Le mode s'arrête
quand la file du tableau est vide (plus aucune ligne ne correspond au filtre ;
l'agent qui reçoit `row: null` conclut de lui-même) ou à N.

⚠️ UN SEUL corps d'exécution. Le travail passe par `worker._un_travail` —
`_traiter`, tel qu'il tourne en production — et les trois verbes vers la file
(`bind_run`, `extend`, `complete`, cf. `file_de_travail`) sont les SEULS points
de variation : servis par la file serveur en flotte, inertes ici (`SansFile`).
Rien d'autre ne diffère — ni le travail construit (`declaration.payload`, le
même que la flotte enfile), ni l'instruction, ni la boucle, ni le journal. Le
mode direct n'appelle JAMAIS `/api/me/runner/jobs`. Un banc qui mesurerait un
autre chemin sous le même nom serait un instrument menteur de plus.

Ce que ce mode NE mesure PAS : la tenue du protocole de réservation des jobs
(aucun job n'existe) ; la concurrence entre agents telle qu'une batterie de
workers la vit (un poste, pas une batterie) ; les délégations de jeton (tout
tourne sous le `OTO_TOKEN` du poste, qui tient lieu de jeton délégué).

⚠️ K > 1 = des PROCESSUS, pas des threads. La deadline murale des requêtes
(`deadline.py`) repose sur SIGALRM, qui n'existe que dans le thread principal :
un thread secondaire n'aurait plus aucune borne de durée sur ses requêtes — ou
lèverait `ValueError` au premier appel. Chaque processus est son propre thread
principal ; un compteur partagé distribue les numéros de travail.

Le bilan de fin a la FORME du bilan de flotte (`bilan.ecrire_bilan`) : lignes
relues au tableau (`par_statut`, `abouties`), refus d'écriture entiers depuis le
journal des appels, jetons — et le modèle RÉELLEMENT servi par le fournisseur
(le champ `model` de ses réponses), à côté du nom demandé. Il se pose en
`<flotte>.direct-<horodatage>.bilan.json`, à côté de la déclaration.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import multiprocessing
import os
import queue
import time
from datetime import datetime, timezone
from typing import Optional

from . import journal, worker
from .backend import Backend
from .bilan import ecrire_bilan
from .declaration import FleetSpec, load_spec, payload, spec_depuis_flotte
from .file_de_travail import SansFile
from .llm_select import get_provider

logger = logging.getLogger("oto_runner.direct")


def horodatage() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def identifiant(stamp: str, n: int) -> str:
    return f"direct-{stamp}-{n}"


def travail(spec: FleetSpec, ident: str, jeton: str) -> dict:
    """Le travail, construit comme la flotte le fait ENFILER : même payload, kind
    `start`. Le jeton du poste tient lieu de jeton délégué — c'est la seule chose
    que la file aurait ajoutée, et elle est nommée ici."""
    return {"id": ident, "kind": "start", "payload": payload(spec),
            "delegated_token": jeton}


class Compteur:
    """Les numéros de travail, distribués un par un — partagé entre processus."""

    def __init__(self, plafond: Optional[int], ctx=multiprocessing):
        self.plafond = plafond
        self._n = ctx.Value("i", 0)
        self._verrou = ctx.Lock()

    def suivant(self) -> Optional[int]:
        with self._verrou:
            if self.plafond is not None and self._n.value >= self.plafond:
                return None
            self._n.value += 1
            return self._n.value


def boucle(spec: FleetSpec, backend, provider, jeton: str, stamp: str,
           compteur: Compteur, file: SansFile) -> str:
    """UN agent : tant qu'il reste un numéro et une ligne dans le tableau, un
    travail — par `worker._un_travail`, le MÊME chemin que le worker. Rend le
    motif d'arrêt de cet agent."""
    while (n := compteur.suivant()) is not None:
        restantes = backend.count_rows(spec.namespace, filter=spec.filter, org=spec.org)
        if restantes == 0:
            logger.info("file du tableau vide — aucun travail de plus n'est lancé")
            return "file vide"
        job = travail(spec, identifiant(stamp, n), jeton)
        logger.info("travail %s lancé — %d ligne(s) encore à traiter", job["id"], restantes)
        worker._un_travail(backend, job, provider, file=file)
    return f"volume atteint ({compteur.plafond} travaux)"


def lancer(spec: FleetSpec, backend, provider, jeton: str, stamp: str,
           plafond: Optional[int], k: int) -> tuple[dict, str, list]:
    """K agents jusqu'à N travaux ou file vide. Rend (conclus, motif, morts) —
    `conclus` a la forme que l'ordonnanceur de flotte lit (`status`, `result`,
    `run_id`), `morts` les agents sortis en erreur (nom, code)."""
    ctx = multiprocessing.get_context("fork")   # cf. l'en-tête : SIGALRM
    compteur = Compteur(plafond, ctx)
    if k == 1:
        file = SansFile()
        motif = boucle(spec, backend, provider, jeton, stamp, compteur, file)
        return file.conclus, motif, []

    retours = ctx.Queue()

    def agent() -> None:
        file = SansFile()
        motif = boucle(spec, backend, provider, jeton, stamp, compteur, file)
        retours.put((file.conclus, motif))

    agents = [ctx.Process(target=agent, name=f"agent-{i + 1}") for i in range(k)]
    for a in agents:
        a.start()
    conclus: dict = {}
    motifs: list = []
    # ⚠️ Vider la file de retour PENDANT que les agents vivent : un processus qui
    # a posé un gros résultat attend qu'il soit lu avant de sortir — le joindre
    # d'abord bloquerait les deux côtés.
    while any(a.is_alive() for a in agents) or not retours.empty():
        try:
            c, m = retours.get(timeout=0.5)
            conclus.update(c)
            motifs.append(m)
        except queue.Empty:
            pass
    for a in agents:
        a.join()
    morts = [(a.name, a.exitcode) for a in agents if a.exitcode != 0]
    motif = "file vide" if "file vide" in motifs else (
        motifs[0] if motifs else f"volume atteint ({plafond} travaux)")
    return conclus, motif, morts


def _relire_journaux(spec: FleetSpec, conclus: dict) -> None:
    """Le journal de chaque travail, RELU avant d'être nommé au bilan — la même
    règle que l'ordonnanceur (`journal.relire`) : absent, il vaut null et se dit."""
    for jid, c in conclus.items():
        chemin = journal.chemin(spec.name, jid)
        try:
            logger.info("travail %s %s — journal complet : %s", jid, c["status"],
                        journal.relu(chemin))
            c["journal"] = chemin
        except journal.JournalIllisible as e:
            logger.error("travail %s %s — SANS JOURNAL RELU (%s)", jid, c["status"], e)
            c["journal"] = None


def _modeles(conclus: dict, demande: str) -> str:
    servis = sorted({str((c.get("result") or {}).get("model")) for c in conclus.values()
                     if (c.get("result") or {}).get("model")})
    if not servis:
        return f"modèle demandé {demande} · servi : non rapporté"
    return (f"modèle demandé {demande} · servi {', '.join(servis)}"
            + (" (identique)" if servis == [demande] else " ⚠️ DIFFÉRENT du nom demandé"))


def jouer(spec: FleetSpec, backend, provider, jeton: str, plafond: Optional[int],
          k: int = 1, stamp: Optional[str] = None) -> dict:
    """Le passage direct entier : les travaux, puis le bilan — quelle que soit la
    sortie, comme la flotte. Rend le bilan ; lève après l'avoir posé si un agent
    est mort."""
    stamp = stamp or horodatage()
    demande = worker._modele_courant(provider)
    lignes_initiales = backend.count_rows(spec.namespace, filter=spec.filter, org=spec.org)
    logger.info("mode direct — %s · tableau %s · filtre %s · %d ligne(s) à traiter · "
                "%s travaux max · %d agent(s) · journaux %s",
                _modeles({}, demande).split(" · ")[0], spec.namespace, spec.filter,
                lignes_initiales, plafond if plafond is not None else "∞", k,
                journal.chemin(spec.name, identifiant(stamp, "<n>")))
    # Le bilan se pose À CÔTÉ de la déclaration, sous le nom du passage direct —
    # jamais par-dessus le bilan de la flotte du même nom.
    spec_bilan = dataclasses.replace(
        spec, source=f"{os.path.splitext(spec.source)[0]}.direct-{stamp}.yaml"
        if spec.source else "")
    t0 = time.monotonic()
    conclus, motif, morts = {}, "interrompu", []
    try:
        conclus, motif, morts = lancer(spec, backend, provider, jeton, stamp, plafond, k)
        if morts:
            motif = f"interrompu : agent(s) mort(s) {morts}"
        return_bilan = None
    finally:
        _relire_journaux(spec, conclus)
        return_bilan = ecrire_bilan(spec_bilan, backend, conclus, agents=k,
                                    lignes_initiales=lignes_initiales,
                                    secondes=time.monotonic() - t0, arret=motif)
        logger.info("mode direct terminé : %s — %d travaux conclus · %s",
                    motif, len(conclus), _modeles(conclus, demande))
    if morts:
        raise RuntimeError(f"mode direct : agent(s) mort(s) {morts} — le bilan est posé, "
                           "la pile de chaque agent est dans sa sortie d'erreur")
    return return_bilan


def _arguments(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m oto_runner.direct",
        description="Joue les travaux d'une flotte ICI, sans file de travaux serveur.")
    p.add_argument("flotte",
                   help="la déclaration : un fichier YAML, ou `#<id>` pour une "
                        "flotte DÉCLARÉE en base (celle que le dashboard montre)")
    p.add_argument("--lignes", type=int, default=None,
                   help="nombre de travaux (défaut : le `volume` de la déclaration ; "
                        "sans volume, jusqu'à la file vide)")
    p.add_argument("--concurrence", type=int, default=None,
                   help="agents en parallèle — des PROCESSUS (défaut : la "
                        "`concurrency` de la déclaration)")
    args = p.parse_args(argv)
    if args.lignes is not None and args.lignes < 1:
        p.error("--lignes : un entier ≥ 1")
    if args.concurrence is not None and args.concurrence < 1:
        p.error("--concurrence : un entier ≥ 1")
    return args


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _arguments(argv)
    jeton = os.environ.get("OTO_TOKEN", "").strip()
    if not jeton:
        raise SystemExit("OTO_TOKEN absent : le mode direct tourne sous le jeton du poste, "
                         "qui tient lieu de jeton délégué")
    provider = get_provider()
    provider.resolve_key()        # échoue FORT avant le premier travail
    journal.preparer()            # le journal est le contrat : vérifié avant
    # Le backend sert le FIL du run et les lectures du tableau — jamais la file
    # de jobs : les trois verbes vont à `SansFile`.
    backend = Backend()
    # La MÊME source que l'ordonnanceur : un YAML, ou `#<id>` pour une flotte
    # déclarée en base. Les deux modes lisaient la déclaration différemment —
    # `fleet` acceptait les deux, `direct` exigeait un fichier. Une flotte créée
    # depuis le dashboard n'était donc jouable QUE par la file de travaux, alors
    # que c'est le mode direct qui sert aujourd'hui. Le fichier reste un moyen
    # de déclarer, jamais la seule façon d'exister.
    if args.flotte.startswith("#"):
        spec = spec_depuis_flotte(backend.lire_flotte(int(args.flotte[1:])))
        logger.info("flotte #%s chargée depuis la base : %s", args.flotte[1:], spec.name)
    else:
        spec = load_spec(args.flotte)
    plafond = args.lignes if args.lignes is not None else spec.volume
    # ⚠️ La concurrence suit la DÉCLARATION, comme le volume juste au-dessus.
    # Elle ne le faisait pas : `--concurrence` valait 1 par défaut et écrasait
    # le `concurrency` du YAML sans rien dire. Une flotte déclarée à 4 agents —
    # y compris déclarée depuis le dashboard, où le champ s'appelle `workers` —
    # tournait donc à 1. Mesuré le 07/09/2026 : les journaux de toutes les
    # vagues portaient « 1 agent(s) », 72 fois, sans exception.
    #
    # Deux options du MÊME parseur ne peuvent pas traiter la déclaration
    # différemment : `--lignes` s'y repliait déjà, `--concurrence` l'écrasait.
    agents = args.concurrence if args.concurrence is not None else spec.concurrency
    jouer(spec, backend, provider, jeton, plafond, k=agents)


if __name__ == "__main__":
    main()
