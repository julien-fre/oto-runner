"""La FLOTTE : un ordonnanceur MINCE au-dessus de la file de jobs.

Un client ordinaire de l'API jobs — aucun kind serveur, aucune boucle d'agent
ici : la boucle vit dans le worker, le claim de ligne vit dans la PROCÉDURE.
Le driver ne fait que trois choses : enfiler des jobs `start` avec une rampe,
maintenir la concurrence déclarée, et s'arrêter proprement sur l'une des
bornes — file vide, volume, budget de jetons, rendement effondré, ou trop
d'échecs consécutifs.

La déclaration (un YAML par flotte, cf. `docs/fleet-example.yaml`) et le travail
qu'elle fait enfiler vivent dans `declaration.py`, partagés avec le mode direct.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import journal
from .backend import Backend, BackendError
from .bilan import ecrire_bilan
# La déclaration et le travail qu'elle construit vivent dans `declaration.py`,
# partagés avec le mode direct ; réexportés ici pour qui lit la flotte.
from .declaration import (FleetSpec, load_spec, spec_depuis_flotte,  # noqa: F401
                          _CHAMPS, _NON_DECLARABLES)
from .declaration import payload as _payload

logger = logging.getLogger("oto_runner.fleet")

_POLL_S = 20
_MAX_FAILED_CONSECUTIFS = 3
_MAX_ERREURS_BACKEND = 10   # ~3-4 min de panne DENSE (reset au 1er succès)
# Faux départs EN SÉRIE (27/08) : un job « done » qui a réservé une ligne sans
# rien écrire n'est pas un succès — et depuis que le run libère la ligne à sa
# conclusion, la ligne ratée est reservie dans la minute au job suivant, qui
# refait le même faux départ : une boucle qui vide le budget sans écrire, et
# qu'aucune borne ne voyait (les jobs sont « done »). N d'affilée ⟹ arrêt
# ANORMAL (relance auto) : « ça tourne à vide » n'est pas « ça tourne ».
# ⚠️ Seuls comptent les jobs qui ont RÉELLEMENT réservé une ligne. Un job à
# CLAIM VIDE (la file n'avait plus rien à rendre) est un non-événement : en fin
# de file il y a toujours plus d'agents que de lignes, et les compter a fait
# échouer une campagne ABOUTIE le 28/08 (18/20 lignes, les 2 dernières sous bail
# ⟹ 5 jobs à un seul appel ⟹ borne mordue, `exit 1`). Il ne remet pas non plus
# le compteur à zéro : la remise à zéro rendrait la borne contournable par
# alternance (un vrai faux départ, un claim à vide, indéfiniment).
_MAX_FAUX_DEPARTS_CONSECUTIFS = 5

# ⚠️ DEUX suffisent, et le seuil est bas exprès. Un relâchement qui échoue laisse
# la ligne sous bail : le travail suivant ne la prend pas, la file paraît avancer
# alors qu'elle se vide en laissant des lignes derrière, et le passage MENT SUR SON
# DÉBIT. Le 29/08, 54 échecs consécutifs sont passés inaperçus parce que le
# relâchement journalisait et continuait — un best-effort sans destination.
# Un échec de relâchement compte donc comme un échec d'écriture.
# La borne « outil critique en échec » (27/08). Un job dont les outils
# échouent conclut quand même « done » — bornes, budget et heartbeat disaient
# « ça tourne » pendant que la flotte marquait 2 395 fiches « enrichies » sans
# une recherche web réussie (Serper à sec, 4 jours). La mesure vient du
# JOURNAL des appels (fenêtre glissante de 15 min) : à la relance après panne,
# aucun appel récent ⟹ pas de verdict, on laisse partir des jobs qui
# ré-alimentent la mesure — sans quoi les vieux échecs re-borneraient à vide.
_SANTE_FENETRE_MIN = 15
_SANTE_MIN_APPELS = 12
_SANTE_TAUX_KO = 0.9
_SANTE_PERIODE_S = 60


def _outil_critique_en_panne(spec, backend, clock) -> "str | None":
    """Le motif de borne si un outil critique échoue massivement, sinon None.
    Sondé au plus toutes les _SANTE_PERIODE_S ; toute erreur de sonde = pas de
    verdict (une sonde en panne n'arrête pas une campagne saine)."""
    if not spec.critical_tools or spec.org is None:
        return None
    now = clock()
    if now - _outil_critique_en_panne.dernier.get("t", -1e9) < _SANTE_PERIODE_S:
        return _outil_critique_en_panne.dernier.get("verdict")
    verdict = None
    for outil in spec.critical_tools:
        try:
            n, ko = backend.tool_health(spec.org, outil, minutes=_SANTE_FENETRE_MIN)
        except Exception as e:  # noqa: BLE001 — la sonde ne tue pas la flotte
            logger.warning("santé de %s illisible : %s", outil, e)
            continue
        if n >= _SANTE_MIN_APPELS and ko / n >= _SANTE_TAUX_KO:
            verdict = (f"outil critique `{outil}` en échec ({ko}/{n} appels "
                       f"sur {_SANTE_FENETRE_MIN} min) — chaque job « done » "
                       "serait faux")
            break
    _outil_critique_en_panne.dernier = {"t": now, "verdict": verdict}
    return verdict


_outil_critique_en_panne.dernier = {}


# ⚠️ Ce qui protège une campagne d'un prix payé sans sortie, ce sont DEUX bornes
# distinctes, et aucune n'est un « rendement » : les FAUX DÉPARTS EN SÉRIE
# (ci-dessus) attrapent « ça tourne à vide », et `max_tokens_per_row` borne la
# ligne elle-même. La borne de rendement — coût rapporté aux écritures sur une
# fenêtre glissante — a été conçue le 27/08 et n'a jamais été écrite ; le
# commentaire qui l'annonçait ici a survécu à sa propre annulation jusqu'au
# 02/09. **Un commentaire qui décrit un mécanisme absent le rend introuvable :
# on cherche le bug dans le code au lieu de constater qu'il n'y a pas de code.**


@dataclass
class FleetBilan:
    done: int = 0
    failed: int = 0
    usage_tokens: int = 0
    lignes_initiales: int = 0
    lignes_restantes: int = 0
    arret: str = ""
    # Combien de fois le passage n'a PAS pu dire où il en était — déclaration,
    # armement, prise, battement, accusé d'arrêt.
    #
    # ⚠️ **C'est le vrai défaut du 03/09**, pas l'armement manquant. Chacun de ces
    # gestes échouait, était journalisé en `warning`, et le passage continuait :
    # « le passage tourne quand même, mais son état ne dira pas en cours ». Le
    # journal disait la vérité à chaque battement. **Personne ne l'a lue pendant
    # huit vagues.**
    #
    # Un compte porté par le BILAN se lit là où on regarde le résultat, et il se
    # conclut : *ce passage a tourné en aveugle N fois*. Une ligne de journal de
    # plus n'aurait rien changé — c'est exactement ce qui a échoué.
    etat_muet: int = 0
    # Pourquoi chaque travail s'est arrêté, compté par motif (`end_turn`,
    # `max_steps`, `max_tokens`, `no_reply`…), tel que le worker l'a DÉCLARÉ.
    #
    # ⚠️ Un travail arrêté sur `max_steps` ou `max_tokens` compte aujourd'hui
    # comme `done` : il a rendu la main sans erreur. Mais il a été COUPÉ, pas
    # conclu — sa ligne porte ce qu'il avait fait au moment où on l'a arrêté, et
    # rien ne le distingue, dans le bilan, d'un travail qui a fini. C'est un
    # succès apparent posé sur un travail tronqué : le compte des lignes traitées
    # reste juste, ce qu'elles valent ne l'est plus.
    #
    # Le motif vient du worker, seul à voir le déroulé. L'ordonnanceur ne le
    # recalcule pas : il ne sait pas ce qu'une étape veut dire.
    arrets: dict = field(default_factory=dict)
    # Les travaux conclus dont le journal JSONL n'a PAS pu être relu d'ici.
    # ⚠️ L'ordonnanceur ne nomme un journal qu'après l'avoir RELU : le 06/09 il a
    # écrit « journal complet : passages/…/12670.jsonl » pour un fichier qui
    # n'existait nulle part — le worker tournait ailleurs, sur une autre version.
    # Un compte, et une conclusion en erreur, là où on lit le résultat.
    journaux_absents: int = 0


def run_fleet(spec: FleetSpec, backend: Backend, *,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic,
              poll_s: int = _POLL_S) -> FleetBilan:
    """La boucle d'ordonnancement. `sleep`/`clock` injectables : les bornes se
    PROUVENT en test, elles ne s'affirment pas — c'est ce qui protège le compte
    pendant une campagne sans surveillance."""
    # La flotte se DÉCLARE en base avant d'enfiler quoi que ce soit : sans
    # identifiant, chaque job part orphelin et `runner.fleets op=state` répond
    # « aucun travail rattaché » pour un passage qui tourne. ⚠️ Une déclaration
    # qui échoue n'arrête PAS le passage — le rattachement sert à LIRE, il ne
    # conditionne pas le travail. Perdre l'observabilité est un moindre mal
    # devant une campagne qui refuse de partir.
    # ⚠️ Compté AVANT que le bilan existe (il naît d'un appel réseau, plus bas) :
    # un aveuglement qui survient pendant le démarrage doit être compté aussi,
    # c'est même là qu'il est le plus probable.
    muet = 0
    # ⚠️ La borne DÉCLARÉE gagne sur le défaut du runner. Résolue ici, une fois,
    # pour qu'elle soit lisible au démarrage plutôt que devinée à la lecture du
    # code : une garde dont on ne sait pas quelle valeur elle applique n'est pas
    # une garde, c'est une intention.
    # ⚠️ L'organisation du passage, posée sur le CLIENT une fois pour toutes :
    # tous les appels qui nomment une `fleet_id` la porteront. La poser appel par
    # appel avait laissé `launch`, le battement et l'enfilage chercher la campagne
    # sous une autre organisation — créée au bon endroit, introuvable au geste
    # suivant, et 560 secondes de `404` toutes les vingt secondes sans qu'un seul
    # travail parte. Un contexte qu'il faut penser à joindre finit par être oublié
    # à l'un des appels.
    if spec.org is not None:
        backend.org = spec.org
    plafond_echecs = spec.max_consecutive_failures or _MAX_FAILED_CONSECUTIFS
    fleet_id = spec.fleet_id
    if fleet_id is None:
        try:
            f = backend.declarer_flotte(
                label=spec.name, procedure=spec.procedure, tools=list(spec.tools),
                namespace=spec.namespace, row_filter=spec.filter or None,
                project_id=spec.project, input=spec.input,
                max_steps=spec.max_steps, workers=spec.concurrency,
                max_rows=spec.volume, max_tokens=spec.budget_tokens,
                max_tokens_per_row=spec.max_tokens_per_row,
                temperature=spec.temperature,
                # L'org DÉCLARÉE, celle-là même que la session pose déjà sur les
                # appels d'outils. Sans elle, la campagne naît ailleurs que là
                # où sa déclaration la met, et ses travaux cherchent leur
                # procédure au mauvais endroit.
                org=spec.org)
            fleet_id = f.get("id")
            logger.info("flotte déclarée en base : id=%s — remettre `fleet_id: %s` "
                        "dans la déclaration pour REPRENDRE ce passage",
                        fleet_id, fleet_id)
        except BackendError as e:
            # ⚠️ REFUS, pas repli. Ce chemin dégradait : le passage partait
            # « sans rattachement », ses travaux tournaient, dépensaient, et
            # restaient hors de portée d'`op=stop` — personne ne pouvait les
            # arrêter avant qu'ils finissent d'eux-mêmes. Le 09/09/2026, cinq
            # d'entre eux ont abouti et la ligne de journal n'a été lue qu'au
            # bilan.
            #
            # La règle de la maison est explicite et je ne l'appliquais pas :
            # « pas de fallback qui masque un problème — lever une erreur ».
            # Un passage qu'on ne peut pas arrêter est précisément le problème
            # que le repli masquait, sous couvert de ne rien bloquer.
            #
            # L'erreur du SERVEUR passe en clair : c'est elle qui dit quoi
            # corriger, et la reformuler ferait perdre le seul texte utile.
            raise RuntimeError(
                "flotte non déclarée, passage ABANDONNÉ avant d'enfiler quoi que "
                f"ce soit — {e}\n"
                "Sans déclaration, les travaux seraient invisibles à `op=state` "
                "et IMPOSSIBLES à arrêter avec `op=stop`. Corrige la déclaration "
                "(ou le serveur) et relance : rien n'a été lancé."
            ) from e
    # ⚠️ PRENDRE la flotte : `armed` → `running`. C'est l'ordonnanceur qui pose ce
    # FAIT, jamais l'opérateur — `armed` veut dire « on a demandé », `running`
    # veut dire « quelqu'un l'a prise et donne signe ». Un refus n'est pas une
    # erreur à retenter : un autre ordonnanceur l'a prise, ou elle n'était pas
    # armée. Partir quand même DOUBLERAIT le passage.
    if fleet_id is not None:
        # ⚠️ ARMER AVANT DE PRENDRE. Une flotte naît `draft` ; `prendre` n'accepte
        # qu'`armed`. Sans ce geste le refus est systématique — et il l'a été sur
        # les 14 campagnes déclarées jusqu'au 03/09, aucune n'ayant jamais atteint
        # `running` alors que huit vagues tournaient.
        try:
            backend.armer_flotte(fleet_id)
        except BackendError as e:
            muet += 1
            logger.warning("flotte #%s non armée (%s) — la prise va donc échouer, "
                           "et l'état restera muet sur ce passage", fleet_id, e)
        try:
            backend.prendre_flotte(fleet_id)
            logger.info("flotte #%s prise — le passage est en cours", fleet_id)
        except BackendError as e:
            muet += 1
            logger.warning("flotte #%s non prise (%s) — le passage tourne quand "
                           "même, mais son état ne dira pas « en cours »",
                           fleet_id, e)
    bilan = FleetBilan(lignes_initiales=backend.count_rows(spec.namespace,
                                                           filter=spec.filter,
                                                           org=spec.org))
    bilan.etat_muet = muet
    en_vol: set[int] = set()
    dernier_depart: Optional[float] = None
    failed_consecutifs = 0
    # L'ordre d'arrêt, LU sur la plateforme — jamais un état purement local :
    # c'est un opérateur qui le pose, depuis un écran ou une conversation.
    arret_demande = False
    # ⚠️ Initialise ICI : un compteur qui n'existe que sur un chemin et qu'on
    # lit sur tous leve au premier passage qui l'emprunte.
    pleine_charge_atteinte = False
    departs = 0
    # (jetons, écritures) des derniers jobs conclus — la matière du rendement.
    # Les erreurs BACKEND consécutives du driver lui-même (count/enqueue) : un
    # 502 isolé sur l'enfilement a TUÉ un vol entier (lot C, 16/08 — la rafale
    # #352 ; 30 min de flotte figée avant détection humaine). Le driver tolère
    # et retente ; seule une panne DENSE et continue l'arrête, PROPREMENT, avec
    # un bilan — jamais un traceback. Les workers en vol, eux, continuent.
    erreurs_backend = 0
    # La matière du BILAN : les jobs conclus (statut + résultat déclaré, jamais
    # leur payload). « Aboutie » se mesure au TABLEAU (départ − restantes), pas
    # au nombre de jobs « done » — le relevé de départ est déjà en `bilan`.
    conclus: dict[int, dict] = {}
    t0 = dernier_bilan = clock()
    logger.info("flotte %s : %d ligne(s) à traiter, concurrence %d, rampe %ds",
                spec.namespace, bilan.lignes_initiales, spec.concurrency,
                spec.ramp_seconds)

    try:
        while True:
            # 0. « Dois-je m'arrêter ? » — la question qui rend `op=stop` RÉEL.
            #
            # ⚠️ Sans cette lecture, l'arrêt demandé resterait une écriture que
            # personne ne lit : l'écran annoncerait `stopping` pour toujours, et
            # le passage continuerait de réserver, d'appeler, de DÉPENSER.
            # C'est exactement pour ça que l'état s'appelle `stopping` et non
            # `stopped` — la plateforme ne ment pas sur ce qu'elle a fait.
            #
            # ⚠️ Et l'arrêt est GRACIEUX, comme celui des agents : on cesse
            # d'enfiler, on laisse finir ce qui est en vol, PUIS on accuse.
            # Couper au milieu laisserait des lignes sous bail et ferait repayer
            # les jobs — le remède serait pire que le mal.
            if fleet_id is not None and not arret_demande:
                try:
                    if backend.battre_flotte(fleet_id):
                        arret_demande = True
                        bilan.arret = "arrêt demandé"
                        logger.info("arrêt DEMANDÉ pour la flotte #%s — plus aucun "
                                    "job enfilé, les %d en vol vont à leur terme",
                                    fleet_id, len(en_vol))
                except BackendError as e:
                    # Une plateforme injoignable n'est pas un ordre d'arrêt : la
                    # confondre éteindrait la flotte à chaque bascule.
                    bilan.etat_muet += 1
                    logger.warning("battement flotte #%s : %s", fleet_id, e)

            # 1. Moissonner les jobs conclus — leur résultat DÉCLARÉ porte le coût.
            for jid in sorted(en_vol):
                try:
                    job = backend.get_job(jid)
                except BackendError as e:
                    logger.warning("get_job %s : %s", jid, e)
                    continue
                st = job.get("status")
                if st not in ("done", "failed"):
                    continue
                en_vol.discard(jid)
                resultat = job.get("result") or {}
                # Le journal JSONL du travail — RELU avant d'être nommé, jamais
                # supposé (cf. `journal.relire`). Absent d'ici : compté, dit en
                # erreur, et `null` pour le bilan.
                chemin = journal.chemin(spec.name, jid, job.get("attempts"))
                niveau = logging.INFO if st == "done" else logging.WARNING
                try:
                    trace = f"journal complet : {journal.relu(chemin)}"
                except journal.JournalIllisible as e:
                    bilan.journaux_absents += 1
                    chemin, niveau = None, logging.ERROR
                    trace = (f"SANS JOURNAL RELU ({e}) — le worker qui a servi ce "
                             "travail tourne ailleurs, ou sans journal")
                # Le `run_id` relie le travail au journal des appels de l'org
                # (chaque refus d'écriture porte le sien) : c'est par lui que le
                # bilan nomme le job — et son journal JSONL — derrière un refus.
                conclus[jid] = {"status": st, "result": resultat,
                                "run_id": job.get("run_id"), "journal": chemin}
                jetons_du_job = int(resultat.get("usage_tokens") or 0)
                bilan.usage_tokens += jetons_du_job
                motif = resultat.get("stopped") or "inconnu"
                bilan.arrets[motif] = bilan.arrets.get(motif, 0) + 1
                # Le journal de flotte reste compact : une ligne par travail
                # conclu, qui POINTE vers le journal complet relu.
                logger.log(niveau, "job %s %s (%s · %d jetons)%s — %s",
                           jid, "conclu" if st == "done" else "FAILED", motif,
                           jetons_du_job,
                           "" if st == "done" else f" : {job.get('last_error')}",
                           trace)
                if st == "done":
                    bilan.done += 1
                    failed_consecutifs = 0
                    # Le claim à vide et le faux départ sont DÉCLARÉS par le
                    # worker — le seul à avoir vu les appels ET leurs sorties.
                    # Un résultat qui ne les porte pas vient d'un worker trop
                    # ancien : on lève, on ne redevine pas en silence.
                    # ⚠️ L'ordonnanceur ne juge plus ce que l'agent a produit.
                    # Il portait ici : le claim à vide, le faux départ, le
                    # relâchement de ligne, le rendement en écritures par jeton.
                    # Tout cela suppose de savoir ce qu'écrire veut dire — ce
                    # n'est pas le sujet d'un exécuteur d'agents.
                    #
                    # Restent les bornes qui ne regardent que l'exécution :
                    # volume, budget de jetons, file vide, échecs consécutifs.
                else:
                    bilan.failed += 1
                    failed_consecutifs += 1

            try:
                restantes = backend.count_rows(spec.namespace, filter=spec.filter,
                                               org=spec.org)
            except BackendError as e:
                erreurs_backend += 1
                if erreurs_backend >= _MAX_ERREURS_BACKEND:
                    bilan.arret = (f"backend indisponible ({erreurs_backend} erreurs "
                                   "consécutives du driver)")
                    logger.warning("flotte arrêtée : %s — %d done, %d failed",
                                   bilan.arret, bilan.done, bilan.failed)
                    return bilan
                logger.warning("count_rows toléré (%d/%d) : %s",
                               erreurs_backend, _MAX_ERREURS_BACKEND, e)
                sleep(poll_s)
                continue
            erreurs_backend = 0
            bilan.lignes_restantes = restantes
            traitees = max(0, bilan.lignes_initiales - restantes)

            # Le bilan INTERMÉDIAIRE : une campagne se pilote PENDANT qu'elle
            # tourne, pas après. Tout le calcul vit dans bilan.py.
            if clock() - dernier_bilan >= spec.bilan_periode_s:
                dernier_bilan = clock()
                ecrire_bilan(spec, backend, conclus, agents=spec.concurrency, secondes=clock() - t0,
                             lignes_initiales=bilan.lignes_initiales)

            # 2. Les bornes — chacune arrête l'ENFILEMENT ; les jobs en vol finissent.
            # ⚠️ L'ordre d'arrêt EST une borne, et il passe par le même chemin
            # qu'elles : cesser d'enfiler, laisser finir ce qui est en vol,
            # conclure. Lui inventer une sortie à part ferait deux façons de
            # s'arrêter, dont une seule serait éprouvée.
            borne = "arrêt demandé" if arret_demande else None
            if borne is None and spec.budget_tokens is not None and bilan.usage_tokens >= spec.budget_tokens:
                borne = f"budget atteint ({bilan.usage_tokens} ≥ {spec.budget_tokens} jetons)"
            elif borne is None and spec.volume is not None and traitees >= spec.volume:
                borne = f"volume atteint ({traitees} ≥ {spec.volume} lignes)"
            elif borne is None and failed_consecutifs >= plafond_echecs:
                borne = (f"{failed_consecutifs} échecs consécutifs — enfiler encore, "
                         "c'est payer pour re-crasher")
            elif (panne := _outil_critique_en_panne(spec, backend, clock)):
                borne = panne
            # ⚠️ Trois bornes ont été retirées ici — relâchements ratés, faux
            # départs, rendement effondré. Les trois jugeaient ce que l'agent
            # avait PRODUIT : avoir réservé sans écrire, ne pas avoir rendu sa
            # ligne, écrire trop peu pour ce qu'il coûte. Un exécuteur d'agents
            # ne sait pas ce qu'écrire veut dire, et ne rien écrire est parfois
            # la bonne réponse.
            elif restantes == 0:
                borne = "file vide"

            if borne:
                if en_vol:
                    logger.info("borne (%s) : %d job(s) en vol finissent, plus un "
                                "n'est enfilé", borne, len(en_vol))
                    sleep(poll_s)
                    continue
                bilan.arret = borne
                logger.info("flotte arrêtée : %s — %d done, %d failed, %d jetons",
                            borne, bilan.done, bilan.failed, bilan.usage_tokens)
                # ⚠️ ACCUSER l'arrêt — le seul geste qui pose le FAIT `stopped`.
                # Sans lui, un arrêt demandé resterait `stopping` pour toujours,
                # ce qui est précisément le symptôme d'un ordonnanceur MORT : on
                # le fabriquerait en étant vivant, et le diagnostic ne vaudrait
                # plus rien.
                if fleet_id is not None and arret_demande:
                    try:
                        backend.accuser_arret(fleet_id)
                    except BackendError as e:
                        bilan.etat_muet += 1
                        logger.warning("accusé d'arrêt flotte #%s : %s — l'état "
                                       "restera `stopping`, à corriger à la main",
                                       fleet_id, e)
                # ⚠️ CONCLURE sur l'aveuglement, pas seulement le journaliser.
                # Chacun de ces ratés était déjà écrit au journal, un par un, et
                # personne ne les a lus pendant huit vagues. Une ligne DE PLUS
                # n'aurait rien changé : ce qui change, c'est qu'un compte
                # remonte dans le BILAN — là où on lit le résultat.
                # ⚠️ Un travail coupé au plafond a rendu la main SANS ERREUR :
                # il est compté `done`, et rien d'autre ne le distingue d'un
                # travail conclu. Le dire ici, à l'endroit où on lit le
                # résultat, est la seule façon de ne pas prendre un passage
                # tronqué pour un passage fini.
                coupes = sum(n for m, n in bilan.arrets.items()
                             if m in ("max_steps", "max_tokens"))
                if coupes:
                    logger.error("⚠️ %d travail/travaux sur %d ont été COUPÉS à "
                                 "leur plafond (%s) : ils comptent comme réussis "
                                 "et ne le sont pas — leur ligne porte ce qui "
                                 "avait été fait au moment de l'arrêt.",
                                 coupes, bilan.done + bilan.failed,
                                 ", ".join(f"{m}×{n}" for m, n in
                                           sorted(bilan.arrets.items())))
                if bilan.journaux_absents:
                    logger.error("⚠️ %d travail/travaux sur %d SANS JOURNAL RELU "
                                 "depuis cet ordonnanceur (%s/<flotte>/<job>.<tentative>.jsonl) : "
                                 "leur déroulé ne se relit que là où leur worker "
                                 "tourne — s'il journalise.",
                                 bilan.journaux_absents, bilan.done + bilan.failed,
                                 journal.dossier())
                if bilan.etat_muet:
                    logger.error("⚠️ ce passage a tourné EN AVEUGLE : %d geste(s) "
                                 "d'état n'ont pas pu être posés. Son avancement "
                                 "et son coût ne sont donc pas lisibles depuis la "
                                 "plateforme — ni maintenant, ni après coup.",
                                 bilan.etat_muet)
                return bilan

            # 3. Enfiler, sous la rampe — un départ au plus par tour de boucle.
            #
            # ⚠️ La rampe est une MONTÉE EN CHARGE, pas un débit permanent.
            # Elle s'appliquait à chaque enfilement, pour toujours : un travail
            # au plus toutes les 60 s, quelle que soit la vitesse des agents.
            # Mesuré le 29/08 : travaux de 107 s, enfilements toutes les 64 s —
            # **c'est l'ordonnanceur qui donnait le tempo, pas les agents**, et
            # le débit plafonnait là où trois agents auraient pu faire trois fois
            # mieux. Sur un lot de 1 136 lignes, c'est des dizaines d'heures.
            #
            # Elle ne s'applique donc que TANT QU'ON MONTE — jusqu'à ce que la
            # concurrence visée soit atteinte une première fois. Ensuite, un
            # travail conclu libère immédiatement sa place : c'est la
            # concurrence qui borne, et elle seule.
            #
            # ⚠️ On ne SUPPRIME pas la rampe : démarrer trois conversations à la
            # même seconde a déjà gelé la plateforme. Elle protège le départ,
            # elle n'a jamais eu à brider la croisière.
            # ⚠️ La rampe se compte en DÉPARTS, pas en pleine charge atteinte.
            #
            # La première version se désactivait quand `len(en_vol)` atteignait la
            # concurrence — et c'est la rampe elle-même qui empêchait d'y arriver :
            # elle n'enfilait qu'un travail par minute, les travaux se concluaient
            # entre-temps, la file restait à deux en vol sur trois, et la
            # désactivation n'arrivait jamais. Un cercle vicieux, mesuré le 29/08 :
            # enfilements toujours à 62 s après le correctif censé les libérer.
            #
            # La rampe couvre donc les `concurrency` PREMIERS départs — le temps
            # de la montée, littéralement — et cesse ensuite.
            monte = departs < spec.concurrency
            pleine_charge_atteinte = not monte
            if (len(en_vol) < spec.concurrency
                    and (dernier_depart is None
                         or not monte
                         or clock() - dernier_depart >= spec.ramp_seconds)):
                try:
                    jid = backend.enqueue("start", _payload(spec),
                                          fleet_id=fleet_id)
                    departs += 1
                except BackendError as e:
                    # ⚠️ Un `404` n'est pas un incident de transport : c'est
                    # une ADRESSE FAUSSE, et la retenter ne la rendra pas vraie.
                    # Le 09/09/2026, une campagne créée sous une organisation et
                    # cherchée sous une autre a produit ce refus toutes les
                    # vingt secondes jusqu'au délai, « toléré » à chaque fois,
                    # sans qu'un seul travail parte. Tolérer ce qui ne se
                    # résoudra jamais fait perdre le passage entier au lieu
                    # d'une seconde.
                    if getattr(e, "status", None) == 404:
                        bilan.arret = f"cible introuvable — {e}"
                        logger.error("flotte ABANDONNÉE : %s — %d done, %d failed",
                                     bilan.arret, bilan.done, bilan.failed)
                        return bilan
                    erreurs_backend += 1
                    if erreurs_backend >= _MAX_ERREURS_BACKEND:
                        bilan.arret = (f"backend indisponible ({erreurs_backend} "
                                       "erreurs consécutives du driver)")
                        logger.warning("flotte arrêtée : %s — %d done, %d failed",
                                       bilan.arret, bilan.done, bilan.failed)
                        return bilan
                    logger.warning("enqueue toléré (%d/%d) : %s",
                                   erreurs_backend, _MAX_ERREURS_BACKEND, e)
                    sleep(poll_s)
                    continue
                erreurs_backend = 0
                en_vol.add(jid)
                dernier_depart = clock()
                logger.info("job %s enfilé (%d/%d en vol)", jid, len(en_vol),
                            spec.concurrency)

            sleep(poll_s)

    finally:
        # Le bilan de FIN tombe quelle que soit la borne — panne et interruption
        # comprises : un pilotage qui n'existe que sur une sortie propre n'existe
        # pas les jours où il sert.
        ecrire_bilan(spec, backend, conclus, agents=spec.concurrency, secondes=clock() - t0,
                     lignes_initiales=bilan.lignes_initiales,
                     arret=bilan.arret or "interrompu")


# Les motifs d'arrêt NORMAUX — la campagne a fini son travail ou sa borne
# planifiée. Tout AUTRE motif (échecs consécutifs, backend indisponible) est une
# PANNE : le process sort en échec pour que systemd (Restart=on-failure +
# RestartSec long) relance la campagne tout seul quand la panne passe — vécu le
# 20/08 : un 402 Mistral (crédit épuisé) a arrêté proprement la flotte à 604
# fiches, et 26 h ont passé avant qu'un humain ne la relance. Le coût d'une
# relance sous panne persistante est quasi nul (les jobs 402 ne consomment pas).
_ARRETS_NORMAUX = ("file vide", "volume atteint", "budget atteint")


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage : python -m oto_runner.fleet <flotte.yaml>\n"
            "        python -m oto_runner.fleet #<id>     (flotte DÉCLARÉE en base)")
    arg = sys.argv[1]
    backend = Backend()
    if arg.startswith("#"):
        # Piloté par la configuration DÉCLARÉE : la même que celle que le
        # dashboard affiche et qu'un agent peut lire. Le fichier YAML reste un
        # moyen de déclarer, pas la seule façon d'exister.
        spec = spec_depuis_flotte(backend.lire_flotte(int(arg[1:])))
        logger.info("flotte #%s chargée depuis la base : %s", arg[1:], spec.name)
    else:
        spec = load_spec(arg)
    bilan = run_fleet(spec, backend)
    logger.info("bilan : %s", bilan)
    if not any(bilan.arret.startswith(m) for m in _ARRETS_NORMAUX):
        sys.exit(1)   # panne → systemd relance (jamais sur une fin normale)


if __name__ == "__main__":
    main()
