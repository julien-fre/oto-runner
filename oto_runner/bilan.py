"""Le BILAN d'une flotte — ce qu'il faut pour PILOTER, rendu par le driver.

Vécu : sur une campagne de plusieurs semaines, tout le pilotage (lignes
abouties, faux départs, refus d'écriture, coût par ligne) est sorti de deux
scripts maison déposés à la main sur la box — un savoir qui ne vivait que dans
une tête et dans un `/root`. Le driver le rend désormais lui-même, à intervalle
régulier ET à la fin quelle que soit la borne : c'est ce qui permet à quelqu'un
d'autre de reprendre la campagne.

Deux règles de lecture, toutes deux payées cher :

- **l'avancement se lit au TABLEAU, jamais aux jobs.** « Aboutie » = une ligne
  qui ne correspond PLUS au filtre de réservation de la flotte. Compter les
  jobs « done » comptait les tours perdus comme des succès ;
- **le coût se lit au résultat DÉCLARÉ des jobs** (`usage_tokens`,
  `tool_counts`), et les refus d'appel au
  JOURNAL des appels de l'org : une écriture refusée (RBAC, quota, schéma) ne
  fait pas échouer le job — l'agent conclut « done » sans une ligne écrite ;
- **la sortie de la file n'est pas l'issue.** Une ligne sort du filtre de
  réservation autant en finissant `echec` — l'état d'ABANDON du cycle de vie,
  posé par la plateforme après trois réservations sans écriture — qu'en
  finissant enrichie. « Abouties 3/3 » sur deux abandons (06/09/2026) : le bilan
  ventile désormais les lignes par valeur FINALE de leur colonne de statut, lue
  au schéma (`role="status"`), et « abouties » ne compte que les états terminaux
  qui ne sont pas l'abandon. Les postes vivent dans `bilan_postes.py`.

Rien ici n'arrête une flotte : une lecture qui échoue devient un poste `null`
assumé et une ligne de journal, jamais un chiffre inventé ni un traceback qui
masquerait l'arrêt en cours (le bilan de fin tombe souvent APRÈS une panne).

⚠️ **PIÈGE DE LECTURE des refus « ligne réservée par … » — deux fois la même
erreur en une semaine, par deux sessions différentes (2026-08-28).** Le message
d'un refus d'écriture nomme le TITULAIRE du moment :

    ligne « 01a0…d018 » réservée par « 06953e8c… » jusqu'à 15:39:04
    ligne « 01a0…d018 » réservée par « 262ea7e2… » jusqu'à 15:44:03

Deux titulaires sur la même ligne à une minute d'intervalle **se lisent
naturellement comme « deux agents la traitent en même temps »** — et l'on en
conclut que la réservation n'est pas atomique. **C'est faux, et coûteux : la
conclusion est remontée jusqu'à une décision de montée en charge.** La prise de
ligne est verrouillée par le datastore ; des titulaires SUCCESSIFS ressemblent
à des titulaires SIMULTANÉS.

**Comment les distinguer, et il n'y a que ce moyen :** reconstituer la
chronologie de la ligne au journal des appels — `data_claim_next`,
`data_release` et `data_write` mêlés, triés par heure. Si un `data_release` du
premier titulaire précède la prise du second, ils se succèdent et le verrou
fonctionne. C'était le cas.

**Et alors, que dit le refus ?** Qu'un agent a écrit sur une ligne qu'il ne
détient pas — relâchée entre-temps, ou dont il a reconstruit l'identifiant au
lieu de reprendre celui que la réservation lui a rendu. **Un refus est donc la
preuve que la garde FONCTIONNE, jamais l'indice qu'elle manque** : prendre un
dispositif de sécurité pour le danger qu'il prévient est l'erreur exacte à ne
pas refaire ici.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Optional

from .bilan_ligne import journaliser_refus as _journaliser_refus, ligne as _ligne
from .bilan_postes import (abouties_de, lignes_par_statut, refus_ecriture,
                           valeur)

__all__ = ["ecrire_bilan", "annoter_lignes_sorties", "controler_fiches",
           "extinction_sans_acte", "valeur", "chemin_json", "PERIODE_S"]

logger = logging.getLogger("oto_runner.bilan")

PERIODE_S = 600            # défaut du champ de déclaration `bilan_periode_s`


def _postes_jobs(jobs: dict) -> dict:
    """Ce que les jobs CONCLUS de la flotte disent : volumes, coût, écritures.

    Un job non conclu n'a pas de résultat : le compter serait compter du vide,
    on lève plutôt que de le taire."""
    # ⚠️ Le bilan ne compte plus les réservations ni les écritures : il
    # devinait le métier à partir des NOMS d'outils. Un exécuteur d'agents ne
    # sait pas ce qu'écrire veut dire — c'est à qui commande le travail de le
    # juger, sur les comptes d'appels que chaque travail déclare.
    postes = {"termines": 0, "echoues": 0, "jetons": 0, "sans_usage": 0}
    for jid, job in sorted(jobs.items()):
        statut = job.get("status")
        if statut not in ("done", "failed"):
            raise ValueError(f"bilan : le job {jid} n'est pas conclu (statut "
                             f"{statut!r}) — le bilan ne compte que des jobs conclus")
        resultat = job.get("result") or {}
        # ⚠️ Un travail sans usage déclaré se compte À PART : l'ajouter comme un zéro
        # rendrait un total qui a l'air complet (cf. `comptage`).
        if resultat.get("usage_tokens") is None:
            postes["sans_usage"] += 1
        else:
            postes["jetons"] += int(resultat["usage_tokens"])
        postes["termines" if statut == "done" else "echoues"] += 1
    return postes


def _restantes(spec, backend) -> Optional[int]:
    """Les lignes qui correspondent ENCORE au filtre de réservation."""
    try:
        return int(backend.count_rows(spec.namespace, filter=spec.filter,
                                      org=spec.org))
    except Exception as e:  # noqa: BLE001 — un compte illisible ≠ un compte faux
        logger.warning("bilan : lignes restantes illisibles : %s", e)
        return None


def chemin_json(spec) -> Optional[str]:
    """Le bilan se pose À CÔTÉ de sa déclaration : `flotte.yaml` →
    `flotte.bilan.json`. Une flotte construite en mémoire (banc de test, appel
    embarqué) n'a pas de déclaration sur disque : il n'y a rien à écrire."""
    source = getattr(spec, "source", "")
    return os.path.splitext(source)[0] + ".bilan.json" if source else None


def _ecrire(chemin: str, bilan: dict) -> None:
    """Réécriture ATOMIQUE : le pilote lit ce fichier pendant que la flotte
    tourne — il ne doit jamais tomber sur un JSON à moitié écrit."""
    dossier = os.path.dirname(os.path.abspath(chemin))
    try:
        fd, tmp = tempfile.mkstemp(dir=dossier, suffix=".bilan.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(bilan, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, chemin)
    except OSError as e:
        logger.error("bilan : écriture de %s impossible : %s", chemin, e)


def annoter_lignes_sorties(spec, backend, jobs: dict) -> dict:
    """Écrit sur les lignes SORTIES de la file ce qui les a traitées et pourquoi
    elles sont sorties. Rend {"sorties": n, "annotees": k}.

    ⚠️ Pourquoi ça existe. Une ligne réservée trois fois sans écriture est
    basculée en « échec » PAR LA PLATEFORME : personne n'écrit à ce moment-là,
    donc la ligne ne porte ni estampille, ni motif, ni la moindre trace de ce
    qui a été tenté. **C'est le seul événement de la campagne sur lequel on a
    vraiment besoin de savoir, et le seul sur lequel on ne sait rien.** Mesuré
    le 28/08 : 2 lignes sur 23 — à l'échelle d'une vague, des centaines de
    fiches muettes, et une fiche muette ne se rattrape pas après coup,
    contrairement à une fiche incomplète.

    Le harnais ne peut pas le faire : le worker qui vient de finir ne sait pas
    quelle ligne il avait, ni qu'elle vient de sortir. L'ordonnanceur, lui, voit
    l'état final du tableau — c'est le seul endroit possible.

    ⚠️ BEST-EFFORT, comme l'estampille : une observation ne bloque jamais une
    file. Tout échec de lecture ou d'écriture est journalisé et le bilan sort
    quand même."""
    if not getattr(spec, "namespace", None):
        return {"sorties": 0, "annotees": 0}
    try:
        sorties = backend.rows(spec.namespace, {"statut": "echec"},
                               org=getattr(spec, "org", None), limit=500)
    except Exception as e:  # noqa: BLE001 — cf. docstring
        logger.warning("bilan : lignes sorties illisibles : %s", e)
        return {"sorties": None, "annotees": 0}
    a_annoter = [r for r in sorties if not r.get("modele")]
    if not a_annoter:
        return {"sorties": len(sorties), "annotees": 0}

    # Le modèle vient des jobs de CETTE flotte ; la version de procédure se lit
    # sur une ligne RÉUSSIE du même lot — l'ordonnanceur ne la connaît pas, mais
    # ses propres fiches la portent. Deux relevés, aucune invention.
    modeles = {(j.get("result") or {}).get("model") for j in jobs.values()}
    modele = next((m for m in modeles if m), None)
    version = None
    try:
        for r in backend.rows(spec.namespace, {"statut": "enrichi"},
                              org=getattr(spec, "org", None), limit=1):
            version = r.get("version_procedure")
    except Exception as e:  # noqa: BLE001
        logger.warning("bilan : version de procédure illisible : %s", e)

    raison = ("sortie de la file : réservée sans écriture jusqu'au plafond de "
              "la plateforme. Aucun agent n'a écrit de fiche ; le motif exact "
              "de chaque tentative se lit au journal des appels de l'org "
              f"(outil data_write, autour du {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC).")
    annotees = 0
    for r in a_annoter:
        valeurs = {"retraitement": "outil", "retraitement_motif": raison}
        if modele:
            valeurs["modele"] = modele
        if version:
            valeurs["version_procedure"] = version
        try:
            backend.patch_row(spec.namespace, str(r.get("_id")), valeurs,
                              org=getattr(spec, "org", None))
            annotees += 1
        except Exception as e:  # noqa: BLE001 — une ligne non annotée ne bloque
            # ni les suivantes ni le bilan : on la compte comme non expliquée.
            logger.warning("bilan : ligne %s non annotée : %s", r.get("siren"), e)
    logger.info("bilan : %d ligne(s) sortie(s), %d annotée(s)", len(sorties), annotees)
    return {"sorties": len(sorties), "annotees": annotees}


ACTE_EXTINCTION = re.compile(
    r"radiation|radié|radie"
    r"|cessation|cessé d'activité|cesse d'activite"
    r"|liquidation|liquidé|liquide judiciaire"
    r"|dissolution|dissous|dissoute"
    # ⚠️ « clôture » ne se prend JAMAIS seul : une clôture d'EXERCICE est un
    # acte de vie normale. On ne retient que les clôtures qui éteignent —
    # attrapé par un test existant, qui rejetait « jugement de clôture ».
    r"|jugement de clôture|jugement de cloture"
    r"|clôture de liquidation|cloture de liquidation"
    r"|clôture pour insuffisance|cloture pour insuffisance"
    r"|jugement d'ouverture|redressement judiciaire"
    r"|reprise par|absorbée par|absorbee par|fusion-absorption",
    re.I)


def extinction_sans_acte(fiche: dict) -> bool:
    """Une fiche déclarée éteinte cite-t-elle un acte d'EXTINCTION ?

    ⚠️ DÉFINITION UNIQUE, volontairement exportée. Elle a d'abord vécu en deux
    exemplaires — ici et dans le script de verdict — et les deux ont divergé dès
    la première correction : le verdict criait sur neuf fiches valides pendant que
    le bilan les acceptait. **Deux contrôles qui font la même chose divergent
    toujours, et le jour où ils divergent on ne sait plus lequel croire.**

    Ce qui fonde une extinction : un acte NOMMÉ — radiation, cessation,
    liquidation, dissolution, jugement de clôture, reprise — lu dans le motif,
    l'écartement, ou la PIÈCE cochée. Ni une date seule, ni un dépôt de comptes,
    qui prouve le contraire ; ni une accumulation d'absences, qui date un vide."""
    fonde = " ".join(str(valeur(fiche.get(c)) or "")
                     for c in ("qualification_motif", "motif_ecartement",
                               "qualification_piece"))
    return not ACTE_EXTINCTION.search(fonde)


def controler_fiches(spec, backend, jobs: dict) -> dict:
    """Deux contrôles DÉTERMINISTES sur les fiches produites, au bilan de fin.

    ⚠️ Ils existent parce que deux défauts ont traversé une grille de six critères
    pourtant tous à zéro (28/08). Aucun n'est une question de jugement : ce sont des
    contradictions internes, qu'une requête attrape et qu'une relecture humaine rate.

    **1. L'estampille est-elle EXACTE ?** Sur ce chemin le harnais ne peut pas
    injecter le modèle — la boucle d'outils tourne chez le fournisseur — il le
    DEMANDE par la consigne. Une fiche a recopié `mistral-large-2407` quand les 144
    travaux du journal disaient tous `2512`. Une estampille absente se voit ; une
    estampille FAUSSE ment, et elle ment sur ce qui sert à trier. Le taux à suivre
    n'est donc pas « posées » mais « exactes ».

    **2. Une fiche se contredit-elle ?** Une entreprise déclarée éteinte dont les
    notes disent « état administratif actif » est fausse par construction, quelle
    que soit la pièce cochée. Le verrou posé la veille forçait l'agent à CHOISIR une
    pièce, pas à en AVOIR une : il a coché « cessation au registre » sans acte, avec
    ses propres notes qui le démentent deux lignes plus haut.

    ⚠️ BEST-EFFORT : une lecture impossible rend des postes `null` qui disent
    POURQUOI ils sont absents, jamais un zéro. Un contrôle n'arrête pas une file."""
    ns = getattr(spec, "namespace", None)
    if not ns:
        return {"estampille_exacte": None, "fiches_contradictoires": None,
                "omis": "déclaration sans tableau"}
    try:
        fiches = backend.rows(ns, {"statut": "enrichi"},
                              org=getattr(spec, "org", None), limit=500)
    except Exception as e:  # noqa: BLE001 — cf. docstring
        logger.warning("bilan : fiches illisibles pour contrôle : %s", e)
        return {"estampille_exacte": None, "fiches_contradictoires": None,
                "omis": f"fiches illisibles : {e}"}


    # ── 1. estampille exacte ────────────────────────────────────────────────
    attendus = {(j.get("result") or {}).get("model") for j in jobs.values()}
    attendus = {m for m in attendus if m}
    fausses = []
    if len(attendus) == 1:
        attendu = next(iter(attendus))
        fausses = [str(f.get("siren")) for f in fiches
                   if valeur(f.get("modele")) and valeur(f.get("modele")) != attendu]
    elif len(attendus) > 1:
        # Plusieurs modèles dans la même flotte : on ne peut pas dire laquelle ment,
        # et l'affirmer serait pire que se taire.
        logger.warning("bilan : %d modèles distincts dans la flotte — contrôle "
                       "d'estampille impossible : %s", len(attendus), attendus)
        fausses = None

    # ── 2. extinction déclarée SANS acte de registre ────────────────────────
    # ⚠️ Première version RETIRÉE avant d'être posée : elle cherchait le mot
    # « actif » dans les notes d'une fiche éteinte. Six alertes, UNE SEULE vraie —
    # deux fiches citaient un acte daté (radiation BODACC du 04/06/2013, jugement de
    # clôture du 08/11/2016) ET signalaient honnêtement que le répertoire affiche
    # encore « actif », parce qu'il retarde les radiations : ce sont de BONNES
    # fiches. Une autre accrochait sur « insuffisance d'ACTIF ». Une garde qui crie
    # à tort cesse d'être lue, et celle-ci aurait crié cinq fois sur six.
    #
    # Le critère juste porte sur ce qui FONDE l'extinction, pas sur un mot : une
    # fiche éteinte doit citer un ÉVÉNEMENT DE REGISTRE DATÉ. Un acte avec sa date,
    # son numéro ou son jugement d'un côté ; une accumulation d'absences — « aucun
    # dépôt, aucun salarié, aucune trace » — de l'autre, qui ne prouve rien.
    # ⚠️ L'ANNÉE SEULE NE COMPTE PAS, et c'est le point qui fait tout le contrôle.
    # Éprouvé sur les huit fiches éteintes d'un palier réel : avec l'année, le
    # critère retenait ZÉRO — y compris le seul vrai manquement, dont le motif dit
    # « aucun dépôt de comptes depuis 2016, aucun salarié, aucune trace ». Cette
    # année-là date une ABSENCE, pas un acte : c'est exactement l'accumulation de
    # riens qu'on veut refuser. Sans elle, le critère retient ce cas et écarte les
    # sept autres, qui citent tous un acte nommé.
    # ⚠️ RESSERRÉ le 29/08 (deuxième affinage, MÊME ligne — voir la note dessous).
    # La version précédente acceptait « une date complète OU un mot de registre ».
    # Elle a donc laissé passer une fiche déclarée éteinte dont la preuve était…
    # un DÉPÔT DE COMPTES daté, publié au BODACC. Un dépôt de comptes est un acte
    # d'ACTIVITÉ : il prouve exactement le contraire de l'extinction, et il était
    # retourné en preuve d'extinction parce que le contrôle ne lisait qu'une date.
    #
    # Le critère porte donc sur la NATURE de l'acte, jamais sur sa date ni sur le
    # support qui le publie. Un événement de registre daté quelconque — dépôt,
    # immatriculation, modification, changement de gérant — ne compte pas, quelle
    # que soit sa date et quel que soit le journal qui l'annonce.

    # ⚠️ NOTE, à lire avant le troisième affinage : c'est la DEUXIÈME fois que la
    # même fiche resserre ce contrôle. Un contrôle qu'un même cas corrige deux fois
    # court après les cas au lieu de porter sur le fond — le signe qu'il approxime
    # une question métier (« cette entreprise est-elle éteinte ? ») par une
    # recherche de mots. S'il faut l'affiner une troisième fois, la bonne réponse
    # ne sera pas un mot de plus : ce sera d'aller lire l'état au registre.
    contradictoires = []
    for f in fiches:
        if valeur(f.get("qualification")) != "dormante_ou_introuvable":
            continue
        # ⚠️ TROISIÈME correction, et cette fois ce n'est pas un mot de plus :
        # c'est le CHAMP. Le contrôle lisait le motif et l'écartement, jamais
        # `qualification_piece` — qui porte précisément la NATURE de l'acte
        # (« radiation », « liquidation », « cessation_registre »). Il a donc
        # crié sur neuf fiches qui cochaient toutes une pièce d'extinction
        # valide : un faux positif intégral, sur le champ le mieux structuré du
        # tableau, pendant que le contrôle fouillait de la prose à côté.
        #
        # La note d'avant disait : « s'il faut affiner une troisième fois, aller
        # lire l'état au registre ». Elle visait le vocabulaire. Le vrai défaut
        # était de chercher dans le texte libre une information qui existait,
        # nommée et codée, dans un champ voisin.
        if extinction_sans_acte(f):
            contradictoires.append(str(f.get("siren")))

    if fausses:
        logger.warning("bilan : %d estampille(s) FAUSSE(S) : %s", len(fausses), fausses[:6])
    if contradictoires:
        logger.warning("bilan : %d fiche(s) déclarée(s) éteinte(s) SANS citer d'acte "
                       "de registre daté : %s", len(contradictoires), contradictoires[:6])
    return {"fiches": len(fiches),
            "estampille_fausse": fausses,
            "estampille_exacte": (None if fausses is None
                                  else len(fiches) - len(fausses)),
            "fiches_contradictoires": contradictoires}


def ecrire_bilan(spec, backend, jobs: dict, *, lignes_initiales: int,
                 secondes: float, arret: str = "", agents: int = None) -> dict:
    """Calcule le bilan de la flotte, le journalise et le pose en JSON.

    `jobs` = les jobs CONCLUS de la flotte (id → job), `lignes_initiales` = le
    compte relevé au lancement, `secondes` = depuis ce lancement, `arret` = le
    motif quand c'est le bilan de FIN (vide pendant la flotte). Rend le bilan,
    pour que l'appelant n'ait rien à recalculer."""
    postes = _postes_jobs(jobs)
    restantes = _restantes(spec, backend)
    # ex-« abouties » : les lignes qui ne correspondent PLUS au filtre — sorties
    # de la file, quelle qu'en soit l'issue. L'issue, c'est le statut.
    sorties = None if restantes is None else max(0, lignes_initiales - restantes)
    statut = lignes_par_statut(spec, backend)
    abouties, abouties_omis = abouties_de(statut, sorties)
    conclus = postes["termines"] + postes["echoues"]
    complet = not postes["sans_usage"]   # un total ne vaut que si chaque travail a déclaré
    refus, refus_omis = refus_ecriture(spec, backend, secondes, jobs)
    # Seulement au bilan de FIN : une ligne peut encore sortir pendant la flotte,
    # et l'annoter à chaque tour ferait du bruit sans rien apprendre.
    annotees = annoter_lignes_sorties(spec, backend, jobs) if arret else None
    controles = controler_fiches(spec, backend, jobs) if arret else None
    bilan = {
        "horodatage": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Le nom de la flotte est le TAG apposé à ses jobs : c'est par lui
        # qu'on retrouve une campagne, pas par le tableau qu'elle draine.
        "flotte": spec.name,
        "namespace": spec.namespace,
        "procedure": spec.procedure,
        "final": bool(arret),
        "arret": arret or None,
        "secondes": round(float(secondes), 1),
        # ⚠️ Combien d'agents ont tourné EN PARALLÈLE. Sans ce nombre, deux
        # bilans ne sont pas comparables : la durée, le débit et le taux
        # d'échec dépendent tous de lui. Il a manqué le 07/09/2026, quand il a
        # fallu relire 72 journaux pour établir que toutes les vagues avaient
        # tourné à un seul agent — le `concurrency` des déclarations, lui,
        # n'est lu QUE par le mode flotte, et le mode direct ne suit que son
        # option de ligne de commande. Écrire ce qui a EU LIEU, pas ce qui
        # était déclaré.
        "agents": agents,
        "lignes": {"depart": lignes_initiales, "restantes": restantes,
                   "sorties": sorties,
                   # La valeur FINALE de la colonne de statut, sur le périmètre
                   # du passage (le filtre sans sa clause de statut).
                   "par_statut": statut.get("par_statut"),
                   "statut": {k: statut[k] for k in ("colonne", "perimetre",
                                                     "abandon", "terminaux", "omis")
                              if k in statut},
                   # Terminales hors abandon — null AVEC sa raison sinon.
                   "abouties": abouties, "abouties_omis": abouties_omis},
        "jobs": {"termines": postes["termines"], "echoues": postes["echoues"]},
        # ⚠️ Compté À PART, jamais fondu dans « abouties » : un bilan qui rend
        # « 98 traitées » sans dire que 2 sont sorties muettes rend un
        # dénominateur amputé qui a l'air complet.
        "lignes_sorties": annotees,
        # Deux contradictions internes qu'une grille de six critères a laissées
        # passer, toutes deux attrapables par une requête : une estampille qui
        # nomme le mauvais modèle, une fiche éteinte dont les notes disent « actif ».
        "controles": controles,
        # ⚠️ Un seul travail sans usage déclaré, et le total n'en est plus un : `total`
        # et les ratios passent à null, `connus` garde la somme déclarée et
        # `travaux_sans_usage` dit combien manquent (cf. `comptage`).
        "jetons": {"total": postes["jetons"] if complet else None,
                   "connus": postes["jetons"],
                   "travaux_sans_usage": postes["sans_usage"],
                   "par_job": round(postes["jetons"] / conclus) if complet and conclus else None,
                   "par_sortie": (round(postes["jetons"] / sorties)
                                  if complet and sorties else None),
                   # Le vrai coût d'une campagne : ce que coûte une ligne qui
                   # ABOUTIT, jamais ce que coûte un job (un job peut n'avoir
                   # rien produit). Aucune aboutie ⟹ null, pas une division.
                   "par_aboutie": (round(postes["jetons"] / abouties)
                                   if complet and abouties else None)},
        "refus_ecriture": refus,
        "refus_ecriture_omis": refus_omis,
    }
    chemin = chemin_json(spec)
    logger.info("%s", _ligne(bilan, chemin))
    if arret:
        _journaliser_refus(refus)
    if chemin:
        _ecrire(chemin, bilan)
    else:
        logger.debug("bilan non posé : la flotte n'a pas de déclaration sur disque")
    return bilan
