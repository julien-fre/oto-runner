"""Ce que le bilan DIT au journal — la ligne compacte, et le déroulé des refus.

Le JSON du bilan porte tout ; une ligne de journal, elle, se lit d'un coup d'œil
dans un `tail -f`. Deux règles la gouvernent, et elles ont chacune un vécu :

- **chaque poste NOMME ce qu'il compte** — « sorties » (de la file), « abouties »
  (état terminal hors abandon), « de cette flotte » (par opposition au journal
  d'appels de l'ORG, qui porte aussi ceux des passages voisins). « Abouties 3/3 »
  sur deux abandons, puis « data_write 3 appels, 2 refusés » sur un travail qui
  n'en avait fait qu'un : deux fois, un chiffre a menti faute d'un nom ;
- **elle abrège, et elle mène au détail** : les motifs y sont coupés, le JSON les
  porte entiers, et la ligne dit où il est. Au bilan de FIN seulement, chaque
  refus est déroulé ENTIER sur sa propre ligne — pendant la flotte, ce déroulé
  noierait le journal.
"""
from __future__ import annotations

import logging
from typing import Optional

from .bilan_postes import REFUS_OUTIL

logger = logging.getLogger("oto_runner.bilan")


def _jetons_lisibles(n: Optional[int]) -> str:
    """Un ordre de grandeur qui se lit d'un coup d'œil : « 1,8 M », « 24,1 k »."""
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M".replace(".", ",")
    if n >= 1_000:
        return f"{n / 1_000:.1f} k".replace(".", ",")
    return str(n)


_MOTIF_AFFICHE = 90   # sur la LIGNE de journal seulement : le JSON porte tout


def _abrege(texte: str) -> str:
    return texte if len(texte) <= _MOTIF_AFFICHE else texte[:_MOTIF_AFFICHE] + "…"


def ligne(bilan: dict, chemin: Optional[str]) -> str:
    """La ligne de journal : des effectifs bruts AVEC leur dénominateur — un
    pourcentage cacherait qu'il porte sur trois lignes — et chaque poste NOMME
    ce qu'il compte : « sorties » (de la file), « abouties » (état terminal hors
    abandon), jamais l'un pour l'autre. Compacte : les motifs y sont abrégés, et
    la ligne pointe vers le JSON qui les porte entiers."""
    lignes, jetons = bilan["lignes"], bilan["jetons"]
    sorties = "?" if lignes["sorties"] is None else lignes["sorties"]
    postes = [f"sorties {sorties}/{lignes['depart']}"]
    if lignes["par_statut"]:
        postes.append("statut final : " + " · ".join(
            f"{k} {n}" for k, n in sorted(lignes["par_statut"].items(),
                                           key=lambda kv: -kv[1])))
    postes.append(f"abouties {lignes['abouties']}" if lignes["abouties"] is not None
                  else f"abouties non mesurées ({lignes['abouties_omis']})")
    if jetons["total"] is None:
        # Un total qui manque se dit avec ce qu'on sait — jamais un « — jetons » muet.
        postes.append(f"{_jetons_lisibles(jetons['connus'])} jetons connus, "
                      f"{jetons['travaux_sans_usage']} travail(aux) sans usage déclaré")
    else:
        postes.append(f"{_jetons_lisibles(jetons['total'])} jetons")
        postes.append(f"{_jetons_lisibles(jetons['par_aboutie'])}/aboutie"
                      if jetons["par_aboutie"] is not None
                      else f"{_jetons_lisibles(jetons['par_sortie'])}/sortie"
                      if jetons["par_sortie"] is not None
                      else "pas de jetons/sortie (0 sortie)")
    refus = bilan["refus_ecriture"]
    if refus:
        # ⚠️ Chaque compte NOMME son périmètre. « data_write 3 appels, 2 refusés »
        # a fait porter à une flotte le refus d'une autre, le 06/09 : ce qui est
        # à elle se dit « de cette flotte », le reste de l'org se dit « org ».
        if refus["refuses"] is None:
            postes.append(f"{refus['outil']} : refus non attribués "
                          f"({refus['refuses_omis']})")
        else:
            postes.append(
                f"{refus['outil']} {refus['refuses']} refusé"
                f"{'s' if refus['refuses'] > 1 else ''} de cette flotte "
                f"(org : {refus['refuses_org']}/{refus['appels_org']} appels "
                f"sur {refus['fenetre_minutes']} min)")
            if refus["refuses_hors_flotte"]:
                postes.append(f"{refus['refuses_hors_flotte']} refus hors de "
                              "cette flotte, non comptés")
        # Le motif qui compte le plus se dit sur la ligne : « 12 refusés » ne dit
        # pas si les agents inventent des entreprises ou oublient un jeton.
        for poste, n in sorted((refus.get("motifs") or {}).items(),
                               key=lambda kv: -kv[1])[:2]:
            postes.append(f"{_abrege(poste)} ×{n}")
    else:
        postes.append(f"{REFUS_OUTIL} non mesuré "
                      f"({bilan['refus_ecriture_omis']})")
    if chemin:
        postes.append(f"détail complet : {chemin}")
    return f"bilan flotte {bilan['flotte']} : " + " · ".join(postes)


def journaliser_refus(refus: Optional[dict]) -> None:
    """Au bilan de FIN : chaque refus ENTIER sur sa ligne, avec le travail et le
    journal JSONL qui le portent — le journal de flotte peut rester compact à
    condition de mener au détail."""
    for r in (refus or {}).get("detail") or []:
        qui = ("HORS de cette flotte (run %s) — non compté" % r["run_id"]
               if r["hors_flotte"] else "job %s" % r["job"])
        logger.info("refus %s à %s UTC — %s : %s — journal du job : %s",
                    refus["outil"], r["quand"], qui, r["erreur"],
                    r["journal"] or "—")
