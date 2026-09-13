"""La CONCLUSION d'un travail : clore le run oto, la dire au journal, rendre la ligne.

Un travail se termine de deux façons, et **les deux passent ici** :

- il **conclut** — la boucle a rendu un résultat : le run se clôt `done` (ou
  `blocked` quand la boucle s'est arrêtée sur une borne), le résultat DÉCLARÉ
  part à la file, l'événement `resultat` ferme le journal ;
- il **meurt en plein vol** — une exception remonte de la boucle. C'est le
  chemin qui manquait. Nuit du 06/09/2026 : deux travaux tués par un
  `ReadTimeout` du fournisseur, journal terminé sur un `erreur`, `run_finish`
  jamais appelé — **et la ligne que le run tenait est restée verrouillée
  jusqu'à l'expiration de son bail** (quinze minutes), pendant que le mode
  direct annonçait « volume atteint ». Un travail mort doit rendre ce qu'il
  tient TOUT DE SUITE : c'est `run_finish(outcome="failed")` qui libère.

⚠️ Clore ne fait jamais échouer davantage. La clôture est un geste de TENUE :
son refus se dit (`run_finish: "refusé : …"`) et n'écrase pas la cause d'origine,
déjà écrite au journal avec sa pile.

⚠️ Ce module ne juge RIEN de ce que l'agent a produit — il ne sait pas ce
qu'écrire veut dire. Il compte des jetons, des pas et des appels, et recopie ce
que la boucle a dit d'elle-même.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .backend import BackendError

logger = logging.getLogger("oto_runner")

_NOTE_MAX = 400        # ce qu'on met dans la note d'un run_finish / d'un complete
_PAUSES_S = (2, 4)     # renvois d'une conclusion dont la réponse s'est perdue


@dataclass
class RunEnCours:
    """Ce qu'un travail TIENT pendant qu'il tourne : sa session MCP et son run.

    Rempli par `worker._traiter` au fur et à mesure, lu par `worker._un_travail`
    quand le travail meurt : sans lui, l'étage qui rattrape l'exception ne sait
    ni quel run est ouvert ni sous quelle identité le clore — et ne libère rien.
    """
    mcp: object = None
    run_id: Optional[str] = None


def clore(tenu: RunEnCours, outcome: str, note: Optional[str] = None,
          job_id=None) -> str:
    """`run_finish` — rend « ok » ou « refusé : <le dire du serveur> ».

    Ne lève JAMAIS : un run_finish refusé ne fait pas échouer un travail que la
    campagne a déjà payé, il se dit au journal et à la ligne de log.

    ⚠️ C'est `run_finish` qui LIBÈRE ce que le run tient — les lignes réservées
    en premier. Sans run ouvert il n'y a rien à clore, et ça se dit aussi :
    jamais un appel à vide, jamais un « ok » qui n'a rien fermé."""
    if tenu.mcp is None or not tenu.run_id:
        return "aucun run ouvert"
    try:
        tenu.mcp.outil("run_finish", {"run_id": tenu.run_id, "outcome": outcome,
                                      "note": note})
        return "ok"
    except Exception as e:  # noqa: BLE001 — cf. docstring
        logger.warning("job %s : run_finish refusé (%s)", job_id, e)
        return f"refusé : {e}"


def resultat_declare(res, modele_par_defaut: str) -> dict:
    """Le résultat DÉCLARÉ (R5) : ce que l'ordonnanceur lit pour ses bornes.

    Un résumé d'EXÉCUTION — jamais du contenu de fil, jamais un jugement sur ce
    que l'agent a produit. `tool_counts` compte les APPELS par outil sans les
    interpréter : il rend le tour perdu lisible d'un coup d'œil (un agent qui
    analyse et conclut en prose sans rien appeler ne produit aucune erreur ; la
    seule trace est l'écart entre ses mots et ses appels)."""
    entree = int(res.usage.get("input_tokens") or 0)
    sortie = int(res.usage.get("output_tokens") or 0)
    # Le cache de prompt se compte À CÔTÉ, jamais dedans : `input_tokens` est le
    # reste NON caché, donc les jetons lus en cache ne sont pas dans `jetons`.
    # `usage_tokens` reste input+output — c'est la base des bornes de flotte
    # (budget, rendement), et la déplacer les fausserait toutes d'un coup.
    compte: dict = {}
    for s in res.steps:
        if s.ok:
            compte[s.tool] = compte.get(s.tool, 0) + 1
    return {
        "usage_tokens": entree + sortie,
        "usage_input": entree,
        "usage_output": sortie,
        "usage_cache_read": int(res.usage.get("cache_read_input_tokens") or 0),
        "usage_cache_write": int(res.usage.get("cache_creation_input_tokens") or 0),
        "stopped": res.stopped,
        "steps": len(res.steps),
        "tool_counts": compte,
        # ⚠️ Repli SUR LE WORKER, pas seulement dans les transports : c'est ce qui
        # ferme la classe. Un transport qui oublierait de poser l'estampille
        # rendrait à nouveau `null` partout — et un `null` ne se distingue pas
        # d'un job qui n'a pas tourné. Ici, au pire, on estampille ce qu'on a
        # DEMANDÉ ; le transport, lui, sait ce qui a été SERVI et gagne.
        "model": res.model or modele_par_defaut,
    }


def echec_nomme(res) -> Optional[str]:
    """Le motif d'ÉCHEC d'une boucle qui a rendu la main sans exception — ou `None`.

    Un seul aujourd'hui : l'appel d'outil rendu en texte (`appel_mal_encode`,
    job 17275). Ce n'est pas un jugement sur le travail : le fournisseur n'a pas
    émis l'appel que son propre message annonçait, la boucle n'a donc rien pu
    exécuter. Conclu `done`, le travail cachait une ligne jamais servie ; conclu
    en échec nommé, il passe par la mécanique existante des tentatives, et se
    voit là où les échecs se lisent."""
    if getattr(res, "stopped", None) != "appel_mal_encode":
        return None
    outil = (getattr(res, "defaut", None) or {}).get("outil") or "outil inconnu"
    return f"appel_outil_mal_encode ({outil})"


def rendre(file, job_id, ok: bool, error: Optional[str], run_id: Optional[str],
           result: Optional[dict], note) -> Optional[dict]:
    """`complete` — et ce qu'on fait quand la file ne prend pas la conclusion.

    Le travail a CONCLU : son run est clos avec sa vraie issue, et `resultat` est
    déjà au journal. Un refus ici n'est donc pas une mort. Le laisser remonter
    jusqu'à `en_echec` re-clôturait le run `failed` et rendait à la file, pour
    être rejoué, un travail qui avait conclu (banc du 13/09/2026 :
    `result_too_large` après un `done`).

    - `result_too_large` — le CODE que le serveur nomme, pas tout 400 : la charge
      est trop grosse, l'issue n'est pas en cause ; renvoyée une fois, réduite
      par `_resume` ;
    - réponse perdue (transport, 5xx) : la conclusion a pu être prise ; renvoyée
      TELLE QUELLE, deux fois au plus (2 s puis 4 s). Un 404 ensuite ne prouve
      pas qu'elle l'a été : il se dit non rendu, avec ce doute ;
    - tout autre refus — un 400 de contrat, un 404 d'emblée — ne se rejoue pas.

    Quatre envois au plus. Ne lève pas sur un refus de la file : chacun s'écrit
    au journal (`conclusion_refusee`), et ce qui reste non rendu se dit en ERREUR
    au log. Ce worker ne rejoue rien ; la file décide du reste à l'expiration du bail."""
    charge, renvois, envoi = result, 0, 0
    while True:
        envoi += 1
        try:
            return file.complete(job_id, ok=ok, error=error, run_id=run_id, result=charge)
        except BackendError as e:
            code = getattr(e, "code", None)
            note("conclusion_refusee", essai=envoi, status=e.status, code=code, erreur=str(e))
            if code == "result_too_large" and result is not None and charge is result:
                charge = _resume(result, e)
                continue
            if (e.status is None or e.status >= 500) and renvois < len(_PAUSES_S):
                time.sleep(_PAUSES_S[renvois])
                renvois += 1
                continue
            logger.error("job %s : conclusion %s NON rendue à la file — %s%s. Le run est "
                         "clos, le résultat est au journal ; rien n'est rejoué ici.",
                         job_id, "ok" if ok else "en échec", e,
                         " (404 après un renvoi : la conclusion d'avant a PU être prise, "
                         "rien ne le prouve)" if envoi > 1 and e.status == 404 else "")
            return None


def _resume(result: dict, e: BackendError) -> dict:
    """La charge d'une conclusion refusée pour sa TAILLE : les valeurs SCALAIRES du
    résultat — compteurs d'usage et leurs ventilations, arrêt, pas, modèle —, sans
    ses conteneurs, et le refus borné. Une consommation connue reste connue : la
    retirer ferait lire zéro là où des jetons ont été payés. Le détail complet
    reste l'événement `resultat` du journal."""
    scalaires = {k: v[:_NOTE_MAX] if isinstance(v, str) else v for k, v in result.items()
                 if v is None or isinstance(v, (bool, int, float, str))}
    return {**scalaires, "conclusion_refusee": {"status": e.status, "code": getattr(e, "code", None),
                                                "erreur": str(e)[:_NOTE_MAX],
                                                "octets": len(json.dumps(result))}}


def en_echec(journal_, tenu: RunEnCours, job: dict, file,
             e: BaseException, modele_demande: str) -> None:
    """Ce qu'un travail MORT doit encore faire : clore son run en `failed` (ce
    qui libère la ligne qu'il tenait), le dire au journal (`resultat`), et rendre
    le travail à sa file en échec.

    L'événement `erreur` — la cause, avec sa pile — est écrit AVANT par
    l'appelant : `resultat` ne le remplace pas, il dit ce qu'on a FAIT de la
    mort. Un journal qui s'arrête sur `erreur` est un travail dont personne n'a
    rien rendu, et c'est précisément ce qu'on ne veut plus."""
    motif = f"{type(e).__name__} : {e}"
    cloture = clore(tenu, "failed", note=f"travail interrompu — {motif}"[:_NOTE_MAX],
                    job_id=job.get("id"))
    if journal_ is not None:
        journal_.ecrire("resultat", outcome="failed", run_id=tenu.run_id,
                        run_finish=cloture,
                        resultat={"stopped": "erreur", "type": type(e).__name__,
                                  "erreur": str(e)},
                        reponse=None, modele_demande=modele_demande,
                        modele_servi=None)
    try:
        # ⚠️ Le `run_id` part AVEC l'échec : c'est lui qui relie ce travail à ses
        # appels dans le journal d'org (le bilan attribue ses refus d'écriture
        # par run). Sans lui, les refus d'un travail mort n'appartenaient à
        # personne — et le bilan les comptait pour la flotte d'à côté.
        file.complete(job["id"], ok=False, error=str(e)[:_NOTE_MAX],
                      run_id=tenu.run_id)
    except BackendError as e2:
        # Bail déjà perdu (re-claimé ailleurs) : le job ne nous appartient plus,
        # on n'insiste pas.
        logger.warning("complete %s : %s", job.get("id"), e2)
