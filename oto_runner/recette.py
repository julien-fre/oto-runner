"""Le travail RECETTE : jouer une recette publiée d'une org, sans modèle.

Un déclencheur peut porter une recette au lieu d'une procédure et d'un modèle
(oto-backend, `runner_triggers.recipe`) : à l'horloge, au webhook ou à la demande,
le backend enfile un travail `start` dont la charge porte `recipe` et la famille
`recette`. Seul un worker qui réserve avec le dépôt `recette` le voit
(`OTO_RUNNER_DEPOT=recette`) : un worker de modèle ne le réserve jamais, sa famille
ne correspond à aucun dépôt de clé.

Le travail ne lance AUCUNE boucle agentique. Il :
1. refuse d'être REJOUÉ — une recette qui a commencé a pu écrire et envoyer ; la
   relancer depuis le début doublerait ses écritures (`recipe_job_not_replayed`) ;
2. prolonge son bail à la durée d'une recette (`BAIL_S`) ;
3. ouvre un run technique (`run_start`), le lie au travail (`bind_run`) ;
4. appelle `POST /api/me/recipes {op: "run"}` avec le jeton DÉLÉGUÉ, l'org du
   travail (`X-Oto-Org`) et le run (`X-Oto-Run`, oto#227) en en-têtes : la recette
   tourne côté serveur, sous l'identité de qui a posé le déclencheur, et ses appels
   se rangent sous ce run ;
5. lit le reçu : rien d'arrêté = `done`, un arrêt (borne, page max, reprise
   possible) = `partial` ; un refus HTTP = `failed`, son code nommé ;
6. clôt le run (`run_finish`, une ligne de comptes) et conclut le travail
   (`complete`, `result = {recipe_receipt, outcome}`).

⚠️ La recette s'exécute dans le BACKEND (le même moteur que `oto_recipe op=run`),
pas ici : ce worker n'est qu'un exécutant de file qui porte une identité.
"""
from __future__ import annotations

import os
from typing import Optional

from . import conclusion
from .backend import BackendError

#: Le dépôt que déclare un worker de recettes à la réservation (contrat §3) : la
#: famille des travaux de recette, que seul ce worker voit.
DEPOT = "recette"
#: L'environnement qui met le worker en mode recettes.
ENV_DEPOT = "OTO_RUNNER_DEPOT"
#: Le bail pendant une recette, et l'échéance murale de son appel : une recette-code
#: est bornée à 600 s côté serveur (`code.MAX_SECONDES`), plus la grâce du pont.
BAIL_S = 900
ECHEANCE_S = 900
#: Les arrêts qui ne sont PAS une borne : la recette est allée au bout.
_FINS_NATURELLES = (None, "", "end", "end_of_results", "done")


def mode_recette() -> bool:
    return (os.environ.get(ENV_DEPOT) or "").strip().lower() == DEPOT


# ── le « provider » d'un worker de recettes ─────────────────────────────────────
# Le worker passe un provider partout (journal, conclusion, boucle de réservation) :
# celui-ci ne sert aucun modèle et ne tient aucune clé. Il dit seulement qui il est.
def model() -> str:
    return DEPOT


def depot() -> str:
    return DEPOT


class RecetteNonRejouee(RuntimeError):
    """`recipe_job_not_replayed` : un travail de recette re-réservé n'est jamais rejoué."""


class RecetteRefusee(RuntimeError):
    """La route a refusé la recette (HTTP ≥ 400, ou transport) : son code nommé."""


class RecetteHorsMode(RuntimeError):
    """Un travail de recette réservé par un worker de MODÈLE — il n'aurait jamais dû
    lui être servi (sa famille ne correspond à aucun dépôt de clé)."""


class TravailNonRecette(RuntimeError):
    """Un travail SANS recette réservé par un worker de recettes."""


def est_recette(job: dict) -> bool:
    return bool((job.get("payload") or {}).get("recipe"))


#: Arrêts qui disent que la recette n'a PAS fait son travail, même sans `error` posé :
#: la forme des données a dérivé (le moteur refuse d'écrire des colonnes vides).
_ARRETS_EN_ECHEC = frozenset({"mapping_drift"})


def issue_du_recu(recu: Optional[dict]) -> str:
    """`done` quand la recette est allée au bout ; `failed` quand le reçu porte une
    ERREUR (outil refusé, arguments invalides, échec répété…) ou un arrêt d'échec ;
    `partial` sur une BORNE (pages, budget, temps : elle reprendra). Vu en e2e local
    (28/11/2026) : `invalid_input` non rejouable sortait `partial`."""
    recu = recu or {}
    arret = recu.get("stopped")
    if arret in _FINS_NATURELLES:
        return "done"
    if recu.get("error") or arret in _ARRETS_EN_ECHEC:
        return "failed"
    return "partial"


def ligne_de_comptes(recu: Optional[dict], issue: str) -> str:
    """La note du `run_finish` : l'issue et les COMPTES du reçu, une ligne."""
    recu = recu or {}
    comptes = [f"{k} {v}" for k, v in recu.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    arret = recu.get("stopped")
    morceaux = [f"recette {issue}"] + comptes[:8] + ([f"arrêt : {arret}"] if arret else [])
    return " · ".join(morceaux)[:conclusion._NOTE_MAX]


def traiter(backend, job: dict, file, note, tenu, mcp_factory) -> None:
    """UN travail de recette, de la garde de reprise à la conclusion.

    Les échecs LÈVENT : `worker._un_travail` les rattrape et conclut en échec
    (`conclusion.en_echec` : run clos `failed`, travail rendu en échec, motif nommé).
    """
    p = job.get("payload") or {}
    recette = p.get("recipe") or {}
    jeton = job.get("delegated_token")
    org = p.get("org_id")
    # 1. Jamais rejoué. Un run déjà lié (une tentative précédente l'a ouvert) se clôt
    #    en échec par le chemin commun : on ouvre la session pour pouvoir le fermer.
    if (job.get("attempts") or 1) > 1 or job.get("run_id"):
        if job.get("run_id") and jeton:
            tenu.mcp = mcp_factory(project=p.get("project_id"), org=org, token=jeton)
            tenu.run_id = job["run_id"]
        raise RecetteNonRejouee(
            f"recipe_job_not_replayed: le travail {job.get('id')} (recette "
            f"`{recette.get('slug')}`) a déjà été réservé une fois — une recette n'est "
            "jamais rejouée, ses écritures ont pu être faites. L'échéance suivante, ou "
            "« Exécuter maintenant », la relancera.")
    if not jeton:
        from .worker import SansPorteur
        raise SansPorteur(
            f"le travail {job.get('id')} n'a pas de jeton délégué : personne à "
            "impersonner. Le worker n'a pas d'identité métier à prêter — "
            "reprogramme-le, il partira au nom de qui le demande.")
    # 2. Le bail couvre la recette entière.
    file.extend(job["id"], lease_seconds=BAIL_S)
    # 3. Un run technique, lié au travail.
    mcp = tenu.mcp = mcp_factory(project=p.get("project_id"), org=org, token=jeton)
    d = mcp.outil("run_start", {"label": p.get("label") or f"recette {recette.get('slug')}"})
    run_id = tenu.run_id = d.get("run_id")
    if not run_id:
        raise RuntimeError(f"run_start sans run_id : réponse dégradée {str(d)[:200]}")
    file.bind_run(job["id"], run_id)
    note("run", run_id=run_id, repris=False, recette=recette.get("slug"),
         version=recette.get("version"))
    # 4. La recette, côté serveur, sous l'identité du demandeur.
    corps = {"op": "run", "slug": recette.get("slug"), "scope": recette.get("scope") or "org"}
    for cle in ("version", "params", "datastore"):
        if recette.get(cle) is not None:
            corps[cle] = recette[cle]
    try:
        reponse = backend.run_recipe(corps, token=jeton, org=org, run=run_id,
                                     wall_s=ECHEANCE_S)
    except BackendError as e:
        raise RecetteRefusee(f"{e.code or 'recipe_run_failed'}: {e}") from e
    recu = reponse.get("receipt") if isinstance(reponse, dict) else None
    # 5–6. L'issue, la clôture, la conclusion.
    issue = issue_du_recu(recu)
    ligne = ligne_de_comptes(recu, issue)
    cloture = conclusion.clore(tenu, issue, note=ligne, job_id=job.get("id"))
    resultat = {"recipe_receipt": recu, "outcome": issue}
    note("resultat", outcome=issue, run_id=run_id, run_finish=cloture, resultat=resultat)
    # Un reçu en échec conclut le travail en échec (alertes, historique) — avec l'erreur
    # du reçu, jamais un succès qui la cacherait.
    erreur = None
    if issue == "failed":
        erreur = f"{(recu or {}).get('stopped') or 'recipe_failed'}: {(recu or {}).get('error') or ''}"[:conclusion._NOTE_MAX]
    file.complete(job["id"], ok=issue != "failed", error=erreur, run_id=run_id, result=resultat)
