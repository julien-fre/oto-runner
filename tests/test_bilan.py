"""Le BILAN d'une flotte — le pilotage se PROUVE, il ne se raconte pas.

Ce que le banc fige : « aboutie » se mesure au TABLEAU (départ − restantes) et
jamais au nombre de jobs conclus ; les faux départs se comptent même quand le
connecteur MCP préfixe les noms d'outils ; le coût par ligne aboutie vaut null
quand rien n'a abouti (jamais une division qui invente un chiffre) ; le JSON se
pose à côté de la déclaration et se réécrit ; et le bilan de FIN tombe quelle
que soit la borne — panne et interruption comprises, parce que c'est
précisément ces jours-là qu'on le lit.
"""
from __future__ import annotations

import json
import logging

import pytest

from oto_runner.bilan import ecrire_bilan
from oto_runner.fleet import run_fleet
from tests.test_fleet import FauxBackend, Horloge, _spec


class BackendBilan:
    """Le backend vu du bilan : un compte de lignes, un journal d'appels."""

    def __init__(self, restantes=0, journal=(0, 0)):
        self.restantes = restantes
        self.journal = journal
        self.sondes: list = []

    def count_rows(self, namespace, filter=None, org=None):
        return self.restantes

    def rows(self, namespace, filter=None, org=None, limit=200):
        return getattr(self, "lignes", [])

    def tool_health(self, org, tool, *, minutes=15, limit=20):
        self.sondes.append((tool, minutes, limit))
        return self.journal


def _job(statut="done", jetons=1000, **outils):
    """Un job conclu et son résultat DÉCLARÉ (les outils sont des kwargs :
    `_job(data_write=1)`, ou préfixés par le connecteur)."""
    return {"status": statut,
            "result": {"usage_tokens": jetons, "tool_counts": dict(outils)}}


def test_les_sorties_se_lisent_au_tableau():
    """Une ligne SORTIE = une ligne qui ne correspond PLUS au filtre de
    réservation. Compter les jobs « done » comptait les tours perdus.

    ⚠️ Ce poste s'appelait « abouties » : il comptait autant une ligne abandonnée
    par la plateforme qu'une ligne enrichie. L'issue se lit au statut (cf.
    `test_bilan_statuts.py`) ; sans schéma lisible, elle est OMISE avec sa
    raison — jamais confondue avec la sortie de file."""
    b = BackendBilan(restantes=18)
    bilan = ecrire_bilan(_spec(), b, {1: _job(), 2: _job()},
                         lignes_initiales=30, secondes=120)
    lignes = bilan["lignes"]
    assert (lignes["depart"], lignes["restantes"], lignes["sorties"]) == (30, 18, 12)
    assert lignes["abouties"] is None and "schéma" in lignes["abouties_omis"]
    assert bilan["jobs"]["termines"] == 2, "2 jobs conclus, 12 lignes sorties"




def test_le_cout_par_ligne_vaut_null_quand_rien_nest_sorti():
    """Zéro sortie : le coût par ligne n'existe pas — null, jamais zéro ni
    une division. Le total, lui, reste dû (il a été payé)."""
    b = BackendBilan(restantes=30)
    bilan = ecrire_bilan(_spec(), b, {1: _job(jetons=4000), 2: _job(jetons=2000)},
                         lignes_initiales=30, secondes=60)
    assert bilan["lignes"]["sorties"] == 0
    assert bilan["jetons"] == {"total": 6000, "connus": 6000, "travaux_sans_usage": 0,
                               "par_job": 3000, "par_sortie": None, "par_aboutie": None}






def test_un_job_non_conclu_leve():
    """Compter un job en vol, ce serait compter du vide : on lève."""
    with pytest.raises(ValueError, match="n'est pas conclu"):
        ecrire_bilan(_spec(), BackendBilan(), {7: {"status": "claimed"}},
                     lignes_initiales=1, secondes=1)




def test_sans_org_le_poste_des_refus_est_omis_explicitement():
    """Un poste absent dit POURQUOI il l'est — il ne se confond jamais avec un
    zéro refus."""
    bilan = ecrire_bilan(_spec(), BackendBilan(restantes=0), {},
                         lignes_initiales=1, secondes=60)
    assert bilan["refus_ecriture"] is None
    assert "sans org" in bilan["refus_ecriture_omis"]


def test_un_backend_muet_ne_fabrique_pas_de_chiffre():
    """Le bilan de fin tombe souvent APRÈS une panne : un compte illisible
    devient null assumé, jamais un chiffre inventé, et rien ne remonte."""
    class Muet(BackendBilan):
        def count_rows(self, namespace, filter=None, org=None):
            raise RuntimeError("backend mort")

    bilan = ecrire_bilan(_spec(), Muet(), {1: _job()}, lignes_initiales=30,
                         secondes=60, arret="backend indisponible")
    assert bilan["lignes"]["restantes"] is None
    assert bilan["lignes"]["sorties"] is None
    assert bilan["jetons"]["par_sortie"] is None
    assert bilan["jetons"]["par_aboutie"] is None






@pytest.mark.parametrize("scenario,attendu", [
    (dict(counts=[2, 2, 0, 0]), "file vide"),
    (dict(counts=[100, 100], statut="failed"), "échecs consécutifs"),
])
def test_le_bilan_de_fin_tombe_quelle_que_soit_la_borne(tmp_path, scenario, attendu):
    """Normale ou ANORMALE, la borne rend son bilan : un pilotage qui n'existe
    que sur une sortie propre n'existe pas les jours où il sert."""
    decl = tmp_path / "flotte.yaml"
    decl.write_text("procedure: p\n")
    h = Horloge()
    bilan = run_fleet(_spec(ramp_seconds=0, source=str(decl)),
                      FauxBackend(**scenario), sleep=h.sleep, clock=h.clock,
                      poll_s=20)
    assert attendu in bilan.arret
    pose = json.loads((tmp_path / "flotte.bilan.json").read_text())
    assert pose["final"] is True and attendu in pose["arret"]


def test_une_flotte_interrompue_rend_quand_meme_son_bilan(tmp_path):
    """Une panne qui n'est pas une BackendError traverse le driver : le bilan
    tombe AVANT que la pile remonte, et dit « interrompu »."""
    decl = tmp_path / "flotte.yaml"
    decl.write_text("procedure: p\n")

    class BackendQuiExplose(FauxBackend):
        def count_rows(self, namespace, filter=None, org=None):
            if self.counts:
                return self.counts.pop(0)
            raise RuntimeError("panne hors contrat")

    h = Horloge()
    with pytest.raises(RuntimeError, match="panne hors contrat"):
        run_fleet(_spec(ramp_seconds=0, source=str(decl)),
                  BackendQuiExplose(counts=[100]), sleep=h.sleep, clock=h.clock,
                  poll_s=20)
    pose = json.loads((tmp_path / "flotte.bilan.json").read_text())
    assert pose["arret"] == "interrompu" and pose["lignes"]["restantes"] is None


def test_le_bilan_intermediaire_tombe_a_la_cadence_declaree(monkeypatch):
    """Une campagne se pilote PENDANT qu'elle tourne : le bilan retombe toutes
    les `bilan_periode_s`, pas seulement à la fin."""
    from oto_runner import fleet as F

    appels: list = []
    monkeypatch.setattr(F, "ecrire_bilan",
                        lambda *a, **kw: appels.append(kw.get("arret", "")))
    h = Horloge()
    run_fleet(_spec(ramp_seconds=0, bilan_periode_s=20), FauxBackend(
        counts=[10, 10, 10, 10, 0, 0]), sleep=h.sleep, clock=h.clock, poll_s=20)
    assert [a for a in appels if not a], "aucun bilan intermédiaire"
    assert appels[-1] == "file vide", "le dernier bilan est celui de la fin"


def test_une_flotte_sans_declaration_sur_disque_ne_pose_rien(tmp_path):
    """Une flotte construite en mémoire n'a pas de déclaration : elle
    journalise son bilan et ne pose aucun fichier."""
    ecrire_bilan(_spec(), BackendBilan(restantes=0), {}, lignes_initiales=1,
                 secondes=60)
    assert list(tmp_path.iterdir()) == []


# ── Les refus : ENTIERS, groupés par texte identique — plus aucun classificateur.
# ⚠️ Il y avait ici quatre tests qui épinglaient une table de fragments rangeant
# les refus sous des libellés inventés (« création refusée par le cran », « ligne
# inconnue »). Relue contre les journaux serveur le 06/09/2026, elle inventait.
# Les bancs des refus vivent dans `test_bilan_statuts.py`.


# ── L'extinction se prouve par un ACTE D'EXTINCTION, pas par une date ───────
# ⚠️ Vécu deux fois sur la MÊME fiche : le contrôle acceptait « une date complète
# ou un mot de registre », et une fiche déclarée éteinte a été validée sur un
# DÉPÔT DE COMPTES daté publié au BODACC — un acte d'ACTIVITÉ retourné en preuve
# d'extinction.

def _fiche_eteinte(motif):
    return {"siren": "111", "qualification": "dormante_ou_introuvable",
            "qualification_motif": motif, "motif_ecartement": ""}


def _eteintes_sans_acte(motif):
    """Rend le nombre de fiches éteintes SANS acte d'extinction retenu."""
    from oto_runner.bilan import controler_fiches
    spec, backend = _spec(org=226), BackendBilan(restantes=0)
    backend.lignes = [_fiche_eteinte(motif)]
    return len(controler_fiches(spec, backend, {})["fiches_contradictoires"])


def test_un_depot_de_comptes_date_ne_prouve_PAS_une_extinction():
    """LE cas qui a fait rater le quatrième passage : la preuve citée était un
    dépôt de comptes — donc une preuve d'ACTIVITÉ."""
    assert _eteintes_sans_acte(
        "Dépôt des comptes annuels du 12/03/2024, publié au BODACC n°2024-0451.") == 1


def test_un_acte_d_extinction_nomme_est_accepte():
    for motif in ("Radiation du RCS le 04/06/2013 (BODACC).",
                  "Jugement de clôture de liquidation du 08/11/2016.",
                  "Cessation d'activité déclarée, dissolution en 2019.",
                  "Reprise par la société X, fusion-absorption."):
        assert _eteintes_sans_acte(motif) == 0, motif


def test_une_accumulation_d_absences_ne_prouve_rien():
    """« Aucun dépôt depuis 2016 » date une ABSENCE, pas un acte."""
    assert _eteintes_sans_acte(
        "Aucun dépôt de comptes depuis 2016, aucun salarié, aucune trace web.") == 1


# ── Combien d'agents ont tourné, et non combien étaient déclarés ─────────────

def test_le_bilan_enregistre_la_concurrence_REELLE():
    """Sans ce nombre, deux bilans ne sont pas comparables : durée, débit et
    taux d'échec en dépendent tous.

    Il a manqué le 07/09/2026 — il a fallu relire 72 journaux pour établir que
    toutes les vagues avaient tourné à UN agent, alors que les déclarations en
    annonçaient quatre. Le `concurrency` d'une déclaration n'est lu QUE par le
    mode flotte ; le mode direct ne suit que son option de ligne de commande.
    D'où la règle : on écrit ce qui a EU LIEU, jamais ce qui était demandé."""
    bilan = ecrire_bilan(_spec(), BackendBilan(restantes=0), {1: _job()},
                         lignes_initiales=1, secondes=10, agents=3)
    assert bilan["agents"] == 3


def test_un_bilan_sans_concurrence_connue_le_dit_au_lieu_de_supposer():
    """`None`, pas 1 : un appelant qui ne renseigne rien ne prouve pas qu'un
    seul agent a tourné — et un défaut plausible se lirait comme une mesure."""
    bilan = ecrire_bilan(_spec(), BackendBilan(restantes=0), {1: _job()},
                         lignes_initiales=1, secondes=10)
    assert bilan["agents"] is None
