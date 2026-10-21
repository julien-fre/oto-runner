"""L'agent travaille sous l'identité du DEMANDEUR, pas sous celle du worker.

Le serveur remet à la réservation un jeton au nom de qui a programmé le travail,
borné à la durée du bail. ⚠️ **Un worker qui l'ignorerait écrirait tout au nom de
son propre compte — et rien ne le signalerait**, puisque les écritures
aboutiraient. C'est le défaut le plus silencieux de ce chemin : il ne produit
aucune erreur, seulement une attribution fausse.
"""
from __future__ import annotations

import pytest

from oto_runner import worker as W


class _Stop(RuntimeError):
    """Sentinelle : ce banc regarde la SESSION, pas le déroulé."""


class _FauxMcp:
    """Retient ce qu'on lui a passé — c'est le jeton qui nous intéresse."""

    dernier = None

    def __init__(self, **kw):
        _FauxMcp.dernier = kw
        self.kw = kw

    def outil(self, *a, **k):
        raise _Stop("ce banc s'arrête au premier appel d'outil")


def test_la_session_porte_le_jeton_du_demandeur(monkeypatch):
    """Le fait central : le jeton du travail gagne sur celui du worker."""
    monkeypatch.setattr(W, "McpSession", _FauxMcp)
    job = {"id": 7, "kind": "start", "payload": {"project_id": 3, "org_id": 42, "procedure": "veille"},
           "delegated_token": "oto_le-jeton-du-demandeur"}
    with pytest.raises(_Stop):      # on s'arrête au premier appel d'outil
        W._traiter(object(), job, object())
    assert _FauxMcp.dernier["token"] == "oto_le-jeton-du-demandeur"


def test_sans_jeton_delegue_le_worker_REFUSE(monkeypatch):
    """⚠️ Renversement assumé du 05/09/2026. Ce banc disait : « travaux d'avant la
    délégation : pas de porteur connu, donc pas de jeton. On ne casse pas — on
    repasse par le comportement d'avant. »

    Ce qui a changé n'est pas le cas, c'est le MODÈLE. Le worker est un serveur
    de boucles agentiques qui impersonnent chacune leur user ; il n'a aucune
    identité métier à prêter. « Repasser par le comportement d'avant », c'était
    faire agir un agent au nom du compte qui héberge le runner — et rien ne le
    signalait, puisque les écritures aboutissaient.
    """
    def _interdit(**kw):
        pytest.fail("une session a été ouverte sans personne à impersonner")

    monkeypatch.setattr(W, "McpSession", _interdit)
    job = {"id": 8, "kind": "start",
           "payload": {"project_id": 3, "org_id": 42, "procedure": "veille"}}
    with pytest.raises(W.SansPorteur) as e:
        W._traiter(object(), job, object())
    assert "personne à impersonner" in str(e.value)
    assert "reprogramme" in str(e.value), "un refus nomme la sortie, pas que le manque"


def test_un_refus_d_identite_arrete_le_travail_sans_le_toucher(monkeypatch):
    """⚠️ Deux choses à la fois : on lève AVANT d'ouvrir quoi que ce soit, et on
    ne conclut pas le travail — le serveur l'a déjà marqué en échec avec sa
    raison. Le conclure ici rendrait une erreur de bail, qui accuserait la file
    alors que le problème est un DROIT."""
    def _interdit(**kw):
        pytest.fail("une session a été ouverte malgré un refus d'identité")

    monkeypatch.setattr(W, "McpSession", _interdit)
    job = {"id": 9, "kind": "start", "payload": {},
           "delegation_refusee": "le compte qui a programmé ce travail n'existe plus"}
    with pytest.raises(W.IdentiteInvalide) as e:
        W._traiter(object(), job, object())
    assert "n'existe plus" in str(e.value), "la RAISON doit remonter, pas juste l'échec"


def test_un_refus_du_serveur_se_journalise_sans_en_presumer_la_cause(caplog):
    """Le refus à la réservation n'est pas toujours un droit perdu : ici, une clé de
    modèle absente. Le message disait « rendre son droit au demandeur, ou le
    reprogrammer sous une autre identité » — une cause présumée, fausse dans ce cas.
    Il dit ce qui s'est passé et rend la raison du serveur, telle quelle."""
    import logging
    from tests.test_worker_reprise import FauxBackend, _job
    raison = "ce travail demande un modèle `anthropic` — Dépose une clé `anthropic`."
    job = {**_job("start"), "delegation_refusee": raison}
    with caplog.at_level(logging.ERROR):
        W._un_travail(FauxBackend(), job, object())
    (message,) = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert "NON exécuté" in message and "refusé par le serveur" in message
    assert raison in message, "la RAISON du serveur remonte, entière"
    assert "droit" not in message and "identité" not in message, message
