"""Le mode direct JOINT la procédure comme l'hébergé — ou refuse de partir, nommément.

Décision du 13/09/2026 : une seule copie de la procédure, jointe au travail, dans
les deux modes ; introuvable = erreur explicite avant exécution. Ce que ces bancs
tiennent :

1. la source est celle de l'hébergé : `GET /api/me/instructions/{slug}`, `body_md`
   BRUT, posé en `job.system` — et le worker l'encadre exactement comme un texte
   joint par la réservation ;
2. le descripteur dit ce qui a été servi (slug, version, empreinte, taille, rendu) ;
3. absente, archivée, vide ou illisible : `ProcedureIntrouvable` avec sa cause, et
   RIEN n'est lancé — ni compte de lignes, ni travail.
"""
from __future__ import annotations

import hashlib

import pytest

from oto_runner import consigne, direct
from oto_runner import worker as W
from oto_runner.backend import BackendError
from tests.test_fleet import _spec

CORPS = "# Passe F\n\nTu conclus la fiche.\n"


class _Source:
    """Ce que le passage direct lit du backend : la procédure, puis le tableau."""

    def __init__(self, reponse=None, erreur=None, restantes=0):
        self.reponse, self.erreur, self.restantes = reponse, erreur, restantes
        self.lectures: list = []

    def _get(self, chemin, params, org=None):
        self.lectures.append((chemin, org))
        if self.erreur:
            raise self.erreur
        return self.reponse

    def count_rows(self, namespace, filter=None, org=None):
        self.lectures.append(("count_rows", org))
        return self.restantes


def _instruction(**kw):
    return {"slug": "audiens-passef", "version": 18, "body_md": CORPS,
            "archived_at": None, **kw}


# ── 1 et 2. la procédure jointe et son descripteur ───────────────────────────

def test_la_procedure_d_org_est_jointe_brute_avec_son_descripteur():
    src = _Source(_instruction())
    j = consigne.jointe(src, "audiens-passef", 226)
    assert src.lectures == [("/api/me/instructions/audiens-passef", 226)]
    assert j["system"] == CORPS, "le corps BRUT, tel que l'hébergé le joint"
    assert j["descripteur"] == {
        "mode": "direct", "slug": "audiens-passef", "scope": "org", "org": 226,
        "version": 18, "sha256": hashlib.sha256(CORPS.encode()).hexdigest(),
        "caracteres": len(CORPS), "rendu": "brut"}


def test_le_travail_direct_porte_la_procedure_la_ou_la_reservation_la_poserait():
    j = consigne.jointe(_Source(_instruction()), "audiens-passef", 226)
    job = direct.travail(_spec(), direct.identifiant("S", 1), "oto_poste", j)
    assert job["system"] == CORPS and job["consigne"]["version"] == 18
    assert job["payload"] == direct.payload(_spec()), "le payload reste celui que la flotte enfile"
    cadre = W._spec_du_job(job).system
    assert "--- LA PROCÉDURE QUI FAIT AUTORITÉ ---\n" + CORPS.strip() in cadre, \
        "le worker l'encadre comme un texte joint par la réservation — un seul cadre"


# ── 3. introuvable : refus nommé, rien de lancé ──────────────────────────────

@pytest.mark.parametrize("source,cause", [
    (_Source(erreur=BackendError("/api/me/instructions/p → 404 : absente", status=404)), "absente"),
    (_Source(erreur=BackendError("/api/me/instructions/p → 500 : boum", status=500)), "illisible"),
    (_Source(erreur=BackendError("réseau : ReadTimeout", status=None)), "illisible"),
    (_Source(_instruction(archived_at="2026-09-10 12:00:00")), "archivee"),
    (_Source(_instruction(body_md="  \n")), "vide"),
    (_Source({}), "illisible"),
])
def test_une_procedure_qu_on_ne_peut_pas_joindre_est_refusee_avec_sa_cause(source, cause):
    with pytest.raises(consigne.ProcedureIntrouvable) as e:
        consigne.jointe(source, "p", 226)
    assert e.value.cause == cause
    assert "procedure_introuvable" in str(e.value) and "`p`" in str(e.value)


def test_le_passage_direct_ne_lance_rien_sans_sa_procedure(monkeypatch):
    monkeypatch.setattr(W, "_un_travail",
                        lambda *a, **k: pytest.fail("aucun travail ne doit partir"))
    src = _Source(erreur=BackendError("→ 404 : absente", status=404), restantes=5)
    with pytest.raises(consigne.ProcedureIntrouvable):
        direct.jouer(_spec(org=226), src, provider=None, jeton="t", plafond=3)
    assert src.lectures == [("/api/me/instructions/p", 226)], \
        "la procédure est lue AVANT tout : pas même un compte de lignes"


def test_tous_les_travaux_d_un_passage_portent_la_meme_version(monkeypatch):
    vus: list = []

    def espion(backend, job, provider, file=None):
        vus.append(job)
        file.complete(job["id"], ok=True, run_id="r", result={"usage_tokens": 1})

    monkeypatch.setattr(W, "_un_travail", espion)
    j = consigne.jointe(_Source(_instruction()), "audiens-passef", 226)
    direct.lancer(_spec(), type("T", (), {"count_rows": lambda *a, **k: 3})(), None,
                  jeton="t", stamp="S", plafond=3, k=1, jointe=j)
    assert len(vus) == 3
    assert {job["consigne"]["sha256"] for job in vus} == {j["descripteur"]["sha256"]}
