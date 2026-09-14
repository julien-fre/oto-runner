"""Un redéploiement de la plateforme ne tue pas la flotte — ÉPROUVÉ, pas espéré.

`test_arret_gracieux` couvre l'autre moitié : l'agent qui reçoit un signal et
finit son travail avant de sortir. Ici, personne ne demande rien à l'agent —
**c'est la plateforme qui disparaît sous lui**, quelques secondes, le temps d'une
bascule de couleur.

⚠️ **Le principe : c'est la PRODUCTION qui doit résister, pas les livraisons qui
doivent s'arrêter.** Un dispositif qui protégerait un passage en interdisant de
déployer déplace son coût sur tout le monde et finit contourné par ceux qu'il
gêne — il a été retiré le 01/09/2026 pour cette raison. Ce qui le remplace est
une propriété du traitement, et une propriété se teste.

Ce que ces tests exigent, dans l'ordre de ce qui coûterait le plus cher :

1. **L'agent SURVIT à une plateforme injoignable** — il ne sort pas, il attend et
   reprend. Un agent qui meurt sur une coupure de trois secondes vide la flotte à
   chaque déploiement, et personne ne le voit avant le bilan.
2. **Il reprend le travail dès que la plateforme revient** — survivre sans
   reprendre serait une flotte vivante et inerte, ce qui est pire : les journaux
   disent « actif ».
3. **Une coupure PENDANT un traitement ne tue pas l'agent** — le travail est rendu
   en échec et sera rejoué par la file ; c'est le job qui repaie, pas la batterie
   qui s'éteint.
4. **Une plateforme qui ne revient jamais laisse l'agent VIVANT** — il n'a aucune
   raison de décider qu'il est mort. C'est systemd qui l'arrête, pas lui.
"""
import oto_runner.worker as worker
from oto_runner.backend import BackendError


def _reset():
    worker._arret_demande = False


class _PlateformeQuiBascule:
    """Une plateforme indisponible pendant `coupures` réservations, puis revenue.

    C'est la forme exacte d'un redéploiement : quelques appels qui échouent net,
    puis le service qui répond de nouveau — sans que rien n'ait prévenu l'agent.
    """

    def __init__(self, coupures: int, jobs: int = 2):
        self.coupures, self.restants = coupures, jobs
        self.claims = self.echecs = 0
        self.completes: list[tuple] = []
        self.base = "https://exemple.invalide"

    def claim(self, lease_seconds=600, **_):
        self.claims += 1
        if self.echecs < self.coupures:
            self.echecs += 1
            raise BackendError("502 Bad Gateway — bascule de couleur en cours")
        if self.restants <= 0:
            worker._demander_arret(15, None)   # file vide : on fait finir le test
            return None
        self.restants -= 1
        return {"id": 200 + self.restants, "kind": "start", "attempts": 1}   # la forme servie

    def complete(self, job_id, ok=True, error=None, run_id=None, result=None):
        # La signature EST celle de `FileDeTravail` : un travail en échec rend
        # aussi son `run_id` (c'est ce qui relie ses refus d'écriture au bilan).
        self.completes.append((job_id, ok))


class _Provider:
    ONE_SHOT = True
    __name__ = "agent_conversations"

    @staticmethod
    def resolve_key():
        return "x"

    @staticmethod
    def model():
        return "un-modele"


def _lancer(monkeypatch, backend, traites, sur_traitement=None):
    monkeypatch.setenv("OTO_RUNNER_ARMED", "1")
    monkeypatch.setenv("OTO_WORKER_SECRET", "otow_banc")
    monkeypatch.setattr(worker, "Backend", lambda *a, **kw: backend)
    monkeypatch.setattr(worker, "get_provider", lambda: _Provider)
    monkeypatch.setattr(worker, "_POLL_S", 0)

    def _traiter(bk, job, prov, **kw):   # journal_, file : la couture du mode direct
        traites.append(job["id"])
        if sur_traitement:
            sur_traitement(job)

    monkeypatch.setattr(worker, "_traiter", _traiter)
    worker.main()


# ── ① survivre, ② reprendre ──────────────────────────────────────────────────

def test_l_agent_survit_a_une_plateforme_injoignable_et_reprend(monkeypatch):
    """Trois réservations qui échouent net, puis le service revient.

    L'agent ne doit ni sortir ni perdre de travail : il attend, réessaie, et
    traite la file dès qu'elle est de nouveau atteignable."""
    _reset()
    b, traites = _PlateformeQuiBascule(coupures=3, jobs=2), []
    _lancer(monkeypatch, b, traites)
    assert b.echecs == 3, "les trois coupures ont bien été rencontrées"
    assert traites == [201, 200], "la file est traitée EN ENTIER après la bascule"


def test_une_coupure_longue_ne_fait_pas_sortir_l_agent(monkeypatch):
    """Vingt appels dans le vide : un agent n'a aucune raison de décider qu'il est
    mort. C'est systemd qui l'arrête, pas lui."""
    _reset()
    b, traites = _PlateformeQuiBascule(coupures=20, jobs=1), []
    _lancer(monkeypatch, b, traites)
    assert b.echecs == 20
    assert traites == [200], "le travail est fait, après vingt échecs de suite"


# ── ③ une coupure PENDANT un traitement ──────────────────────────────────────

def test_une_coupure_pendant_un_traitement_ne_tue_pas_la_batterie(monkeypatch):
    """La plateforme tombe pendant que l'agent travaille : le travail échoue et
    sera rejoué par la file, mais **la batterie ne s'éteint pas**.

    C'est le job qui repaie, jamais l'agent qui meurt — sans quoi un déploiement
    viderait la flotte au lieu de coûter un rejeu."""
    _reset()
    b, traites = _PlateformeQuiBascule(coupures=0, jobs=3), []

    def _tombe_sur_le_premier(job):
        if job["id"] == 202:
            raise BackendError("503 — la plateforme redéploie pendant le travail")

    _lancer(monkeypatch, b, traites, sur_traitement=_tombe_sur_le_premier)
    assert traites == [202, 201, 200], "les DEUX suivants ont été traités"
    assert (202, False) in b.completes, "le travail interrompu est rendu en échec"
    assert len(b.completes) == 1, (
        "seul le travail INTERROMPU est conclu par la boucle — les autres se "
        f"concluent dans `_traiter`, que ce test remplace. Reçu : {b.completes}")


# ── ④ ce que l'agent ne doit PAS faire ───────────────────────────────────────

def test_l_agent_ne_confond_pas_plateforme_absente_et_file_vide(monkeypatch):
    """Une plateforme injoignable n'est pas une file vide.

    ⚠️ Si l'agent traitait un refus réseau comme « plus rien à faire », il
    sortirait sur un déploiement — et le bilan dirait « flotte terminée » pour une
    flotte qui n'a rien fait. Le pire des deux mondes : silencieux et faux."""
    _reset()
    b, traites = _PlateformeQuiBascule(coupures=5, jobs=1), []
    _lancer(monkeypatch, b, traites)
    # l'agent n'est sorti QUE sur file vide (le drapeau posé par `claim`), et
    # après avoir fait son travail — pas sur les cinq erreurs.
    assert traites == [200]
    assert b.claims == 5 + 1 + 1, "5 échecs, 1 travail, 1 file vide"
