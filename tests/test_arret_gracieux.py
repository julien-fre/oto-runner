"""L'arrêt gracieux d'un agent : au signal, ne plus réserver, finir, sortir.

Sans ça, un déploiement tue les agents en plein travail — vécu le 28/08 sur une
campagne de 100 lignes : trois traitements tués, repris seize minutes plus tard par
expiration de bail. Rien n'a été perdu, la reprise étant le design, mais la
protection n'était qu'une DISCIPLINE (« ne pas déployer pendant une campagne ») au
lieu d'une propriété du système.

⚠️ Ce que ces tests protègent en priorité : que le signal **n'interrompe jamais** le
travail en cours. Un agent tué au milieu laisse sa ligne sous bail et fait repayer
le job — c'est exactement ce qu'on corrige, et un correctif qui le referait serait
pire que rien.
"""
import oto_runner.worker as worker


def _reset():
    worker._arret_demande = False


class _BackendFactice:
    """Rend un job à chaque réservation, et note ce qui a été traité."""

    def __init__(self, jobs=3):
        self.restants = jobs
        self.claims = 0
        self.base = "https://exemple.invalide"

    def claim(self, lease_seconds=600, **_):
        self.claims += 1
        if self.restants <= 0:
            # File vide : en vrai l'agent attendrait indéfiniment, ce qui est le
            # BON comportement — ici on lève le drapeau pour que le test finisse.
            worker._demander_arret(15, None)
            return None
        self.restants -= 1
        return {"id": 100 + self.restants, "kind": "start", "attempts": 1}   # la forme servie

    def complete(self, *a, **kw):
        pass


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


def test_le_signal_leve_un_drapeau_sans_rien_interrompre():
    _reset()
    worker._demander_arret(15, None)
    assert worker._arret_demande is True
    # Un second signal ne change rien : c'est systemd qui tuera passé sa patience,
    # pas nous qui abrégerions le travail en cours.
    worker._demander_arret(15, None)
    assert worker._arret_demande is True


def test_le_travail_en_cours_va_a_son_terme_puis_l_agent_sort(monkeypatch):
    """LE cas qui compte : le signal tombe PENDANT un traitement. Celui-ci doit
    finir, et aucun autre ne doit être réservé."""
    _reset()
    b, traites = _BackendFactice(jobs=5), []
    _lancer(monkeypatch, b, traites,
            sur_traitement=lambda job: worker._demander_arret(15, None))
    assert traites == [104], "un seul travail traité, et il est allé à son terme"
    assert b.restants == 4, "aucun autre travail n'a été réservé"


def test_sans_signal_l_agent_traite_toute_la_file(monkeypatch):
    _reset()
    b, traites = _BackendFactice(jobs=3), []
    _lancer(monkeypatch, b, traites)
    assert traites == [102, 101, 100]




def test_un_signal_pendant_la_reservation_rend_la_ligne_sans_l_entamer(monkeypatch):
    """Entre la décision de réserver et le retour du backend, le signal peut
    arriver. Rendre la ligne tout de suite vaut mieux que la garder sous bail
    pendant que l'agent s'éteint."""
    _reset()

    class _BackendQuiSignale(_BackendFactice):
        def claim(self, lease_seconds=600, **_):
            job = super().claim(lease_seconds)
            worker._demander_arret(15, None)   # le signal arrive PENDANT le claim
            return job

    b, traites = _BackendQuiSignale(jobs=5), []
    _lancer(monkeypatch, b, traites)
    assert traites == [], "le travail réservé n'a pas été entamé"
