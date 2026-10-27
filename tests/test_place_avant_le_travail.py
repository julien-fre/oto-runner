"""La place sur la box AVANT le travail (claude-sandbox-manager, file des places) : le
worker attend son tour sur la ferme sans tenir de travail, puis son run consomme la place
réservée. Plus de travail pris puis refusé en 429, plus de tentative perdue à une box pleine."""
from __future__ import annotations

import logging
import time
import types

import pytest

import oto_runner.worker as worker
from oto_runner import agent_abonnement as A
from oto_runner import agent_ferme as F

from test_agent_ferme import FIN, INIT, MCP, Reponse, ferme  # noqa: F401 — la fixture


@pytest.fixture(autouse=True)
def _sans_place():
    A._place = None
    worker._arret_demande = False
    yield
    A._arreter_entretien()
    A._place = None
    worker._arret_demande = False


def test_la_place_part_avec_le_run_et_une_seule_fois(ferme):
    ferme["reponses"] += [Reponse(200, [INIT, FIN]), Reponse(200, [INIT, FIN])]
    A.tenir_place("p" * 32)
    base = dict(instructions="i", inputs="go", tools=[], api_key="sk-ant-api03-org",
                modele="claude-sonnet-5", mcp=MCP, org=4242, attendre=lambda s: None)
    F.run_once(**base)
    F.run_once(**base)
    assert ferme["post"][0][1]["place"] == "p" * 32
    assert "place" not in ferme["post"][1][1]          # consommée par le premier run


def test_une_place_echue_retombe_sur_les_essais_d_avant(ferme):
    ferme["reponses"] += [Reponse(429), Reponse(200, [INIT, FIN])]
    A.tenir_place("p" * 32)
    res = F.run_once(instructions="i", inputs="go", tools=[], api_key="sk-ant-api03-org",
                     modele="claude-sonnet-5", mcp=MCP, org=4242, attendre=lambda s: None)
    assert res.stopped == "end_turn" and len(ferme["post"]) == 2


@pytest.mark.parametrize("prolongee, pretee", [(True, True), (None, False), (False, False)])
def test_la_sonde_du_forfait_emprunte_la_place_que_la_ferme_sait_garder(
        monkeypatch, prolongee, pretee):
    """Ferme qui sait garder (prolonger répond) : la sonde tourne SUR la place du travail,
    `garder_place` — sur une box pleine, elle ne bute plus sur un 429 pendant que la place
    échoit. Ferme antérieure (404) ou place perdue : la sonde fait comme avant."""
    vu = {}

    def poster(url, corps, entetes, lecture, **kw):
        vu.update(place=kw.get("place"), garder=corps.get("garder_place"))
        return Reponse(200, [])
    monkeypatch.setenv("OTO_FERME_URL", "http://ferme")
    monkeypatch.setenv("OTO_FERME_TOKEN", "t")
    monkeypatch.setattr(A, "poster_le_run", poster)
    monkeypatch.setattr(A, "prolonger_place", lambda jeton: prolongee)
    monkeypatch.setattr(A, "_PROLONGER_PLACE_S", 3600)
    A.tenir_place("p" * 32)
    A.verifier_forfait("u1")
    assert vu == ({"place": "p" * 32, "garder": True} if pretee
                  else {"place": None, "garder": None})
    assert A._place == "p" * 32                     # toujours au travail, pour son run


def test_la_place_tenue_est_prolongee_jusqu_au_run(monkeypatch):
    """Un pré-run plus long que la tenue (30 s) : la place est prolongée tant qu'elle est
    tenue, et plus du tout une fois prise par le run."""
    monkeypatch.setenv("OTO_FERME_URL", "http://ferme")
    monkeypatch.setenv("OTO_FERME_TOKEN", "t")
    monkeypatch.setattr(A, "_PROLONGER_PLACE_S", 0.01)
    battements = []

    def post(url, headers=None, timeout=None, **kw):
        battements.append(url)
        return types.SimpleNamespace(status_code=200, json=lambda: {"prolongee": True})
    monkeypatch.setattr(A.requests, "post", post)
    A.tenir_place("p" * 32)
    debut = time.monotonic()
    while len(battements) < 3 and time.monotonic() - debut < 2:
        time.sleep(0.01)
    assert battements[0] == f"http://ferme/api/places/{'p' * 32}/prolonger"
    assert A.prendre_place() == "p" * 32
    time.sleep(0.05)
    n = len(battements)
    time.sleep(0.05)
    assert len(battements) == n                     # prise par le run : plus prolongée


@pytest.mark.parametrize("statut, corps, message", [
    (404, {}, "ne prolonge pas"), (200, {"prolongee": False}, "perdue")])
def test_l_entretien_s_arrete_sur_une_ferme_ancienne_ou_une_place_perdue(
        monkeypatch, caplog, statut, corps, message):
    monkeypatch.setenv("OTO_FERME_URL", "http://ferme")
    monkeypatch.setenv("OTO_FERME_TOKEN", "t")
    monkeypatch.setattr(A, "_PROLONGER_PLACE_S", 0.01)
    battements = []

    def post(url, **kw):
        battements.append(url)
        return types.SimpleNamespace(status_code=statut, text="", json=lambda: corps)
    monkeypatch.setattr(A.requests, "post", post)
    with caplog.at_level(logging.INFO, "oto_runner"):
        A.tenir_place("p" * 32)
        entretien = A._entretien
        entretien.join(2)
    assert not entretien.is_alive() and len(battements) == 1
    assert message in caplog.text
    assert A._place == "p" * 32                     # le run la présentera quand même


def test_une_ferme_injoignable_n_est_pas_une_ferme_pleine(monkeypatch):
    monkeypatch.setenv("OTO_FERME_URL", "http://ferme/")
    monkeypatch.setenv("OTO_FERME_TOKEN", "t")

    def coupe(*a, **kw):
        raise A.requests.ConnectionError("connexion refusée")
    monkeypatch.setattr(A.requests, "post", coupe)
    with pytest.raises(A.FermeInjoignable, match="ConnectionError"):
        A.reserver_place(20)
    with pytest.raises(A.FermeInjoignable, match="injoignable"):
        A.poster_le_run("http://ferme/api/sandboxes/u/runs", {}, {}, 10)
    monkeypatch.setattr(A.requests, "post", lambda *a, **kw: types.SimpleNamespace(
        status_code=502, text="Bad Gateway"))
    with pytest.raises(A.FermeInjoignable, match="502"):
        A.reserver_place(20)
    monkeypatch.setattr(A.requests, "post", lambda *a, **kw: types.SimpleNamespace(
        status_code=401, text="jeton"))
    with pytest.raises(RuntimeError) as e:
        A.reserver_place(20)
    assert not isinstance(e.value, A.FermeInjoignable)


def test_reserver_et_rendre_parlent_a_la_ferme(monkeypatch):
    monkeypatch.setenv("OTO_FERME_URL", "http://ferme/")
    monkeypatch.setenv("OTO_FERME_TOKEN", "t")
    vu = []
    reponse = {"statut": 200, "corps": {"place": "j" * 32, "tenue_s": 30}}

    def post(url, json=None, headers=None, timeout=None):
        vu.append(("POST", url, json, timeout))
        return types.SimpleNamespace(status_code=reponse["statut"], text="route inconnue",
                                     json=lambda: reponse["corps"])
    monkeypatch.setattr(A.requests, "post", post)
    monkeypatch.setattr(A.requests, "delete", lambda url, **kw: vu.append(("DELETE", url)))
    monkeypatch.setenv("OTO_RUNNER_QUI", "oto-runner-abonnement@2")
    assert A.reserver_place(20) == "j" * 32
    assert vu[-1] == ("POST", "http://ferme/api/places",
                      {"attente_s": 20, "qui": "oto-runner-abonnement@2"}, (10, 35))
    monkeypatch.delenv("OTO_RUNNER_QUI")
    assert A.qui() == f"oto-runner:{A.os.getpid()}"
    reponse["corps"] = {"place": None}
    assert A.reserver_place(20) is None
    A.rendre_place("j" * 32)
    assert vu[-1] == ("DELETE", f"http://ferme/api/places/{'j' * 32}")
    reponse["statut"] = 404
    with pytest.raises(A.PlacesNonServies):
        A.reserver_place(20)


class _Ferme:
    """Le provider vu du worker : une file de places scriptée, et ce qui en est fait."""
    ONE_SHOT = True
    RESERVE_SA_PLACE = True
    PlacesNonServies = A.PlacesNonServies
    FermeInjoignable = A.FermeInjoignable
    __name__ = "agent_ferme"

    def __init__(self, places):
        self.places, self.rendues, self.tenues = list(places), [], []
        self.tenue = None

    def resolve_key(self):
        return ""

    def model(self):
        return "claude-sonnet-5"

    def reserver_place(self):
        p = self.places.pop(0)
        if isinstance(p, Exception):
            raise p
        return p

    def rendre_place(self, jeton):
        self.rendues.append(jeton)

    def tenir_place(self, jeton):
        self.tenue = jeton
        self.tenues.append(jeton)

    def lacher_place(self):
        if self.tenue:
            self.rendues.append(self.tenue)
        self.tenue = None


class _Backend:
    base = "https://exemple.invalide"

    def __init__(self, jobs):
        self.jobs, self.claims = list(jobs), 0

    def claim(self, **_):
        self.claims += 1
        if not self.jobs:
            worker._demander_arret(15, None)
            return None
        return self.jobs.pop(0)


def _lancer(monkeypatch, provider, backend, consomme=True):
    traites = []
    monkeypatch.setenv("OTO_RUNNER_ARMED", "1")
    monkeypatch.setenv("OTO_WORKER_SECRET", "otow_banc")
    monkeypatch.setattr(worker, "Backend", lambda *a, **kw: backend)
    monkeypatch.setattr(worker, "get_provider", lambda: provider)
    monkeypatch.setattr(worker, "_POLL_S", 0)
    monkeypatch.setattr(worker, "_verifier_cle_au_demarrage", lambda *a: None)

    def un_travail(bk, job, prov, file=None):
        traites.append((job["id"], prov.tenue))
        if consomme:
            prov.tenue = None          # son run a pris la place
    monkeypatch.setattr(worker, "_un_travail", un_travail)
    worker.main()
    return traites


JOB = {"id": 1, "kind": "start", "attempts": 1}


def test_le_worker_attend_sa_place_avant_de_prendre_un_travail(monkeypatch):
    prov = _Ferme([None, None, "a" * 32, "b" * 32])
    bk = _Backend([JOB])
    traites = _lancer(monkeypatch, prov, bk)
    # deux attentes sans place : aucun claim ; puis le travail part avec SA place
    assert traites == [(1, "a" * 32)]
    assert bk.claims == 2                              # le second : file vide
    assert prov.rendues == ["b" * 32]                  # sans travail, la place revient


def test_un_travail_rendu_avant_son_run_rend_sa_place(monkeypatch):
    prov = _Ferme(["a" * 32, "b" * 32])
    traites = _lancer(monkeypatch, prov, _Backend([JOB]), consomme=False)
    assert traites == [(1, "a" * 32)]
    assert prov.rendues == ["a" * 32, "b" * 32]


def test_une_ferme_sans_file_revient_au_chemin_d_avant(monkeypatch):
    prov = _Ferme([A.PlacesNonServies("404")])
    bk = _Backend([JOB, dict(JOB, id=2)])
    traites = _lancer(monkeypatch, prov, bk)
    assert traites == [(1, None), (2, None)]           # plus de demande de place
    assert prov.tenues == [] and prov.places == []


def test_une_ferme_injoignable_ne_fait_prendre_aucun_travail(monkeypatch, caplog):
    prov = _Ferme([A.FermeInjoignable("ConnectionError : connexion refusée"),
                   RuntimeError("agent de la ferme (place) : 401 jeton"), None,
                   "a" * 32, "b" * 32])
    bk = _Backend([JOB])
    with caplog.at_level(logging.INFO, "oto_runner"):
        traites = _lancer(monkeypatch, prov, bk)
    assert traites == [(1, "a" * 32)] and bk.claims == 2
    # trois causes, trois messages : injoignable ≠ refusée ≠ pleine
    messages = [r.getMessage() for r in caplog.records]
    injoignable = [m for m in messages if m.startswith("ferme injoignable")]
    refus = [m for m in messages if m.startswith("demande de place refusée")]
    pleine = [m for m in messages if m.startswith("ferme pleine")]
    assert len(injoignable) == len(refus) == len(pleine) == 1
    assert "connexion refusée" in injoignable[0] and "401" in refus[0]


def test_un_provider_sans_ferme_ne_demande_pas_de_place(monkeypatch):
    prov = _Ferme([])
    prov.RESERVE_SA_PLACE = False
    traites = _lancer(monkeypatch, prov, _Backend([JOB]))
    assert traites == [(1, None)]
