"""Le travail RECETTE : une recette publiée jouée sans modèle (`oto_runner/recette.py`).

Ce que ce banc tient :
- le reçu décide de l'issue (`done` / `partial`), un refus HTTP fait `failed` nommé ;
- un travail re-réservé n'appelle JAMAIS la recette une seconde fois ;
- le bail est prolongé AVANT l'appel, le run est ouvert, lié, clos ;
- l'appel part au nom du jeton délégué, avec l'org et le run en en-têtes ;
- le worker de recettes n'a besoin d'aucun provider ni d'aucune clé de modèle.
"""
from __future__ import annotations

import pytest

from oto_runner import recette
from oto_runner import worker as W
from oto_runner.backend import Backend, BackendError


class _FauxMcp:
    def __init__(self, **kw):
        self.kw = kw
        self.appels: list = []

    def outil(self, nom, args=None):
        self.appels.append((nom, args))
        if nom == "run_start":
            return {"run_id": "run-42"}
        return {}

    def fermer(self):
        pass


class _FauxBackend:
    """La file ET la route des recettes : note chaque geste, dans l'ordre."""

    def __init__(self, reponse=None, refus: BackendError | None = None):
        self.gestes: list = []
        self.reponse = reponse if reponse is not None else {"receipt": {"stopped": None}}
        self.refus = refus

    def extend(self, job_id, lease_seconds=600):
        self.gestes.append(("extend", job_id, lease_seconds))

    def bind_run(self, job_id, run_id):
        self.gestes.append(("bind_run", job_id, run_id))

    def run_recipe(self, corps, *, token, org, run=None, wall_s=900):
        self.gestes.append(("run_recipe", corps, token, org, run, wall_s))
        if self.refus:
            raise self.refus
        return self.reponse

    def complete(self, job_id, ok, error=None, run_id=None, result=None):
        self.gestes.append(("complete", job_id, ok, error, run_id, result))
        return {}

    def noms(self):
        return [g[0] for g in self.gestes]


def _job(**surcharges):
    job = {"id": 11, "kind": "start", "attempts": 1, "delegated_token": "oto_jeton-du-porteur",
           "payload": {"org_id": 196, "project_id": None, "label": "planifié — recette tam",
                       "model_family": "recette",
                       "recipe": {"slug": "tam", "version": 3, "scope": "org",
                                  "params": {"q": "x"}, "datastore": 1771}}}
    job.update(surcharges)
    return job


@pytest.fixture
def mcps(monkeypatch):
    vus: list = []

    def _fabrique(**kw):
        m = _FauxMcp(**kw)
        vus.append(m)
        return m

    monkeypatch.setattr(W, "McpSession", _fabrique)
    return vus


def test_une_recette_au_bout_conclut_done(mcps):
    b = _FauxBackend({"receipt": {"stopped": None, "rows_built": 12, "items_seen": 40}})
    W._traiter(b, _job(), recette)
    # Le bail AVANT l'appel, le run lié avant l'appel, la conclusion à la fin.
    assert b.noms() == ["extend", "bind_run", "run_recipe", "complete"]
    assert b.gestes[0] == ("extend", 11, recette.BAIL_S)
    _, corps, jeton, org, run, wall = b.gestes[2]
    assert corps == {"op": "run", "slug": "tam", "scope": "org", "version": 3,
                     "params": {"q": "x"}, "datastore": 1771}
    assert "_run_id" not in corps, "le run part en en-tête X-Oto-Run, jamais dans le corps"
    assert (jeton, org, run, wall) == ("oto_jeton-du-porteur", 196, "run-42", recette.ECHEANCE_S)
    _, _, ok, err, run_id, res = b.gestes[3]
    assert ok is True and err is None and run_id == "run-42"
    assert res["outcome"] == "done" and res["recipe_receipt"]["rows_built"] == 12
    fin = [a for a in mcps[0].appels if a[0] == "run_finish"][0][1]
    assert fin["outcome"] == "done" and "rows_built 12" in fin["note"]
    # La session porte le jeton DÉLÉGUÉ, pas le secret du worker.
    assert mcps[0].kw["token"] == "oto_jeton-du-porteur"


def test_une_borne_atteinte_conclut_partial(mcps):
    b = _FauxBackend({"receipt": {"stopped": "max_pages", "rows_built": 50, "resume": "tok"}})
    W._traiter(b, _job(), recette)
    _, _, ok, _, _, res = b.gestes[-1]
    assert ok is True and res["outcome"] == "partial"
    fin = [a for a in mcps[0].appels if a[0] == "run_finish"][0][1]
    assert fin["outcome"] == "partial" and "max_pages" in fin["note"]


def test_un_recu_en_echec_conclut_le_travail_en_echec(mcps):
    b = _FauxBackend({"receipt": {"stopped": "invalid_input", "error": "unrecognized field(s): q",
                                  "rows_built": 0}})
    W._traiter(b, _job(), recette)
    _, _, ok, erreur, _, res = b.gestes[-1]
    assert ok is False and res["outcome"] == "failed"
    assert "invalid_input" in erreur and "unrecognized" in erreur
    fin = [a for a in mcps[0].appels if a[0] == "run_finish"][0][1]
    assert fin["outcome"] == "failed"


def test_un_refus_de_la_route_echoue_en_nommant_son_code(mcps):
    b = _FauxBackend(refus=BackendError("/api/me/recipes → 409 : pas publiée",
                                        status=409, code="recipe_not_published"))
    with pytest.raises(recette.RecetteRefusee) as e:
        W._traiter(b, _job(), recette)
    assert "recipe_not_published" in str(e.value)
    assert "complete" not in b.noms(), "l'échec se conclut par le chemin commun (en_echec)"


def test_un_echec_passe_par_la_conclusion_commune(mcps, tmp_path, monkeypatch):
    """`_un_travail` rattrape l'échec : run clos `failed`, travail rendu en échec."""
    monkeypatch.setenv("OTO_RUNNER_PASSAGES_DIR", str(tmp_path))
    b = _FauxBackend(refus=BackendError("x", status=500, code="boom"))
    W._un_travail(b, _job(), recette, file=b)
    fin = [a for a in mcps[0].appels if a[0] == "run_finish"][0][1]
    assert fin["outcome"] == "failed"
    _, _, ok, err, run_id, _ = b.gestes[-1]
    assert ok is False and "boom" in err and run_id == "run-42"


@pytest.mark.parametrize("reprise", [{"attempts": 2}, {"run_id": "run-ancien"}])
def test_un_travail_re_reserve_n_est_jamais_rejoue(mcps, reprise):
    b = _FauxBackend()
    with pytest.raises(recette.RecetteNonRejouee) as e:
        W._traiter(b, _job(**reprise), recette)
    assert "recipe_job_not_replayed" in str(e.value)
    assert "run_recipe" not in b.noms() and "extend" not in b.noms()


def test_un_run_lie_d_avant_est_rendu_pour_etre_clos(mcps):
    """La tentative précédente a ouvert un run : il se clôt en échec par en_echec."""
    tenu = W.RunEnCours()
    with pytest.raises(recette.RecetteNonRejouee):
        W._traiter(_FauxBackend(), _job(attempts=2, run_id="run-ancien"), recette, tenu=tenu)
    assert tenu.run_id == "run-ancien" and tenu.mcp is not None


def test_un_worker_de_modele_refuse_une_recette(mcps):
    with pytest.raises(recette.RecetteHorsMode):
        W._traiter(_FauxBackend(), _job(), object())
    assert mcps == [], "rien n'est ouvert avant le refus"


def test_un_worker_de_recettes_refuse_un_travail_d_agent(mcps):
    job = _job()
    job["payload"] = {"org_id": 196, "procedure": "veille", "input": "fais"}
    with pytest.raises(recette.TravailNonRecette):
        W._traiter(_FauxBackend(), job, recette)
    assert mcps == []


def test_le_mode_recette_demarre_sans_provider_ni_cle(monkeypatch):
    """`OTO_RUNNER_DEPOT=recette` : `main` n'appelle jamais `get_provider`, ne vérifie
    aucune clé, et réserve avec le dépôt `recette`."""
    monkeypatch.setenv("OTO_RUNNER_ARMED", "1")
    monkeypatch.setenv(recette.ENV_DEPOT, recette.DEPOT)
    monkeypatch.setenv("OTO_WORKER_SECRET", "otow_secret-du-worker")
    monkeypatch.setattr(W, "get_provider",
                        lambda: pytest.fail("un worker de recettes n'a pas de provider"))
    monkeypatch.setattr(W, "_verifier_cle_au_demarrage",
                        lambda *a: pytest.fail("aucune clé de modèle à vérifier"))
    monkeypatch.setattr(W.journal, "preparer", lambda: "/tmp")
    claims: list = []

    def _claim(self, **kw):
        claims.append(kw)
        W._arret_demande = True     # un tour, puis on sort
        return None

    monkeypatch.setattr(Backend, "claim", _claim)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setattr(W.signal, "signal", lambda *a: None)
    monkeypatch.setattr(W, "_arret_demande", False)
    W.main()
    assert claims == [{"lease_seconds": recette.BAIL_S, "depot": recette.DEPOT}]


def test_le_run_part_en_en_tete(monkeypatch):
    """`run_recipe` pose X-Oto-Org et X-Oto-Run, le jeton délégué, et une échéance longue."""
    monkeypatch.setenv("OTO_TOKEN", "secret")
    b = Backend(base="https://exemple.invalide")
    vu = {}

    class _R:
        status_code, content = 200, b'{"receipt": {}}'

        def json(self):
            return {"receipt": {}}

    def _reseau(chemin, fn, **kw):
        vu["chemin"], vu["kw"] = chemin, kw
        return _R()

    monkeypatch.setattr(b, "_reseau", _reseau)
    b.run_recipe({"op": "run", "slug": "tam"}, token="delegue", org=196, run="run-9", wall_s=900)
    h = vu["kw"]["headers"]
    assert vu["chemin"] == "/api/me/recipes"
    assert h["Authorization"] == "Bearer delegue" and h["X-Oto-Org"] == "196"
    assert h["X-Oto-Run"] == "run-9"
    assert vu["kw"]["wall_s"] == 900 and vu["kw"]["timeout"] == (10, 900)


@pytest.mark.parametrize("arret,issue", [(None, "done"), ("", "done"), ("end", "done"),
                                         ("max_units", "partial"), ("max_pages", "partial"),
                                         ("mapping_drift", "failed")])
def test_issue_du_recu(arret, issue):
    assert recette.issue_du_recu({"stopped": arret}) == issue


def test_un_recu_qui_porte_une_erreur_est_un_echec():
    """Vu en e2e local : `invalid_input` (arguments refusés, non rejouable) sortait `partial`."""
    assert recette.issue_du_recu({"stopped": "invalid_input", "error": "unrecognized field"}) == "failed"
    assert recette.issue_du_recu({"stopped": "call_refused", "error": "x"}) == "failed"
