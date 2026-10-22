"""La SONDE du forfait : avant un run d'abonnement, le fournisseur dit l'état du forfait,
le backend tranche, et un forfait indisponible ne lance rien."""
import json

import pytest

from oto_runner import agent_abonnement as A
from oto_runner import worker
from oto_runner.backend import BackendError

REFUS = {"status": "rejected", "rateLimitType": "five_hour", "resetsAt": 1790216400,
         "unifiedWindows": {"five_hour": {"utilization": 1.0, "resetsAt": 1790216400},
                            "seven_day": {"utilization": 0.31, "resetsAt": 1790776800}}}


def test_le_rapport_nomme_la_fenetre_qui_a_mordu():
    assert A.rapport_du_forfait(REFUS) == {
        "etat": "rejected", "fenetres": REFUS["unifiedWindows"],
        "fenetre": "five_hour", "echeance": 1790216400}
    assert A.rapport_du_forfait({"status": "allowed"}) == {"etat": "allowed", "fenetres": None}
    assert A.rapport_du_forfait(None) is None


class _Reponse:
    def __init__(self, evenements, status_code=200):
        self.status_code, self.text = status_code, ""
        self._lignes = [json.dumps(e) for e in evenements]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self, decode_unicode=True):
        return iter(self._lignes)


@pytest.fixture
def _ferme(monkeypatch):
    monkeypatch.setenv("OTO_FERME_URL", "http://ferme")
    monkeypatch.setenv("OTO_FERME_TOKEN", "t")
    envoyes = []

    def poster(url, corps, entetes, lecture, **kw):
        envoyes.append((url, corps))
        return _Reponse(poster.evenements)
    poster.evenements = []
    monkeypatch.setattr(A, "poster_le_run", poster)
    return poster, envoyes


def test_la_sonde_est_un_tour_sans_outil_ni_relais(_ferme):
    poster, envoyes = _ferme
    poster.evenements = [{"type": "system", "subtype": "init"},
                         {"type": "rate_limit_event", "rate_limit_info": REFUS},
                         {"type": "result", "is_error": True}]
    assert A.verifier_forfait("sb-1")["etat"] == "rejected"
    ((url, corps),) = envoyes
    assert url == "http://ferme/api/sandboxes/sb-1/runs"
    assert corps["max_turns"] == 1 and "mcp" not in corps and "system" not in corps


def test_sans_evenement_de_forfait_la_sonde_ne_sait_rien(_ferme):
    poster, _ = _ferme
    poster.evenements = [{"type": "result", "is_error": True, "result": "API Error: 429"}]
    assert A.verifier_forfait("sb-1") is None


class _Provider:
    VERIFIE_LE_FORFAIT = True

    def __init__(self, rapport=None, leve=None):
        self.rapport, self.leve = rapport, leve

    def verifier_forfait(self, sandbox):
        if self.leve:
            raise self.leve
        return self.rapport


class _File:
    def __init__(self, verdict=None, leve=None):
        self.verdict, self.leve, self.recus = verdict, leve, []

    def forfait(self, job_id, rapport):
        self.recus.append((job_id, rapport))
        if self.leve:
            raise self.leve
        return self.verdict


_JOB = {"id": 7, "sandbox_id": "sb-1", "forfait_a_sonder": True}
_RAPPORT = A.rapport_du_forfait(REFUS)


@pytest.fixture(autouse=True)
def _backend_neuf(monkeypatch):
    monkeypatch.setattr(worker, "_forfait_sans_backend", False)
    monkeypatch.setenv("OTO_RUNNER_SONDE_FORFAIT", "1")


def _sonder(provider, file):
    return worker._sonder_le_forfait(_JOB, provider, file, lambda *a, **k: None)


def test_un_forfait_indisponible_ne_lance_rien():
    file = _File({"servable": False, "limit_reset_at": "2026-10-21T20:00:00Z"})
    assert _sonder(_Provider(_RAPPORT), file) is False
    assert file.recus == [(7, _RAPPORT)]


def test_un_forfait_servable_part():
    assert _sonder(_Provider(_RAPPORT), _File({"servable": True})) is True


@pytest.mark.parametrize("provider", [_Provider(None), _Provider(leve=RuntimeError("ferme"))])
def test_dans_le_doute_le_travail_part(provider):
    file = _File({"servable": False})
    assert _sonder(provider, file) is True
    assert file.recus == []


def test_un_backend_sans_l_operation_suspend_la_sonde_une_fois():
    # Un backend antérieur à `op=forfait` : sa route refuse la valeur d'`op`.
    file = _File(leve=BackendError("op inconnue", status=400, code="invalid_input"))
    assert _sonder(_Provider(_RAPPORT), file) is True
    assert _sonder(_Provider(_RAPPORT), file) is True
    assert len(file.recus) == 1


def test_un_rapport_refuse_ne_suspend_pas_la_sonde(caplog):
    # Le refus porte sur CE rapport : le travail part, et le suivant est encore sondé.
    file = _File(leve=BackendError("rapport illisible", status=400,
                                   code="forfait_report_unreadable"))
    with caplog.at_level("WARNING", logger="oto_runner"):
        assert _sonder(_Provider(_RAPPORT), file) is True
        assert _sonder(_Provider(_RAPPORT), file) is True
    assert len(file.recus) == 2
    assert worker._forfait_sans_backend is False
    # Journalisé en métadonnées : la forme du rapport, pas ses valeurs.
    (ligne,) = {r.getMessage() for r in caplog.records}
    assert "rapport de sonde refusé" in ligne
    assert "'fenetres': ['five_hour', 'seven_day']" in ligne
    assert "0.31" not in ligne and "1790216400" not in ligne


def test_une_autre_400_ne_suspend_pas_la_sonde():
    file = _File(leve=BackendError("champ manquant", status=400, code="missing_fields"))
    assert _sonder(_Provider(_RAPPORT), file) is True
    assert _sonder(_Provider(_RAPPORT), file) is True
    assert len(file.recus) == 2


def test_un_travail_sans_forfait_a_sonder_ne_sonde_pas():
    # Forfait connecté, ou backend antérieur à la sonde : le champ est absent.
    provider = _Provider(leve=AssertionError("ne doit pas sonder"))
    file = _File({"servable": False})
    sans = {k: v for k, v in _JOB.items() if k != "forfait_a_sonder"}
    assert worker._sonder_le_forfait(sans, provider, file, lambda *a, **k: None) is True
    assert worker._sonder_le_forfait({**sans, "forfait_a_sonder": False}, provider, file,
                                     lambda *a, **k: None) is True
    assert file.recus == []


def test_un_travail_a_sonder_est_sonde():
    file = _File({"servable": True})
    assert _sonder(_Provider(_RAPPORT), file) is True
    assert file.recus == [(7, _RAPPORT)]


def test_une_file_sans_verdict_ne_sonde_pas():
    class SansFile:
        pass
    provider = _Provider(leve=AssertionError("ne doit pas sonder"))
    assert _sonder(provider, SansFile()) is True


def test_un_provider_par_cle_ne_sonde_pas():
    class ParCle:
        pass
    assert _sonder(ParCle(), _File({"servable": False})) is True


def test_sans_la_variable_rien_n_est_sonde(monkeypatch):
    monkeypatch.delenv("OTO_RUNNER_SONDE_FORFAIT")
    provider = _Provider(leve=AssertionError("ne doit pas sonder"))
    assert _sonder(provider, _File({"servable": False})) is True
