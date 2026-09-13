"""Un worker qui ne tient AUCUNE clé de modèle à lui — `OTO_RUNNER_ORG_KEYS_ONLY=1`.

C'est ce qui ouvre une famille de modèles (Anthropic) aux organisations qui
apportent leur clé, sans que la plateforme finance un seul jeton. Ces bancs
tiennent les trois gestes du mode :

1. **il démarre sans clé d'environnement** — mais pas sans dépôt nommé ;
2. **il le dit au serveur à chaque réservation** (`org_key_only`) — et un worker
   ordinaire n'envoie PAS le champ, pour rester compatible avec un backend qui ne
   le déclare pas (incident du 04/09) ;
3. **il ne part jamais sans la clé de l'org** — filet local, avant session ou run.
"""
from __future__ import annotations

import pytest

from oto_runner import worker as W
from oto_runner.backend import Backend


class _Provider:
    def __init__(self, depot="anthropic"):
        self._depot = depot
        self.cle_demandee = False

    def depot(self):
        return self._depot

    def resolve_key(self):
        self.cle_demandee = True
        raise RuntimeError("ANTHROPIC_API_KEY absente de l'environnement du worker")


# ── 1. le démarrage ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("valeur,attendu", [("1", True), ("", False), ("0", False),
                                            (" 1 ", True)])
def test_le_mode_se_lit_de_l_environnement(monkeypatch, valeur, attendu):
    monkeypatch.setenv("OTO_RUNNER_ORG_KEYS_ONLY", valeur)
    assert W._cles_clients_seules() is attendu


def test_un_worker_ORDINAIRE_exige_toujours_sa_cle_au_demarrage(monkeypatch):
    """Le comportement d'avant, intact : sans clé, il échoue au boot."""
    monkeypatch.delenv("OTO_RUNNER_ORG_KEYS_ONLY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        W._verifier_cle_au_demarrage(_Provider(), "anthropic")


def test_en_mode_cles_clients_il_demarre_SANS_cle_d_environnement(monkeypatch):
    """⚠️ LE banc du démarrage. Sans ce geste, ouvrir Anthropic aux clés clients
    obligeait à poser une clé de plateforme — donc à payer le repli."""
    monkeypatch.setenv("OTO_RUNNER_ORG_KEYS_ONLY", "1")
    p = _Provider()
    W._verifier_cle_au_demarrage(p, "anthropic")
    assert p.cle_demandee is False, "la clé d'environnement n'est même pas lue"


def test_en_mode_cles_clients_un_worker_SANS_depot_ne_demarre_pas(monkeypatch):
    """Aucune clé d'org ne lui serait jamais remise : il sonderait à vide."""
    monkeypatch.setenv("OTO_RUNNER_ORG_KEYS_ONLY", "1")
    with pytest.raises(SystemExit) as e:
        W._verifier_cle_au_demarrage(_Provider(depot=""), "")
    assert "OTO_RUNNER_PROVIDER" in str(e.value), "le refus dit où regarder"


def test_le_journal_de_demarrage_dit_QUI_paie(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_ORG_KEYS_ONLY", "1")
    assert "UNIQUEMENT" in W._dire_la_cle("anthropic")
    monkeypatch.delenv("OTO_RUNNER_ORG_KEYS_ONLY")
    assert "quand elle en dépose une" in W._dire_la_cle("anthropic")


# ── 2. ce que la réservation envoie ───────────────────────────────────────────

def _corps_du_claim(monkeypatch, **kw):
    b = Backend(base="https://exemple.invalide", token="otow_banc")
    vu = {}
    monkeypatch.setattr(b, "_post", lambda chemin, corps, *a, **k: vu.update(corps) or {})
    b.claim(lease_seconds=600, depot="anthropic", **kw)
    return vu


def test_un_worker_ordinaire_n_envoie_PAS_le_champ(monkeypatch):
    """⚠️ Un backend qui ne déclare pas `org_key_only` répondrait `unknown_fields`
    à chaque réservation — l'incident du 04/09, sur le champ `provider`."""
    assert "org_key_only" not in _corps_du_claim(monkeypatch)


def test_le_mode_cles_clients_l_envoie(monkeypatch):
    corps = _corps_du_claim(monkeypatch, org_key_only=True)
    assert corps["org_key_only"] is True and corps["provider"] == "anthropic"


# ── 3. le filet local ─────────────────────────────────────────────────────────

class _Sentinelle(Exception):
    """Levée par la session MCP : prouve qu'on est allé AU-DELÀ des gardes."""


def _job(**kw):
    return {"id": 7, "kind": "start", "run_id": None, "delegated_token": "otd_x",
            "payload": {"procedure": "p", "tools": ["data_rows"], "input": "Vas-y.",
                        "max_steps": 3, "model": "claude-sonnet-5",
                        "model_family": "anthropic"}, **kw}


@pytest.fixture
def session_espion(monkeypatch):
    def _ouvrir(**kw):
        raise _Sentinelle()
    monkeypatch.setattr(W, "McpSession", _ouvrir)


def test_sans_cle_d_org_le_travail_est_refuse_AVANT_toute_session(monkeypatch, session_espion):
    """Filet contre un backend qui ne connaîtrait pas le mode : le travail n'ouvre
    ni session ni run, donc ne dépense rien."""
    monkeypatch.setenv("OTO_RUNNER_ORG_KEYS_ONLY", "1")
    with pytest.raises(W.SansCleDeposee):
        W._traiter(object(), _job(), _Provider())


def test_avec_la_cle_d_org_le_travail_depasse_les_gardes(monkeypatch, session_espion):
    monkeypatch.setenv("OTO_RUNNER_ORG_KEYS_ONLY", "1")
    with pytest.raises(_Sentinelle):
        W._traiter(object(), _job(model_key="sk-ant-de-l-org"), _Provider())


def test_un_worker_ORDINAIRE_part_sans_cle_d_org_comme_avant(monkeypatch, session_espion):
    monkeypatch.delenv("OTO_RUNNER_ORG_KEYS_ONLY", raising=False)
    with pytest.raises(_Sentinelle):
        W._traiter(object(), _job(), _Provider())
