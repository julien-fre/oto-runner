"""Chaque session MCP ouverte se FERME : `DELETE` avec son `Mcp-Session-Id`.

Le runner ouvrait une session par travail (et une de plus par relais de sandbox)
et n'en fermait aucune. Côté serveur, chaque session garde sa mémoire jusqu'à sa
fermeture pour inactivité. Ce que ces bancs tiennent :

1. `fermer` envoie `DELETE` sur l'URL MCP, avec l'en-tête de session et le même jeton ;
2. elle est idempotente, et ne lève jamais : 404, 405, 500 ou erreur réseau se disent
   au journal, sans remonter ;
3. le worker la ferme à la fin de CHAQUE travail — conclu, en échec ou interrompu —
   et après la clôture du run, qui passe encore par elle ;
4. le relais la ferme quand son entrée se referme, et quand il lève.
"""
from __future__ import annotations

import io
import logging
import types

import pytest
import requests

from oto_runner import mcp as M
from oto_runner import relais_mcp
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult
from oto_runner.deadline import DeadlineExceeded
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

_PROVIDER = types.SimpleNamespace(__name__="agent_llm_openai", model=lambda: "mistral-large-2512")


def _session(monkeypatch, reponse=None, leve=None):
    """Une vraie `McpSession` sans réseau : l'initialize est scripté, le DELETE capturé."""
    deletes = []

    def _post(self, corps, avec_entetes=False):
        if corps.get("method") == "initialize":
            return ({"mcp-session-id": "sess-abcdef12"}, {}) if avec_entetes else {}
        return {}

    def _delete(url, **kw):
        deletes.append({"url": url, **kw})
        if leve is not None:
            raise leve
        return types.SimpleNamespace(status_code=200 if reponse is None else reponse)

    monkeypatch.setattr(M.McpSession, "_post", _post)
    monkeypatch.setattr(M, "delete_with_deadline", _delete)
    return M.McpSession(url="http://x/mcp", token="oto_delegue"), deletes


# ── 1. le DELETE, son en-tête, son jeton ─────────────────────────────────────

def test_fermer_envoie_delete_avec_la_session_et_le_jeton(monkeypatch):
    s, deletes = _session(monkeypatch)
    s.fermer()
    (d,) = deletes
    assert d["url"] == "http://x/mcp"
    assert d["headers"]["Mcp-Session-Id"] == "sess-abcdef12"
    assert d["headers"]["Authorization"] == "Bearer oto_delegue", \
        "le serveur n'accepte une session qu'avec le jeton qui l'a ouverte"
    assert d["wall_s"] <= 10 and max(d["timeout"]) <= 10, "un délai COURT : une politesse"
    assert s.session is None


# ── 2. idempotente, et jamais une exception ──────────────────────────────────

def test_fermer_deux_fois_n_envoie_qu_un_delete(monkeypatch):
    s, deletes = _session(monkeypatch)
    s.fermer()
    s.fermer()
    assert len(deletes) == 1


@pytest.mark.parametrize("statut", [404, 405])
def test_un_404_ou_405_se_dit_en_debug_sans_lever(monkeypatch, caplog, statut):
    s, _ = _session(monkeypatch, reponse=statut)
    with caplog.at_level(logging.DEBUG, logger="oto_runner"):
        s.fermer()
    (rec,) = [r for r in caplog.records if "DELETE" in r.getMessage()]
    assert rec.levelno == logging.DEBUG and str(statut) in rec.getMessage()


def test_un_refus_serveur_se_dit_en_warning_sans_lever(monkeypatch, caplog):
    s, _ = _session(monkeypatch, reponse=500)
    with caplog.at_level(logging.DEBUG, logger="oto_runner"):
        s.fermer()
    assert any(r.levelno == logging.WARNING and "500" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("erreur", [requests.ConnectionError("réseau coupé"),
                                    DeadlineExceeded("mur"),
                                    ValueError("signal only works in main thread")])
def test_une_erreur_reseau_ne_remonte_pas_et_se_dit(monkeypatch, caplog, erreur):
    s, deletes = _session(monkeypatch, leve=erreur)
    with caplog.at_level(logging.WARNING, logger="oto_runner"):
        s.fermer()
    assert "non fermée" in caplog.text
    s.fermer()
    assert len(deletes) == 1, "une fermeture ratée ne se retente pas : l'id est parti"


def test_une_session_jamais_ouverte_ne_ferme_rien(monkeypatch):
    s, deletes = _session(monkeypatch)
    s.session = None
    s.fermer()
    assert deletes == []


# ── 3. le worker ferme à la fin de chaque travail ────────────────────────────

class _McpFermable(FauxMcp):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.journal_des_gestes = []
        _McpFermable.vues.append(self)

    def outil(self, name, args=None):
        self.journal_des_gestes.append(name)
        return super().outil(name, args)

    def fermer(self):
        self.journal_des_gestes.append("fermer")


def _jouer(monkeypatch, run):
    _McpFermable.vues = []
    monkeypatch.setattr(W, "McpSession", _McpFermable)
    monkeypatch.setattr(W.agent_runtime, "run", run)
    W._un_travail(FauxBackend(), _job("start"), _PROVIDER)
    (mcp,) = _McpFermable.vues
    return mcp.journal_des_gestes


def test_un_travail_conclu_ferme_sa_session_en_dernier(monkeypatch):
    gestes = _jouer(monkeypatch, lambda *a, **k: AgentResult(reply="fini", stopped="end_turn"))
    assert gestes[-1] == "fermer" and gestes.count("fermer") == 1
    assert "run_finish" in gestes


def test_un_travail_qui_leve_ferme_sa_session_apres_avoir_clos_son_run(monkeypatch):
    def run(*a, **k):
        raise RuntimeError("le modèle a lâché")
    gestes = _jouer(monkeypatch, run)
    assert gestes[-1] == "fermer", "fermée même quand le travail meurt"
    assert gestes.index("run_finish") < gestes.index("fermer"), \
        "la clôture du run passe encore par la session : elle se ferme APRÈS"


def test_un_travail_interrompu_ferme_sa_session_et_laisse_passer_l_interruption(monkeypatch):
    def run(*a, **k):
        raise KeyboardInterrupt
    _McpFermable.vues = []
    monkeypatch.setattr(W, "McpSession", _McpFermable)
    monkeypatch.setattr(W.agent_runtime, "run", run)
    with pytest.raises(KeyboardInterrupt):
        W._un_travail(FauxBackend(), _job("start"), _PROVIDER)
    (mcp,) = _McpFermable.vues
    assert mcp.journal_des_gestes[-1] == "fermer"


def test_un_delete_qui_echoue_ne_change_pas_l_issue_du_travail(monkeypatch):
    """La vraie `fermer`, branchée sur un DELETE qui lève : le travail conclut quand même."""
    def _delete(url, **kw):
        raise requests.ConnectionError("réseau coupé")
    monkeypatch.setattr(M, "delete_with_deadline", _delete)

    class _McpReelle(_McpFermable):
        def fermer(self):
            self.session, self.url, self.token = "sess-abcdef12", "http://x/mcp", "t"
            M.McpSession.fermer(self)
            self.journal_des_gestes.append("fermer")

    _McpFermable.vues = []
    monkeypatch.setattr(W, "McpSession", _McpReelle)
    monkeypatch.setattr(W.agent_runtime, "run",
                        lambda *a, **k: AgentResult(reply="fini", stopped="end_turn"))
    b = FauxBackend()
    W._un_travail(b, _job("start"), _PROVIDER)
    assert ("complete", True, "r-NEUF") in b.appels


# ── 4. le relais ferme à son arrêt ───────────────────────────────────────────

class _SessionDuRelais:
    def __init__(self):
        self.fermee = 0

    def fermer(self):
        self.fermee += 1


def test_le_relais_ferme_sa_session_quand_son_entree_se_referme():
    session = _SessionDuRelais()
    sortie = io.StringIO()
    relais_mcp.servir(relais_mcp.Relais(session, frozenset()),
                      io.StringIO('{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n'), sortie)
    assert '"id": 1' in sortie.getvalue()
    assert session.fermee == 1


def test_le_relais_ferme_sa_session_quand_il_leve():
    session = _SessionDuRelais()
    with pytest.raises(ValueError):
        relais_mcp.servir(relais_mcp.Relais(session, frozenset()),
                          io.StringIO("pas du json\n"), io.StringIO())
    assert session.fermee == 1


def test_sigterm_fait_sortir_le_relais_par_son_finally():
    with pytest.raises(SystemExit):
        relais_mcp._sortir_sur_signal(15, None)
