"""La voie `claude-subscription` : le flux du CLI lu, le forfait rapporté, le relais borné."""
import pytest

from oto_runner import agent_abonnement as A
from oto_runner import conclusion, worker
from oto_runner.relais_mcp import Relais

INIT = {"type": "system", "subtype": "init", "apiKeySource": "none", "model": "claude-sonnet-5"}
FORFAIT = {"type": "rate_limit_event", "rate_limit_info": {
    "status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.13, "resetsAt": 1790216400},
                                            "seven_day": {"utilization": 0.31, "resetsAt": 1790776800}}}}
APPEL = {"type": "assistant", "message": {"content": [
    {"type": "tool_use", "id": "t1", "name": "mcp__oto__data_rows", "input": {}}]}}
RETOUR = {"type": "user", "message": {"content": [
    {"type": "tool_result", "tool_use_id": "t1", "is_error": False}]}}
FIN = {"type": "result", "subtype": "success", "is_error": False, "result": "fait",
       "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100}}


def test_un_run_conclu_rend_reponse_pas_usage_et_forfait():
    res = A.lire_flux([INIT, FORFAIT, APPEL, RETOUR, FIN])
    assert (res.reply, res.stopped, res.model) == ("fait", "end_turn", "claude-sonnet-5")
    assert [(s.tool, s.ok) for s in res.steps] == [("data_rows", True)]
    assert res.abonnement == {"etat": "allowed",
                              "fenetres": FORFAIT["rate_limit_info"]["unifiedWindows"]}
    declare = conclusion.resultat_declare(res, "sub:sonnet")
    assert declare["abonnement"]["etat"] == "allowed"
    assert declare["usage_tokens"] == 15 and declare["tool_counts"] == {"data_rows": 1}


def test_un_run_paye_par_une_cle_est_refuse():
    with pytest.raises(RuntimeError, match="apiKeySource"):
        A.lire_flux([dict(INIT, apiKeySource="ANTHROPIC_API_KEY"), FIN])


def test_un_sandbox_deconnecte_conclut_en_echec_nomme_et_le_dit_au_backend():
    res = A.lire_flux([{"type": "ferme_resume", "connecte": False, "erreur": "Not logged in"}])
    assert res.abonnement == {"deconnecte": True}
    assert conclusion.echec_nomme(res) == "fin_anormale (sandbox_deconnecte)"


def test_une_erreur_du_cli_est_une_fin_anormale():
    res = A.lire_flux([INIT, dict(FIN, is_error=True, result="API Error: 529")])
    assert conclusion.echec_nomme(res) == "fin_anormale (success)"


def test_un_flux_sans_resultat_echoue_franchement():
    with pytest.raises(RuntimeError, match="aucun résultat"):
        A.lire_flux([INIT, FORFAIT])


def test_le_bail_se_prolonge_pendant_le_run_au_plus_une_fois_par_minute():
    temps = iter([0, 10, 70, 80, 200, 210])   # départ, puis un instant par événement
    appels = []
    A.lire_flux([INIT, FORFAIT, APPEL, RETOUR, FIN], prolonger=lambda: appels.append(1),
                horloge=lambda: next(temps))
    assert len(appels) == 2


def test_le_modele_du_catalogue_perd_son_prefixe_et_une_autre_famille_est_refusee():
    assert A._modele_du_cli("sub:opus") == "opus"
    with pytest.raises(ValueError):
        A._modele_du_cli("claude-sonnet-5")


def test_seule_la_voie_abonnement_recoit_le_contexte_du_sandbox():
    class Autre:
        pass
    assert worker._contexte_du_sandbox({"id": 1}, Autre, object(), object(), print) == {}
    ctx = worker._contexte_du_sandbox({"id": 1, "sandbox_id": "u42"}, A, "mcp", object(), print)
    assert ctx["sandbox"] == "u42" and ctx["mcp"] == "mcp" and callable(ctx["prolonger"])
    assert ctx["apposer"] is print


class SessionFactice:
    def __init__(self):
        self.appels = []

    def schemas(self, noms):
        return [{"name": n, "description": "d", "input_schema": {"type": "object"}}
                for n in sorted(noms)]

    def call(self, nom, args):
        self.appels.append((nom, args))
        return "ok", False, ()


def test_le_relais_ne_sert_que_l_allowlist():
    session = SessionFactice()
    relais = Relais(session, frozenset({"data_rows"}))
    liste = relais.repondre({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert [t["name"] for t in liste["result"]["tools"]] == ["data_rows"]
    refus = relais.repondre({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                             "params": {"name": "data_write", "arguments": {}}})
    assert refus["result"]["isError"] and session.appels == []
    ok = relais.repondre({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                          "params": {"name": "data_rows", "arguments": {"q": 1}}})
    assert ok["result"] == {"content": [{"type": "text", "text": "ok"}], "isError": False}
    assert session.appels == [("data_rows", {"q": 1})]


def test_le_relais_repond_a_l_initialize_et_ignore_les_notifications():
    relais = Relais(SessionFactice(), frozenset())
    init = relais.repondre({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                            "params": {"protocolVersion": "2025-06-18"}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert relais.repondre({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


TEXTE = {"type": "assistant", "message": {"role": "assistant", "content": [
    {"type": "text", "text": "Je regarde."},
    {"type": "tool_use", "id": "t1", "name": "mcp__oto__data_rows", "input": {}}]}}
GROS_RETOUR = {"type": "user", "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t1", "is_error": False, "content": "x" * 80_000}]}}


def test_le_fil_se_tient_EN_DIRECT_message_par_message():
    fil = []
    temps = iter([0, 1, 2, 5, 6, 7])
    A.lire_flux([INIT, TEXTE, GROS_RETOUR, FIN],
                apposer=lambda role, neutre, brut: fil.append((role, neutre, brut)),
                horloge=lambda: next(temps))
    assert [r for r, _, _ in fil] == ["assistant", "tool"]
    (_, neutre_a, brut_a), (_, neutre_t, brut_t) = fil
    assert neutre_a == {"text": "Je regarde.", "tool_calls": [{"name": "data_rows"}]}
    assert brut_a == TEXTE["message"], "le brut est le message Anthropic, tel quel"
    assert neutre_t == {"tool_calls": [{"name": "data_rows", "ok": True, "duration_ms": 3000}]}
    garde = brut_t["content"][0]["content"]
    assert len(garde) < 51_000 and "coupé pour le fil : 80000 caractères" in garde
    assert GROS_RETOUR["message"]["content"][0]["content"] == "x" * 80_000, "l'original reste intact"


class _Stop(RuntimeError):
    pass


class _BackendQuiNote:
    base = "https://exemple.invalide"

    def __init__(self):
        self.fil = []

    def thread_append(self, run_id, role, neutre, provider_raw=None, token=None):
        self.fil.append((role, neutre))
        return len(self.fil)

    def __getattr__(self, _nom):
        return lambda *a, **k: {}


class _McpMuet:
    def __init__(self, **kw):
        self.run_id = None

    def outil(self, nom, args=None):
        return {"run_id": "run-1"}


def test_la_voie_sandbox_appose_l_ordre_puis_le_direct_et_JAMAIS_de_synthese(monkeypatch):
    backend = _BackendQuiNote()
    monkeypatch.setattr(worker, "McpSession", _McpMuet)

    class _P:
        __name__ = "agent_abonnement"
        ONE_SHOT = True
        SANDBOX = True

        @staticmethod
        def depot():
            return "claude_subscription"

        @staticmethod
        def model():
            return "sub:sonnet"

        @staticmethod
        def run_once(**kw):
            kw["apposer"]("assistant", {"text": "fait", "tool_calls": []}, {"role": "assistant"})
            raise _Stop

    job = {"id": 7, "kind": "start", "attempts": 0, "delegated_token": "oto_delegue",
           "sandbox_id": "u42", "payload": {"input": "vas-y", "model_family": "claude_subscription"}}
    with pytest.raises(_Stop):
        worker._traiter(backend, job, _P)
    assert backend.fil == [("user", {"text": "vas-y"}),
                           ("assistant", {"text": "fait", "tool_calls": []})]
