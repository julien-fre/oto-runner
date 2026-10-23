"""Les bornes que la ferme tient elle-même (tours, budget) concluent `blocked` sans rejeu,
une échéance DÉCLARÉE reste une borne sur Conversations, et un `continue` n'est jamais
servi sans son fil. Flux synthétiques (dépôt public)."""
from __future__ import annotations

import json
import types

import pytest

from oto_runner import agent_abonnement as A
from oto_runner import agent_conversations as C
from oto_runner import agent_ferme as F
from oto_runner import conclusion, worker
from oto_runner.agent_runtime import AgentSpec
from oto_runner.deadline import DeadlineExceeded
from oto_runner.declaration import load_spec, spec_depuis_flotte

INIT = {"type": "system", "subtype": "init", "apiKeySource": "ANTHROPIC_API_KEY",
        "model": "claude-x"}
MCP = types.SimpleNamespace(url="https://mcp", token="jeton-delegue", org=4242, project=7,
                            run_id="r1")


class Reponse:
    def __init__(self, statut, evenements=()):
        self.status_code, self._ev, self.text = statut, list(evenements), ""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return (json.dumps(e) for e in self._ev)


@pytest.fixture
def ferme(monkeypatch):
    monkeypatch.setenv("OTO_FERME_URL", "http://ferme:8190/")
    monkeypatch.setenv("OTO_FERME_TOKEN", "t")
    F._sandboxes_creees.clear()
    vu = {"post": [], "reponses": []}
    monkeypatch.setattr(F.requests, "put",
                        lambda url, **kw: types.SimpleNamespace(status_code=200, text=""))

    def post(url, json=None, **kw):
        vu["post"].append(json)
        return vu["reponses"].pop(0)
    monkeypatch.setattr(A.requests, "post", post)
    return vu


def _lancer(**kw):
    base = dict(instructions="i", inputs="go", tools=["data_rows"], api_key="sk-ant-api03-org",
                modele="claude-sonnet-5", mcp=MCP, org=4242, attendre=lambda s: None)
    return F.run_once(**{**base, **kw})


def _fin(subtype):
    return {"type": "result", "subtype": subtype, "is_error": True,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


# ── B1 : une borne de la ferme conclut `blocked`, jamais un échec rejoué ─────────

@pytest.mark.parametrize("fin,arret", [("error_max_turns", "max_steps"),
                                       ("error_max_budget_usd", "max_budget")])
def test_une_borne_de_la_ferme_est_bloquee_pas_echouee(ferme, fin, arret):
    ferme["reponses"].append(Reponse(200, [INIT, _fin(fin)]))
    evenements = []
    res = _lancer(on_event=lambda e, c: evenements.append((e, c)))
    assert res.stopped == arret
    assert conclusion.echec_nomme(res) is None          # ni échec, ni rejeu
    assert res.usage is not None                        # l'usage payé reste relevé
    assert ("borne_atteinte", {"borne": arret, "par": "ferme", "fin": fin}) in evenements


def test_une_vraie_erreur_du_cli_reste_une_fin_anormale(ferme):
    ferme["reponses"].append(Reponse(200, [INIT, _fin("error_during_execution")]))
    assert conclusion.echec_nomme(_lancer()).startswith("fin_anormale")


def test_auth_mismatch_echoue_en_le_nommant(ferme):
    ferme["reponses"].append(Reponse(200, [
        {"type": "ferme_resume", "erreur": "auth_mismatch : apiKeySource=none"}]))
    with pytest.raises(RuntimeError, match="auth_mismatch"):
        _lancer()


# ── Les tours déclarés partent à la ferme ; le budget reste le sien ──────────────

def test_les_tours_declares_partent_en_max_turns(ferme):
    ferme["reponses"].append(Reponse(200, [INIT, dict(_fin("success"), is_error=False)]))
    _lancer(max_turns=40)
    assert ferme["post"][0]["max_turns"] == 40
    assert "max_budget_usd" not in ferme["post"][0]


def test_sans_tours_declares_la_ferme_garde_son_defaut(ferme):
    ferme["reponses"].append(Reponse(200, [INIT, dict(_fin("success"), is_error=False)]))
    _lancer()
    assert "max_turns" not in ferme["post"][0]


def test_les_tours_declares_passent_tels_quels():
    """Pas de plafond haut : la borne est celle de qui déclare l'agent."""
    assert A.champ_des_tours(5000) == {"max_turns": 5000}
    assert A.champ_des_tours(None) == {}


def test_sans_max_steps_la_boucle_n_a_pas_de_borne_en_tours():
    """Ni défaut (24) ni plafond dur (64) : déclaré, tel quel ; absent, aucun."""
    assert AgentSpec(system="s", tools=frozenset()).max_steps is None
    assert worker._spec_du_job({"payload": {"max_steps": 500}}).max_steps == 500
    assert worker._spec_du_job({"payload": {}}).max_steps is None


def test_max_steps_zero_leve_absent_reste_sans_borne(tmp_path):
    """`0` n'est pas « sans borne » : il lève, sur chaque chemin qui lit la déclaration
    (charge du travail, flotte en base, fichier). Absent (None) reste sans borne."""
    with pytest.raises(ValueError, match="max_steps = 0"):
        worker._spec_du_job({"payload": {"max_steps": 0}})
    with pytest.raises(ValueError, match="max_steps = 0"):
        spec_depuis_flotte({"id": 3, "procedure": "p", "input": "fais ceci",
                            "tools": ["oto_procedure"], "max_steps": 0})
    y = tmp_path / "f.yaml"
    y.write_text("procedure: p\nnamespace: t\ntools: [oto_procedure]\ninput: fais\n"
                 "max_steps: 0\n")
    with pytest.raises(ValueError, match="max_steps = 0"):
        load_spec(str(y))
    assert worker._spec_du_job({"payload": {"max_steps": None}}).max_steps is None
    assert spec_depuis_flotte({"id": 3, "procedure": "p", "input": "fais ceci",
                               "tools": ["oto_procedure"], "max_steps": None}).max_steps is None
    y.write_text("procedure: p\nnamespace: t\ntools: [oto_procedure]\ninput: fais\n")
    assert load_spec(str(y)).max_steps is None


def test_le_worker_ne_transmet_que_les_tours_declares():
    spec = AgentSpec(system="s", tools=frozenset())
    note = lambda e, **c: None  # noqa: E731
    assert worker._bornes_du_one_shot(spec, F, note, max_steps=40) == {"max_turns": 40}
    assert worker._bornes_du_one_shot(spec, F, note) == {}
    assert worker._bornes_du_one_shot(spec, C, note, max_steps=40) == {}


# ── L'attente d'une place ne se prend pas sur la durée du run ────────────────────

def test_l_echeance_part_de_l_acceptation_pas_de_la_file(ferme):
    t = {"now": 0.0}

    def attendre(s):
        t["now"] += s
    ferme["reponses"] += [Reponse(429), Reponse(429),
                          Reponse(200, [INIT, dict(_fin("success"), is_error=False)])]
    # 20 + 40 s d'attente pour une échéance de 50 s : le run tourne quand même.
    res = _lancer(max_seconds=50, horloge=lambda: t["now"], attendre=attendre)
    assert res.stopped == "end_turn"


# ── B2 : Conversations, une échéance déclarée est TOUJOURS une borne ─────────────

@pytest.mark.parametrize("declaree", [600, 900, 1800])
def test_une_echeance_declaree_depassee_est_une_borne_meme_rabotee(monkeypatch, declaree):
    monkeypatch.setattr(C, "resolve_key", lambda: "k")
    monkeypatch.setattr(C, "connector_id", lambda: "c")
    monkeypatch.setattr(C, "modele_resolu", lambda nom: nom)

    def coupe(url, corps, entetes, wall_s=C._WALL_S):
        raise DeadlineExceeded("wall")
    monkeypatch.setattr(C, "_poster", coupe)
    res = C.run_once(instructions="i", inputs="x", tools=["t"], max_seconds=declaree)
    assert res.stopped == "max_seconds"


def test_les_rejeux_se_partagent_l_echeance(monkeypatch):
    t = {"now": 0.0}
    vus = []

    class R:
        status_code, text = 502, "bad gateway"

    def post(url, wall_s, **kw):
        vus.append(wall_s)
        t["now"] += wall_s                      # le transitoire a mangé tout son temps
        return R()
    monkeypatch.setattr(C, "post_with_deadline", post)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    with pytest.raises(DeadlineExceeded):
        C._poster("u", {}, {}, wall_s=120, horloge=lambda: t["now"])
    assert vus == [120]                         # pas trois fois l'échéance


# ── B3 : un `continue` n'est jamais servi sans son fil ───────────────────────────

@pytest.mark.parametrize("moteur", [F, A, C])
def test_un_continue_sur_un_moteur_one_shot_est_refuse(moteur):
    with pytest.raises(worker.ReprisePerdue, match="continue"):
        worker._exiger_reprise_servie({"id": 9, "kind": "continue"}, moteur)


def test_un_start_repris_et_la_boucle_passent():
    worker._exiger_reprise_servie({"id": 9, "kind": "start", "run_id": "r"}, F)
    worker._exiger_reprise_servie({"id": 9, "kind": "continue"},
                                  types.SimpleNamespace(ONE_SHOT=False))
