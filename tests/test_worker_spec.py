"""Le cadre système du worker : le sien, SEUL — un texte joint est refusé, pas encadré.

⚠️ Le worker ne lit aucun objet d'Oto — c'est la règle d'ADR 0064, et elle tient.
Du 09 au 13/09/2026, le travail pouvait porter un `system` — le texte de la
procédure, joint par la réservation — que le cadre encadrait avec « ne la recharge
pas ». Retiré le 13/09/2026 (décision d'Alexis) : l'instruction dit de lire la
procédure, l'AGENT la lit par MCP, et la plateforme n'en injecte aucune seconde
copie. Un `system` encore servi dit qu'un producteur ancien tourne : il est refusé
AVANT toute session ou run, jamais ignoré en silence.

Ces bancs tiennent sur le code d'avant le retrait : leur rouge y est celui du défaut.
"""
from __future__ import annotations

import pytest

from oto_runner import worker
from oto_runner.agent_runtime import AgentResult
from tests.test_worker_reprise import FauxBackend, FauxMcp
from tests.test_worker_reprise import _job as _travail

LECTURE = "Lis la procédure `passe-f` avec oto_procedure et applique-la."


def _job(**kw):
    base = {"id": 1, "payload": {"tools": ["data_rows"], "max_steps": 5}}
    base.update(kw)
    return base


def test_le_cadre_est_celui_du_worker_SEUL():
    spec = worker._spec_du_job(_job())
    assert spec.system == worker._SYSTEM_FRAME, (
        "aucun travail ne doit hériter d'un cadre qu'on ne lui a pas donné")


def test_l_instruction_de_lecture_part_telle_quelle_et_rien_n_est_preinjecte(monkeypatch):
    vu: dict = {}

    def faux_run(spec, transport, provider, prompt=None, **_):
        vu.update(system=spec.system, prompt=prompt, outils=spec.tools)
        return AgentResult(reply="fini", stopped="end_turn")

    monkeypatch.setattr(worker, "McpSession", FauxMcp)
    monkeypatch.setattr(worker.agent_runtime, "run", faux_run)
    job = _travail("start")
    job["payload"].update(input=LECTURE, tools=["oto_procedure", "data_rows"])
    worker._traiter(FauxBackend(), job, provider=None)
    assert vu["prompt"] == LECTURE, "l'instruction de lecture est le premier message, telle quelle"
    assert vu["system"] == worker._SYSTEM_FRAME, "aucun texte de procédure dans le cadre"
    assert "oto_procedure" in vu["outils"], "l'agent lit par l'outil que son travail autorise"


@pytest.mark.parametrize("texte", ["LA CONSIGNE MÉTIER", "   "])
def test_un_system_encore_servi_est_refuse_avant_toute_session(monkeypatch, texte):
    def aucune_session(**kw):
        pytest.fail("aucune session MCP ne doit s'ouvrir sur un travail qui porte un texte joint")

    monkeypatch.setattr(worker, "McpSession", aucune_session)
    b = FauxBackend()
    job = _travail("start")
    job["system"] = texte
    with pytest.raises(Exception) as e:
        worker._traiter(b, job, provider=None)
    assert type(e.value).__name__ == "TexteJointIncompatible", "refusé nommément, pas encadré"
    assert "`system`" in str(e.value)
    assert b.appels == [], "ni run, ni fil, ni conclusion par ce chemin"


@pytest.mark.parametrize("vide", ["", None])
def test_un_system_absent_ou_vide_n_est_pas_un_texte_joint(monkeypatch, vide):
    """Garde négative : prouvée par son vert seul, sans chute."""
    monkeypatch.setattr(worker, "McpSession", FauxMcp)
    monkeypatch.setattr(worker.agent_runtime, "run",
                        lambda *a, **k: AgentResult(reply="fini", stopped="end_turn"))
    b = FauxBackend()
    job = _travail("start")
    job["system"] = vide
    worker._traiter(b, job, provider=None)
    assert ("bind_run", "r-NEUF") in b.appels
