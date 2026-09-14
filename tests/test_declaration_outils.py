"""Une déclaration qui demande de LIRE ce qu'elle n'autorise pas à lire ne part pas.

Trouvé le 06/09/2026 dans le journal d'un travail (événement #5 : « Outil
`oto_procedure` indisponible pour ce run ») : l'instruction disait « Lis d'abord
la procédure … avec `oto_procedure` », et `oto_procedure` n'était pas dans
`tools`. Le worker n'injecte rien de métier, l'allowlist est fail-closed : l'agent
n'a JAMAIS lu la consigne, dans aucune flotte de la campagne — et chaque travail
concluait « done ».

Ce que ces bancs figent : le refus à la déclaration (YAML et flotte en base),
franc et nommé ; l'exemple documenté qui passe ; et, au journal du travail, les
outils autorisés dès `debut` puis l'ÉCART entre ce que l'instruction nomme et ce
que la liste autorise — confronté au catalogue réel, jamais deviné.
"""
from __future__ import annotations

import json
import types

import pytest

from oto_runner import journal
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult
from oto_runner.declaration import load_spec, spec_depuis_flotte, verifier_outils
from tests.test_fleet import _spec
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

_YAML = "procedure: enrichissement\nnamespace: n\ninput: {input}\ntools: [{tools}]\n"


def _ecrire(tmp_path, input="Fais ceci.", tools="data_claim_next, data_write"):
    d = tmp_path / "campagne.yaml"
    d.write_text(_YAML.format(input=input, tools=tools))
    return str(d)


def test_une_procedure_nommee_sans_oto_procedure_est_REFUSEE_au_chargement(tmp_path):
    with pytest.raises(ValueError) as e:
        load_spec(_ecrire(tmp_path))
    dit = str(e.value)
    assert "n'autorise pas l'outil qui la lit" in dit
    assert "oto_procedure" in dit and "enrichissement" in dit, "le refus NOMME le défaut"


def test_une_instruction_qui_nomme_oto_procedure_sans_l_autoriser_est_REFUSEE(tmp_path):
    chemin = _ecrire(tmp_path, input="Lis d'abord la procédure avec oto_procedure.",
                     tools="data_write")
    with pytest.raises(ValueError, match="son instruction nomme `oto_procedure`"):
        load_spec(chemin)


def test_avec_oto_procedure_dans_tools_la_declaration_passe(tmp_path):
    spec = load_spec(_ecrire(tmp_path, input="Lis la procédure avec oto_procedure.",
                             tools="oto_procedure, data_claim_next, data_write"))
    assert "oto_procedure" in spec.tools


def test_la_flotte_declaree_en_base_subit_le_meme_refus():
    with pytest.raises(ValueError, match="oto_procedure"):
        spec_depuis_flotte({"id": 1, "procedure": "p", "namespace": "n",
                            "tools": ["data_write"], "input": "fais ceci"})


def test_une_spec_construite_en_code_se_verifie_de_la_meme_facon():
    verifier_outils(_spec(tools=("oto_procedure", "data_claim_next")))
    with pytest.raises(ValueError):
        verifier_outils(_spec())          # procedure="p", tools sans oto_procedure


def test_l_exemple_documente_est_une_declaration_qui_PART(caplog):
    """`docs/fleet-example.yaml` est ce que tout le monde copie : il doit passer
    la validation — avec `oto_procedure` ET `data_release` — et n'avertir de RIEN
    (il portait encore deux réglages qui n'ont jamais existé dans le code)."""
    import logging
    import pathlib
    exemple = pathlib.Path(__file__).resolve().parents[1] / "docs" / "fleet-example.yaml"
    with caplog.at_level(logging.WARNING):
        spec = load_spec(str(exemple))
    assert {"oto_procedure", "data_claim_next", "data_release"} <= set(spec.tools)
    assert spec.input.strip(), "l'instruction est OBLIGATOIRE : l'exemple en porte une"
    assert not caplog.records, [r.getMessage() for r in caplog.records]


# ── Au journal du travail : les outils autorisés, et l'écart ──────────────────

CATALOGUE = frozenset({"oto_procedure", "data_claim_next", "data_release", "data_write",
                       "data_rows", "fr_get", "serper_search", "run_start", "run_finish"})


def test_l_ecart_confronte_l_instruction_au_catalogue_reel():
    instruction = ("Lis d'abord la procedure avec oto_procedure. Ta file est le tableau "
                   "`vivier`, filtre {\"statut\": \"a_enrichir\", \"lot_test\": \"banc\"}. "
                   "Reserve UNE ligne avec data_claim_next, puis ecris avec data_write.")
    ecart = journal.ecart_instruction(CATALOGUE, ("data_claim_next", "data_write",
                                                  "outil_retire_du_catalogue"), instruction)
    assert ecart["nommes_hors_liste"] == ["oto_procedure"], "LE cas du 04–06/09"
    assert ecart["autorises_inconnus"] == ["outil_retire_du_catalogue"]
    assert ecart["catalogue"] == len(CATALOGUE)


def test_l_ecart_ne_devine_pas_un_outil_dans_un_mot_qui_lui_ressemble():
    """`a_enrichir`, `lot_test` ressemblent à des outils : seul le catalogue
    décide. Et `data_write` dans `mon_data_write_perso` n'est pas un appel."""
    ecart = journal.ecart_instruction(CATALOGUE, ("data_write",),
                                      "statut a_enrichir, lot_test, mon_data_write_perso, data_rows.")
    assert ecart["nommes_hors_liste"] == ["data_rows"]


def test_sans_catalogue_l_ecart_est_OMIS_avec_sa_raison():
    ecart = journal.ecart_instruction(None, ("data_write",), "avec oto_procedure")
    assert ecart == {"nommes_hors_liste": None, "autorises_inconnus": None,
                     "ecart_omis": "transport sans catalogue d'outils"}


def test_le_journal_du_travail_porte_l_ecart_des_l_ouverture(monkeypatch, tmp_path):
    """L'événement qui aurait tout dit dès le 04/09 : `debut` liste les outils
    autorisés, `outils` (sitôt la session MCP ouverte) dit ce que l'instruction
    nomme hors de la liste, d'après le catalogue que la session voit."""
    class McpAvecCatalogue(FauxMcp):
        def catalogue(self):
            return CATALOGUE

    monkeypatch.setattr(W, "McpSession", McpAvecCatalogue)
    monkeypatch.setattr(W.agent_runtime, "run",
                        lambda *a, **k: AgentResult(reply="fini", stopped="end_turn"))
    job = _job("start")
    job["payload"].update(fleet="banc", tools=["data_claim_next", "data_write"],
                          input="Lis d'abord la procédure avec oto_procedure, puis "
                                "réserve une ligne avec data_claim_next.")
    W._un_travail(FauxBackend(), job, types.SimpleNamespace(__name__="p", model=lambda: "m"))
    evs = [json.loads(l) for l in open(journal.chemin("banc", 7, 1))]
    assert [e["ev"] for e in evs][:3] == ["debut", "outils", "run"]
    assert evs[0]["outils_autorises"] == ["data_claim_next", "data_write"]
    assert evs[1]["autorises"] == ["data_claim_next", "data_write"]
    assert evs[1]["nommes_hors_liste"] == ["oto_procedure"]
    assert evs[1]["autorises_inconnus"] == []
