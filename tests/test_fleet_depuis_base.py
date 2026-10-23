"""Une flotte pilotée par sa configuration DÉCLARÉE, pas par un fichier local.

Jusqu'ici la source d'un passage était un YAML posé à côté de l'exécutable : il
n'existait que pour qui avait accès à la machine. La base le rendait visible
depuis le dashboard, mais en COPIE — le fichier restait la vérité.

> **Un passage piloté par sa configuration en base est le même objet pour
> l'ordonnanceur, pour le dashboard et pour un agent.** Piloté par un fichier, il
> en a autant de versions que de machines.

⚠️ Ce que ces tests gardent surtout, c'est la FRONTIÈRE : tout ne remonte pas en
base. `ramp_seconds`, `critical_tools` et la cadence du bilan sont des réglages
d'EXÉCUTION locale, pas de la configuration déclarée d'un passage. Les faire
remonter « pour tout avoir au même endroit » mélangerait ce qu'un opérateur
DÉCLARE et ce qu'une machine RÈGLE — et le second change sans que le premier
n'ait bougé.
"""
from __future__ import annotations

import pytest

from oto_runner.fleet import FleetSpec, spec_depuis_flotte


def _flotte(**over) -> dict:
    base = {
        "id": 7, "label": "passage-3", "procedure": "enrichissement",
        # `oto_procedure` : une flotte qui nomme une procédure doit autoriser
        # l'outil qui la lit, sinon la déclaration est refusée (06/09).
        "namespace": "edition", "row_filter": {"lot": "a"}, "tools": ["oto_kb", "oto_procedure"],
        "project_id": 12, "org_id": 226, "workers": 4, "max_rows": 150,
        "max_tokens": 2_000_000, "max_steps": 30, "input": "traite la ligne",
    }
    base.update(over)
    return base


def test_la_configuration_declaree_devient_la_spec():
    s = spec_depuis_flotte(_flotte())
    assert isinstance(s, FleetSpec)
    assert s.procedure == "enrichissement" and s.namespace == "edition"
    assert s.filter == {"lot": "a"} and s.tools == ("oto_kb", "oto_procedure")
    assert s.project == 12 and s.org == 226
    assert s.concurrency == 4 and s.volume == 150
    assert s.budget_tokens == 2_000_000 and s.max_steps == 30
    assert s.input == "traite la ligne"
    assert s.name == "passage-3"


def test_la_flotte_chargee_est_REPRISE_pas_redeclaree():
    """⚠️ Elle existe déjà : la piloter ne doit pas en ouvrir une seconde, sinon
    son état serait éclaté sur deux lignes dont aucune ne dirait la vérité."""
    s = spec_depuis_flotte(_flotte())
    assert s.fleet_id == 7


def test_la_source_dit_d_ou_vient_le_passage():
    """Le bilan se pose à côté de sa source. Une flotte lue en base n'a pas de
    fichier : elle porte son identité, pas un chemin qui n'existe pas."""
    assert spec_depuis_flotte(_flotte()).source == "flotte #7"


def test_une_flotte_sans_label_reste_nommable():
    """Le nom sert de tag et de préfixe de bilan : jamais vide, jamais deviné
    depuis autre chose que la flotte elle-même."""
    assert spec_depuis_flotte(_flotte(label=None)).name == "flotte-7"


# ── ce qui NE remonte pas en base ────────────────────────────────────────────

def test_les_reglages_d_execution_locale_restent_aux_defauts_du_runner():
    """⚠️ La frontière. `ramp_seconds`, `critical_tools` et la cadence du bilan
    règlent la MACHINE, pas le passage. Les lire en base donnerait l'illusion
    qu'un opérateur les a déclarés — alors qu'ils changent avec l'infrastructure,
    sans que la configuration du passage n'ait bougé."""
    s = spec_depuis_flotte(_flotte())
    vierge = FleetSpec(procedure="p", namespace="n", tools=(), name="x",
                       input="fais ceci")
    assert s.ramp_seconds == vierge.ramp_seconds
    assert s.critical_tools == vierge.critical_tools
    assert s.bilan_periode_s == vierge.bilan_periode_s


def test_un_champ_absent_retombe_sur_le_defaut_sans_inventer():
    """⚠️ Les défauts d'EXÉCUTION se supposent (combien d'agents, combien de
    tours) ; l'INSTRUCTION ne se suppose pas — elle dit ce que l'agent doit
    faire, et le worker ne le sait pas."""
    s = spec_depuis_flotte({"id": 3, "procedure": "p", "input": "fais ceci",
                            "tools": ["oto_procedure"]})   # exigé dès qu'une procédure est nommée
    assert s.namespace == "" and s.tools == ("oto_procedure",) and s.filter == {}
    assert s.concurrency == 3 and s.max_steps is None   # pas de borne en tours imposée
    assert s.volume is None and s.budget_tokens is None


def test_une_flotte_SANS_instruction_est_refusee():
    """Le lot du 04/09 : il n'y a plus d'instruction par défaut. Une campagne
    sans instruction partait avec un texte que personne n'avait écrit ni relu —
    et dont la FORME enseignait à sauter la recherche (7 jobs sur 11 sans aucun
    appel d'outil, mesuré dans la nuit du 03 au 04/09)."""
    import pytest
    with pytest.raises(ValueError) as e:
        spec_depuis_flotte({"id": 4, "procedure": "p"})
    assert "instruction" in str(e.value)
    # ⚠️ Le refus dit ce qui MANQUE, pas ce qu'il faudrait écrire : le worker ne
    # sait pas ce qu'une instruction doit contenir.
    assert "source" not in str(e.value).lower()


# ── une flotte illisible se REFUSE, elle ne se devine pas ────────────────────

@pytest.mark.parametrize("manquant", ["id", "procedure"])
def test_une_flotte_sans_identifiant_ni_procedure_est_refusee(manquant):
    """⚠️ Sans identifiant, le passage se redéclarerait ; sans procédure, il
    partirait sans savoir quoi jouer. Les deux se refusent NOMMÉMENT plutôt que
    de retomber sur un défaut — un défaut ici produirait un passage qui tourne et
    ne fait rien, ce qui est plus coûteux qu'un refus."""
    f = _flotte()
    f.pop(manquant)
    with pytest.raises(ValueError) as e:
        spec_depuis_flotte(f)
    assert manquant in str(e.value)


def test_une_reponse_vide_est_refusee_et_non_traitee_comme_une_flotte_neuve():
    """Une flotte inconnue rend `{}` côté client : le prendre pour une flotte
    valide ferait partir un passage sur des défauts, sous un identifiant qui
    n'existe pas."""
    with pytest.raises(ValueError):
        spec_depuis_flotte({})
