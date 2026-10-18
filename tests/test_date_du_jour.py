"""La date du jour, donnée à l'agent par sa consigne.

⚠️ Ce fichier existe parce que l'agent ne recevait AUCUNE date : il datait ses
recherches du 09/09/2026, recopié des exemples de sa procédure, quel que soit le
jour (63 fois sur le lot 01 du vivier, le 11/09/2026). Le marqueur
`{date_du_jour}` de la consigne est remplacé quand le travail se construit.
"""
from __future__ import annotations

from datetime import date

from oto_runner.declaration import load_spec, payload


def _consigne(tmp_path, texte):
    y = tmp_path / "f.yaml"
    y.write_text("procedure: p\nnamespace: t\ntools: [oto_procedure]\n"
                 f"input: {texte!r}\n")
    return payload(load_spec(str(y)))["input"]


def test_la_date_du_jour_remplace_le_marqueur(tmp_path):
    assert (_consigne(tmp_path, "Nous sommes le {date_du_jour}.")
            == f"Nous sommes le {date.today().strftime('%Y-%m-%d')}.")


def test_sans_marqueur_la_consigne_reste_identique(tmp_path):
    assert _consigne(tmp_path, "Lis la procedure.") == "Lis la procedure."
