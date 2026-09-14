"""La borne des descriptions d'outils servies au modèle, outil par outil.

Réglée par la DÉCLARATION du passage (`descriptions_outils: {defaut: 1024, entieres: [data_write]}`),
jamais par l'environnement : c'est un choix de passage, comme la température (« il doit être
paramétrable », décision d'Alexis du 09/09/2026 pour celle-ci ; même doctrine ici, 11/09/2026).

Mesuré le 11/09/2026 sur la passe A du banc des textes courts (13 outils) :
- servies entières (borne 8 192), les descriptions pèsent 34,4 k caractères : l'entrée hors cache
  triple et le coût de la passe double ;
- coupées toutes à 1 024, `data_write` (7 865) perdait ses règles (« @keep seul dans la couche »,
  `@empty`, l'écriture par `id`) : 14 cases « @keep — … » sur le lot 01 du vivier.
D'où le défaut quand la déclaration se tait : `data_write` entière, les autres à 1 024
(18,9 k caractères sur la même passe).

⚠️ **La même grammaire vit dans le backend** (oto-backend
`oto_mcp/capabilities/_descriptions_outils.py`, autre dépôt, oto#241) : une campagne hébergée
la déclare sur `oto_fleet op=create`, le backend la valide à la pose puis la passe au travail
(`payload["descriptions_outils"]`). Qui change cette forme d'un côté change l'autre (14/09/2026).
"""
from __future__ import annotations

from typing import Optional

DESCRIPTIONS_PAR_DEFAUT = {"defaut": 1024, "entieres": ("data_write",)}


def reglage(brut: Optional[dict]) -> dict:
    """Le réglage déclaré, complété des défauts. Une forme fausse LÈVE : pas de repli qui la masque."""
    if brut is None:
        return dict(DESCRIPTIONS_PAR_DEFAUT)
    if not isinstance(brut, dict) or set(brut) - {"defaut", "entieres"}:
        raise ValueError(f"descriptions_outils = {brut!r} : deux clés permises, `defaut` "
                         "(un entier ≥ 1) et `entieres` (une liste de noms d'outils)")
    defaut = brut.get("defaut", DESCRIPTIONS_PAR_DEFAUT["defaut"])
    if isinstance(defaut, bool) or not isinstance(defaut, int) or defaut < 1:
        raise ValueError(f"descriptions_outils.defaut = {defaut!r} : un entier ≥ 1 est attendu")
    entieres = brut.get("entieres", DESCRIPTIONS_PAR_DEFAUT["entieres"])
    if not isinstance(entieres, (list, tuple)) or not all(isinstance(x, str) and x for x in entieres):
        raise ValueError(f"descriptions_outils.entieres = {entieres!r} : une liste de noms d'outils est attendue")
    return {"defaut": defaut, "entieres": tuple(entieres)}


def borne(outil: str, r: dict, longueur: int) -> int:
    """Ce qu'on sert de la description de cet outil, selon le réglage `r` (celui de `reglage`)."""
    return longueur if outil in r["entieres"] else min(longueur, r["defaut"])
