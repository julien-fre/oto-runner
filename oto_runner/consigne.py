"""La procédure JOINTE au travail en mode direct — la même que l'hébergé, ou un refus nommé.

Décision du 13/09/2026 (référent plan) : **une seule copie de la procédure, jointe
au travail, dans les deux modes**, et une procédure déclarée introuvable est une
erreur EXPLICITE avant toute exécution — jamais une dégradation silencieuse.

- **Hébergé** : le backend joint le texte à la réservation (`job.system`).
- **Direct** : ce module lit la MÊME source — la procédure d'org, `body_md` BRUT,
  par `GET /api/me/instructions/{slug}` (hors canal MCP, pas de rendu allégé) — et
  la pose au même endroit, `job.system`. Le travail passe ensuite par le même
  `worker._traiter` et le même cadre : l'agent lit le même texte qu'en hébergé.

Elle est lue UNE fois par passage, au lancement : tous les travaux d'un passage
direct portent la même version, et le descripteur le dit.

⚠️ Mesuré le 13/09/2026 (692 contre 690) : en direct, l'agent chargeait la
procédure par `oto_procedure` pendant qu'en hébergé elle était jointe ET relue —
deux chemins, deux coûts, et une version servie qui n'était écrite nulle part.
"""
from __future__ import annotations

import hashlib
from typing import Optional


class ProcedureIntrouvable(RuntimeError):
    """La procédure déclarée ne peut pas être jointe — `cause` dit pourquoi."""

    def __init__(self, slug: str, org: Optional[int], cause: str, detail: str = ""):
        self.slug, self.org, self.cause = slug, org, cause
        super().__init__(
            f"procedure_introuvable ({cause}) : `{slug}` dans l'org "
            f"{org if org is not None else 'active du jeton'}"
            + (f" — {detail}" if detail else "")
            + ". Aucun travail n'est lancé : un agent sans sa procédure n'exécuterait "
              "pas ce que la déclaration demande.")


def jointe(backend, slug: str, org: Optional[int]) -> dict:
    """`{"system": <body_md brut>, "descripteur": {…}}` — ou `ProcedureIntrouvable`.

    Causes : `absente` (404), `archivee` (retirée, elle ne se sert plus),
    `vide` (aucun corps), `illisible` (toute autre erreur de lecture)."""
    from .backend import BackendError
    try:
        instr = backend._get(f"/api/me/instructions/{slug}", {}, org=org)
    except BackendError as e:
        cause = "absente" if e.status == 404 else "illisible"
        raise ProcedureIntrouvable(slug, org, cause, str(e)) from e
    if not isinstance(instr, dict) or not instr:
        raise ProcedureIntrouvable(slug, org, "illisible", "réponse vide")
    if instr.get("archived_at"):
        raise ProcedureIntrouvable(slug, org, "archivee", f"retirée le {instr['archived_at']}")
    corps = instr.get("body_md") or ""
    if not corps.strip():
        raise ProcedureIntrouvable(slug, org, "vide", f"version {instr.get('version')}")
    return {"system": corps,
            "descripteur": {"mode": "direct", "slug": instr.get("slug") or slug,
                            "scope": "org", "org": org, "version": instr.get("version"),
                            "sha256": hashlib.sha256(corps.encode("utf-8")).hexdigest(),
                            "caracteres": len(corps), "rendu": "brut"}}
