"""Le corps envoyé au serveur ne porte QUE des champs que la route déclare.

⚠️ Incident du 04/09/2026, 15 minutes de production à l'arrêt. Un champ
`provider` ajouté au corps du claim a fait répondre à la route REST
`400 : unknown_fields` — à CHAQUE réservation, sur les trois workers, en boucle.
Rien ne l'avait vu :

- la mesure faite avant le déploiement portait sur le modèle pydantic de la
  capacité (`JobsInput`, `extra` ignoré par défaut) — **la mauvaise couche**.
  C'est le pont REST qui tranche, et lui refuse ce qu'il ne déclare pas ;
- les 203 bancs du runner passaient au vert : aucun ne regardait le CORPS
  réellement envoyé. Les doublures de test acceptent `**kw` et notent l'appel,
  pas la requête.

D'où ce banc, qui tient la classe : ajouter un champ à un corps servi est un
changement de CONTRAT, et il se déploie côté serveur AVANT le client.
"""
from __future__ import annotations

import pytest

from oto_runner.backend import Backend


@pytest.fixture
def _client(monkeypatch):
    monkeypatch.setenv("OTO_TOKEN", "jeton-de-test")
    return Backend(base="https://exemple.invalide")


def _corps_de(client, appel, monkeypatch):
    vu = {}

    def _post(chemin, corps, *a, **k):
        vu["chemin"], vu["corps"] = chemin, corps
        return {}

    monkeypatch.setattr(client, "_post", _post)
    appel()
    return vu


# Les champs que la route POST /api/me/runner/jobs DÉCLARE, relevés sur l'OpenAPI
# servi par la production (`GET /api/openapi.json`, v1.195.0+984f8f68, 04/09/2026).
# L'adaptateur REST du backend construit sa garde sur la MÊME collection
# (`cap.Input.model_fields`, `capabilities/_rest_adapter.py`) : ce que l'OpenAPI
# déclare est exactement ce que la garde accepte.
#
# ⚠️ Ajouter un champ au corps sans l'ajouter ici fait échouer ce banc — et c'est
# le but : la question à se poser n'est pas « pydantic l'ignorera-t-il ? » mais
# « la route servie le DÉCLARE-t-elle ? ». La réponse se lit sur la prod.
_DECLARES_PAR_LA_ROUTE = {
    "cursor", "error", "fleet_id", "job_id", "kind", "lease_seconds", "limit",
    "max_attempts", "ok", "op", "payload", "provider", "result", "run_id", "status",
    # 13/09/2026 — déclaré par `runner.jobs` depuis le lot « worker sans clé propre »
    # d'oto-backend. ⚠️ N'est ENVOYÉ que par un worker `OTO_RUNNER_ORG_KEYS_ONLY=1` :
    # un tel worker exige ce backend-là ; un worker ordinaire n'envoie pas le champ.
    "org_key_only",
}


def test_le_claim_n_envoie_que_des_champs_que_la_route_declare(_client, monkeypatch):
    vu = _corps_de(_client, lambda: _client.claim(lease_seconds=600, depot="anthropic"),
                   monkeypatch)
    assert vu["chemin"] == "/api/me/runner/jobs"
    inconnus = set(vu["corps"]) - _DECLARES_PAR_LA_ROUTE
    assert not inconnus, (
        f"{inconnus} n'est pas déclaré par la route servie : la réservation "
        "partira en `400 : unknown_fields`, sur toute la flotte, en boucle")
    assert vu["corps"]["provider"] == "anthropic"


def test_sans_depot_le_champ_ne_part_pas(_client, monkeypatch):
    """Un worker dont l'hôte n'a pas de dépôt connu n'envoie pas un `provider`
    vide : le serveur n'aurait rien à en faire, et un champ vide se lit comme un
    dépôt nommé qui n'existe pas."""
    vu = _corps_de(_client, lambda: _client.claim(lease_seconds=600, depot=""),
                   monkeypatch)
    assert set(vu["corps"]) == {"op", "lease_seconds"}
