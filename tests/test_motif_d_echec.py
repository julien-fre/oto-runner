"""La cause d'une mort, EN TÊTE de ce qui la dit.

⚠️ Le 02/10/2026, un agent planifié est mort trois fois en une seconde sur un refus
du fournisseur. Le motif rendu commençait par `Error code: 400 - {'type': 'error',
'error': {'type': 'invalid_request_error', '` : l'enveloppe. La lentille des agents
en échec n'en garde que la tête, et la raison du fournisseur n'y entrait pas.
"""
from __future__ import annotations

from oto_runner import conclusion
from oto_runner.conclusion import RunEnCours


class BadRequestError(Exception):
    """La forme des erreurs de statut des SDK Anthropic et OpenAI : `status_code`
    et `body`, le message en `Error code: N - {…}`."""

    def __init__(self, body):
        super().__init__(f"Error code: 400 - {body}")
        self.status_code = 400
        self.body = body


_ANTHROPIC = {"type": "error", "request_id": "req_1",
              "error": {"type": "invalid_request_error",
                        "message": "model: claude-inconnu n'existe pas"}}


def test_la_raison_du_fournisseur_passe_devant_l_enveloppe():
    motif = conclusion.motif_d_echec(BadRequestError(_ANTHROPIC))
    assert motif == ("BadRequestError : HTTP 400 invalid_request_error — "
                     "model: claude-inconnu n'existe pas")
    assert "model: claude-inconnu" in motif[:80], "la tête que garde `agents_en_echec` porte la cause"


def test_un_corps_sans_enveloppe_error_se_lit_aussi():
    """OpenAI rend parfois l'objet d'erreur lui-même en `body`."""
    motif = conclusion.motif_d_echec(BadRequestError(
        {"message": "max_tokens trop grand", "code": "invalid_value"}))
    assert motif == "BadRequestError : HTTP 400 invalid_value — max_tokens trop grand"


def test_une_exception_ordinaire_reste_dite_telle_quelle():
    assert (conclusion.motif_d_echec(RuntimeError("Read timed out."))
            == "RuntimeError : Read timed out.")


def test_un_corps_sans_message_n_est_pas_reconstruit():
    e = BadRequestError({"type": "error"})
    assert conclusion.motif_d_echec(e) == f"BadRequestError : {e}"


class _Mcp:
    def __init__(self):
        self.vus = []

    def outil(self, nom, args):
        self.vus.append((nom, args))
        return {}


class _File:
    def __init__(self):
        self.vus = []

    def complete(self, job_id, **kw):
        self.vus.append(kw)


def test_le_run_et_la_file_recoivent_la_cause_en_tete():
    mcp, file = _Mcp(), _File()
    conclusion.en_echec(None, RunEnCours(mcp=mcp, run_id="r1"), {"id": 7}, file,
                        BadRequestError(_ANTHROPIC), "claude-inconnu")
    (nom, args), = mcp.vus
    assert nom == "run_finish" and args["outcome"] == "failed"
    assert args["note"].startswith(
        "travail interrompu — BadRequestError : HTTP 400 invalid_request_error — model:")
    assert file.vus[0]["error"].startswith("BadRequestError : HTTP 400 invalid_request_error")
