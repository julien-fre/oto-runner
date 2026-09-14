"""La clé de modèle vient AVEC le travail, et ne lui survit pas.

Une org dépose sa clé chez Otomata comme elle dépose celle de Folk ou de Zoho ;
le backend la remet avec chaque travail que ses agents réservent. Le worker ne
la cherche jamais : il ne parle pas au coffre, il reçoit de quoi exécuter CE
travail-là et rien de plus.

Ce que ces bancs protègent :

1. **Le worker nomme son dépôt** — sans quoi le backend ne peut pas savoir
   quelle clé servir, et toutes les orgs continuent de tourner sur la nôtre sans
   que rien ne le dise.
2. **La clé reçue est celle du travail en cours**, jamais gardée pour le
   suivant : une clé qui traîne fait payer une org pour le travail d'une autre,
   et le seul endroit où ça se verrait est sa facture.
3. **On ne devine pas à qui appartient un hôte.** `OTO_RUNNER_OPENAI_API_KEY`
   sert Scaleway comme La Plateforme : réclamer « la clé mistral » d'une org qui
   tourne chez Scaleway lui demanderait la clé d'un fournisseur qu'elle n'utilise
   pas.
"""
import oto_runner.worker as worker
from oto_runner import agent_conversations, agent_llm, agent_llm_openai


# ── qui sait consommer quoi ───────────────────────────────────────────────────

def test_le_provider_anthropic_nomme_son_depot():
    assert agent_llm.depot() == "anthropic"


def test_un_hote_sans_depot_connu_n_en_reclame_aucun(monkeypatch):
    """Scaleway par défaut : aucune org ne dépose de clé « scaleway » aujourd'hui,
    et en réclamer une au hasard ferait servir la mauvaise."""
    monkeypatch.delenv("OTO_RUNNER_OPENAI_BASE", raising=False)
    assert agent_llm_openai.depot() == ""


def test_l_hote_decide_du_depot_pas_le_nom_du_module(monkeypatch):
    """Les deux providers lisent la MÊME variable d'hôte : c'est elle qui dit
    chez qui l'org tourne, et donc quelle clé lui demander."""
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", "https://api.mistral.ai/v1")
    assert agent_llm_openai.depot() == "mistral"
    monkeypatch.delenv("OTO_RUNNER_OPENAI_BASE")
    assert agent_conversations.depot() == "mistral", "La Plateforme par défaut"
    monkeypatch.setenv("OTO_RUNNER_OPENAI_BASE", "https://api.scaleway.ai")
    assert agent_conversations.depot() == ""


# ── le worker le nomme à la réservation ───────────────────────────────────────

class _BackendQuiNote:
    def __init__(self):
        self.base = "https://exemple.invalide"
        self.depots = []

    def claim(self, lease_seconds=600, depot=""):
        self.depots.append(depot)
        worker._demander_arret(15, None)
        return None

    def complete(self, *a, **kw):
        pass


class _Provider:
    ONE_SHOT = True
    __name__ = "agent_llm"

    @staticmethod
    def resolve_key():
        return "cle-de-la-plateforme"

    @staticmethod
    def model():
        return "un-modele"

    @staticmethod
    def depot():
        return "anthropic"


def test_le_worker_nomme_son_depot_a_chaque_reservation(monkeypatch):
    worker._arret_demande = False
    bk = _BackendQuiNote()
    monkeypatch.setenv("OTO_RUNNER_ARMED", "1")
    monkeypatch.setenv("OTO_WORKER_SECRET", "otow_banc")
    monkeypatch.setattr(worker, "Backend", lambda *a, **kw: bk)
    monkeypatch.setattr(worker, "get_provider", lambda: _Provider)
    monkeypatch.setattr(worker, "_POLL_S", 0)
    worker.main()
    assert bk.depots == ["anthropic"]


def test_un_provider_sans_depot_ne_fait_pas_tomber_le_worker(monkeypatch):
    """Un provider d'un autre âge (ou d'un hôte inconnu) n'a pas de `depot` : le
    worker réserve comme avant et tourne sur la clé de la plateforme."""
    worker._arret_demande = False
    bk = _BackendQuiNote()

    class _Vieux(_Provider):
        depot = None

    monkeypatch.setenv("OTO_RUNNER_ARMED", "1")
    monkeypatch.setenv("OTO_WORKER_SECRET", "otow_banc")
    monkeypatch.setattr(worker, "Backend", lambda *a, **kw: bk)
    monkeypatch.setattr(worker, "get_provider", lambda: type(
        "P", (), {"ONE_SHOT": True, "__name__": "agent_llm",
                  "resolve_key": staticmethod(lambda: "x"),
                  "model": staticmethod(lambda: "m")}))
    monkeypatch.setattr(worker, "_POLL_S", 0)
    worker.main()
    assert bk.depots == [""]


# ── la clé du travail, et rien qu'elle ────────────────────────────────────────

class _Stop(RuntimeError):
    """Sentinelle : ce banc regarde la CLÉ passée, pas le déroulé."""


class _BackendMuet:
    base = "https://exemple.invalide"

    def __getattr__(self, _nom):
        return lambda *a, **k: {}


class _McpMuet:
    def __init__(self, **kw):
        self.run_id = None

    def outil(self, nom, args=None):
        return {"run_id": "run-1", "body_md": "la procédure"}


def _cle_passee_a_la_boucle(monkeypatch, job):
    """Joue `_traiter` jusqu'à la boucle de modèle et rend l'`api_key` reçue."""
    return _vu_par_la_boucle(monkeypatch, job).get("api_key", "PAS D'APPEL")


def _vu_par_la_boucle(monkeypatch, job) -> dict:
    """Joue `_traiter` jusqu'à la boucle de modèle et rend la clé ET le workspace reçus."""
    vue = {}

    def _run(*a, **kw):
        vue["api_key"] = kw.get("api_key")
        vue["workspace"] = kw.get("workspace")
        raise _Stop

    monkeypatch.setattr(worker, "McpSession", _McpMuet)
    monkeypatch.setattr(worker.agent_runtime, "run", _run)

    class _P:
        __name__ = "agent_llm"
        ONE_SHOT = False

        @staticmethod
        def model():
            return "un-modele"

    try:
        worker._traiter(_BackendMuet(), job, _P)
    except _Stop:
        pass
    return vue


def test_la_cle_du_travail_est_celle_qui_paie_le_modele(monkeypatch):
    cle = _cle_passee_a_la_boucle(monkeypatch, {
        "id": 21, "kind": "start", "delegated_token": "oto_delegue", "model_key": "sk-de-l-org",
        "payload": {"project_id": 3, "org_id": 42, "procedure": "veille",
                    "input": "Lis la procédure `veille` et applique-la."}})
    assert cle == "sk-de-l-org"


def test_sans_cle_deposee_le_worker_tourne_sur_celle_de_la_plateforme(monkeypatch):
    """`None`, pas chaîne vide : le provider doit RETOMBER sur son environnement,
    pas présenter une clé vide au fournisseur."""
    cle = _cle_passee_a_la_boucle(monkeypatch, {
        "id": 22, "kind": "start", "delegated_token": "oto_delegue",
        "payload": {"project_id": 3, "org_id": 42, "procedure": "veille",
                    "input": "Lis la procédure `veille` et applique-la."}})
    assert cle is None


def test_la_cle_ne_survit_pas_au_travail(monkeypatch):
    """Un travail avec clé, puis un travail sans : le second ne doit pas hériter.
    Une clé qui traîne fait payer une org pour le travail d'une autre — et le
    seul endroit où ça se verrait est sa facture."""
    _cle_passee_a_la_boucle(monkeypatch, {
        "id": 23, "kind": "start", "delegated_token": "oto_delegue", "model_key": "sk-de-l-org",
        "payload": {"project_id": 3, "org_id": 42, "procedure": "veille",
                    "input": "Lis la procédure `veille` et applique-la."}})
    suivante = _cle_passee_a_la_boucle(monkeypatch, {
        "id": 24, "kind": "start", "delegated_token": "oto_delegue",
        "payload": {"project_id": 3, "org_id": 99, "procedure": "veille",
                    "input": "Lis la procédure `veille` et applique-la."}})
    assert suivante is None

