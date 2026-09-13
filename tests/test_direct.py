"""Le mode DIRECT : le MÊME corps d'exécution que le worker, sans file de travaux.

« Soit prise par DB, soit direct. » Ce que ces bancs figent : les trois verbes
vers la file (`bind_run`, `extend`, `complete`) sont le SEUL point de variation —
le mode direct et le worker appellent la même fonction de traitement avec le
même travail construit, et deux travaux joués l'un sous la file serveur, l'autre
sans file, laissent le même journal ; le mode direct n'appelle JAMAIS
`/api/me/runner/jobs` ; la concurrence se fait en processus (SIGALRM) avec un
compteur partagé ; le bilan de fin a la forme du bilan de flotte et porte le
modèle réellement servi à côté du nom demandé.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import types

import pytest

from oto_runner import direct, journal
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult
from oto_runner.backend import Backend
from oto_runner.bilan import ecrire_bilan
from oto_runner.declaration import payload
from oto_runner.file_de_travail import FileDeTravail, SansFile
from tests.test_bilan_statuts import BackendStatuts
from tests.test_fleet import _spec
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job

_PROVIDER = types.SimpleNamespace(__name__="agent_llm_openai",
                                  model=lambda: "gpt-oss-120b")


class _Table:
    """Ce que le mode direct lit du tableau : un compte de lignes. Rien d'autre."""

    def __init__(self, restantes=10):
        self.restantes = restantes

    def count_rows(self, namespace, filter=None, org=None):
        return self.restantes


def _boucle_scriptee(monkeypatch, modele="gpt-oss-120b-2508"):
    monkeypatch.setattr(W, "McpSession", FauxMcp)

    def faux_run(spec, transport, provider, prompt=None, history=None,
                 on_turn=None, on_event=None, **_):
        on_event("modele", {"texte": "je réponds", "modele": modele})
        if on_turn:
            on_turn("assistant", {"text": "je réponds"}, {"role": "assistant"})
        return AgentResult(reply="fini", stopped="end_turn",
                           usage={"input_tokens": 10, "output_tokens": 2}, model=modele)

    monkeypatch.setattr(W.agent_runtime, "run", faux_run)


# ── Le protocole « aucune file » ─────────────────────────────────────────────

def test_sans_file_sert_les_trois_verbes_journalise_et_conserve_le_resultat(caplog):
    f = SansFile()
    assert isinstance(f, FileDeTravail)
    with caplog.at_level(logging.DEBUG):
        f.bind_run("direct-S-1", "r-1")
        f.extend("direct-S-1", 600)
        rep = f.complete("direct-S-1", ok=True, run_id="r-1",
                         result={"usage_tokens": 12, "model": "m-servi"})
        f.complete("direct-S-2", ok=False, error="boum")
    assert f.conclus == {
        "direct-S-1": {"status": "done", "run_id": "r-1", "error": None,
                       "result": {"usage_tokens": 12, "model": "m-servi"}},
        "direct-S-2": {"status": "failed", "run_id": None, "error": "boum", "result": {}},
    }
    assert rep == {"ok": True, "aucune_file": True, "job_id": "direct-S-1"}
    dits = "\n".join(r.getMessage() for r in caplog.records)
    assert "aucune file : le run r-1 aurait été lié au travail direct-S-1" in dits
    assert "aucun bail à prolonger" in dits and "direct-S-2 conclu — failed : boum" in dits


def test_la_file_serveur_sert_le_meme_protocole():
    """Les deux implémentations, nommées : le backend (POST /api/me/runner/jobs)
    et « aucune file » — un seul contrat, deux façons de le servir."""
    assert isinstance(Backend(base="http://x", token="t"), FileDeTravail)


# ── UN SEUL corps d'exécution ────────────────────────────────────────────────

def test_le_mode_direct_et_le_worker_appellent_la_MEME_fonction_avec_le_MEME_travail(monkeypatch):
    """Le mode direct passe par `worker._un_travail` — la fonction que la boucle du
    worker appelle (`main` → `_un_travail(backend, job, provider)`) — avec un
    travail dont le payload est CELUI que la flotte enfile. Seule la file change."""
    spec = _spec(volume=2)
    appels: list = []

    def espion(backend, job, provider, file=None):
        appels.append((job, file))
        file.complete(job["id"], ok=True, run_id=f"r-{job['id']}", result={"usage_tokens": 1})

    monkeypatch.setattr(W, "_un_travail", espion)
    conclus, motif, morts = direct.lancer(spec, _Table(), _PROVIDER, jeton="oto_poste",
                                          stamp="S", plafond=2, k=1)
    assert [j["id"] for j, _ in appels] == ["direct-S-1", "direct-S-2"]
    for job, file in appels:
        assert job["payload"] == payload(spec), "le MÊME travail que la flotte enfile"
        assert job["kind"] == "start" and job["delegated_token"] == "oto_poste"
        assert isinstance(file, SansFile)
    assert set(conclus) == {"direct-S-1", "direct-S-2"} and morts == []
    assert motif == "volume atteint (2 travaux)"
    src = open(W.__file__).read()
    assert "_un_travail(backend, job, provider)" in src, "la boucle du worker appelle la même fonction"


def test_seuls_les_trois_verbes_varient_le_journal_est_IDENTIQUE(monkeypatch, tmp_path):
    """Le même travail joué deux fois par le vrai `_traiter` : sous la file (le
    backend sert les trois verbes) puis sans file (`SansFile`). Les journaux
    sont identiques à l'horodatage près ; seule la destination des trois verbes
    diffère."""
    _boucle_scriptee(monkeypatch)

    def jouer(file, flotte):
        job = _job("start")
        job["payload"]["fleet"] = flotte
        fil = FauxBackend()
        W._un_travail(fil, job, _PROVIDER, file=file)
        evs = [json.loads(l) for l in open(journal.chemin(flotte, 7))]
        for e in evs:
            e.pop("t")
            if e["ev"] == "debut":
                e["job"]["payload"].pop("fleet")
        return fil, evs

    fil_serveur, evs_serveur = jouer(None, "sous-file")       # None = le backend sert la file
    sans = SansFile()
    fil_direct, evs_direct = jouer(sans, "sans-file")
    assert evs_serveur == evs_direct, "rien d'autre ne diffère : même journal"
    assert [e["ev"] for e in evs_direct] == ["debut", "outils", "run", "modele", "resultat"]
    assert ("bind_run", "r-NEUF") in fil_serveur.appels
    assert ("complete", True, "r-NEUF") in fil_serveur.appels
    assert not [a for a in fil_direct.appels if a[0] in ("bind_run", "complete")], (
        "sans file, le backend ne reçoit AUCUN des trois verbes")
    assert [a[0] for a in fil_direct.appels] == ["append"], "le FIL, lui, est posé"
    assert sans.conclus[7]["status"] == "done" and sans.conclus[7]["run_id"] == "r-NEUF"
    assert sans.conclus[7]["result"]["model"] == "gpt-oss-120b-2508"


def test_sans_file_un_travail_MORT_clot_son_run_et_rend_le_sien(monkeypatch, tmp_path):
    """Le chemin d'échec ne varie pas non plus : le run est clos `failed` (ce
    qui libère la ligne réservée), le journal se termine par `resultat`, et le
    travail conclu porte son `run_id` — sans lui, ses refus d'écriture
    n'appartiendraient à personne au bilan.

    ⚠️ En flotte, un travail mort se rejoue plus tard ; en direct il est perdu
    (« volume atteint »). C'est là que la libération immédiate compte le plus."""
    _boucle_scriptee(monkeypatch)

    def meurt(*a, **kw):
        raise RuntimeError("Read timed out. (read timeout=10)")

    monkeypatch.setattr(W.agent_runtime, "run", meurt)
    job = _job("start")
    job["payload"]["fleet"] = "sans-file"
    sans = SansFile()
    W._un_travail(FauxBackend(), job, _PROVIDER, file=sans)

    assert sans.conclus[7]["status"] == "failed"
    assert sans.conclus[7]["run_id"] == "r-NEUF"
    assert "read timeout=10" in sans.conclus[7]["error"]
    evs = [json.loads(l) for l in open(journal.chemin("sans-file", 7))]
    assert [e["ev"] for e in evs[-2:]] == ["erreur", "resultat"]
    assert evs[-1]["outcome"] == "failed" and evs[-1]["run_finish"] == "ok"


def test_le_mode_direct_n_appelle_JAMAIS_la_file_de_jobs(monkeypatch):
    """Le vrai `_traiter`, un vrai `Backend` dont on espionne les routes : le fil du
    run est posé (`/api/me/runs/thread`), le tableau est lu — et pas UNE requête
    vers `/api/me/runner/jobs`."""
    _boucle_scriptee(monkeypatch)

    class BackendEspion(Backend):
        def __init__(self):
            self.base, self.token, self.chemins = "http://x", "t", []

        def _post(self, chemin, corps, token=None):
            self.chemins.append(chemin)
            return {"seq": 1}

        def _get(self, chemin, params, org=None):
            self.chemins.append(chemin)
            return {"total": 3}

    b = BackendEspion()
    conclus, motif, _ = direct.lancer(_spec(namespace="vivier"), b, _PROVIDER,
                                      jeton="t", stamp="S", plafond=1, k=1)
    assert conclus["direct-S-1"]["status"] == "done"
    assert not [c for c in b.chemins if c.startswith("/api/me/runner/jobs")], b.chemins
    assert "/api/me/runs/thread" in b.chemins
    assert "/api/datastore/namespaces/vivier/rows" in b.chemins


def test_la_file_du_tableau_vide_arrete_le_mode_direct(monkeypatch):
    monkeypatch.setattr(W, "_un_travail", lambda *a, **k: pytest.fail("aucun travail ne doit partir"))
    conclus, motif, _ = direct.lancer(_spec(), _Table(restantes=0), _PROVIDER, jeton="t",
                                      stamp="S", plafond=5, k=1)
    assert conclus == {} and motif == "file vide"


def test_le_travail_direct_porte_le_jeton_du_poste_et_le_tag_de_flotte():
    spec = _spec()
    job = direct.travail(spec, direct.identifiant("S", 3), "oto_poste")
    assert job["id"] == "direct-S-3" and job["delegated_token"] == "oto_poste"
    assert job["payload"]["fleet"] == "flotte-demo", "le journal ira sous passages/flotte-demo/"
    assert journal.chemin("flotte-demo", job["id"]).endswith("flotte-demo/direct-S-3.jsonl")


# ── La concurrence : des processus, un compteur partagé ───────────────────────

def test_la_concurrence_se_fait_en_processus_et_le_compteur_est_partage(monkeypatch):
    """SIGALRM n'existe que dans le thread principal : K agents = K processus, et
    le compteur partagé garantit N travaux au TOTAL, numérotés une seule fois."""
    def faux(backend, job, provider, file=None):
        file.complete(job["id"], ok=True, run_id="r",
                      result={"usage_tokens": 1, "pid": os.getpid()})

    monkeypatch.setattr(W, "_un_travail", faux)
    conclus, motif, morts = direct.lancer(_spec(), _Table(), None, jeton="t", stamp="S",
                                          plafond=4, k=2)
    assert sorted(conclus) == ["direct-S-1", "direct-S-2", "direct-S-3", "direct-S-4"]
    assert all(c["result"]["pid"] != os.getpid() for c in conclus.values()), (
        "les travaux ont tourné dans des processus enfants")
    assert motif == "volume atteint (4 travaux)" and morts == []


def test_un_agent_mort_est_nomme_et_les_autres_conclus_restent(monkeypatch):
    def faux(backend, job, provider, file=None):
        if job["id"].endswith("-2"):
            raise RuntimeError("agent planté")
        file.complete(job["id"], ok=True, run_id="r", result={"usage_tokens": 1})

    monkeypatch.setattr(W, "_un_travail", faux)
    conclus, motif, morts = direct.lancer(_spec(), _Table(), None, jeton="t", stamp="S",
                                          plafond=3, k=2)
    assert len(morts) == 1 and morts[0][1] != 0
    assert "direct-S-1" in conclus or "direct-S-3" in conclus


# ── Le bilan de fin : la forme du bilan de flotte, et le modèle SERVI ─────────

class _BackendDirect(BackendStatuts):
    """Le tableau vu du mode direct : des comptes successifs, un schéma, un agrégat."""

    def __init__(self, counts, groupes):
        super().__init__(groupes)
        self.counts = list(counts)

    def count_rows(self, namespace, filter=None, org=None):
        return self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]


def test_le_bilan_direct_a_la_forme_du_bilan_de_flotte_et_porte_le_modele_servi(monkeypatch, tmp_path, caplog):
    decl = tmp_path / "banc.yaml"
    decl.write_text("procedure: p\n")
    spec = _spec(source=str(decl), org=2, volume=2,
                 filter={"statut": "a_enrichir", "lot_test": "banc"})
    b = _BackendDirect(counts=[2, 2, 1, 0], groupes=[{"statut": "enrichi", "count": 2}])

    def faux_un_travail(backend, job, provider, file=None):
        journal.Journal(journal.chemin(spec.name, job["id"])).ecrire("resultat", outcome="done")
        file.complete(job["id"], ok=True, run_id=f"r-{job['id']}",
                      result={"usage_tokens": 1500, "model": "gpt-oss-120b-2508",
                              "stopped": "end_turn"})

    monkeypatch.setattr(W, "_un_travail", faux_un_travail)
    with caplog.at_level(logging.INFO):
        bilan = direct.jouer(spec, b, _PROVIDER, jeton="t", plafond=2, k=1, stamp="S")

    reference = ecrire_bilan(dataclasses.replace(spec, source=""), b, {},
                             lignes_initiales=2, secondes=1, arret="x")
    assert set(bilan) == set(reference), "la même FORME que le bilan de flotte"
    assert bilan["flotte"] == "flotte-demo" and bilan["final"] is True
    assert bilan["arret"] == "volume atteint (2 travaux)"
    assert bilan["lignes"] == {**bilan["lignes"], "depart": 2, "restantes": 0, "sorties": 2,
                               "par_statut": {"enrichi": 2}, "abouties": 2}
    assert bilan["jobs"] == {"termines": 2, "echoues": 0}
    assert bilan["jetons"]["total"] == 3000 and bilan["jetons"]["par_aboutie"] == 1500
    pose = json.loads((tmp_path / "banc.direct-S.bilan.json").read_text())
    assert pose["final"] is True, "posé À CÔTÉ de la déclaration, sous le nom du passage direct"
    assert not (tmp_path / "banc.bilan.json").exists(), "jamais par-dessus le bilan de la flotte"
    dits = [r.getMessage() for r in caplog.records]
    fin = next(d for d in dits if d.startswith("mode direct terminé"))
    assert "modèle demandé gpt-oss-120b · servi gpt-oss-120b-2508 ⚠️ DIFFÉRENT" in fin
    assert any("journal complet : " in d and "direct-S-1.jsonl (1 événement" in d for d in dits), (
        "le journal de chaque travail est RELU avant d'être nommé")


def test_le_bilan_direct_tombe_meme_quand_un_travail_plante(monkeypatch, tmp_path):
    decl = tmp_path / "banc.yaml"
    decl.write_text("procedure: p\n")
    spec = _spec(source=str(decl), org=2, filter={"statut": "a_enrichir", "lot": "x"})
    b = _BackendDirect(counts=[3, 3, 0], groupes=[])

    def explose(backend, job, provider, file=None):
        raise RuntimeError("panne hors contrat")

    monkeypatch.setattr(W, "_un_travail", explose)
    with pytest.raises(RuntimeError, match="panne hors contrat"):
        direct.jouer(spec, b, _PROVIDER, jeton="t", plafond=3, k=1, stamp="S")
    pose = json.loads((tmp_path / "banc.direct-S.bilan.json").read_text())
    assert pose["arret"] == "interrompu" and pose["jobs"] == {"termines": 0, "echoues": 0}


# ── La commande ──────────────────────────────────────────────────────────────

def test_les_arguments_de_la_commande():
    a = direct._arguments(["banc.yaml"])
    # `None` des DEUX côtés : ni le volume ni la concurrence ne sont décidés
    # ici. Ils retombent sur la déclaration, qui est leur domicile. Avant le
    # 07/09/2026 la concurrence valait 1 — un défaut qui écrasait en silence
    # ce que la déclaration disait, et les deux options du même parseur ne
    # traitaient donc pas la déclaration de la même façon.
    assert (a.flotte, a.lignes, a.concurrence) == ("banc.yaml", None, None)
    a = direct._arguments(["banc.yaml", "--lignes", "3", "--concurrence", "2"])
    assert (a.lignes, a.concurrence) == (3, 2)
    with pytest.raises(SystemExit):
        direct._arguments(["banc.yaml", "--concurrence", "0"])


def test_la_commande_refuse_de_partir_sans_jeton(monkeypatch, tmp_path):
    decl = tmp_path / "banc.yaml"
    decl.write_text("procedure: p\nnamespace: n\ninput: fais ceci\ntools: [oto_procedure]\n")
    monkeypatch.delenv("OTO_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="OTO_TOKEN"):
        direct.main([str(decl)])


# ── La concurrence suit la DÉCLARATION, comme le volume ─────────────────────
# `--concurrence` valait 1 par défaut et écrasait le `concurrency` du YAML sans
# rien dire. Une flotte déclarée à 4 agents — y compris déclarée depuis le
# dashboard, où le champ s'appelle `workers` et devient `spec.concurrency` —
# tournait à 1. Mesuré le 07/09/2026 : 72 journaux, tous « 1 agent(s) ».

def test_sans_option_la_concurrence_vient_de_la_declaration(monkeypatch, tmp_path):
    from oto_runner import direct
    vu = {}
    yaml = tmp_path / "f.yaml"
    yaml.write_text(
        "procedure: p\nnamespace: n\ninput: i\ntools: [oto_procedure, data_write]\n"
        "concurrency: 4\n")
    monkeypatch.setenv("OTO_TOKEN", "t")
    monkeypatch.setattr(direct, "get_provider", lambda: _Provider())
    monkeypatch.setattr(direct, "Backend", lambda: object())
    monkeypatch.setattr(direct.journal, "preparer", lambda: None)
    monkeypatch.setattr(direct, "jouer",
                        lambda *a, **k: vu.update(k) or {"ok": True})
    direct.main([str(yaml)])
    assert vu["k"] == 4, (
        "la déclaration dit 4 agents et le passage en lance 1 : le réglage est "
        "inerte, et il l'est en silence")


def test_l_option_reste_prioritaire_sur_la_declaration(monkeypatch, tmp_path):
    """Écraser reste possible — mais explicitement, pas par un défaut."""
    from oto_runner import direct
    vu = {}
    yaml = tmp_path / "f.yaml"
    yaml.write_text(
        "procedure: p\nnamespace: n\ninput: i\ntools: [oto_procedure, data_write]\n"
        "concurrency: 4\n")
    monkeypatch.setenv("OTO_TOKEN", "t")
    monkeypatch.setattr(direct, "get_provider", lambda: _Provider())
    monkeypatch.setattr(direct, "Backend", lambda: object())
    monkeypatch.setattr(direct.journal, "preparer", lambda: None)
    monkeypatch.setattr(direct, "jouer",
                        lambda *a, **k: vu.update(k) or {"ok": True})
    direct.main([str(yaml), "--concurrence", "2"])
    assert vu["k"] == 2


class _Provider:
    @staticmethod
    def resolve_key(): return None


# ── La même source de déclaration que l'ordonnanceur ─────────────────────────
# `fleet` acceptait un YAML OU `#<id>` (une flotte déclarée en base, celle que
# le dashboard montre) ; `direct` exigeait un fichier. Une flotte créée depuis
# l'interface n'était donc jouable que par la file de travaux — alors que c'est
# le mode direct qui sert. Deux modes du même runner ne peuvent pas lire la
# déclaration différemment.

def test_le_mode_direct_joue_une_flotte_DECLAREE_en_base(monkeypatch):
    from oto_runner import direct
    vu = {}
    declaree = {"id": 42, "label": "campagne", "procedure": "p", "namespace": "n",
                "tools": ["oto_procedure", "data_write"], "input": "fais ceci",
                "workers": 3, "org_id": 226, "project_id": 219}

    class _B:
        @staticmethod
        def lire_flotte(fid):
            vu["lue"] = fid
            return declaree

    monkeypatch.setenv("OTO_TOKEN", "t")
    monkeypatch.setattr(direct, "get_provider", lambda: _Provider())
    monkeypatch.setattr(direct, "Backend", lambda: _B())
    monkeypatch.setattr(direct.journal, "preparer", lambda: None)
    monkeypatch.setattr(direct, "jouer", lambda *a, **k: vu.update(k) or {})
    direct.main(["#42"])
    assert vu["lue"] == 42, "le mode direct n'est pas allé chercher la flotte déclarée"
    assert vu["k"] == 3, "les `workers` de la déclaration doivent piloter les agents"
