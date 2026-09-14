"""Le runner CONSERVE TOUT : un journal JSONL par travail, au fil de l'eau, jamais tronqué.

Nuit du 05 au 06/09/2026 : sept écritures refusées sur huit, et rien après coup
pour dire ce que le modèle avait envoyé ni ce que le schéma avait répondu — le
motif était coupé à soixante caractères, le fil ne garde que la sortie tronquée
pour le modèle, la plateforme n'offre aucune lecture des jobs. Et sur le chemin
OpenAI-compatible, sans état chez le fournisseur, ce journal est la SEULE trace.

Ce que ces bancs figent : la sortie d'outil journalisée est ENTIÈRE quand le
modèle, lui, lit la version plafonnée ; les arguments d'appel sont complets ;
chaque événement est écrit dès qu'il a lieu (un plantage laisse tout ce qui le
précède) ; le travail est recopié SANS ses secrets ; le fichier est en 0600 ; le
répertoire se déclare par l'environnement et se refuse au boot s'il n'est pas
inscriptible.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import types

import pytest

from oto_runner import agent_runtime, journal
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentResult, AgentSpec
from tests.test_agent_runtime import FauxProvider, FauxTransport, _turn

SPEC = AgentSpec(system="le cadre", tools=frozenset({"data_rows", "data_write"}),
                 max_steps=4, label="job:1")


def _lignes(chemin) -> list[dict]:
    return [json.loads(l) for l in pathlib.Path(chemin).read_text().splitlines()]


# ── La boucle : tout, entier ─────────────────────────────────────────────────

def test_une_erreur_d_outil_longue_est_journalisee_ENTIERE_et_plafonnee_pour_le_modele(
        tmp_path, monkeypatch):
    """LE cas : le refus d'un schéma nomme la colonne et la raison — en fin de
    message. Le modèle lit la version plafonnée ; le journal garde tout.

    Le plafond est abaissé ici pour tenir en un test : à son défaut (120 000),
    ce refus de 42 k passerait entier — c'est le sujet d'un autre banc."""
    monkeypatch.setenv("OTO_RUNNER_MAX_TOOL_OUTPUT", "12000")
    refus = ("écriture refusée par le schéma : `qualification_piece` = "
             "`cessation_registre` n'est pas une option — " + "détail " * 6_000)
    assert len(refus) > agent_runtime.max_tool_output()
    t = FauxTransport({"data_write": (refus, True)})
    arguments = {"namespace": "vivier", "key": "@claimed",
                 "patch": {"qualification_piece": "cessation_registre"}}
    p = FauxProvider([_turn(text="j'écris", calls=[("data_write", arguments)]),
                      _turn(text="fini")])
    j = journal.Journal(str(tmp_path / "flotte" / "1.jsonl"))
    res = agent_runtime.run(SPEC, t, p, prompt="vas-y", on_event=j.evenement)

    evs = _lignes(j.chemin)
    assert [e["ev"] for e in evs] == ["systeme", "utilisateur", "modele", "outil",
                                      "modele", "fin"]
    assert all(e["t"].endswith("Z") for e in evs), "chaque événement est horodaté"
    outil, = [e for e in evs if e["ev"] == "outil"]
    assert outil["texte"] == refus, "le journal porte la sortie ENTIÈRE"
    assert outil["ok"] is False and outil["tronque_pour_le_modele"] is True
    assert outil["servi_chars"] < len(refus), (
        "le journal dit AUSSI la longueur réellement servie au modèle — sans "
        "elle, « qu'a-t-il lu ? » n'a pas de réponse après coup")
    assert evs[0]["max_tool_output"] == 12_000, "sous quel plafond ce déroulé a tourné"
    assert outil["arguments"] == arguments, "les arguments sont complets, pas résumés"
    lu_par_le_modele = res.messages[-2]["content"][0]["content"]
    assert len(lu_par_le_modele) == outil["servi_chars"]
    assert "SORTIE TRONQUÉE" in lu_par_le_modele
    assert evs[0]["texte"] == "le cadre" and evs[1]["texte"] == "vas-y"
    modele = evs[2]
    assert modele["appels"] == [{"id": "t0", "nom": "data_write", "arguments": arguments}]
    assert evs[-1]["stopped"] == "end_turn" and evs[-1]["reponse"] == "fini"
    assert evs[-1]["pas"] == 1 and "usage" in evs[-1]


def test_le_journal_s_ecrit_au_fil_de_l_eau_un_plantage_laisse_sa_trace(tmp_path):
    """Écrit à CHAQUE événement, pas à la fin : ce qui précède le plantage est
    sur disque quand la pile remonte."""
    class Meurt(FauxProvider):
        def complete(self, **kw):
            if not self.file:
                raise RuntimeError("fournisseur mort au 2e tour")
            return super().complete(**kw)

    p = Meurt([_turn(text="je cherche", calls=[("data_rows", {"limit": 1})])])
    j = journal.Journal(str(tmp_path / "flotte" / "2.jsonl"))
    with pytest.raises(RuntimeError, match="2e tour"):
        agent_runtime.run(SPEC, FauxTransport(), p, prompt="go", on_event=j.evenement)
    assert [e["ev"] for e in _lignes(j.chemin)] == ["systeme", "utilisateur",
                                                     "modele", "outil"]


def test_sans_journal_la_boucle_est_inchangee():
    p = FauxProvider([_turn(text="fini")])
    assert agent_runtime.run(SPEC, FauxTransport(), p, prompt="go").reply == "fini"


# ── Le worker : du début à la conclusion, ou au plantage ─────────────────────

# Un fournisseur est un MODULE (`agent_llm_openai`) : on le double par un objet à
# attributs, pas par une classe — `type.__name__` l'emporterait sur le nôtre.
_Provider = types.SimpleNamespace(__name__="agent_llm_openai",
                                  model=lambda: "gpt-oss-120b")


def _boucle_scriptee(monkeypatch, resultat=None, leve=None):
    from tests.test_worker_reprise import FauxMcp
    monkeypatch.setattr(W, "McpSession", FauxMcp)

    def faux_run(spec, transport, provider, prompt=None, history=None,
                 on_turn=None, on_event=None, **_):
        on_event("modele", {"texte": "je réponds"})
        if leve:
            raise leve
        return resultat or AgentResult(reply="fini", stopped="end_turn",
                                       usage={"input_tokens": 10, "output_tokens": 2})

    monkeypatch.setattr(W.agent_runtime, "run", faux_run)


def _job():
    from tests.test_worker_reprise import _job as job_de_base
    job = job_de_base("start")
    job["payload"]["fleet"] = "banc-demo"
    job["model_key"] = "sk-la-cle-de-l-org"
    job["model_workspace"] = "wrkspc_du_banc"
    return job


def test_le_journal_du_worker_ouvre_conclut_et_ne_porte_aucun_secret(monkeypatch, tmp_path, caplog):
    from tests.test_worker_reprise import FauxBackend
    _boucle_scriptee(monkeypatch)
    with caplog.at_level(logging.INFO):
        W._un_travail(FauxBackend(), _job(), _Provider)
    annonce = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("job 7 : journal"))
    assert "7.1.jsonl (5 événements, dernier : resultat)" in annonce, (
        "le worker n'annonce son journal qu'après l'avoir RELU")

    chemin = tmp_path / "passages" / "banc-demo" / "7.1.jsonl"
    assert chemin.exists(), "un répertoire par flotte, un fichier par travail"
    assert oct(chemin.stat().st_mode & 0o777) == "0o600", "donnée de file : 0600"
    evs = _lignes(chemin)
    assert [e["ev"] for e in evs] == ["debut", "outils", "run", "modele", "resultat"]
    debut = evs[0]
    assert debut["outils_autorises"] == ["data_rows"], "les outils AUTORISÉS, dès l'ouverture"
    assert evs[1]["ecart_omis"] == "transport sans catalogue d'outils", (
        "sans catalogue, l'écart est OMIS avec sa raison — jamais une liste vide")
    assert debut["job"]["payload"]["input"] == "Vas-y.", "le message initial, tel que reçu"
    assert debut["provider"] == "agent_llm_openai" and debut["modele_demande"] == "gpt-oss-120b"
    brut = chemin.read_text()
    assert "oto_delegue" not in brut and "sk-la-cle" not in brut, "JAMAIS un secret"
    assert "delegated_token" not in brut and "model_key" not in brut
    assert "wrkspc_du_banc" not in brut and "model_workspace" not in brut, (
        "le workspace d'une clé d'organisation voyage avec la clé, et comme elle, jamais "
        "dans le journal")
    fin = evs[-1]
    assert fin["outcome"] == "done" and fin["run_id"] == "r-NEUF"
    assert fin["resultat"]["stopped"] == "end_turn" and fin["resultat"]["usage_tokens"] == 12
    assert fin["run_finish"] == "ok"


def test_un_plantage_laisse_type_message_ENTIER_et_pile(monkeypatch, tmp_path):
    from tests.test_worker_reprise import FauxBackend
    message = "chat/completions → 400 : " + "le fournisseur dit " * 200
    _boucle_scriptee(monkeypatch, leve=RuntimeError(message))
    b = FauxBackend()
    W._un_travail(b, _job(), _Provider)

    evs = _lignes(tmp_path / "passages" / "banc-demo" / "7.1.jsonl")
    assert [e["ev"] for e in evs] == ["debut", "outils", "run", "modele",
                                      "erreur", "resultat"]
    erreur = evs[-2]
    assert erreur["type"] == "RuntimeError" and erreur["message"] == message
    assert "Traceback" in erreur["traceback"] and "faux_run" in erreur["traceback"]
    assert ("complete", False, "r-NEUF") in b.appels, (
        "le travail est conclu en échec AVEC son run — c'est ce qui relie ses "
        "refus d'écriture à lui, au bilan")


def test_un_travail_mort_en_plein_vol_CLOT_son_run_et_le_dit(monkeypatch, tmp_path):
    """⚠️ Nuit du 06/09 : deux travaux tués par un `ReadTimeout` du fournisseur.
    Journal terminé sur `erreur`, `run_finish` jamais appelé — **et la ligne que
    le run tenait est restée verrouillée jusqu'à l'expiration de son bail**
    (quinze minutes), pendant que le mode direct annonçait « volume atteint ».
    Un travail mort rend ce qu'il tient tout de suite."""
    from tests.test_worker_reprise import FauxBackend, FauxMcp

    vus: list = []

    class McpEspion(FauxMcp):
        def __init__(self, **kw):
            super().__init__(**kw)
            vus.append(self)

    _boucle_scriptee(monkeypatch, leve=RuntimeError("Read timed out. (read timeout=300)"))
    monkeypatch.setattr(W, "McpSession", McpEspion)
    W._un_travail(FauxBackend(), _job(), _Provider)

    mcp, = vus
    assert mcp.outils == ["run_start", "run_finish"], "le run ouvert est CLOS"
    nom, args = mcp.appels[-1]
    assert args["run_id"] == "r-NEUF" and args["outcome"] == "failed"
    assert "Read timed out" in args["note"], "la clôture porte le motif"

    fin = _lignes(tmp_path / "passages" / "banc-demo" / "7.1.jsonl")[-1]
    assert fin["ev"] == "resultat" and fin["outcome"] == "failed"
    assert fin["run_finish"] == "ok" and fin["run_id"] == "r-NEUF"
    assert fin["resultat"]["type"] == "RuntimeError"
    assert "read timeout=300" in fin["resultat"]["erreur"]


def test_un_run_finish_refuse_se_DIT_et_ne_masque_pas_la_cause(monkeypatch, tmp_path):
    """La clôture est un geste de tenue : son refus se journalise, il n'écrase
    pas l'erreur d'origine et ne fait pas échouer davantage."""
    from tests.test_worker_reprise import FauxBackend, FauxMcp

    class McpQuiRefuse(FauxMcp):
        def outil(self, name, args=None):
            d = super().outil(name, args)
            if name == "run_finish":
                raise RuntimeError("run_finish : 403 le run n'est plus à vous")
            return d

    _boucle_scriptee(monkeypatch, leve=RuntimeError("boum"))
    monkeypatch.setattr(W, "McpSession", McpQuiRefuse)
    W._un_travail(FauxBackend(), _job(), _Provider)

    evs = _lignes(tmp_path / "passages" / "banc-demo" / "7.1.jsonl")
    assert evs[-2]["ev"] == "erreur" and evs[-2]["message"] == "boum"
    assert evs[-1]["run_finish"].startswith("refusé : ")


def test_un_travail_sans_flotte_va_dans_hors_flotte_et_un_tag_douteux_ne_fait_pas_de_chemin():
    assert journal.chemin(None, 5).endswith(os.path.join("hors-flotte", "5.jsonl"))
    assert journal.chemin("", 5).endswith(os.path.join("hors-flotte", "5.jsonl"))
    assert journal.nom_de_flotte("../banc v151/large") == "banc_v151_large"


# ── Un fichier par tentative (oto#196) ───────────────────────────────────────

def test_deux_tentatives_d_un_meme_travail_ont_deux_journaux_distincts(monkeypatch, tmp_path):
    """Une reprise garde l'identifiant du travail : nommée par lui seul, la seconde
    tentative s'ajoutait au journal de la première. Le compteur rendu par la
    réservation (`attempts`) les sépare, et la première reste intacte."""
    monkeypatch.setenv("OTO_RUNNER_PASSAGES_DIR", str(tmp_path))
    premiere = journal.du_travail({"id": 7, "attempts": 1, "payload": {"fleet": "f"}})
    premiere.ecrire("debut"); premiere.ecrire("erreur", message="bail perdu")
    seconde = journal.du_travail({"id": 7, "attempts": 2, "payload": {"fleet": "f"}})
    seconde.ecrire("debut"); seconde.ecrire("resultat", outcome="done")

    assert premiere.chemin.endswith(os.path.join("f", "7.1.jsonl"))
    assert seconde.chemin.endswith(os.path.join("f", "7.2.jsonl"))
    assert journal.relire(premiere.chemin) == (2, "erreur"), "la première tentative n'a pas bougé"
    assert journal.relire(seconde.chemin) == (2, "resultat")


def test_sans_tentative_le_nom_reste_celui_du_mode_direct():
    """Le mode direct ne reprend jamais un travail : pas de compteur, pas de suffixe."""
    assert journal.chemin("f", 3).endswith(os.path.join("f", "3.jsonl"))
    assert journal.chemin("f", 3, 1).endswith(os.path.join("f", "3.1.jsonl"))


# ── Relire avant d'annoncer ──────────────────────────────────────────────────

def test_relire_leve_en_nommant_le_chemin_quand_le_journal_n_est_pas_la(tmp_path):
    """Un instrument qui dit « journal complet » sans fichier est pire que pas
    d'instrument : chaque annonce vient APRÈS une relecture qui a réussi."""
    absent = str(tmp_path / "x" / "9.jsonl")
    with pytest.raises(journal.JournalIllisible, match="9.jsonl"):
        journal.relire(absent)
    vide = tmp_path / "vide.jsonl"
    vide.write_text("")
    with pytest.raises(journal.JournalIllisible, match="vide"):
        journal.relire(str(vide))
    casse = tmp_path / "casse.jsonl"
    casse.write_text('{"t": "…", "ev": "debut"}\n{pas du json')
    with pytest.raises(journal.JournalIllisible, match="casse.jsonl"):
        journal.relire(str(casse))
    j = journal.Journal(str(tmp_path / "ok.jsonl"))
    j.ecrire("debut"); j.ecrire("fin", stopped="end_turn")
    assert journal.relire(j.chemin) == (2, "fin")
    assert journal.relu(j.chemin).endswith("ok.jsonl (2 événements, dernier : fin)")


# ── Le répertoire : déclaré, préparé au boot, refusé s'il n'est pas inscriptible ──

def test_le_repertoire_se_declare_par_l_environnement(monkeypatch, tmp_path):
    monkeypatch.setenv("OTO_RUNNER_PASSAGES_DIR", str(tmp_path / "ailleurs"))
    assert journal.chemin("f", 3) == str(tmp_path / "ailleurs" / "f" / "3.jsonl")
    racine = journal.preparer()
    assert racine == str(tmp_path / "ailleurs") and os.path.isdir(racine)
    monkeypatch.delenv("OTO_RUNNER_PASSAGES_DIR")
    assert journal.chemin("f", 3) == os.path.join("passages", "f", "3.jsonl"), \
        "sans variable : `passages/`, relatif au répertoire courant, comme le bilan " \
        "est relatif à sa déclaration"


@pytest.mark.skipif(os.geteuid() == 0, reason="root écrit partout")
def test_un_repertoire_non_inscriptible_refuse_AU_BOOT(monkeypatch, tmp_path):
    """Un journal impossible se dit au démarrage du worker — pas en faisant
    échouer le premier travail payé."""
    verrouille = tmp_path / "ro"
    verrouille.mkdir(mode=0o500)
    monkeypatch.setenv("OTO_RUNNER_PASSAGES_DIR", str(verrouille))
    try:
        with pytest.raises(PermissionError, match="OTO_RUNNER_PASSAGES_DIR"):
            journal.preparer()
    finally:
        verrouille.chmod(0o700)


# ── Le chemin Conversations : la requête et la réponse, entières ─────────────

def test_le_chemin_conversations_journalise_la_requete_et_les_outputs_bruts(monkeypatch):
    from oto_runner import agent_conversations as C
    from tests.test_agent_conversations import _R, _REPONSE, _env
    _env(monkeypatch)
    monkeypatch.setattr(C, "post_with_deadline", lambda url, **kw: _R(corps=_REPONSE))
    monkeypatch.setattr(C, "modele_resolu", lambda nom: "mistral-large-2512")
    evs: list = []
    C.run_once(instructions="le cadre", inputs="vas-y", tools=("data_rows",),
               on_event=lambda ev, champs: evs.append((ev, champs)))
    assert [e for e, _ in evs] == ["conversation", "reponse"]
    requete = evs[0][1]["corps"]
    assert requete["instructions"] == "le cadre" and requete["inputs"] == "vas-y"
    assert "Authorization" not in json.dumps(evs), "jamais la clé"
    assert evs[1][1]["outputs"] == _REPONSE["outputs"], "les outputs BRUTS, entiers"


# ── Un travail réservé sans compteur est refusé (D′, suite d'oto#196) ────────

def test_un_travail_de_la_file_SANS_attempts_est_refuse_en_le_nommant(monkeypatch, tmp_path):
    """⚠️ `du_travail` retombait en silence sur `<job>.jsonl` quand `attempts` manquait :
    exactement le mélange que le fichier par tentative supprime. Sur la file serveur, la
    réservation rend toujours ce compteur — son absence est un contrat cassé. Le travail
    est conclu en échec nommé, sans rien exécuter, et la boucle du worker ne tombe pas."""
    from tests.test_worker_reprise import FauxBackend
    joue: list = []
    monkeypatch.setattr(W.agent_runtime, "run", lambda *a, **k: joue.append(1))
    job = _job()
    del job["attempts"]
    b = FauxBackend()
    W._un_travail(b, job, _Provider)

    assert joue == [], "rien n'est exécuté"
    assert ("complete", False, None) in b.appels, "le travail est rendu en échec à la file"
    evs = _lignes(tmp_path / "passages" / "banc-demo" / "7.jsonl")
    (erreur,) = [e for e in evs if e["ev"] == "erreur"]
    assert erreur["type"] == "TentativeInconnue" and "attempts" in erreur["message"]


def test_en_direct_un_travail_sans_attempts_tourne_et_garde_le_nom_court(monkeypatch, tmp_path):
    """Le pendant : le mode direct ne reprend jamais un travail et n'a pas de compteur.
    La garde ne vaut que pour la file serveur."""
    from oto_runner.file_de_travail import SansFile
    from tests.test_worker_reprise import FauxBackend
    _boucle_scriptee(monkeypatch)
    job = _job()
    del job["attempts"]
    sans = SansFile()
    W._un_travail(FauxBackend(), job, _Provider, file=sans)

    assert sans.conclus[7]["status"] == "done"
    assert (tmp_path / "passages" / "banc-demo" / "7.jsonl").exists()
