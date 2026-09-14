"""Les jetons de contexte se posent SÉLECTIVEMENT, d'après le schéma du tool.

Ils sont advertisés par tool (ADR 0038 côté serveur) et la face MCP valide
contre le schéma déclaré : poser `_project` sur un tool qui ne le déclare pas
fait refuser l'appel ENTIER. Vécu au premier vol de flotte — `oto_procedure`
ne déclare pas `_project`, 4 jobs en échec avant une seule ligne traitée.
"""
from __future__ import annotations

from oto_runner.mcp import McpSession


def _session(monkeypatch, tools):
    """Une session sans réseau : tools/list scripté, appels capturés."""
    vu = {}

    def _post(self, corps, avec_entetes=False):
        methode = corps.get("method")
        if methode == "initialize":
            return ({"mcp-session-id": "s1"}, {}) if avec_entetes else {}
        if methode == "tools/list":
            return {"result": {"tools": [
                {"name": n, "inputSchema": {"type": "object",
                                            "properties": {k: {} for k in props}}}
                for n, props in tools.items()]}}
        if methode == "tools/call":
            vu["appel"] = corps["params"]
            return {"result": {"content": [{"type": "text", "text": "{}"}]}}
        return {}

    monkeypatch.setattr(McpSession, "_post", _post)
    s = McpSession(url="http://x", token="t", project=248, run_id="r-1")
    return s, vu




def test_un_tool_qui_ne_declare_pas_project_ne_le_recoit_jamais(monkeypatch):
    s, vu = _session(monkeypatch, {"oto_procedure": ["op", "slug", "_org"]})
    s.call("oto_procedure", {"op": "get", "slug": "demo"})
    assert "_project" not in vu["appel"]["arguments"], \
        "poser un jeton non déclaré fait refuser l'appel ENTIER (vécu, 4 jobs)"
    assert "_run_id" not in vu["appel"]["arguments"]


def test_un_tool_inconnu_du_cache_ne_recoit_aucun_jeton(monkeypatch):
    s, vu = _session(monkeypatch, {"autre_tool": ["x"]})
    s.call("tool_inconnu", {"x": 1})
    assert "_project" not in vu["appel"]["arguments"], \
        "fail-safe : un appel sans contexte vaut mieux qu'un refus"




def _session_org(monkeypatch, tools):
    vu = {}

    def _post(self, corps, avec_entetes=False):
        methode = corps.get("method")
        if methode == "initialize":
            return ({"mcp-session-id": "s1"}, {}) if avec_entetes else {}
        if methode == "tools/list":
            return {"result": {"tools": [
                {"name": n, "inputSchema": {"type": "object",
                                            "properties": {k: {} for k in props}}}
                for n, props in tools.items()]}}
        if methode == "tools/call":
            vu["appel"] = corps["params"]
            return {"result": {"content": [{"type": "text", "text": "{}"}]}}
        return {}

    monkeypatch.setattr(McpSession, "_post", _post)
    return McpSession(url="http://x", token="t", project=248, run_id="r-1",
                      org=226), vu


def test_un_initialize_muet_echoue_net_apres_trois_essais(monkeypatch):
    """Un 502 pendant l'initialize rendait une session MUETTE — puis chaque
    appel mourait en « Missing session ID » cryptique. Trois essais, échec NET."""
    import pytest

    import oto_runner.mcp as M

    essais = {"n": 0}

    def _post(self, corps, avec_entetes=False):
        essais["n"] += 1
        return ({}, {}) if avec_entetes else {}

    monkeypatch.setattr(McpSession, "_post", _post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError) as e:
        McpSession(url="http://x", token="t")
    assert "session id" in str(e.value) and essais["n"] == 3


def test_toute_requete_sortante_a_une_deadline_dure(monkeypatch):
    """Deux workers pendus UNE HEURE en SSL (le read timeout se réarme, le
    handshake pend) : chaque client HTTP du runner passe par la deadline
    SIGALRM — plus aucun requests nu sur le chemin des requêtes."""
    import time as _t

    import pytest

    from oto_runner import deadline as D

    monkeypatch.setattr(D, "_DEFAULT_WALL_S", 1)
    with pytest.raises(D.DeadlineExceeded):
        D._with_deadline(lambda url, **k: _t.sleep(5), "http://lent", wall_s=1)


def test_un_tools_list_vide_echoue_net(monkeypatch):
    """Un tools/list en erreur (502 en vol) laissait un cache VIDE — le
    fail-safe ne posait plus aucun jeton et le job mourait sur une erreur
    MÉTIER trompeuse jamais rejouée (« Aucune doctrine (scope org) », vécu
    job 27). Session dégradée = échec NET, le retry de job fait le reste."""
    import pytest

    def _post(self, corps, avec_entetes=False):
        if corps.get("method") == "initialize":
            return ({"mcp-session-id": "s1"}, {}) if avec_entetes else {}
        return {"_brut": "502 Bad Gateway"}

    monkeypatch.setattr(McpSession, "_post", _post)
    s = McpSession(url="http://x", token="t")
    with pytest.raises(RuntimeError, match="session dégradée"):
        s.schemas(frozenset())


# ── L'org se pose même quand le tool déclare AUSSI `_project` ────────────────
# Mesuré le 07/09/2026 sur 333 travaux de la campagne : `data_claim_next`
# déclare les deux. La règle d'avant ne posait `_org` QUE si `_project` était
# absent, en supposant qu'un projet résout son org. Le projet était bien
# transmis, et l'appel se résolvait quand même dans l'org du jeton — refus, puis
# le modèle relisait l'erreur et reposait `_org` lui-même au tour suivant.
# UN TOUR PERDU PAR FICHE, 333 fois sur 333.

def test_un_tool_qui_declare_les_deux_recoit_les_deux(monkeypatch):
    s, vu = _session_org(monkeypatch, {
        "data_claim_next": ["namespace", "worker", "filter", "_project", "_org", "_run_id"]})
    s.call("data_claim_next", {"namespace": "t", "worker": "w"})
    args = vu["appel"]["arguments"]
    assert args.get("_project") == 248
    assert args.get("_org") == 226, (
        "l'org n'est pas posée alors que le tool la déclare : le projet ne la "
        "résout pas pour tous les tools, et la déduire coûte un tour de modèle")


def test_un_org_INVENTE_par_le_modele_est_REMPLACE(monkeypatch):
    """Renversement du 08/09/2026, sur mesure et non sur relecture.

    Ce banc affirmait l'inverse : « viser une autre org reste possible, à
    condition que ce soit écrit dans l'appel ». Défendable tant que personne ne
    l'avait exercé. Le premier passage en mode file l'a exercé : le modèle a
    posé un `_org` INVENTÉ — une organisation dont il n'est membre d'aucune —
    quinze fois sur quatre-vingt-un appels, et le `setdefault` a respecté son invention.

    Il n'a pas visé ailleurs : il a rempli un champ qu'on lui tendait, avec une
    valeur plausible. Un `setdefault` fait confiance à ce que le modèle
    fournit ; une valeur imposée ne lui laisse pas l'occasion d'inventer."""
    s, vu = _session_org(monkeypatch, {
        "data_rows": ["namespace", "_project", "_org"]})
    s.call("data_rows", {"namespace": "t", "_org": 999})
    assert vu["appel"]["arguments"]["_org"] == 226, (
        "l'org du runner s'impose : ce que le runner SAIT, le modèle ne le "
        "choisit pas")


def test_un_tool_qui_ne_declare_PAS_l_org_ne_la_recoit_toujours_pas(monkeypatch):
    """La sélectivité par schéma reste la règle — poser un jeton non déclaré
    fait refuser l'appel entier (4 jobs perdus au premier vol de flotte)."""
    s, vu = _session_org(monkeypatch, {"data_write": ["namespace", "_project"]})
    s.call("data_write", {"namespace": "t"})
    assert "_org" not in vu["appel"]["arguments"]


# ── Ce qu'un agent ne choisit JAMAIS ─────────────────────────────────────────
# `max_claims` et `lease_s` ne sont pas des options d'appel : ce sont des
# réglages de CYCLE DE VIE du tableau. Le paramètre passé à la réservation
# l'emporte sur la déclaration du schéma ET s'applique à toute la table.
#
# Mesuré le 07/09/2026 : le modèle les pose de lui-même — `lease_s` 812 fois,
# `max_claims` 209 fois dans nos journaux. Il lit le schéma et choisit ; la
# porte lui était ouverte. Côté plateforme : `max_claims` a ARMÉ une garde sur
# 322 tableaux qui n'en déclarent aucune.

def test_le_modele_ne_choisit_pas_le_cycle_de_vie_d_un_tableau(monkeypatch):
    s, vu = _session_org(monkeypatch, {
        "data_claim_next": ["namespace", "worker", "max_claims", "lease_s", "_org"]})
    s.call("data_claim_next", {"namespace": "t", "worker": "w",
                               "max_claims": 1, "lease_s": 3600})
    args = vu["appel"]["arguments"]
    assert "max_claims" not in args, (
        "un plafond de reprises posé à l'appel s'applique à TOUTE la table et "
        "l'emporte sur sa déclaration — ce n'est pas au modèle d'en décider")
    assert "lease_s" not in args
    assert args["namespace"] == "t", "le reste de l'appel passe intact"


def test_le_retrait_se_DIT(monkeypatch, caplog):
    """Un paramètre qu'on enlève en silence ferait chercher longtemps pourquoi la
    consigne semble ignorée."""
    s, _ = _session_org(monkeypatch, {
        "data_claim_next": ["namespace", "max_claims", "_org"]})
    with caplog.at_level("WARNING"):
        s.call("data_claim_next", {"namespace": "t", "max_claims": 1})
    assert any("max_claims" in r.message and "RETIRÉ" in r.message
               for r in caplog.records)


def test_un_autre_tool_garde_ses_arguments(monkeypatch):
    """La liste est nominative, pas une règle de nommage : `data_write` qui
    porterait un `lease_s` — il n'en porte pas — ne serait pas amputé."""
    s, vu = _session_org(monkeypatch, {"data_write": ["namespace", "lease_s", "_org"]})
    s.call("data_write", {"namespace": "t", "lease_s": 99})
    assert vu["appel"]["arguments"]["lease_s"] == 99


# ── Ce que le modèle ne VOIT pas, il ne peut pas l'inventer ──────────────────
# `_org`, `_project` et `_run_id` sont posés par le runner depuis le travail.
# Tant qu'ils figuraient au schéma servi, le modèle les remplissait : `_org`
# inventé quinze fois sur quatre-vingt-un appels le 08/09/2026, et sur un autre
# banc une organisation nommée de toutes pièces dans un compte rendu. Retirer le
# champ ferme la classe ; un `setdefault` ne fermait qu'un cas.

def test_le_schema_servi_au_modele_ne_porte_AUCUN_jeton_de_contexte(monkeypatch):
    s, _ = _session_org(monkeypatch, {
        "data_rows": ["namespace", "_org", "_project", "_run_id"]})

    servi = s.schemas(frozenset({"data_rows"}))[0]["input_schema"]["properties"]

    assert "namespace" in servi, "les vrais paramètres restent servis"
    for jeton in ("_org", "_project", "_run_id", "_group", "_instance"):
        assert jeton not in servi, (
            f"`{jeton}` est tendu au modèle : il le remplira, avec une valeur "
            "plausible et fausse")


def test_le_runner_POSE_encore_ce_qu_il_a_retire_du_schema(monkeypatch):
    """Le pendant, et il est vital : `_declares` lit le catalogue pour savoir
    quoi poser. Si le nettoyage mutait le cache au lieu de le copier, le poseur
    deviendrait aveugle — on aurait retiré une capacité au lieu d'une occasion
    de se tromper."""
    s, vu = _session_org(monkeypatch, {
        "data_rows": ["namespace", "_org", "_project", "_run_id"]})
    s.schemas(frozenset({"data_rows"}))          # le modèle a vu le schéma nettoyé

    s.call("data_rows", {"namespace": "t"})

    args = vu["appel"]["arguments"]
    assert args["_org"] == 226 and args["_project"] == 248 and args["_run_id"] == "r-1"


def test_un_jeton_de_contexte_REQUIS_sort_aussi_du_required(monkeypatch):
    """Sinon le schéma servi exige un champ qu'il ne décrit plus — un contrat
    qui se contredit, et le modèle n'a aucun moyen de le satisfaire."""
    s, _ = _session_org(monkeypatch, {"data_rows": ["namespace", "_org"]})
    s.schemas(frozenset())                    # peuple le catalogue de la session
    s._outils[0]["inputSchema"]["required"] = ["namespace", "_org"]

    servi = s.schemas(frozenset({"data_rows"}))[0]["input_schema"]

    assert servi["required"] == ["namespace"]


def test_un_group_INVENTE_par_le_modele_est_RETIRE(monkeypatch):
    """Mesuré le 09/09/2026 en mode direct (jetable 634, passe D) : le modèle a
    posé `_group=226` — l'org recopiée dans le champ voisin — et l'écriture a
    été refusée (« groupe inconnu »). Le runner ne connaît aucune équipe : il
    ne pose pas `_group`, et retire celui que le modèle invente."""
    s, vu = _session_org(monkeypatch, {
        "data_write": ["namespace", "row", "_project", "_org", "_group"]})
    s.call("data_write", {"namespace": "t", "row": {}, "_group": 226})
    args = vu["appel"]["arguments"]
    assert "_group" not in args, "un `_group` inventé ne part pas vers la plateforme"
    assert args["_org"] == 226, "l'org du runner reste posée"


def test_une_instance_INVENTEE_par_le_modele_est_RETIREE(monkeypatch):
    """29 appels d'une passe E (11/09/2026) portaient un `_instance` inventé,
    tous refusés par la plateforme. Même geste que pour `_group` : le runner
    ne le pose pas, ne le sert pas au modèle, et retire celui qu'il invente."""
    s, vu = _session_org(monkeypatch, {
        "data_write": ["namespace", "row", "_project", "_org", "_instance"]})
    s.call("data_write", {"namespace": "t", "row": {}, "_instance": "x"})
    args = vu["appel"]["arguments"]
    assert "_instance" not in args, "un `_instance` inventé ne part pas vers la plateforme"
    assert args["_org"] == 226, "l'org du runner reste posée"
