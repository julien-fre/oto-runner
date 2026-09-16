"""Le moteur Claude Code — ce que le banc fige sans appeler le service.

Le SDK est remplacé par une doublure : ce qui est vérifié est ce que le WORKER
décide (outils exposés, permissions, environnement, plafonds), ce qu'il fait des
appels d'outils (tout passe par la session MCP du travail), et comment il lit le
résultat. L'essai réel valide contre Claude Code ce que le banc ne peut pas savoir.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from oto_runner import agent_claude_code as AC
from oto_runner import conclusion, llm_select
from oto_runner import worker as W
from oto_runner.agent_runtime import AgentSpec
from tests.test_worker_reprise import FauxBackend, FauxMcp, _job


# ── La doublure du SDK ───────────────────────────────────────────────────────

class ToolUseBlock(SimpleNamespace):
    pass


class TextBlock(SimpleNamespace):
    pass


class AssistantMessage(SimpleNamespace):
    pass


class ResultMessage(SimpleNamespace):
    pass


def _resultat(**kw):
    base = dict(subtype="success", is_error=False, num_turns=3, total_cost_usd=0.42,
                duration_ms=1000, permission_denials=[], stop_reason="end_turn",
                result="Fini : 2 lignes écrites.",
                usage={"input_tokens": 10, "output_tokens": 5,
                       "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20},
                model_usage=None)
    base.update(kw)
    return ResultMessage(**base)


class FauxSdk:
    """`script(options, outils)` est une coroutine-générateur : elle rend les messages
    et peut appeler les outils exposés, comme le ferait Claude Code."""

    def __init__(self, script):
        self.script = script
        self.options = None
        self.outils = {}

    def tool(self, nom, description, schema):
        def deco(fn):
            outil = SimpleNamespace(name=nom, description=description, schema=schema,
                                    handler=fn)
            self.outils[nom] = outil
            return outil
        return deco

    def create_sdk_mcp_server(self, nom, tools=None):
        return {"nom": nom, "outils": list(tools or ())}

    def ClaudeAgentOptions(self, **kw):  # noqa: N802 — le nom du SDK
        self.options = kw
        return SimpleNamespace(**kw)

    def query(self, prompt, options):
        return self.script(self, prompt, options)


class Mcp:
    def __init__(self, catalogue=("data_rows", "data_write", "slack_post_message")):
        self.catalogue_ = catalogue
        self.appels = []
        self.panne = None

    def schemas(self, noms):
        return [{"name": n, "description": f"desc {n}",
                 "input_schema": {"type": "object", "properties": {}}}
                for n in self.catalogue_ if n in noms]

    def call(self, nom, args):
        if self.panne:
            raise self.panne
        self.appels.append((nom, args))
        return f"sortie de {nom}", False


def _lancer(monkeypatch, script, *, tools=("data_rows", "data_write"), spec=None, mcp=None,
            workspace=None, heartbeat=None, cle="sk-org"):
    sdk = FauxSdk(script)
    monkeypatch.setattr(AC, "_sdk", lambda: sdk)
    evenements = []
    res = AC.run_once(instructions="cadre", inputs="Fais le travail.", tools=tools,
                      api_key=cle, modele="claude-sonnet-5",
                      on_event=lambda ev, c: evenements.append((ev, c)),
                      mcp=mcp or Mcp(), spec=spec or AgentSpec(system="s", tools=frozenset(tools)),
                      workspace=workspace, heartbeat=heartbeat)
    return res, sdk, evenements


async def _simple(sdk, prompt, options):
    yield AssistantMessage(content=[TextBlock(text="je commence")], model="claude-sonnet-5",
                           parent_tool_use_id=None, message_id="m1", usage={})
    yield _resultat()


# ── Ce que Claude Code reçoit ────────────────────────────────────────────────

def test_seule_l_allowlist_est_exposee_et_autorisee(monkeypatch):
    res, sdk, _ = _lancer(monkeypatch, _simple, mcp=Mcp())
    assert set(sdk.outils) == {"data_rows", "data_write"}, "slack_post_message hors allowlist"
    o = sdk.options
    assert o["mcp_servers"] == {"oto": {"nom": "oto", "outils": list(sdk.outils.values())}}
    assert o["strict_mcp_config"] is True
    assert o["tools"] == ["Agent"], "ni shell, ni fichiers, ni web"
    assert "mcp__oto__data_rows" in o["allowed_tools"] and "Agent" in o["allowed_tools"]
    assert not any("slack" in t for t in o["allowed_tools"])
    assert o["permission_mode"] == "dontAsk"
    assert o["setting_sources"] == []
    assert o["system_prompt"] == {"type": "preset", "preset": "claude_code", "append": "cadre"}


def test_la_cle_du_travail_part_et_les_secrets_du_worker_non(monkeypatch):
    monkeypatch.setenv("OTO_WORKER_SECRET", "otow_secret")
    _, sdk, _ = _lancer(monkeypatch, _simple, workspace="wrkspc_1")
    env = sdk.options["env"]
    assert env["ANTHROPIC_API_KEY"] == "sk-org"
    assert env["OTO_WORKER_SECRET"] == "" and env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "anthropic-workspace-id: wrkspc_1"
    assert env["CLAUDE_CONFIG_DIR"].startswith(sdk.options["cwd"])


def test_le_repertoire_du_travail_est_efface(monkeypatch):
    import os
    _, sdk, _ = _lancer(monkeypatch, _simple)
    assert not os.path.exists(sdk.options["cwd"])


def test_le_plafond_de_tours_du_travail_est_servi_au_dela_de_64(monkeypatch):
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=200)
    _, sdk, _ = _lancer(monkeypatch, _simple, spec=spec)
    assert sdk.options["max_turns"] == 200


def test_un_plafond_au_dela_du_moteur_est_dit(monkeypatch):
    spec = AgentSpec(system="s", tools=frozenset({"data_rows"}), max_steps=1000)
    _, sdk, ev = _lancer(monkeypatch, _simple, spec=spec)
    assert sdk.options["max_turns"] == AC.PLAFOND_TOURS
    (note,) = [c for e, c in ev if e == "plafond_tours"]
    assert note == {"demande": 1000, "servi": AC.PLAFOND_TOURS, "plafond": AC.PLAFOND_TOURS}


def test_l_effort_du_travail_est_servi(monkeypatch):
    spec = AgentSpec(system="s", tools=frozenset(), effort="high")
    _, sdk, _ = _lancer(monkeypatch, _simple, spec=spec)
    assert sdk.options["effort"] == "high"


def test_une_temperature_declaree_est_dite_non_servie(monkeypatch):
    spec = AgentSpec(system="s", tools=frozenset(), temperature=0)
    _, sdk, ev = _lancer(monkeypatch, _simple, spec=spec)
    assert "temperature" not in sdk.options
    assert [c for e, c in ev if e == "temperature_non_servie"] == [{"temperature": 0}]


# ── Les appels d'outils ──────────────────────────────────────────────────────

async def _appelle(sdk, prompt, options):
    sortie = await sdk.outils["data_write"].handler({"id": 3, "row": {"statut": "ok"}})
    yield AssistantMessage(content=[ToolUseBlock(name="Agent", id="t1", input={})],
                           model="claude-sonnet-5", parent_tool_use_id=None,
                           message_id="m1", usage={})
    yield AssistantMessage(content=[TextBlock(text=sortie["content"][0]["text"])],
                           model="claude-sonnet-5", parent_tool_use_id="t1",
                           message_id="m2", usage={})
    yield _resultat()


def test_un_appel_passe_par_la_session_mcp_du_travail(monkeypatch):
    mcp = Mcp()
    res, _, ev = _lancer(monkeypatch, _appelle, mcp=mcp)
    assert mcp.appels == [("data_write", {"id": 3, "row": {"statut": "ok"}})]
    assert [s.tool for s in res.steps] == ["data_write"]
    assert conclusion.resultat_declare(res, "x")["tool_counts"] == {"data_write": 1}
    assert [c["parent_principal"] for e, c in ev if e == "delegation"] == [True]


def test_une_sortie_trop_longue_est_coupee_en_le_disant(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_MAX_TOOL_OUTPUT", "10")
    lu = {}

    async def script(sdk, prompt, options):
        lu["sortie"] = await sdk.outils["data_rows"].handler({})
        yield _resultat()

    _lancer(monkeypatch, script)
    assert "SORTIE TRONQUÉE" in lu["sortie"]["content"][0]["text"]


def test_un_transport_mort_fait_echouer_le_travail(monkeypatch):
    mcp = Mcp()
    mcp.panne = RuntimeError("session MCP rouverte mais refusée")
    suite = []

    async def script(sdk, prompt, options):
        await sdk.outils["data_rows"].handler({})
        yield AssistantMessage(content=[], model=None, parent_tool_use_id=None,
                               message_id="m1", usage={})
        suite.append("continué")
        yield _resultat()

    with pytest.raises(RuntimeError, match="transport MCP mort"):
        _lancer(monkeypatch, script, mcp=mcp)
    assert suite == [], "le déroulé s'arrête au premier message après la panne"


# ── La lecture du résultat ───────────────────────────────────────────────────

def test_un_succes_conclut_end_turn_avec_l_usage_de_la_session(monkeypatch):
    res, _, ev = _lancer(monkeypatch, _simple)
    assert res.stopped == "end_turn" and res.reply == "Fini : 2 lignes écrites."
    assert res.usage["input_tokens"] == 10 and res.usage["cache_read_input_tokens"] == 100
    assert res.usage["input_total_tokens"] == 130
    assert res.model == "claude-sonnet-5"
    (fin,) = [c for e, c in ev if e == "resultat_claude_code"]
    assert fin["cout_usd"] == 0.42


def test_l_usage_par_modele_couvre_les_sous_agents(monkeypatch):
    async def script(sdk, prompt, options):
        yield _resultat(model_usage={
            "claude-sonnet-5": {"inputTokens": 10, "outputTokens": 5,
                                "cacheReadInputTokens": 100, "cacheCreationInputTokens": 0},
            "claude-haiku-4-5": {"inputTokens": 7, "outputTokens": 3,
                                 "cacheReadInputTokens": 0, "cacheCreationInputTokens": 1}})

    res, _, ev = _lancer(monkeypatch, script)
    assert res.usage["input_tokens"] == 17 and res.usage["output_tokens"] == 8
    assert res.usage["input_total_tokens"] == 118
    (fin,) = [c for e, c in ev if e == "resultat_claude_code"]
    assert fin["modeles"] == ["claude-haiku-4-5", "claude-sonnet-5"]


def test_le_plafond_de_tours_atteint_conclut_max_steps(monkeypatch):
    async def script(sdk, prompt, options):
        yield _resultat(subtype="error_max_turns", is_error=True, result=None)

    res, _, _ = _lancer(monkeypatch, script)
    assert res.stopped == "max_steps" and conclusion.echec_nomme(res) is None


def test_une_erreur_d_execution_est_un_echec_nomme(monkeypatch):
    async def script(sdk, prompt, options):
        yield _resultat(subtype="error_during_execution", is_error=True, result=None)

    res, _, _ = _lancer(monkeypatch, script)
    assert res.stopped == "fin_anormale"
    assert conclusion.echec_nomme(res) == "fin_anormale (error_during_execution)"


def test_sans_message_de_resultat_le_travail_echoue(monkeypatch):
    async def script(sdk, prompt, options):
        yield AssistantMessage(content=[], model=None, parent_tool_use_id=None,
                               message_id="m1", usage={})

    with pytest.raises(RuntimeError, match="sans message de résultat"):
        _lancer(monkeypatch, script)


def test_le_budget_de_jetons_arrete_le_deroule_et_compte_un_message_une_fois(monkeypatch):
    async def script(sdk, prompt, options):
        for _ in range(3):   # le même message d'API, découpé en trois blocs
            yield AssistantMessage(content=[], model="claude-sonnet-5",
                                   parent_tool_use_id=None, message_id="m1",
                                   usage={"input_tokens": 400, "output_tokens": 100})
        yield AssistantMessage(content=[], model="claude-sonnet-5",
                               parent_tool_use_id=None, message_id="m2",
                               usage={"input_tokens": 400, "output_tokens": 200})
        yield _resultat()

    spec = AgentSpec(system="s", tools=frozenset(), max_tokens=1000)
    res, _, ev = _lancer(monkeypatch, script, spec=spec)
    assert res.stopped == "max_tokens"
    assert [c["jetons"] for e, c in ev if e == "budget_depasse"] == [1100]


def test_la_deadline_murale_coupe(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_CLAUDE_CODE_WALL_S", "1")

    async def script(sdk, prompt, options):
        await asyncio.sleep(5)
        yield _resultat()

    with pytest.raises(AC.DeadlineExceeded):
        _lancer(monkeypatch, script)


def test_le_bail_est_prolonge_pendant_le_deroule(monkeypatch):
    battements = []
    _lancer(monkeypatch, _appelle, heartbeat=lambda: battements.append(1))
    assert battements == [1], "un battement, puis au plus un par minute"


# ── Le branchement au worker ─────────────────────────────────────────────────

def test_le_moteur_se_choisit_par_l_environnement(monkeypatch):
    monkeypatch.setenv("OTO_RUNNER_PROVIDER", "claude-code")
    assert llm_select.get_provider() is AC
    assert AC.depot() == "anthropic"


def test_l_effort_du_travail_n_est_pas_refuse_sur_ce_moteur():
    W._exiger_effort_servi({"effort": "high"}, AC)


def test_le_worker_remet_la_session_le_cadre_le_workspace_et_le_bail(monkeypatch):
    class McpDuTravail(FauxMcp):
        def schemas(self, noms):
            return Mcp().schemas(noms)

    monkeypatch.setattr(W, "McpSession", McpDuTravail)
    vu = {}

    def faux_run_once(**kw):
        vu.update(kw)
        from oto_runner.agent_runtime import AgentResult
        return AgentResult(reply="fini", stopped="end_turn", model="claude-sonnet-5")

    monkeypatch.setattr(AC, "run_once", faux_run_once)
    job = _job("start")
    job["model_workspace"] = "wrkspc_9"
    job["payload"]["max_steps"] = 200
    backend = FauxBackend()
    W._traiter(backend, job, AC)
    assert isinstance(vu["mcp"], McpDuTravail) and vu["mcp"].run_id == "r-NEUF"
    assert vu["spec"].max_steps == 200 and vu["workspace"] == "wrkspc_9"
    assert callable(vu["heartbeat"])
    assert ("complete", True, "r-NEUF") in backend.appels


class SystemMessage(SimpleNamespace):
    pass


class ResultError(Exception):
    pass


def test_l_exception_du_sdk_apres_un_resultat_en_erreur_ne_masque_pas_le_resultat(monkeypatch):
    """Relevé sur le SDK 0.2.153 : un résultat `is_error` fait sortir le CLI en code 1, et
    le SDK lève `ResultError` après l'avoir rendu. Le résultat conclut."""
    async def script(sdk, prompt, options):
        yield _resultat(subtype="error_max_turns", is_error=True, result=None)
        raise ResultError("Claude Code returned an error result")

    res, _, _ = _lancer(monkeypatch, script)
    assert res.stopped == "max_steps"


def test_une_exception_du_sdk_sans_resultat_remonte(monkeypatch):
    async def script(sdk, prompt, options):
        raise ResultError("démarrage refusé")
        yield  # noqa — générateur

    with pytest.raises(ResultError):
        _lancer(monkeypatch, script)


def test_une_cle_refusee_arrete_le_travail_sans_attendre_les_rejeux(monkeypatch):
    rejeux = []

    async def script(sdk, prompt, options):
        for tentative in range(10):
            rejeux.append(tentative)
            yield SystemMessage(subtype="api_retry", data={"error_status": 401})
        yield _resultat()

    with pytest.raises(RuntimeError, match="clé de modèle refusée par Anthropic \\(401\\)"):
        _lancer(monkeypatch, script)
    assert rejeux == [0]


def test_un_rejeu_transitoire_ne_coupe_rien(monkeypatch):
    async def script(sdk, prompt, options):
        yield SystemMessage(subtype="api_retry", data={"error_status": 529})
        yield _resultat()

    res, _, _ = _lancer(monkeypatch, script)
    assert res.stopped == "end_turn"


def test_une_sortie_anticipee_ferme_le_flux(monkeypatch):
    ferme = []

    async def script(sdk, prompt, options):
        try:
            yield AssistantMessage(content=[], model="claude-sonnet-5", parent_tool_use_id=None,
                                   message_id="m1",
                                   usage={"input_tokens": 5000, "output_tokens": 1})
            yield _resultat()
        finally:
            ferme.append(True)

    spec = AgentSpec(system="s", tools=frozenset(), max_tokens=10)
    res, _, _ = _lancer(monkeypatch, script, spec=spec)
    assert res.stopped == "max_tokens" and ferme == [True]


def test_la_boucle_classique_dit_le_plafond_qu_elle_impose():
    from oto_runner import agent_runtime
    from oto_runner.llm_types import Turn
    from tests.test_effort_du_travail import FauxProvider, FauxTransport
    ev = []
    agent_runtime.run(AgentSpec(system="s", tools=frozenset(), max_steps=200), FauxTransport(),
                      FauxProvider([Turn(text="fini", raw_content=[])]),
                      prompt="go", on_event=lambda e, c: ev.append((e, c)))
    assert [c for e, c in ev if e == "plafond_tours"] == [
        {"demande": 200, "servi": 64, "plafond": 64}]
