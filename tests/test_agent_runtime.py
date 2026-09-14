"""La boucle du worker — les invariants, désormais prouvés provider-agnostiques.

Fakes purs : un provider scripté (forme Anthropic pour les assertions de fil) et
un transport MCP espion. La boucle ne connaît AUCUNE forme de fil — un test le
prouve en la faisant tourner sur un provider à forme OpenAI.
"""
from __future__ import annotations

import pytest

from oto_runner import agent_runtime
from oto_runner.agent_runtime import AgentSpec
from oto_runner.llm_types import ToolCall, Turn


class FauxTransport:
    def __init__(self, reponses=None):
        self.reponses = dict(reponses or {})
        self.appels = []

    def schemas(self, names):
        return [{"name": n, "description": "", "input_schema": {"type": "object"}}
                for n in sorted(names)]

    def call(self, name, arguments):
        self.appels.append((name, arguments))
        return self.reponses.get(name, ("ok", False))


class FauxProvider:
    """Forme Anthropic : tour assistant = blocs, résultats = UN message user."""

    def __init__(self, tours):
        self.file = list(tours)

    def complete(self, **kwargs):
        return self.file.pop(0)

    def user_message(self, text):
        return {"role": "user", "content": text}

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.raw_content}

    def tool_messages(self, results):
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r["id"],
             "content": r["text"], "is_error": r["is_error"]} for r in results]}]

    def format_tools(self, schemas):
        return list(schemas)


def _turn(text="", calls=(), stop="end_turn"):
    return Turn(
        text=text,
        tool_calls=tuple(ToolCall(id=f"t{i}", name=n, arguments=a)
                         for i, (n, a) in enumerate(calls)),
        stop_reason=stop,
        raw_content=[{"type": "text", "text": text}] if text else [])


SPEC = AgentSpec(system="s", tools=frozenset({"data_rows", "oto_procedure"}),
                 max_steps=3)


def test_un_outil_hors_allowlist_nest_jamais_transporte():
    t = FauxTransport()
    p = FauxProvider([_turn(calls=[("email_send", {"to": "x"})]),
                      _turn(text="compris, j'arrête")])
    res = agent_runtime.run(SPEC, t, p, prompt="go")
    assert t.appels == [], "l'outil interdit ne doit jamais atteindre le transport"
    assert res.steps[0].ok is False and "indisponible" in (res.steps[0].error or "")
    assert res.stopped == "end_turn"


def _servi(sortie: str) -> str:
    """Ce que le modèle a RÉELLEMENT reçu pour une sortie d'outil donnée."""
    t = FauxTransport({"data_rows": (sortie, False)})
    p = FauxProvider([_turn(calls=[("data_rows", {})]), _turn(text="fini")])
    res = agent_runtime.run(SPEC, t, p, prompt="go")
    return res.messages[-2]["content"][0]["content"]


def test_une_consigne_de_45k_passe_ENTIERE_au_modele():
    """⚠️ LE défaut de la nuit du 06/09 : à 12 000 caractères, la consigne
    métier (44 818) et le schéma du tableau (38 079) arrivaient coupés au
    premier tiers — le modèle inventait des options, oubliait des colonnes, et
    concluait « le schéma ne contient pas de colonne contacts » sur la partie
    qu'il n'avait pas lue. Le défaut passe à 120 000 : ces deux-là entrent."""
    procedure = "consigne " * 5_000            # ≈ 45 000 caractères
    assert 44_000 < len(procedure) < 46_000
    assert _servi(procedure) == procedure, "aucune coupe, aucun marqueur"


def test_au_dela_du_plafond_la_coupe_est_DITE_avec_ce_qui_manque(monkeypatch):
    """Une troncature muette est un bug à part entière : le modèle croit avoir
    tout lu. La phrase de fin nomme ce qui manque, en caractères."""
    monkeypatch.setenv("OTO_RUNNER_MAX_TOOL_OUTPUT", "1000")
    contenu = _servi("x" * 4_500)
    assert contenu.startswith("x" * 1_000)
    assert "SORTIE TRONQUÉE" in contenu
    assert "les 1000 premiers caractères sur 4500" in contenu
    assert "il en manque 3500" in contenu
    assert "n'existe pas" in contenu, "la conclusion par absence est interdite"


def test_le_plafond_se_regle_par_l_environnement(monkeypatch):
    monkeypatch.delenv("OTO_RUNNER_MAX_TOOL_OUTPUT", raising=False)
    assert agent_runtime.max_tool_output() == 120_000 \
        == agent_runtime.DEFAULT_MAX_TOOL_OUTPUT_CHARS
    monkeypatch.setenv("OTO_RUNNER_MAX_TOOL_OUTPUT", "250000")
    assert agent_runtime.max_tool_output() == 250_000
    assert len(_servi("y" * 200_000)) == 200_000, "le réglage est SUIVI"
    monkeypatch.setenv("OTO_RUNNER_MAX_TOOL_OUTPUT", "beaucoup")
    with pytest.raises(ValueError, match="OTO_RUNNER_MAX_TOOL_OUTPUT"):
        agent_runtime.max_tool_output()


def test_chaque_tour_de_modele_est_CHRONOMETRE():
    """Sans `duree_ms`, un journal ne dit pas si un tour a pris deux secondes ou
    cinq minutes — la première question devant un travail mort sur un délai."""
    import time as _t

    class Lent(FauxProvider):
        def complete(self, **kw):
            _t.sleep(0.02)
            return super().complete(**kw)

    evs: list = []
    agent_runtime.run(SPEC, FauxTransport(), Lent([_turn(text="fini")]),
                      prompt="go", on_event=lambda ev, ch: evs.append((ev, ch)))
    tours = [ch for ev, ch in evs if ev == "modele"]
    assert len(tours) == 1 and tours[0]["duree_ms"] >= 20


def test_le_plafond_de_tours_arrete_proprement():
    t = FauxTransport()
    p = FauxProvider([_turn(text="je continue", calls=[("data_rows", {})])
                      for _ in range(10)])
    res = agent_runtime.run(SPEC, t, p, prompt="go")
    assert res.stopped == "max_steps"
    assert res.reply == "je continue", "le texte intermédiaire est le repli"


def test_un_refus_est_terminal_jamais_un_crash():
    t = FauxTransport()
    p = FauxProvider([_turn(stop="refusal")])
    res = agent_runtime.run(SPEC, t, p, prompt="go")
    assert res.stopped == "refusal" and res.reply == "" and t.appels == []


def test_le_fil_est_appose_dans_lordre_et_en_double_etage():
    t = FauxTransport({"data_rows": ('{"rows": []}', False)})
    p = FauxProvider([_turn(text="je regarde", calls=[("data_rows", {"limit": 1})]),
                      _turn(text="terminé")])
    fil = []
    agent_runtime.run(SPEC, t, p, prompt="vas-y",
                      on_turn=lambda role, neutre, brut: fil.append((role, neutre, brut)))
    assert [f[0] for f in fil] == ["user", "assistant", "tool", "assistant"]
    assert fil[1][1]["tool_calls"] == [{"name": "data_rows"}]
    assert fil[2][2]["content"][0]["type"] == "tool_result"


def test_une_erreur_de_transport_est_un_resultat_pas_une_panne():
    class Casse(FauxTransport):
        def call(self, name, arguments):
            raise RuntimeError("connexion perdue")
    p = FauxProvider([_turn(calls=[("data_rows", {})]),
                      _turn(text="je conclus sans")])
    res = agent_runtime.run(SPEC, Casse(), p, prompt="go")
    assert res.stopped == "end_turn"
    assert res.steps[0].ok is False and "connexion perdue" in res.steps[0].error


def test_la_continuation_repart_de_lhistorique_sans_nouveau_prompt():
    p = FauxProvider([_turn(text="je reprends et je conclus")])
    historique = [{"role": "user", "content": "démarre"},
                  {"role": "assistant", "content": [{"type": "text", "text": "début"}]}]
    res = agent_runtime.run(SPEC, FauxTransport(), p, prompt=None, history=historique)
    assert res.reply == "je reprends et je conclus"
    assert res.messages[0] == historique[0]
    assert [m["role"] for m in res.messages].count("user") == 1


def test_la_boucle_ne_connait_pas_la_forme_du_fil():
    """LE test du seam : le même run sur un provider à forme OpenAI produit des
    messages `role:tool` SÉPARÉS — si la boucle codait encore la forme Anthropic
    en dur, ce test la prendrait la main dans le sac."""
    class ProviderOpenAI(FauxProvider):
        def assistant_message(self, turn):
            return turn.raw_content if isinstance(turn.raw_content, dict) else \
                {"role": "assistant", "content": turn.text}

        def tool_messages(self, results):
            return [{"role": "tool", "tool_call_id": r["id"], "content": r["text"]}
                    for r in results]

        def format_tools(self, schemas):
            return [{"type": "function", "function": {"name": s["name"]}}
                    for s in schemas]

    t = FauxTransport({"data_rows": ("ok", False), "oto_procedure": ("doc", False)})
    tours = [Turn(text="", stop_reason="end_turn",
                  tool_calls=(ToolCall(id="a1", name="data_rows", arguments={}),
                              ToolCall(id="a2", name="oto_procedure", arguments={})),
                  raw_content={"role": "assistant", "content": None,
                               "tool_calls": [{"id": "a1"}, {"id": "a2"}]}),
             _turn(text="fini")]
    res = agent_runtime.run(SPEC, t, ProviderOpenAI(tours), prompt="go")
    roles = [m["role"] for m in res.messages]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"], \
        "deux résultats → DEUX messages role:tool (forme OpenAI respectée)"
    assert res.messages[2]["tool_call_id"] == "a1"
    assert res.messages[3]["tool_call_id"] == "a2"


def test_une_erreur_transitoire_est_rejouee_en_silence(monkeypatch):
    """La politique de reprise est de la MÉCANIQUE, pas de la consigne : un
    timeout d'outil se rejoue UNE fois, le modèle ne voit que la 2e réponse."""
    import oto_runner.agent_runtime as ar
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)

    class Capricieux(FauxTransport):
        def __init__(self):
            super().__init__()
            self.essais = 0

        def call(self, name, arguments):
            self.essais += 1
            if self.essais == 1:
                return ("serpapi_jobs : timeout lors de la recherche", True)
            return ('{"jobs": []}', False)

    t = Capricieux()
    p = FauxProvider([_turn(calls=[("data_rows", {})]), _turn(text="fini")])
    res = agent_runtime.run(SPEC, t, p, prompt="go")
    assert t.essais == 2, "rejoué une fois"
    assert res.steps[0].ok is True, "le modèle ne voit que la seconde réponse"


def test_une_erreur_metier_nest_jamais_rejouee():
    """not_found est une RÉPONSE, pas un accident — la rejouer coûterait un
    crédit pour la même réponse."""
    class Metier(FauxTransport):
        def __init__(self):
            super().__init__()
            self.essais = 0

        def call(self, name, arguments):
            self.essais += 1
            return ('{"error": "not_found", "siren": "123"}', True)

    t = Metier()
    p = FauxProvider([_turn(calls=[("data_rows", {})]), _turn(text="fini")])
    agent_runtime.run(SPEC, t, p, prompt="go")
    assert t.essais == 1


def test_les_postes_de_cache_se_cumulent_sur_tout_le_run():
    """Un run qui cache bien affiche un `input_tokens` MINUSCULE : sans les deux
    postes de cache, son volume d'entrée réel serait illisible au résultat."""
    def _u(**kw):
        return Turn(text=kw.pop("text", ""), tool_calls=(), stop_reason="end_turn",
                    raw_content=[], usage=kw)

    p = FauxProvider([
        Turn(text="", tool_calls=(ToolCall(id="t0", name="data_rows", arguments={}),),
             stop_reason="tool_use", raw_content=[],
             usage={"input_tokens": 9000, "output_tokens": 120,
                    "cache_creation_input_tokens": 8800,
                    "cache_read_input_tokens": 0}),
        _u(text="fini", input_tokens=300, output_tokens=80,
           cache_creation_input_tokens=400, cache_read_input_tokens=8800),
    ])
    res = agent_runtime.run(SPEC, FauxTransport(), p, prompt="go")
    assert res.usage == {"input_tokens": 9300, "output_tokens": 200,
                         "cache_creation_input_tokens": 9200,
                         "cache_read_input_tokens": 8800,
                         # Aucun tour ne déclare d'entrée totale : inconnue, pas zéro.
                         "input_total_tokens": None}


def test_un_provider_sans_poste_de_cache_laisse_le_cache_inconnu():
    """Un tour qui ne déclare pas le cache n'en déclare pas zéro. Ce banc voulait les
    postes de cache « à zéro, jamais absents » : des zéros fabriqués (cf. `comptage`).
    Ils restent inconnus, la couverture le dit, et l'entrée et la sortie restent connues."""
    p = FauxProvider([Turn(text="fini", raw_content=[],
                           usage={"input_tokens": 10, "output_tokens": 3})])
    res = agent_runtime.run(SPEC, FauxTransport(), p, prompt="go")
    assert (res.usage["input_tokens"], res.usage["output_tokens"]) == (10, 3)
    assert res.usage["cache_read_input_tokens"] is None
    assert res.usage["cache_creation_input_tokens"] is None
    assert res.couverture["tours"] == 1
    assert res.couverture["declares"]["cache_read_input_tokens"] == 0


# ── LA BORNE DE JETONS, appliquée par l'agent lui-même ───────────────────────
#
# ⚠️ Elle vivait sur la flotte, et seul un ordonnanceur savait la lire — donc
# personne dès qu'un passage tourne sans lui. L'agent ne connaissait qu'un
# plafond d'ÉTAPES, **qui ne dit rien de ce qu'une étape coûte** : une ligne
# mesurée à 65 571 jetons le 01/09 tenait largement sous ses 40 pas.
#
# Sans cette borne, un passage non observé peut dépenser le triple pendant vingt
# minutes sans que personne ne le sache avant de lire le relevé.

def _tour(text="", **usage):
    """Un tour qui APPELLE un outil — donc la boucle continue.

    ⚠️ Un tour sans appel d'outil conclut le déroulé : la borne ne serait jamais
    atteinte et le test passerait au vert sans rien éprouver. C'est le défaut
    qu'avait la première version de ces tests."""
    return Turn(text=text, tool_calls=(ToolCall(id="t0", name="data_rows",
                                                arguments={}),),
                stop_reason="tool_use", raw_content=[], usage=usage)


def _fin(text="fini", **usage):
    return Turn(text=text, tool_calls=(), stop_reason="end_turn",
                raw_content=[{"type": "text", "text": text}], usage=usage)


def test_la_borne_de_jetons_arrete_le_deroule():
    spec = AgentSpec(system="s", tools=frozenset(), max_steps=10, max_tokens=1000)
    p = FauxProvider([
        _tour(input_tokens=600, output_tokens=100),          # 700 — sous la borne
        _tour(input_tokens=400, output_tokens=100),          # 1200 — dépasse
        _fin(text="ce tour ne doit JAMAIS avoir lieu", input_tokens=99_999),
    ])
    res = agent_runtime.run(spec, FauxTransport(), p, prompt="go")
    assert res.stopped == "max_tokens"
    assert len(p.file) == 1, "le tour suivant n'a pas été joué"


def test_la_borne_se_verifie_APRES_le_tour_donc_elle_borne_la_derive():
    """⚠️ On ne connaît le coût d'un tour qu'une fois qu'il a eu lieu.

    La borne empêche donc le tour SUIVANT — c'est le plus tôt qu'on puisse
    s'arrêter. Elle borne la dérive à UN tour de dépassement, au lieu d'un
    déroulé entier. Prétendre l'appliquer avant serait mentir sur ce qu'elle
    peut."""
    spec = AgentSpec(system="s", tools=frozenset(), max_steps=10, max_tokens=100)
    p = FauxProvider([_tour(input_tokens=5000, output_tokens=1000),
                      _fin(text="jamais")])
    res = agent_runtime.run(spec, FauxTransport(), p, prompt="go")
    assert res.stopped == "max_tokens"
    assert res.usage["input_tokens"] == 5000, (
        "le premier tour a bien été payé — la borne ne l'annule pas, elle "
        "empêche le suivant")


def test_les_jetons_LUS_EN_CACHE_ne_comptent_PAS_dans_la_borne():
    """⚠️ LE piège, et il couperait exactement les passages qu'on veut protéger.

    Les jetons lus en cache coûtent une fraction du tarif d'entrée. Les compter
    ferait dépasser la borne à un déroulé BIEN caché — c'est-à-dire le plus
    économe. Une borne qui punit l'économie est pire qu'aucune borne : elle
    pousse à la désactiver."""
    spec = AgentSpec(system="s", tools=frozenset(), max_steps=10, max_tokens=5000)
    p = FauxProvider([
        # 130 000 lus en cache — largement au-dessus de la borne s'ils comptaient
        _tour(input_tokens=1000, output_tokens=200, cache_read_input_tokens=130_000),
        _fin(input_tokens=500, output_tokens=100,
             cache_read_input_tokens=130_000),
    ])
    res = agent_runtime.run(spec, FauxTransport(), p, prompt="go")
    assert res.stopped == "end_turn", (
        f"coupé à tort : {res.stopped} — un déroulé bien caché a été puni pour "
        "son cache")
    assert res.usage["cache_read_input_tokens"] == 260_000


def test_l_ecriture_de_cache_COMPTE_elle():
    """Elle se paie plus cher que l'entrée : l'exclure laisserait un déroulé
    dépenser sans borne en (re)construisant son cache."""
    spec = AgentSpec(system="s", tools=frozenset(), max_steps=10, max_tokens=1000)
    # Entrée et sortie DÉCLARÉES (à zéro) : sans elles la borne ne se suit plus, et
    # l'arrêt serait `max_tokens_non_mesurable` (cf. `test_comptage_usages`).
    p = FauxProvider([_tour(input_tokens=0, output_tokens=0,
                            cache_creation_input_tokens=1500),
                      _fin(text="jamais")])
    res = agent_runtime.run(spec, FauxTransport(), p, prompt="go")
    assert res.stopped == "max_tokens"


def test_sans_borne_le_comportement_est_INCHANGÉ():
    """La borne est facultative : un travail isolé peut légitimement ne pas en
    porter. Ce qui ne doit jamais arriver, c'est qu'un passage sur des données
    clientes parte sans."""
    spec = AgentSpec(system="s", tools=frozenset(), max_steps=2)
    p = FauxProvider([_tour(input_tokens=10_000_000),
                      _fin(input_tokens=10_000_000)])
    res = agent_runtime.run(spec, FauxTransport(), p, prompt="go")
    assert res.stopped == "end_turn"
