"""Un déroulé qui CASSE a dépensé des jetons — et il doit le dire.

Jusqu'ici, `run` levait et son compteur d'usage mourait avec la pile :
`en_echec` concluait le travail sans `result`, et le serveur comptait ZÉRO sur
les déroulés qui partent en vrille. C'est-à-dire sur les plus chers — un travail
qui brûle deux cent mille jetons puis casse au dernier pas, et qui peut
recommencer à chaque tentative.

Ce que ces bancs tiennent :

1. les jetons déjà dépensés SURVIVENT à l'exception ;
2. `run` LÈVE toujours — on n'a pas transformé un échec en résultat ;
3. un déroulé mort avant d'avoir rien dépensé ne rend pas des zéros, qui se
   liraient comme une mesure.
"""
from __future__ import annotations

import pytest

from oto_runner import agent_runtime, conclusion


class _Transport:
    """Le transport d'outils, réduit à ce que la boucle en demande."""

    def schemas(self, names):
        return [{"name": n, "description": "", "input_schema": {"type": "object"}}
                for n in sorted(names or ())]

    def call(self, name, arguments):
        return ("ok", False)


class _ProviderQuiCasse:
    """Sert `tours` tours FACTURÉS — chacun appelant un outil, donc la boucle
    continue — puis casse. C'est le cas qui coûte : des jetons déjà dépensés au
    moment où ça lâche."""

    def __init__(self, tours_avant_la_casse=1):
        self.restants = tours_avant_la_casse

    def format_tools(self, schemas):
        return list(schemas or ())

    def user_message(self, text):
        return {"role": "user", "content": text}

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.raw_content}

    def tool_messages(self, results):
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r["id"],
             "content": r["text"], "is_error": r["is_error"]} for r in results]}]

    def complete(self, **kw):
        if self.restants <= 0:
            raise RuntimeError("le fournisseur a lâché")
        self.restants -= 1
        return agent_runtime.Turn(
            text="",
            tool_calls=(agent_runtime.ToolCall(id="t0", name="outil",
                                               arguments={}),),
            stop_reason="tool_use",
            usage={"input_tokens": 1000, "output_tokens": 200,
                   "cache_read_input_tokens": 50,
                   "cache_creation_input_tokens": 10},
            raw_content=[], model="claude-opus-5")


def _lever(tours=1):
    """Joue un déroulé jusqu'à la casse et rend l'exception."""
    spec = agent_runtime.AgentSpec(system="s", tools=frozenset({"outil"}),
                                   max_steps=8)
    with pytest.raises(RuntimeError) as e:
        agent_runtime.run(spec, _Transport(), _ProviderQuiCasse(tours),
                          prompt="fais quelque chose")
    return e.value


# ── 1. les jetons survivent ───────────────────────────────────────────────────

def test_les_jetons_DEJA_DEPENSES_survivent_a_l_exception():
    """⚠️ LE banc du lot. Ils l'étaient chez le fournisseur quoi qu'il arrive
    ensuite ; les perdre avec la pile, c'est compter zéro sur un déroulé cher."""
    e = _lever(tours=1)
    assert e.usage_partiel["input_tokens"] == 1000
    assert e.usage_partiel["output_tokens"] == 200
    assert e.usage_partiel["cache_read_input_tokens"] == 50


def test_ils_s_ACCUMULENT_sur_plusieurs_tours():
    e = _lever(tours=3)
    assert e.usage_partiel["input_tokens"] == 3000


def test_run_LEVE_toujours():
    """Le contrat de `run` est de lever quand ça casse. Le changer ferait passer
    un échec pour un résultat auprès de TOUS ses appelants."""
    assert isinstance(_lever(), RuntimeError)


def test_le_modele_SERVI_survit_aussi():
    """Sans lui, on saurait ce qu'on a dépensé sans savoir à quel tarif."""
    assert _lever().modele_partiel == "claude-opus-5"


# ── 2. ce que la conclusion rend ──────────────────────────────────────────────

def test_la_conclusion_rend_les_QUATRE_postes():
    """La même forme que `resultat_declare` : le serveur ne doit avoir QU'UNE
    façon de lire un coût."""
    r = conclusion.resultat_partiel(_lever(), "claude-sonnet-5")
    assert r["usage_input"] == 1000 and r["usage_output"] == 200
    assert r["usage_cache_read"] == 50 and r["usage_cache_write"] == 10
    assert r["usage_tokens"] == 1200, "entrée + sortie, comme partout"
    assert r["model"] == "claude-opus-5", "le SERVI l'emporte sur le demandé"


def test_le_motif_d_arret_est_le_TYPE_de_l_exception():
    """La ligne dit à la fois ce qui a été dépensé et pourquoi ça s'est arrêté."""
    assert conclusion.resultat_partiel(_lever(), "m")["stopped"] == "RuntimeError"


def test_un_deroule_mort_AVANT_son_premier_tour_ne_rend_RIEN():
    """⚠️ Pas des zéros : un zéro se lirait « mesuré, et nul ». `None` laisse le
    serveur écrire NULL, qui se lit « non mesuré » — la distinction sur laquelle
    tout le compteur repose."""
    assert conclusion.resultat_partiel(_lever(tours=0), "m") is None


def test_une_exception_SANS_usage_ne_rend_rien():
    """Une exception qui n'a pas traversé `run` (transport, backend) n'a pas de
    jetons à déclarer — et n'en invente pas."""
    assert conclusion.resultat_partiel(ValueError("ailleurs"), "m") is None


def test_le_modele_DEMANDE_sert_de_repli():
    e = _lever()
    e.modele_partiel = None
    assert conclusion.resultat_partiel(e, "claude-haiku-4-5")["model"] == "claude-haiku-4-5"


# ── 3. la conclusion ENVOIE ce qu'elle a mesuré ───────────────────────────────

class _FileEspionne:
    """La file, réduite à la conclusion — et à ce qu'elle porte."""

    def __init__(self):
        self.conclusions = []

    def complete(self, job_id, ok, error=None, run_id=None, result=None):
        self.conclusions.append({"job_id": job_id, "ok": ok, "result": result})
        return {"ok": True}


def test_la_conclusion_d_un_echec_PORTE_le_cout_jusqu_au_serveur():
    """⚠️ Ce banc existe parce qu'une épreuve de chute l'a réclamé : retirer
    `result=` de l'appel à `complete` ne faisait rougir AUCUN test. Mesurer le
    coût sans jamais l'envoyer aurait laissé le serveur compter zéro, avec un
    calcul parfaitement juste en amont."""
    tenu = conclusion.RunEnCours(mcp=None, run_id=None)
    file = _FileEspionne()
    conclusion.en_echec(None, tenu, {"id": 77}, file, _lever(), "claude-opus-5")
    envoye = file.conclusions[0]
    assert envoye["job_id"] == 77 and envoye["ok"] is False
    assert envoye["result"]["usage_input"] == 1000, "le coût part avec l'échec"
    assert envoye["result"]["usage_cache_read"] == 50


def test_un_echec_SANS_jetons_conclut_quand_meme_sans_resultat():
    """Le bord opposé : une mort avant le premier tour conclut le travail — elle
    ne doit simplement rien déclarer, plutôt que des zéros."""
    tenu = conclusion.RunEnCours(mcp=None, run_id=None)
    file = _FileEspionne()
    conclusion.en_echec(None, tenu, {"id": 78}, file, _lever(tours=0), "m")
    assert file.conclusions[0]["result"] is None
    assert file.conclusions[0]["ok"] is False


# ── 4. un déroulé mort garde l'inconnu inconnu (comptage des usages × #16) ────

class _ProviderPeuBavard(_ProviderQuiCasse):
    """La même casse, mais le fournisseur ne déclare que les postes qu'on lui donne :
    les autres restent inconnus (cf. `comptage`)."""

    def __init__(self, tours_avant_la_casse=1, usage=None):
        super().__init__(tours_avant_la_casse)
        self.usage = {"input_tokens": 1000} if usage is None else usage

    def complete(self, **kw):
        if self.restants <= 0:
            raise RuntimeError("le fournisseur a lâché")
        self.restants -= 1
        return agent_runtime.Turn(
            text="", tool_calls=(agent_runtime.ToolCall(id="t0", name="outil", arguments={}),),
            stop_reason="tool_use", usage=dict(self.usage), raw_content=[],
            model="claude-opus-5")


def _lever_avec(provider):
    spec = agent_runtime.AgentSpec(system="s", tools=frozenset({"outil"}), max_steps=8)
    with pytest.raises(RuntimeError) as e:
        agent_runtime.run(spec, _Transport(), provider, prompt="fais quelque chose")
    return e.value


def test_un_poste_non_declare_d_un_deroule_mort_reste_inconnu_jamais_zero():
    """⚠️ La conclusion d'un échec additionnait `int(x or 0)` : une sortie que le
    fournisseur n'a pas déclarée devenait 0, et `usage_tokens` se lisait 1000 — un
    total présenté comme complet. Il reste `None`, l'entrée connue reste connue, et
    la couverture dit ce qui fonde les postes."""
    r = conclusion.resultat_partiel(_lever_avec(_ProviderPeuBavard(1, {"input_tokens": 1000})), "m")
    assert r is not None, "un tour a été facturé : ce n'est pas « rien dépensé »"
    assert r["usage_input"] == 1000
    assert r["usage_output"] is None and r["usage_tokens"] is None
    assert r["usage_cache_read"] is None and r["usage_cache_write"] is None
    assert r["usage_couverture"]["tours"] == 1


def test_un_tour_facture_sans_aucun_poste_declare_n_est_pas_rien_depense():
    """⚠️ « Rien dépensé » se jugeait sur les VALEURS (`not any(usage.values())`) : un
    tour joué dont le fournisseur n'a rien déclaré se lisait comme un déroulé mort avant
    son premier tour, et son coût disparaissait. Le critère est le nombre de tours."""
    r = conclusion.resultat_partiel(_lever_avec(_ProviderPeuBavard(2, {})), "m")
    assert r is not None and r["usage_couverture"]["tours"] == 2
    assert all(r[k] is None for k in ("usage_tokens", "usage_input", "usage_input_total",
                                      "usage_output", "usage_cache_read", "usage_cache_write"))


def test_mort_ou_conclu_le_serveur_lit_les_memes_postes():
    """Une seule façon de lire un coût : les postes d'usage d'un déroulé mort sont ceux
    d'un déroulé conclu, clé pour clé."""
    mort = conclusion.resultat_partiel(_lever(), "m")
    conclu = conclusion.resultat_declare(
        agent_runtime.AgentResult(reply="fini", stopped="end_turn", usage={},
                                  couverture={"tours": 1}), "m")
    assert {k for k in mort if k.startswith("usage_")} == {k for k in conclu if k.startswith("usage_")}
