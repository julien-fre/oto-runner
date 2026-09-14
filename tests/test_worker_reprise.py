"""La reprise après mort : un start re-claimé REPREND son fil, il ne rouvre rien.

La mort du worker n'est jamais un événement : le bail expire, un pair
re-claime, recharge le fil, continue. Le discriminant est le RUN LIÉ (bind_run
a eu lieu avant la mort), pas le kind — sans ça, chaque kill -9 sur un start
entamé fabriquait un run orphelin et un doublon de travail.
"""
from __future__ import annotations

from oto_runner import worker as W


class FauxBackend:
    def __init__(self, fil=None):
        self.fil = fil or []
        self.appels = []

    def bind_run(self, job_id, run_id):
        self.appels.append(("bind_run", run_id))

    def thread_read(self, run_id, include_raw=False, token=None):
        self.appels.append(("thread_read", run_id))
        return self.fil

    def thread_append(self, run_id, role, content, provider_raw=None, token=None):
        self.appels.append(("append", role))
        return 1

    def extend(self, job_id, lease_seconds=600):
        pass

    def complete(self, job_id, ok, error=None, run_id=None, result=None):
        self.appels.append(("complete", ok, run_id))
        self.appels.append(("complete_result", result))
        return "done"


class FauxMcp:
    def __init__(self, **kw):
        self.run_id = None
        self.outils = []
        self.appels = []      # (nom, arguments) — la clôture se lit à ses ARGS

    def outil(self, name, args=None):
        self.outils.append(name)
        self.appels.append((name, dict(args or {})))
        if name == "oto_procedure":
            return {"body_md": "la procédure"}
        if name == "run_start":
            return {"run_id": "r-NEUF"}
        return {}

    def schemas(self, names):
        return []

    def call(self, name, arguments):
        return "ok", False


def _job(kind, run_id=None):
    # Le jeton délégué est la forme SERVIE depuis le 02/09 : chaque boucle
    # impersonne son user. Une doublure qui l'omet décrit un travail que le
    # serveur refuse désormais de traiter. Même chose pour `attempts` : la réservation
    # le rend toujours, et le worker refuse un travail réservé qui ne le porte pas.
    return {"id": 7, "kind": kind, "run_id": run_id, "attempts": 1,
            "delegated_token": "oto_delegue",
            "payload": {"procedure": "demo", "tools": ["data_rows"],
                        "input": "Vas-y.", "max_steps": 3}}


def _run_stub(monkeypatch, resultat_prompt):
    """Capture le prompt/history passés à la boucle, sans tour de modèle."""
    vu = {}

    def faux_run(spec, transport, provider, prompt=None, history=None, on_turn=None, **_):
        vu.update(prompt=prompt, history=list(history or []))
        from oto_runner.agent_runtime import AgentResult
        return AgentResult(reply="fini", stopped="end_turn")

    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    monkeypatch.setattr(W, "McpSession", FauxMcp)
    return vu


def test_un_start_vierge_ouvre_son_run(monkeypatch):
    vu = _run_stub(monkeypatch, None)
    b = FauxBackend()
    W._traiter(b, _job("start"), provider=None)
    assert ("bind_run", "r-NEUF") in b.appels, "le start vierge ouvre et lie son run"
    assert vu["prompt"] == "Vas-y." and vu["history"] == []


def test_un_start_reclaime_apres_mort_reprend_son_fil(monkeypatch):
    """LE cas kill -9 : le job start porte déjà run_id (lié avant la mort) —
    le fil se recharge, AUCUN run neuf, AUCUN prompt rejoué (le message initial
    est déjà dans le fil, apposé au premier vol)."""
    vu = _run_stub(monkeypatch, None)
    fil = [{"provider_raw": {"role": "user", "content": "Vas-y."}},
           {"provider_raw": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "a"}]}},
           {"provider_raw": {"role": "tool", "tool_call_id": "a", "content": "ok"}}]
    b = FauxBackend(fil=fil)
    W._traiter(b, _job("start", run_id="r-SURVIVANT"), provider=None)
    assert ("thread_read", "r-SURVIVANT") in b.appels, "le fil est rechargé"
    assert not any(a[0] == "bind_run" for a in b.appels), "jamais un run neuf"
    assert vu["prompt"] is None, "reprise PURE — rien à rejouer"
    assert len(vu["history"]) == 3, "l'historique EST le fil du run survivant " \
        "(un fil finissant par user/tool passe INTÉGRAL — la troncature ne " \
        "vise que le tour assistant orphelin, cf. test dédié)"
    assert ("complete", True, "r-SURVIVANT") in b.appels


def test_un_continue_garde_son_message(monkeypatch):
    vu = _run_stub(monkeypatch, None)
    b = FauxBackend(fil=[{"provider_raw": {"role": "user", "content": "avant"}}])
    W._traiter(b, _job("continue", run_id="r-1"), provider=None)
    assert vu["prompt"] == "Vas-y.", "le continue porte SON message user"






def test_la_reprise_tronque_un_fil_finissant_par_assistant(monkeypatch):
    """Mort entre l'appose du tour assistant et celle de ses résultats : le fil
    finit par assistant, que les API de complétion refusent. Le transport est
    tronqué au dernier user/tool — le fil persisté, lui, reste intact."""
    vu = _run_stub(monkeypatch, None)
    fil = [{"provider_raw": {"role": "user", "content": "go"}},
           {"provider_raw": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "a"}]}},
           {"provider_raw": {"role": "tool", "tool_call_id": "a", "content": "ok"}},
           {"provider_raw": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "b"}]}}]
    b = FauxBackend(fil=fil)
    W._traiter(b, _job("start", run_id="r-x"), provider=None)
    assert vu["history"][-1]["role"] == "tool", \
        "le dernier message transporté est user/tool, jamais assistant"
    assert len(vu["history"]) == 3


def test_la_reprise_tronque_un_tour_multi_appels_incomplet(monkeypatch):
    """400 Mistral « Not the same number of function calls and responses »
    (vécu, job 27) : mort entre deux apposes de résultats d'un tour à N appels.
    Sans troncature le refus est PERSISTANT — chaque re-claim re-frappe le 400
    jusqu'à l'échec définitif. On tronque depuis le tour assistant incomplet."""
    vu = _run_stub(monkeypatch, None)
    fil = [{"provider_raw": {"role": "user", "content": "go"}},
           {"provider_raw": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "a"}, {"id": "b"}]}},
           {"provider_raw": {"role": "tool", "tool_call_id": "a", "content": "ok"}}]
    b = FauxBackend(fil=fil)
    W._traiter(b, _job("start", run_id="r-x"), provider=None)
    assert vu["history"] == [{"role": "user", "content": "go"}], \
        "le tour à 2 appels / 1 réponse saute ENTIER — le modèle le rejoue"


def test_un_tour_multi_appels_complet_passe_integral():
    from oto_runner.worker import _assainir_pour_transport
    fil = [{"role": "user", "content": "go"},
           {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
           {"role": "tool", "tool_call_id": "a", "content": "ok"},
           {"role": "tool", "tool_call_id": "b", "content": "ok"}]
    assert _assainir_pour_transport(fil) == fil


def test_un_echec_dextend_ne_tue_pas_le_run(monkeypatch):
    """Un 502 isolé sur le HEARTBEAT de bail tuait le run entier (vécu ×2,
    jets déjà payés) : le bail a 10 min de marge, le prochain tour le
    prolonge — on logge et on continue."""
    from oto_runner.backend import BackendError

    monkeypatch.setattr(W, "McpSession", FauxMcp)

    class BackendExtendMort(FauxBackend):
        def extend(self, job_id, lease_seconds=600):
            raise BackendError("/api → 502 : bad gateway", status=502)

    apposes = []

    def faux_run(spec, transport, provider, prompt=None, history=None, on_turn=None, **_):
        on_turn("assistant", {"content": "un tour"}, {"role": "assistant"})
        from oto_runner.agent_runtime import AgentResult
        return AgentResult(reply="fini", stopped="end_turn")

    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    b = BackendExtendMort()
    b.appels = apposes
    W._traiter(b, _job("start"), provider=None)   # ne lève PAS
    assert ("append", "assistant") in apposes and ("complete", True, "r-NEUF") in apposes


def test_lappose_du_fil_est_rejouee_avant_de_tuer(monkeypatch):
    """L'appose EST la persistance : 2 rejeux espacés avant d'abandonner
    (un 502 de « balle perdue » du pool Caddy ne vaut pas un run mort)."""
    from oto_runner.backend import BackendError

    monkeypatch.setattr(W, "McpSession", FauxMcp)
    monkeypatch.setattr("time.sleep", lambda s: None)

    class BackendAppendFragile(FauxBackend):
        rates = 0

        def thread_append(self, run_id, role, content, provider_raw=None, token=None):
            if self.rates < 2:
                self.rates += 1
                raise BackendError("/api → 502 : bad gateway", status=502)
            return super().thread_append(run_id, role, content, provider_raw)

    def faux_run(spec, transport, provider, prompt=None, history=None, on_turn=None, **_):
        on_turn("assistant", {"content": "t"}, {"role": "assistant"})
        from oto_runner.agent_runtime import AgentResult
        return AgentResult(reply="fini", stopped="end_turn")

    monkeypatch.setattr(W.agent_runtime, "run", faux_run)
    b = BackendAppendFragile()
    W._traiter(b, _job("start"), provider=None)
    assert b.rates == 2 and ("append", "assistant") in b.appels, \
        "2 échecs absorbés, le 3e essai a écrit"


def test_un_resultat_doutil_orphelin_est_ecarte():
    """400 « Unexpected tool call id in tool results » (vécu, job 28) : un 502
    rendu APRÈS que l'écriture a réussi côté serveur fait doubler ou orpheliner
    un résultat au rejeu. Le transport n'embarque que les résultats qui
    répondent à un appel du tour assistant ouvert — premier gagne."""
    from oto_runner.worker import _assainir_pour_transport
    fil = [{"role": "user", "content": "go"},
           {"role": "assistant", "tool_calls": [{"id": "a"}]},
           {"role": "tool", "tool_call_id": "a", "content": "ok"},
           {"role": "tool", "tool_call_id": "a", "content": "ok (doublon)"},
           {"role": "tool", "tool_call_id": "X75Hps2uA", "content": "orphelin"},
           {"role": "assistant", "content": "suite"},
           {"role": "user", "content": "continue"}]
    assert _assainir_pour_transport(fil) == [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "assistant", "content": "suite"},
        {"role": "user", "content": "continue"}]


def test_un_segment_incomplet_en_milieu_de_fil_saute_entier():
    """Le fil PERSISTÉ garde le tour qu'une reprise antérieure avait écarté de
    son transport — et la suite s'appose APRÈS lui. La vue cohérente est celle
    que le modèle repris a réellement eue : sans ce segment."""
    from oto_runner.worker import _assainir_pour_transport
    fil = [{"role": "user", "content": "go"},
           {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
           {"role": "tool", "tool_call_id": "a", "content": "ok"},
           # (b jamais apposé — mort ici ; la reprise est repartie d'avant)
           {"role": "assistant", "tool_calls": [{"id": "c"}]},
           {"role": "tool", "tool_call_id": "c", "content": "ok"}]
    assert _assainir_pour_transport(fil) == [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"}]


def test_un_run_start_degrade_echoue_avec_une_erreur_parlante(monkeypatch):
    """Un blip transport rend un run_start au contenu dégradé ({"_texte": …}
    sans isError) : le KeyError brut qui suivait maquillait un transitoire en
    mystère (vécu, job 49). Erreur NETTE, le retry de job fait le reste."""
    import pytest

    class McpDegrade(FauxMcp):
        def outil(self, name, args=None):
            if name == "run_start":
                return {"_texte": ""}
            return super().outil(name, args)

    monkeypatch.setattr(W, "McpSession", McpDegrade)
    with pytest.raises(RuntimeError, match="run_start sans run_id"):
        W._traiter(FauxBackend(), _job("start"), provider=None)


def test_le_resultat_declare_le_cache_a_cote_des_jetons(monkeypatch):
    """`usage_tokens` reste input+output — c'est la base des bornes de flotte, et
    la gonfler du cache les fausserait toutes. Le cache se déclare À CÔTÉ, parce
    que `input_tokens` ne compte QUE le reste non caché : sans ces deux postes,
    un run bien caché a l'air gratuit."""
    import oto_runner.worker as W2
    from oto_runner.agent_runtime import AgentResult

    def faux_run(spec, transport, provider, prompt=None, history=None, on_turn=None, **_):
        return AgentResult(reply="fini", stopped="end_turn",
                           usage={"input_tokens": 2400, "output_tokens": 600,
                                  "cache_creation_input_tokens": 9200,
                                  "cache_read_input_tokens": 148000})

    monkeypatch.setattr(W2.agent_runtime, "run", faux_run)
    monkeypatch.setattr(W2, "McpSession", FauxMcp)
    b = FauxBackend()
    W2._traiter(b, _job("start"), provider=None)
    result = next(a for a in b.appels if a[0] == "complete_result")[1]
    assert result["usage_tokens"] == 3000
    assert result["usage_cache_read"] == 148000
    assert result["usage_cache_write"] == 9200
