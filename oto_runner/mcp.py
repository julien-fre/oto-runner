"""Le troisième contrat : la face MCP du backend — outils, gates et rédaction inclus.

Client streamable-http minimal (initialize → Mcp-Session-Id → tools/list,
tools/call), porté du harnais de campagne (`mcp_oto.py`), en requests. Ce qui
compte n'est pas ce qu'il fait mais ce qu'il N'A PAS à faire : credential, RBAC,
activation, rédaction de champs, journal d'audit — tout est appliqué CÔTÉ SERVEUR
au passage de l'appel, parce que ce client est un client comme un autre.

⚠️ Une session MCP ne survit pas au REDÉPLOIEMENT du service : le serveur ne la
connaît plus (`-32600` « Session not found ») et tous les appels suivants
échouent d'un coup. L'agent, lui, lit ça comme une réponse — il l'annonce
poliment et conclut : job « done » sans écriture, donc jamais rejoué, et la
ligne reste « à traiter » sans que personne ne le sache (2 fiches perdues en
silence le 28/08). La session se ROUVRE donc ici, une seule fois par appel, et
l'appel est rejoué ; si la réouverture échoue, on LÈVE — le job échoue et le
backend le rejoue, ce qui est la seule issue honnête.

Tout ce qui revient du serveur se décode en UTF-8 EXPLICITEMENT (28/08/2026) : le
flux SSE arrive en `text/event-stream` SANS charset, et requests applique alors le
défaut HTTP des `text/*` — ISO-8859-1 — donc « é » ressortait en « Ã© ». Le modèle
RECOPIE ses résultats d'outils : la corruption finissait dans les fiches produites.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import requests  # noqa: F401 — la forme des kwargs

from .agent_runtime import serialize
from .contenu_outil import Image, lire
from .deadline import post_with_deadline
from .descriptions import borne as _borne, reglage as _reglage_descriptions

logger = logging.getLogger("oto_runner")

_TIMEOUT = (10, 180)
# La signature d'une session que le serveur ne connaît plus : son message, et le
# code JSON-RPC qu'il rend quand l'en-tête `Mcp-Session-Id` est refusé. Un
# `-32600` (« Invalid Request ») n'a de toute façon jamais été exécuté : le
# rejouer après réouverture ne peut pas doubler une écriture.
_SESSION_PERDUE = re.compile(r"session not found|missing session id", re.I)
_CODE_REQUETE_INVALIDE = -32600


def _session_perdue(d: dict) -> bool:
    """La réponse dit-elle que notre session n'existe plus côté serveur ?"""
    err = (d or {}).get("error")
    if isinstance(err, dict) and (err.get("code") == _CODE_REQUETE_INVALIDE
                                  or _SESSION_PERDUE.search(str(err.get("message")
                                                                or ""))):
        return True
    return bool(_SESSION_PERDUE.search(str((d or {}).get("_brut") or "")))


def _utf8(r) -> str:
    """Le corps d'une réponse du serveur, décodé en UTF-8 — quoi qu'annoncent les
    en-têtes. `r.text` ne convient pas : il suit le charset déclaré, et le flux
    SSE n'en déclare AUCUN. `errors="strict"` : un corps qui n'est pas de l'UTF-8
    est un échec NET, jamais des remplacements muets au milieu d'une fiche."""
    try:
        return r.content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(
            f"réponse MCP non décodable en UTF-8 (octet {e.start}, {e.reason}) — "
            f"content-type « {r.headers.get('Content-Type', '?')} »") from e


#: Ce qu'un agent ne choisit JAMAIS, même si le schéma l'accepte.
#:
#: ⚠️ Ces deux-là ne sont pas des options d'appel : ce sont des réglages de CYCLE
#: DE VIE du tableau. Le paramètre passé à la réservation l'emporte sur ce que le
#: schéma déclare **et s'applique à toute la table**, pas à la ligne réservée.
#:
#: Mesuré le 07/09/2026 sur nos journaux : le modèle les pose de lui-même —
#: `lease_s` 812 fois, `max_claims` 209 fois — en lisant le schéma de l'outil et
#: en choisissant. Il n'invente rien : la porte lui est ouverte. Côté plateforme,
#: la mesure est pire encore : `max_claims` a ARMÉ une garde sur 322 tableaux qui
#: n'en déclarent aucune, 219 fois, toujours à la même valeur — des lignes ont pu
#: sortir de files où rien n'aurait dû les faire sortir.
#:
#: La plateforme a borné le dégât (le paramètre ne peut plus qu'assouplir). Ceci
#: est l'autre couche, et c'est la bonne pour ce défaut : **un réglage
#: irréversible ne doit pas être OFFERT au modèle**, pas seulement rendu
#: inoffensif. Mesurer et empêcher se ressemblent dans un compte rendu, jamais
#: dans les faits.
#:
#: Retiré, jamais réécrit : le serveur applique alors la déclaration du tableau,
#: qui est son domicile. Et le retrait se DIT — un paramètre qu'on enlève en
#: silence ferait chercher longtemps pourquoi la consigne semble ignorée.
_JAMAIS_AU_MODELE = {
    "data_claim_next": ("max_claims", "lease_s"),
}


#: Les jetons de CONTEXTE que le runner pose lui-même depuis le travail. Ils
#: sont retirés du schéma servi au modèle : un paramètre tendu à un modèle est
#: un paramètre qu'il remplit. Mesuré le 08/09/2026 — `_org` inventé quinze fois
#: sur quatre-vingt-un appels d'outil, et, sur un autre banc, une organisation
#: nommée de toutes pièces dans le compte rendu d'un agent (« l'organisation
#: <nom> (org <n>) »), suivie d'une fiche qui a l'air d'en être une. Le modèle
#: ne ment pas : il complète une forme qu'on lui a tendue. Ce qu'il ne voit pas,
#: il ne peut pas l'inventer.
#:
#: ⚠️ Un jeton ajouté ici DOIT être posé par `call()` — sinon on retire une
#: capacité au lieu d'une occasion de se tromper.
_POSES_PAR_LE_RUNNER = ("_org", "_project", "_run_id", "_group", "_instance")

#: `_group` est un jeton de contexte que le runner ne pose JAMAIS : un agent de
#: flotte travaille dans l'org de sa flotte, sans équipe. Il sort du schéma
#: servi pour la même raison que `_org`, et s'il arrive quand même, il est
#: RETIRÉ. Mesuré le 09/09/2026 (mode direct, jetable 634, passe D) : le modèle
#: a posé `_group=226` — l'org, recopiée dans le champ voisin — et l'écriture
#: de la fiche a été refusée (« groupe inconnu »), la ligne est restée en D.
#:
#: `_instance` pareil, le 11/09/2026 : 29 appels d'une passe E portaient un
#: `_instance` inventé, tous refusés. Un agent de flotte n'a pas d'instance à
#: nommer — le contexte est celui du travail, et le runner le pose.
_RETIRES_SANS_ETRE_POSES = ("_group", "_instance")


def _sans_jetons_de_contexte(schema: dict) -> dict:
    """Le schéma d'un outil, privé des jetons que le runner pose lui-même.

    Copie : le catalogue est mis en cache pour toute la session et `_declares`
    le relit pour savoir QUOI poser — le muter aveuglerait le poseur."""
    props = schema.get("properties") or {}
    if not any(j in props for j in _POSES_PAR_LE_RUNNER):
        return schema
    net = dict(schema)
    net["properties"] = {k: v for k, v in props.items()
                         if k not in _POSES_PAR_LE_RUNNER}
    if schema.get("required"):
        net["required"] = [r for r in schema["required"]
                           if r not in _POSES_PAR_LE_RUNNER]
    return net


#: La description d'un outil servie au modèle est bornée outil par outil, par le réglage du travail
#: (`descriptions.py` : `data_write` entière, les autres à 1 024 quand la déclaration se tait).
#: Mesuré le 11/09/2026 : l'API Mistral accepte au moins 65 536 caractères ; l'ancienne coupe muette
#: à 1 024 masquait les règles de `data_write`. Une description coupée est DITE au journal du travail
#: (`descriptions_outils`).


class McpSession:
    """Une session MCP réutilisable — le transport d'outils de la boucle."""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None,
                 project: Optional[int] = None, run_id: Optional[str] = None,
                 org: Optional[int] = None, descriptions: Optional[dict] = None):
        # La borne des descriptions servies, réglée par le travail ; une forme fausse lève ICI.
        self._descriptions = _reglage_descriptions(descriptions)
        self.url = url or os.environ.get("OTO_MCP_URL", "https://mcp.oto.cx/mcp")
        self.token = (token or os.environ.get("OTO_TOKEN", "")).strip()
        # Les jetons de contexte d'appel (ADR 0038) : posés sur CHAQUE appel de
        # travail — le projet résout l'org et les identités, le run corrèle le
        # journal. C'est le worker qui les porte, pas le modèle.
        self.project = project
        self.org = org      # l'org de la MISSION — sert les tools qui déclarent
        # `_org` mais pas `_project` (oto_procedure : une doctrine d'org se
        # charge dans SON org, pas dans l'org maison du jeton)
        self.run_id = run_id
        self.session: Optional[str] = None
        self._n = 0
        self._props: Optional[dict] = None   # tool → propriétés d'entrée déclarées
        self._outils: Optional[list] = None  # le tools/list de la session, lu UNE fois
        self._ouvrir()

    def _post(self, corps: dict, avec_entetes: bool = False):
        entetes = {"Authorization": f"Bearer {self.token}",
                   "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self.session:
            entetes["Mcp-Session-Id"] = self.session
        if corps.get("method") == "initialize" and self.org is not None:
            # La boîte à outils d'une session se calcule au HANDSHAKE : sans cet
            # en-tête, elle suit l'org maison du compte porteur, pas celle de la
            # mission (#1058 : la maison a basculé 2 → 226 le 22/09, et les
            # connecteurs de toutes les flottes ont disparu ~2h). Le backend
            # vérifie l'appartenance ; il ne le lit qu'à l'initialize — les appels
            # portent déjà leur org en `_org`, c'est elle qui fait foi ensuite.
            entetes["X-Oto-Org"] = str(self.org)
        r = post_with_deadline(self.url, json=corps, headers=entetes,
                               timeout=_TIMEOUT, wall_s=300)
        brut = _utf8(r)
        charge = "".join(l[5:].strip() for l in brut.splitlines()
                         if l.startswith("data:")) or brut
        try:
            data = json.loads(charge) if charge.strip() else {}
        except Exception:  # noqa: BLE001
            data = {"_brut": charge[:400]}
        return (r.headers, data) if avec_entetes else data

    def _ouvrir(self):
        # L'ancien id part AVANT le premier POST : un serveur qui refuse une
        # session inconnue refuserait aussi l'initialize qui la porte.
        self.session = None
        # Un 502 pendant l'initialize rendait une session MUETTE (session id
        # absent avalé) : tous les appels suivants mouraient en « Missing
        # session ID » cryptique (vécu, nuit du 15/08). Trois essais espacés,
        # puis un échec NET — le retry de job fait le reste.
        import time as _t
        for essai in range(3):
            self._n += 1
            entetes, _ = self._post(
                {"jsonrpc": "2.0", "id": self._n, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": {"name": "oto-runner", "version": "0.1"}}},
                avec_entetes=True)
            self.session = entetes.get("mcp-session-id") or entetes.get("Mcp-Session-Id")
            if self.session:
                self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
                return
            _t.sleep(5 * (essai + 1))
        raise RuntimeError(
            "initialize MCP sans session id après 3 essais — backend indisponible ?")

    # ── le contrat ToolTransport de la boucle ────────────────────────────────
    def schemas(self, names: frozenset) -> list[dict]:
        """Les schémas de l'allowlist, au format modèle — lus du tools/list de la
        session (donc déjà filtrés par la visibilité du COMPTE du worker : deux
        crans, le compte puis l'allowlist du job)."""
        if self._outils is None:
            # Une session vit un travail : le catalogue se lit UNE fois, et sert
            # aux schémas comme au relevé d'écart (`catalogue`).
            self._n += 1
            d = self._post({"jsonrpc": "2.0", "id": self._n,
                            "method": "tools/list", "params": {}})
            outils = (d.get("result") or {}).get("tools")
            if not outils:
                # Un tools/list qui échoue (502 en vol) laissait un cache VIDE :
                # le fail-safe ne posait plus AUCUN jeton, et le job mourait plus
                # loin sur une erreur MÉTIER trompeuse (« Aucune doctrine (scope
                # org) », vécu — jamais rejouée car non transitoire). Échec NET
                # ici : le retry de job repart d'une session saine.
                raise RuntimeError(
                    f"tools/list vide ou en erreur ({str(d)[:120]}) — session dégradée")
            self._outils = list(outils)
        out = []
        self._props = {}
        self.descriptions_servies = []
        for t in self._outils:
            props = ((t.get("inputSchema") or {}).get("properties") or {})
            self._props[t.get("name") or ""] = frozenset(props)
            if t.get("name") in names:
                desc = t.get("description") or ""
                servie = _borne(t["name"], self._descriptions, len(desc))
                self.descriptions_servies.append(
                    {"outil": t["name"], "longueur": len(desc), "servie": servie})
                out.append({"name": t["name"],
                            "description": desc[:servie],
                            "input_schema": _sans_jetons_de_contexte(
                                t.get("inputSchema")
                                or {"type": "object", "properties": {}})})
        return out

    def catalogue(self) -> frozenset:
        """TOUS les noms d'outils que la session voit — pour dire, au journal du
        travail, l'ÉCART entre ce que l'instruction nomme et ce que l'allowlist
        autorise. 06/09/2026 : « avec `oto_procedure` » dans l'instruction,
        `oto_procedure` absent de `tools` — et l'agent n'a jamais lu la consigne."""
        if self._props is None:
            self.schemas(frozenset())
        return frozenset(self._props)

    def _declares(self, name: str) -> frozenset:
        """Les propriétés d'entrée DÉCLARÉES par ce tool. C'est ce qui rend la
        pose des jetons de contexte SÉLECTIVE : ils sont advertisés par tool
        (ADR 0038), et les poser à l'aveugle fait refuser l'appel ENTIER à la
        validation — vécu au premier vol de flotte : `oto_procedure` ne déclare
        pas `_project`, 4 jobs en échec avant une seule ligne traitée. Un tool
        absent du cache ne reçoit AUCUN jeton (un appel sans contexte vaut
        mieux qu'un refus)."""
        if self._props is None:
            self.schemas(frozenset())
        return self._props.get(name, frozenset())

    def call(self, name: str, arguments: dict) -> tuple[str, bool, tuple[Image, ...]]:
        """UN appel d'outil → (texte pour le fil, is_error, images). Les jetons de
        contexte sont posés ici — le modèle n'a pas à les connaître.

        ⚠️ TOUS les blocs du résultat sont lus (`contenu_outil.lire`) : on ne
        gardait que le premier bloc texte, et l'image de `web_image` n'atteignait
        jamais le modèle (#32). Un bloc que le runner ne sait pas transporter lève
        `BlocNonGere` — l'appel échoue en le nommant, il n'est jamais amputé."""
        args = dict(arguments or {})
        for interdit in _JAMAIS_AU_MODELE.get(name, ()):
            if args.pop(interdit, None) is not None:
                logger.warning(
                    "%s : `%s` posé par le modèle a été RETIRÉ — le cycle de vie "
                    "d'un tableau se déclare à son schéma, pas dans un appel",
                    name, interdit)
        for jeton in _RETIRES_SANS_ETRE_POSES:
            if args.pop(jeton, None) is not None:
                logger.warning(
                    "%s : `%s` posé par le modèle a été RETIRÉ — un agent de "
                    "flotte travaille dans l'org de sa flotte, sans équipe",
                    name, jeton)
        declares = self._declares(name)
        if self.project is not None and "_project" in declares:
            args.setdefault("_project", self.project)
        if self.org is not None and "_org" in declares:
            # ⚠️ L'org est posée dès que le tool la DÉCLARE, même s'il déclare
            # aussi `_project`. La règle d'avant ne la posait que faute de
            # projet, en supposant qu'un projet résout son org. Mesuré le
            # 07/09/2026 sur 333 travaux : c'est FAUX pour `data_claim_next`.
            # Le projet (219) était bien transmis, et l'appel se résolvait quand
            # même dans l'org du jeton (2 « Otomata Admin ») au lieu de celle de
            # la mission (226) — refus, puis le modèle relisait l'erreur et
            # reposait `_org` lui-même au tour suivant. UN TOUR PERDU PAR FICHE,
            # 333 fois sur 333, soit 13 % de tous les appels d'outil.
            #
            # Un jeton redondant coûte quelques octets ; le déduire coûtait un
            # aller-retour de modèle complet, à chaque ligne.
            #
            # ⚠️ IMPOSÉE, et non `setdefault` — renversement du 08/09/2026, sur
            # mesure et non sur relecture. La règle d'avant laissait au modèle
            # un `_org` explicite, au motif que « viser ailleurs doit rester
            # possible ». C'était défendable tant que personne ne l'avait
            # exercé. Le premier passage en mode file l'a exercé : le modèle a
            # posé un `_org` INVENTÉ — une organisation dont il n'est membre
            # d'aucune — quinze fois sur quatre-vingt-un appels d'outil, et le `setdefault`
            # a respecté son invention. Il n'a pas visé une autre org : il a
            # rempli un champ qu'on lui tendait, avec une valeur plausible.
            #
            # Un agent de flotte travaille dans l'org de sa flotte. Le besoin de
            # viser ailleurs n'a jamais été exercé volontairement ; l'accident,
            # lui, l'a été 15 fois au premier essai. Même traitement que
            # `max_claims` et `lease_s` : ce que le runner sait, le modèle ne le
            # choisit pas.
            pose = args.get("_org")
            if pose is not None and pose != self.org:
                logger.warning(
                    "%s : `_org`=%r posé par le modèle a été REMPLACÉ par %r — "
                    "un agent de flotte travaille dans l'org de sa flotte",
                    name, pose, self.org)
            args["_org"] = self.org
        if self.run_id is not None and "_run_id" in declares:
            args.setdefault("_run_id", self.run_id)
        self._n += 1
        corps = {"jsonrpc": "2.0", "id": self._n, "method": "tools/call",
                 "params": {"name": name, "arguments": args}}
        d = self._post(corps)
        if _session_perdue(d):
            # UNE réouverture par appel, jamais une boucle : `_ouvrir` lève
            # après 3 initialize muets, et le job échoue — c'est voulu. Laisser
            # l'erreur revenir au modèle ferait conclure « done » sans écriture.
            logger.warning("session MCP perdue sur %s : réouverture", name)
            self._ouvrir()
            logger.info("session MCP rouverte après « Session not found » — "
                        "%s rejoué", name)
            self._n += 1
            d = self._post(dict(corps, id=self._n))
            if _session_perdue(d):
                raise RuntimeError(
                    f"session MCP rouverte mais {name} reste refusé "
                    f"({str((d or {}).get('error') or d)[:200]}) — le job échoue "
                    "pour être rejoué")
        res = (d or {}).get("result") or {}
        texte, images = lire(res.get("content"))
        if res.get("isError"):
            return texte or serialize(res), True, images
        if res.get("structuredContent") is not None:
            return serialize(res["structuredContent"]), False, images
        if res.get("content"):
            return texte, False, images
        err = d.get("error")
        if err:
            return serialize(err), True, ()
        return serialize(d), False, ()

    def outil(self, name: str, arguments: Optional[dict] = None) -> dict:
        """Appel direct hors boucle (run_start, run_finish…) — rend le payload."""
        texte, is_error, images = self.call(name, arguments or {})
        if images:
            raise RuntimeError(f"{name} : {len(images)} image(s) rendue(s) à un appel "
                               "direct, qui ne lit que du JSON")
        try:
            data = json.loads(texte)
        except Exception:  # noqa: BLE001
            data = {"_texte": texte}
        if is_error:
            raise RuntimeError(f"{name} : {texte[:300]}")
        return data
