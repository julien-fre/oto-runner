# oto-runner

Le **worker externe** des runs hébergés oto (chantier runner, R2) : il réclame des
jobs, joue la boucle agent (modèle ↔ outils), et appose chaque tour au **fil** du
run. Un process = un worker ; N processes = la batterie.

## Le design en une règle

Le runner est un **client pur** — trois contrats, zéro import du backend :

1. **l'API de fil et de jobs** (REST `POST /api/me/runs/thread`, `/api/me/runner/jobs`) ;
2. **la face MCP** (`mcp.oto.cx/mcp`, jeton `oto_`) — credential, RBAC, activation,
   rédaction et journal s'appliquent CÔTÉ SERVEUR au passage de chaque appel,
   précisément parce que le worker est un client comme un autre ;
3. **une clé de modèle** (env — V1 : un worker = une org, c'est la clé qui décide
   qui paie).

Si ces contrats tiennent, le worker est remplaçable. La mort d'un worker n'est
jamais un événement : le bail du job expire, un pair re-claime, **recharge le fil**
(`provider_raw` rejoués verbatim) et continue.

## Ce que le worker possède

**Deux choses : l'adresse du backend et un secret de machine.** Rien d'autre —
ni compte, ni organisation, ni appartenance. Le secret (`otow_…`) se déclare côté
backend par `oto_admin_runner_worker op=create` (il est montré une fois, le
backend n'en garde que le haché) et se révoque de même. Tout ce que le worker
fait ensuite lui est **commandé par le backend avec chaque travail** :
l'organisation, le jeton **délégué** émis au nom de qui a déclaré la campagne
(borné au bail), la clé de modèle de l'organisation, la procédure, la
température. Le worker ne nomme rien et ne déduit rien.

⚠️ **Ce que ça remplace, et pourquoi c'est écrit ici.** Jusqu'au 09/09/2026 les
trois agents tournaient sous le **jeton personnel d'un compte** admin de quatorze
organisations. Un worker qui porte un jeton de compte sonde l'org **active** de
ce compte — silencieusement : rien ne refuse, la file paraît vide parce qu'on
regarde la mauvaise. Trois conceptions se sont succédé en deux jours (un compte
marqué worker, puis un compte qui nomme son org, puis un compte qui lit ses
orgs) ; chacune faisait *porter* ou *déduire* quelque chose au worker, et c'est
ce pli qui était faux. Le worker refuse donc de démarrer avec un `oto_…` dans
`OTO_WORKER_SECRET`, en le disant.

Le fil du run se lit et s'écrit avec le jeton **délégué** du travail : il vit
dans l'organisation du déclarant, où le secret du worker n'a aucun droit.

Sur la box, deux fichiers d'environnement, et la séparation est le point :
`/opt/oto-runner/.env` (600) est celui des **unités des workers** — il porte
`OTO_WORKER_SECRET` et aucun jeton de compte ; `/opt/oto-runner/.env.fleet` (600)
porte `OTO_TOKEN` pour **déclarer** une flotte (`scripts/flotte.sh`, unité
`oto-fleet-campagne`), un geste de compte que les workers ne font pas. Le
critère de preuve n'est pas le fichier mais l'environnement des processus :
`/proc/<pid>/environ` d'un worker ne contient pas `OTO_TOKEN` (vérifié le
11/09/2026).

## ⚠️ Cran d'armement

Sans `OTO_RUNNER_ARMED=1`, le worker refuse de démarrer. Le premier run hébergé
réel est gaté par une relecture d'architecture — ce cran rend la gate mécanique.

## Environnement

```
OTO_BASE=https://mcp.oto.cx          # REST (fil + jobs)
OTO_MCP_URL=https://mcp.oto.cx/mcp   # face MCP (outils)
OTO_WORKER_SECRET=otow_…             # le secret de MACHINE du worker — pas un jeton de compte (cf. « Ce que le worker possède »)
ANTHROPIC_API_KEY=…                  # la clé de modèle = qui paie
OTO_RUNNER_MODEL=claude-sonnet-5     # le DÉFAUT du worker — un agent qui déclare son modèle le remplace (cf. « Le modèle »)
OTO_RUNNER_ARMED=1                   # cf. ci-dessus
OTO_RUNNER_PASSAGES_DIR=passages     # où le worker écrit le JOURNAL de chaque travail
OTO_RUNNER_RELANCES_MAX=0            # relances d'un fil qui rend un appel au client
OTO_RUNNER_EFFORT=…                  # profondeur de raisonnement — Anthropic `output_config.effort`,
                                     # OpenAI-compatible `reasoning_effort` ; ABSENT = rien n'est envoyé
OTO_RUNNER_MAX_TOKENS=8192           # plafond de COMPLÉTION d'un tour (les deux providers)
OTO_RUNNER_MAX_TOOL_OUTPUT=120000    # plafond, en CARACTÈRES, d'une sortie d'outil servie au modèle
OTO_RUNNER_PARALLEL_TOOLS=1          # 1 (défaut) = le modèle groupe ses appels d'outils dans un
                                     # tour ; 0 = UN SEUL par tour (`parallel_tool_calls: false`)
OTO_RUNNER_TEMPERATURE=0             # DERNIER RECOURS seulement : la campagne prime (cf. ci-dessous)
```

⚠️ **La température se déclare sur la CAMPAGNE, pas sur le worker.** Elle est un
choix de procédure et non d'hôte : deux campagnes servies par le même worker n'en
veulent pas la même. Le backend la sert dans le travail depuis `v1.244.0`
(colonne `temperature` de la flotte), et l'ordre est **le passage, puis l'hôte
(`OTO_RUNNER_TEMPERATURE`), puis le fournisseur** — du plus proche du métier au
plus lointain. Ne rien déclarer nulle part n'envoie **rien**, et le fournisseur
applique son défaut : ce n'est pas la même chose qu'envoyer une valeur, parce que
ce défaut bouge sans prévenir. Mesuré le 06/09/2026 : sans température posée,
deux passages de la MÊME procédure sur le MÊME banc de trois lignes vont de 11 à
18 sur 18 — un bruit qui avalait entièrement l'écart qu'on croyait mesurer entre
deux versions du texte. Poser `0` rend les passages comparables.

⚠️ Le provider `anthropic` **refuse** une campagne qui en déclare une, avec un
message qui le dit : il règle la profondeur par `output_config.effort`. Un
réglage offert qui n'agit sur rien coûte plus cher qu'un réglage absent — on
l'ajuste des heures en croyant mesurer.

⚠️ **Le plafond de sortie d'outil se règle, et son défaut a changé.** À 12 000
caractères, la consigne métier rendue par `oto_procedure` (44 818 caractères) et
le schéma du tableau (38 079) arrivaient **coupés à leur premier tiers** : le
modèle inventait des options et une colonne, oubliait `statut: enrichi`, relisait
la procédure trois fois pour tenter d'en voir plus, et concluait « le schéma ne
contient pas de colonne contacts » — elle existait, dans la partie coupée (nuit
du 06/09/2026, tous les passages). Le défaut est passé à **120 000 caractères
≈ 30 k jetons** : une procédure de 45 k et un schéma de 40 k passent entiers, et
on reste très en dessous des fenêtres des modèles servis. Quand une troncature a
tout de même lieu, le texte servi se termine par une phrase qui **dit combien de
caractères manquent** et interdit de conclure par absence ; le journal, lui,
garde le texte ENTIER et note la longueur réellement servie (`servi_chars`).

⚠️ **Deux réglages que Scaleway facture ou coupe.** `OTO_RUNNER_EFFORT` n'était lu
que côté Anthropic ; Scaleway active le raisonnement **par défaut** et le facture
— sans ce réglage, on paie un raisonnement qu'on n'a pas choisi. Il n'a pas de
défaut : absent, rien n'est envoyé et le fournisseur applique le sien. Et sur
Scaleway les jetons de raisonnement **partagent le plafond de complétion** avec la
réponse : une fiche fait 3–6 k, et `8192` coupe la réponse quand le modèle a
raisonné avant (`finish_reason: length`, lisible au `stop_reason` du tour dans le
journal du travail). `OTO_RUNNER_MAX_TOKENS` se règle par worker, comme le modèle.

⚠️ **Le groupement des appels d'outils se règle** (chemin OpenAI-compatible).
Mistral Large 3 groupe jusqu'à **13 appels dans un même tour**, puis écrit la
fiche sans jamais reformuler une requête après un résultat décevant (mesuré le
06/09/2026). `OTO_RUNNER_PARALLEL_TOOLS=0` envoie `parallel_tool_calls: false` —
un appel par tour, chaque résultat lu avant le suivant ; absent ou `1` = le
comportement actuel, rien n'est envoyé ; toute autre valeur **lève**. Le réglage
est dit au journal du travail, dans l'événement `systeme`, à côté de
`max_tool_output`. Si le serveur rend tout de même plusieurs appels dans un tour,
le runner les exécute comme d'habitude — aucune garde ne masque son écart.

## Le runner conserve TOUT : un journal par travail

Nuit du 05 au 06/09/2026 : trois lignes, « abouties 3/3 », **7 écritures
refusées sur 8** — et rien, après coup, pour dire ce que le modèle avait envoyé
ni ce que le schéma lui avait répondu. Le bilan portait les motifs coupés à
soixante caractères, juste avant la colonne et la raison ; le fil du run porte
des sorties d'outils déjà tronquées pour le modèle ; la plateforme n'offre
aucune lecture des jobs ; et le fournisseur OpenAI-compatible est sans état.
*« Il faut que le runner conserve tout. »*

Le worker écrit donc, pour chaque travail, un **JSONL** — un événement par
ligne, horodaté, **écrit à chaque événement** (un travail qui plante laisse sa
trace jusqu'au plantage) — dans `<OTO_RUNNER_PASSAGES_DIR>/<flotte>/<job_id>.jsonl`
(`passages/` par défaut, relatif au répertoire courant du worker ; `<flotte>` est
le nom de la déclaration, celui de `<flotte>.bilan.json` et `<flotte>.log` ;
`hors-flotte` pour un travail sans flotte). Rien n'y est tronqué :

```
debut        le travail reçu (sans ses secrets), le fournisseur, le modèle demandé
run          le run ouvert ou repris (et la taille du fil rechargé)
systeme      le prompt système, l'allowlist, les plafonds
historique   le fil rechargé, tel que transporté au modèle (reprise)
utilisateur  le message initial (ou le message d'un `continue`)
modele       un par tour : texte, appels AVEC leurs arguments complets, usage, modèle servi,
             durée du tour (`duree_ms`), blocs bruts
outil        un par appel : arguments, sortie ENTIÈRE telle que rendue par le transport
             (`tronque_pour_le_modele` dit si le modèle en a lu une version plafonnée,
             `servi_chars` combien il en a reçu), ok, durée
fin          la raison d'arrêt, la réponse, l'usage cumulé
resultat     le résultat DÉCLARÉ à la plateforme, l'outcome, la clôture du run —
             **y compris quand le travail meurt** (`outcome: failed`), après l'`erreur`
erreur       le plantage : type, message ENTIER, pile — la CAUSE, jamais le dernier mot
```

Un `systeme` supplémentaire dit chaque **retentative** d'un tour de modèle (essai,
motif, attente). Un incident de transport — connexion perdue, délai d'attente,
429, 5xx — est rejoué trois fois (5 s puis 20 s) avant d'échouer : deux travaux
sont morts sur un `ReadTimeout` isolé le 06/09, l'un après cinq minutes de
silence du fournisseur. Et **un travail qui meurt clôt son run** (`run_finish`
`failed`, ce qui libère la ligne qu'il tenait) avant de se conclure en échec :
sans ça, la ligne réservée restait verrouillée tout son bail (quinze minutes).

⚠️ **Tuer une vague, en revanche, n'est pas neutre — et le symptôme est muet.**
Ce filet ne joue que si le travail MEURT ; un processus tué de l'extérieur
n'exécute rien, donc ne clôt pas son run et ne libère pas sa ligne. Elle reste
réservée jusqu'à l'expiration du bail. Mesuré le 07/09/2026 : une vague
interrompue à la main a laissé une ligne verrouillée, et le banc de six qui a
suivi n'en a traité que **cinq** — le sixième travail a réservé le vide (2
appels, 484 jetons) et s'est conclu normalement. Aucune erreur, aucun message :
juste une ligne absente du résultat.

C'est le pendant de « demander l'arrêt n'arrête rien » : demander est sans
effet immédiat, tuer a un effet différé qu'on ne voit pas. Après une vague tuée,
attendre le bail avant de relancer, ou lire le compte de lignes plutôt que le
nombre de travaux.

Sur le chemin Conversations : `conversation` (la requête entière, sans la clé),
`reponse` (les `outputs` bruts), `relance`. Une ligne :

```json
{"t": "2026-09-06T01:23:10.412Z", "ev": "outil", "id": "call_1", "nom": "data_write", "arguments": {"namespace": "vivier", "key": "@claimed", "patch": {"qualification_piece": "cessation_registre"}}, "ok": false, "transport_ko": false, "duree_ms": 212, "texte": "Error calling tool 'data_write': écriture refusée par le schéma : …", "tronque_pour_le_modele": false}
```

⚠️ Trois règles. **Le fichier porte de la donnée de la file de travail** : 0600,
répertoire 0700, jamais un secret (le jeton délégué et la clé de modèle n'y
passent pas). **Une écriture qui échoue lève** — un runner qui cesserait de
conserver en silence violerait le contrat ; le répertoire est vérifié au boot.
**Personne n'annonce un journal sans l'avoir relu** : le worker comme
l'ordonnanceur relisent le fichier (il existe, n'est pas vide, sa dernière ligne
se parse) avant d'écrire « journal complet : … » ; sinon l'ordonnanceur dit
`SANS JOURNAL RELU`, en erreur, compte (`journaux_absents`) et conclut dessus.
Le journal est écrit **là où le worker tourne** : quand les workers sont sur une
autre machine que l'ordonnanceur, c'est là qu'il se lit — et c'est précisément
ce que « sans journal relu » signale.

En mode Conversations, les outils sont exécutés par le connecteur : un appel
d'outil qui REVIENT au client interrompt le fil, et le job conclut sans avoir
rien écrit après avoir payé le run entier. `OTO_RUNNER_RELANCES_MAX` (défaut
**0**, donc inactif) autorise autant de relances du fil — on répond à l'appel
rendu et on repart de là.

## Cache de prompt (Anthropic)

Le modèle est sans état : chaque tour lui renvoie la procédure, les schémas
d'outils et tous les résultats d'outils déjà lus. Le provider Anthropic pose donc
**trois points de cache** (limite dure : 4) — dernier bloc du `system`, dernière
définition d'`tools`, dernier bloc du dernier message — et une lecture en cache
se facture ~0,1× le prix d'entrée. ⚠️ Un préfixe sous le minimum du modèle (1024
jetons sur Sonnet 5) n'est **pas** caché, sans erreur ni avertissement : le seul
juge est `usage_cache_read` au résultat du job.

## Un job

`start` : `{procedure, project_id, tools: […], input?, label?, max_steps?, model?, model_family?}` —
le worker ouvre le run, le lie au job, charge la procédure d'oto (jamais copiée),
joue la boucle sur un fil neuf. `continue` : `{run_id, input?}` — il recharge le
fil et continue ; `input` absent = reprise pure après une mort en plein tour.
Un job porte des **références, jamais un secret**.

### Session MCP perdue, et le job qui n'a rien écrit

Une session MCP ne survit pas au **redéploiement** du service : le serveur ne la
connaît plus (`-32600` « Session not found ») et tous les appels d'outils
suivants échouent d'un coup. L'agent lit ça comme une réponse, l'annonce
proprement et conclut — job « done » sans une écriture, donc **jamais rejoué**,
et la ligne reste « à traiter » sans que personne ne le sache (2 fiches perdues
en silence le 28/08).

Deux crans. Le client MCP **rouvre** la session et rejoue l'appel — une seule
fois par appel, journalisée, jamais en boucle ; un `-32600` n'ayant pas été
exécuté, le rejeu ne peut pas doubler une écriture. Si la réouverture échoue, il
**lève**. Et le worker **échoue le job** (`ok=False`) dès lors qu'il a réservé
une ligne, n'a rien écrit, et porte des appels morts au transport : le backend
le rejoue. Il n'existe pas d'issue légitime « conclu, rien écrit ».

## Le modèle : déclaré par l'agent, servi par le worker

Jusqu'au 12/09/2026 le modèle était épinglé par worker (`OTO_RUNNER_MODEL`), et
lui seul. Un agent pouvait en déclarer un — le serveur le stockait, le validait,
et le runner le **jetait** : `tests/test_champs_servis_et_lus.py` inscrivait
`model` dans `_NON_LUS` en toutes lettres. Un champ inerte est une dette ; un
champ qui PROMET une attribution qui n'arrive pas coûte plus cher, parce que la
promesse se lit dans les relevés.

```
payload.model         le modèle DEMANDÉ pour ce travail    → AgentSpec.model → le fournisseur
payload.model_family  la famille (= le dépôt de clé)       → refusée si ce n'est pas la sienne
absent                le modèle du worker, comme avant     (tout agent déclaré sans modèle)
```

⚠️ **Le worker ne valide pas le NOM.** Il ne connaît pas le catalogue — qui vit
côté serveur — et deviner ferait refuser ici un modèle que le fournisseur sert
très bien. Un nom inconnu remonte comme l'erreur du fournisseur, qui le nomme.

⚠️ **Il valide la FAMILLE, et c'est un refus franc.** Le serveur filtre déjà la
file (un worker ne réserve que les travaux de son dépôt, plus ceux qui n'en
déclarent aucun) ; ce second verrou existe pour la mauvaise CONFIGURATION — un
`OTO_RUNNER_OPENAI_BASE` changé sans la clé ferait réserver des travaux qu'on
servirait chez le mauvais fournisseur, sans rien casser de visible. Servir le
modèle du worker à la place de celui qu'on demande est exactement ce que ce lot
retire : un agent qui déclare Opus et reçoit Sonnet en silence est un mensonge
qui ne se lit que sur une facture.

⚠️ **Ce que ça coûte, dit franchement** : le cache de prompt est indexé par
modèle. Deux passages de modèles différents sur le même worker ne partagent plus
de préfixe — chacun garde le sien, aucun n'invalide l'autre, mais le premier
tour de chacun se paie plein tarif. Le pool reste homogène par FAMILLE ; c'est
la variation À L'INTÉRIEUR d'une famille qui segmente le cache.

## Deux modes : par la file, ou direct

*« Soit prise par DB, soit direct. »* Le **même** travail — même instruction,
mêmes outils, même boucle, même journal — se joue de deux façons :

| | **par la file** (`python -m oto_runner.fleet <flotte.yaml\|#id>`) | **direct** (`python -m oto_runner.direct <flotte.yaml\|#id> [--lignes N] [--concurrence K]`) |
|---|---|---|
| qui exécute | les workers (`oto-runner@{1,2,3}` sur la box), qui **réservent** des jobs | ce processus, ici, tout de suite |
| combien en parallèle | **trois — le nombre d'unités systemd ACTIVÉES**, pas ce que la déclaration demande (⚠️ cf. plus bas) | `--concurrence K`, ou la `concurrency` de la déclaration : K **processus** forkés, sans borne d'unités |
| la file de jobs | `POST /api/me/runner/jobs` : enfiler, réserver, lier, battre, conclure | **aucune** — jamais un appel à cette route |
| le jeton | le jeton **délégué** remis avec chaque job (l'agent agit pour le demandeur) | `OTO_TOKEN` du poste, qui tient lieu de jeton délégué |
| le modèle | celui que l'agent DÉCLARE, à défaut celui de l'env des **workers** | idem, sur l'env de **ce processus** (`OTO_RUNNER_MODEL`) |
| le journal JSONL | `passages/<flotte>/<job_id>.jsonl`, **là où le worker tourne** | `passages/<flotte>/direct-<horodatage>-<n>.jsonl`, ici |
| le bilan | `<flotte>.bilan.json` | `<flotte>.direct-<horodatage>.bilan.json` — même forme |

⚠️ **`workers` ne crée pas de workers — c'est une profondeur de file, pas un
nombre d'exécutants.** Le champ que porte une flotte déclarée (`workers` côté
plateforme, `concurrency` dans un YAML) règle **combien de travaux
l'ordonnanceur garde en vol**. Ce qui les exécute, ce sont les unités systemd
de la box, et **il y en a trois activées** (`oto-runner@{1,2,3}`, relevé le
07/09/2026). Déclarer `workers: 10` enfile dix travaux et n'en fait traiter que
trois à la fois : le dixième attend, il n'existe pas de dixième agent. Pour
élargir la batterie, il faut activer des unités (`systemctl enable --now
oto-runner@4`), et ça ne se décide pas depuis une déclaration.

En mode **direct**, le même chiffre a un sens différent et littéral : chaque
agent est un **processus forké sur le poste**, donc `--concurrence 4` fait bien
quatre agents. C'est la seule des deux voies où le nombre déclaré est le nombre
qui s'exécute.

⚠️ Et ce nombre n'est pas le plafond utile. Le fournisseur de modèle borne, lui,
le **débit de requêtes** : mesuré le 07/09/2026 sur le compte servi, **75
requêtes par minute** contre un million de jetons par minute — donc quatorze
fois plus de marge sur les jetons que sur les appels. **Optimiser le poids des
fiches ne débloque donc rien ; seul le nombre d'appels compte.**

Un dépassement ne se voit pas comme une lenteur : les 429 sont rejoués trois
fois, puis le travail échoue, puis `max_claims` sort la ligne de la file — **des
lignes manquent, sans erreur lisible**.

**Ce plafond ne se convertit pas en un nombre d'agents fixe** : il dépend du
temps qu'une fiche passe AILLEURS que chez le modèle. Deux mesures du même jour
le montrent :

| profil de passe | requêtes/min par agent | agents avant 75/min |
|---|---|---|
| enrichissement dense, peu de web (06/09, avant le correctif d'org) | 31 à 47 | **1,6 à 2,4** |
| passe à quatre ouvertures de page (07/09, après) | ~11 | **~7** |

Une passe qui scrute le web est **douce** pour le quota : chaque page ouverte est
du temps pendant lequel on ne demande rien au modèle. À l'inverse, une passe
**sans aucun appel web enchaîne ses tours sans respirer** — c'est la plus courte
en durée et la plus agressive par minute. Compter les agents sans regarder ce
profil donne un chiffre qui ne vaut que pour la passe qui l'a produit.

La bonne façon de le dire : diviser 75 par les requêtes/minute qu'UNE passe
consomme réellement, relevé sur ses propres journaux (`ev: "modele"` par travail,
divisé par la durée du bilan). C'est gratuit et ça ne casse aucune fiche.

**Quand utiliser lequel.** La file pour une campagne : plusieurs machines, reprise
après une mort, baux, délégation d'identité, arrêt gracieux — tout ce que la
plateforme garantit. Le direct pour **mesurer un passage sous la main** : une
consigne, un modèle, un réglage, sur N lignes, avec le journal complet et le
bilan sous les yeux dans la minute, sans déployer un worker ni ouvrir de job.

⚠️ **Un seul corps d'exécution.** Le mode direct passe par `worker._un_travail`
(`_traiter`, tel qu'il tourne en production) avec le travail que la flotte
construit (`declaration.payload`). Les trois verbes vers la file — `bind_run`,
`extend`, `complete` (`file_de_travail.FileDeTravail`) — sont le **seul** point
de variation : servis par la file serveur dans un mode, inertes dans l'autre
(`SansFile`, qui journalise ce qu'il aurait fait). Un banc qui mesurerait un
autre chemin sous le même nom serait un instrument menteur de plus ; un test le
tient (`tests/test_direct.py`).

**Ce que le mode direct NE mesure PAS** : la tenue du protocole de réservation
des jobs (aucun job n'existe) ; la concurrence entre agents telle qu'une batterie
de workers la vit (`--concurrence K` lance K **processus** sur un poste — pas des
threads : la deadline murale des requêtes repose sur `SIGALRM`, thread principal
seulement) ; les délégations de jeton (tout tourne sous le jeton du `.env`).

**Le bilan direct dit le modèle réellement servi.** Le fournisseur rapporte dans
chaque réponse le modèle qu'il a servi (`model`) ; le journal du travail le porte
à chaque tour et à la conclusion (`modele_servi`, à côté de `modele_demande`), et
le bilan direct conclut « modèle demandé X · servi Y » — **différent** quand il
l'est. Une étiquette de flotte a trompé deux heures de mesures une nuit ; le
nom demandé n'est pas une mesure, la réponse du fournisseur en est une.

## La flotte

`python -m oto_runner.fleet flotte.yaml` enfile des jobs `start` sur une file de
travail datastore et s'arrête sur une **borne** (déclaration complète :
`docs/fleet-example.yaml`). Le **nom du fichier** sans son extension est le nom
de la flotte : il est apposé à chaque job (`fleet`), et c'est par lui qu'on
retrouve les jobs d'une campagne — plus par « id ≥ N ».

Bornes **normales** (exit 0) : file vide, volume atteint, budget atteint. Toute
autre borne est une **panne** — exit 1, pour que systemd relance la campagne
quand la panne passe : échecs consécutifs, backend indisponible, outil critique
en échec, faux départs en série, **rendement effondré**.

Le plafond par LIGNE borne le prix d'un passage là où il dérape vraiment — une
ligne seule a coûté **65 571 jetons** le 01/09 :

```yaml
max_tokens_per_row: 40000   # plafond de jetons d'UNE ligne (absent ⟹ pas de borne)
```

Il est **descendu dans chaque travail enfilé**, donc appliqué par l'agent
lui-même : il vaut quel que soit le chemin qui a mis la ligne en file, y compris
sans ordonnanceur. L'agent s'arrête sur `stopped: max_tokens`, conclut, et la
ligne suivante repart à zéro. ⚠️ Ce qui compte dans ce total est ce qui est
**facturé** — entrée non cachée + sortie + écriture de cache : les jetons *lus*
en cache coûtent une fraction et gonfleraient le compteur d'un facteur trois sur
un déroulé bien caché, coupant les passages les plus économes.

Effet mesuré sur une campagne réelle (02/09) : sans borne, médiane 59 123 jetons
et **maximum 265 658**, dispersion ×48 ; borné à 80 000, médiane 86 303 et
maximum 107 204 — plus aucune ligne folle.

> ⚠️ **Corrigé le 02/09.** Ce paragraphe a décrit du 27/08 au 02/09 une borne de
> « rendement » — coût rapporté aux écritures produites, sur une fenêtre
> glissante — sous les noms `jetons_par_ecriture_max` et `rendement_fenetre`.
> **Ces deux réglages n'ont jamais existé dans le code** : le mécanisme a été
> conçu puis remplacé par la borne par ligne, sans que ce README suive. Une
> déclaration écrite depuis cette page repartait donc **sans aucune borne**, en
> croyant en avoir une — et le runner avertissait par-dessus que
> `max_tokens_per_row` était « ignoré », ce qui achevait de convaincre. Ce qui
> attrape « ça tourne à vide », ce sont les **faux départs en série**, décrits
> ci-dessus.

Chaque job conclu déclare son coût et sa sortie (`usage_tokens`,
`usage_cache_read`, `usage_cache_write`, `tool_counts`, `claims`, `writes`,
`claim_vide`, `faux_depart`, `model`) : c'est ce que l'ordonnanceur lit, sans jamais ouvrir un
fil. `usage_tokens` reste **input + output** — la base des bornes de flotte
(budget, rendement) ne bouge pas ; le cache se compte à côté.

## ⚠️ Le 05/09/2026 à 13:05 UTC, `oto_procedure` a changé de sens

Le worker **chargeait** la procédure nommée par un travail et l'injectait dans le
prompt système. Depuis le déploiement `dc78c10c`, il ne connaît plus cet objet :
si un travail suppose de lire une procédure, c'est **son instruction qui le dit**
et **l'agent qui la lit**.

Conséquence pour quiconque relit un relevé d'outils (`tool_counts`), et c'est un
piège d'interprétation à double sens :

| dans un travail… | absence d'appel à `oto_procedure` |
|---|---|
| **avant** le 05/09 13:05 UTC | **normale** — l'agent n'avait aucune raison d'appeler ce qu'il avait déjà dans son prompt |
| **après** | **signale une consigne jamais lue** — l'agent a travaillé sur sa seule instruction |

**Même chiffre, sens opposé, et la bascule est un déploiement — donc invisible
dans les données.** Relire un ancien log avec la grille d'aujourd'hui fait
conclure à un désastre qui n'a pas eu lieu ; relire un log récent avec la grille
d'hier fait se rassurer sur un précédent qui ne vaut plus.

Le cas qui a fait écrire ce paragraphe : une campagne de 504 fiches clientes dont
l'instruction, réécrite la veille pour une autre raison, avait perdu la seule
phrase qui nommait la procédure. Elle fonctionnait parce que le worker injectait ;
après ce déploiement, elle aurait produit 504 fiches improvisées **sans une seule
erreur**, et le bilan aurait annoncé « 504 abouties ».

## Une déclaration qui demande de lire ce qu'elle n'autorise pas à lire ne part pas

Trouvé le 06/09/2026 dans le journal d'un travail (événement #5 : *« Outil
`oto_procedure` indisponible pour ce run »*) : l'instruction disait *« Lis d'abord
la procédure … avec `oto_procedure` »*, et `oto_procedure` n'était pas dans
`tools`. L'allowlist est fail-closed, le worker n'injecte rien de métier :
**l'agent n'a jamais lu la consigne, dans aucune flotte de la campagne**, et
chaque travail concluait « done ». `load_spec` (et la flotte lue en base)
**refuse** désormais une déclaration qui nomme une `procedure` sans autoriser
`oto_procedure`, ou dont l'instruction nomme `oto_procedure` sans l'autoriser —
un refus franc qui nomme le défaut, jamais une correction silencieuse. Et le
journal de chaque travail porte, dès l'ouverture, les outils autorisés puis
l'**écart** entre ce que l'instruction nomme et ce que la liste autorise,
confronté au catalogue réel de la session (`outils.nommes_hors_liste`).

## Il n'y a PAS d'instruction par défaut

Le worker **exécute** une instruction, il n'en **compose** pas. Il ne sait pas ce
que l'instruction contient ni ce que l'agent va faire — donc il ne peut pas en
écrire une qui vaille. Une campagne sans instruction est **refusée** au chargement.

⚠️ Il en existait une, en dur, sept lignes : *« ta file est ce tableau, réserve
chaque ligne, traite-les selon la procédure, puis conclus »*. Deux défauts, et le
second a coûté cher :

**① Du métier dans le transport** — un tableau, des lignes, une réservation,
alors que le worker ne sait rien de tout ça. Même faute que les crochets retirés
le 03/09.

**② Sa FORME enseignait un court-circuit.** *Réserve → traite → conclus* est une
partition en trois temps où « chercher » n'apparaît nulle part, sinon caché dans
« selon la procédure ». Mesuré dans la nuit du 03 au 04/09 sur des vagues
réelles : **7 jobs sur 11 n'appelaient AUCUN outil** et écrivaient quand même une
fiche complète — le modèle *racontait* les appels au lieu de les émettre, avec
dirigeants et dates inventés, dans un compte rendu parfaitement structuré. Avec
une instruction qui dit d'où viennent les données : **1 sur 9, puis 1 sur 20**, à
consigne et modèle identiques.

⚠️ **Et la garde d'alors visait à côté** : « n'invente jamais une ligne ni un
identifiant » protège l'EXISTENCE d'une ligne, pas le CONTENU d'une fiche.
Inventer un dirigeant ne violait aucune consigne.

### Ce qu'une instruction d'enrichissement doit faire — quatre gestes

⚠️ **Ceci est un savoir de MÉTIER, pas une règle du worker** : il vaut pour les
passages qui enrichissent des fiches depuis des sources ouvertes, pas pour tous.
Il vit ici et dans ce que le serveur DÉRIVE, jamais dans le code du runner — et
le refus de démarrage ne le récite pas, il dit seulement ce qui manque.

**1. Nommer ce que la ligne porte DÉJÀ.** Sans ça l'agent ne distingue pas ce
qu'il a reçu de ce qu'il a trouvé — et les fabrications mesurées sont toutes des
embellissements de données reçues.

**2. Dire que RECOPIER ne compte pas.** Sinon c'est la sortie de moindre effort,
et elle a l'air d'un travail fini.

**3. Nommer les outils comme la SOURCE, pas comme une étape** : « ce que tu
ajoutes vient d'une source ouverte pendant ce run ». C'est le cœur — l'ancienne
version faisait de la recherche une étape d'une partition ; celle-ci en fait la
condition d'existence de la donnée.

**4. Viser la PLAUSIBILITÉ**, qui est le vrai piège : « une fiche écrite sans
avoir appelé ces outils est inventée, même quand elle est plausible, et surtout
quand elle est plausible. » Le court-circuit ne produit pas du délire, il produit
du **crédible**.

⚠️ **Une consigne plus DÉTAILLÉE aggrave le défaut** : à écrire des règles de
vérification de plus en plus fines, on obtient des simulations de plus en plus
convaincantes — la règle récitée mot pour mot pour justifier deux noms inventés.
Ce qui corrige n'est pas une règle de plus, c'est de retirer l'ambiguïté sur *d'où
viennent les données*.

## Ce qu'une campagne déclare est LU, ou son absence est justifiée

Trois champs servis par le serveur étaient **ignorés** par le runner sans que
rien ne le dise : `provider`, `model`, `max_consecutive_failures`.

⚠️ **Les deux premiers sont des étiquettes ; le troisième était une garde.**
Quelqu'un déclare « arrête après N échecs d'affilée », le serveur *valide* la
valeur — et le runner appliquait sa constante. *Une borne déclarée qui n'est pas
appliquée ne se découvre que le jour où on comptait dessus* : elle ne fausse pas
un relevé, elle laisse tourner une campagne qu'on croyait bornée. Et la
validation côté serveur achevait de convaincre qu'elle était prise en compte.

**La borne est désormais lue et appliquée** (`_MAX_FAILED_CONSECUTIFS` n'est plus
qu'un défaut). Les deux étiquettes restent non lues, **avec leur raison écrite** :
le worker est un pool homogène, son fournisseur et son modèle viennent de son
environnement, et un fil commencé chez un fournisseur ne se continue pas chez un
autre.

⚠️ **Et un contrôle tient la classe** (`tests/test_champs_servis_et_lus.py`) :
tout champ servi doit être lu **ou** inscrit comme non lu **avec sa raison**. Un
champ ajouté demain côté serveur et oublié ici fait rougir le contrôle — au lieu
d'attendre qu'un utilisateur s'aperçoive que sa déclaration n'a aucun effet. Avec
le contrôle symétrique : *un champ qu'on finit par lire doit sortir de la liste
des exclus*, sinon elle devient un cimetière qu'on ne relit plus.

*Mesuré avant de corriger : les 14 campagnes déclarées laissaient la borne à
`null`. Personne ne s'était cru protégé — c'est ce qui distingue ce cas d'un
incident.*

## L'état d'un passage, et ce qu'on n'a pas pu dire

La séquence est **déclarer → armer → prendre**. Une flotte naît `draft` ; la
prendre exige `armed`. ⚠️ **L'armement manquait jusqu'au 03/09** : le runner
déclarait puis tentait de prendre, le refus était systématique, rattrapé et
poursuivi. Mesure du jour : **14 campagnes en base, toutes `draft`, aucune jamais
`running`** — alors que huit vagues avaient réellement tourné.

⚠️ **Le défaut n'était pas l'armement manquant, c'était l'échec avalé.** Le
journal disait la vérité à chaque battement — *« le passage tourne quand même,
mais son état ne dira pas en cours »* — et personne ne l'a lue pendant huit
vagues. Une ligne de journal de plus n'y aurait rien changé.

D'où `etat_muet` dans le bilan : **combien de fois le passage n'a pas pu dire où
il en était** (déclaration, armement, prise, battement, accusé d'arrêt). Le
passage continue — perdre l'observabilité vaut mieux qu'une campagne qui refuse
de partir — mais il **conclut** dessus, en erreur, là où on lit le résultat :

```
⚠️ ce passage a tourné EN AVEUGLE : N geste(s) d'état n'ont pas pu être posés.
```

Un passage sain ne dit rien : *une alerte qui se déclenche toujours ne se
distingue pas d'un décor.*

`model` porte ce que le fournisseur **dit avoir servi**, relevé sur le tour
lui-même (le dernier tour fait foi si un alias bascule en cours de déroulé) : un
alias flotte quand le fournisseur le décide, et sans ce champ une anomalie de
campagne ne se date pas — on ignore quels jobs ont tourné avant la bascule et
lesquels après. À défaut de réponse du fournisseur, le worker estampille ce
qu'il a **demandé** : une estampille approchée vaut mieux qu'un `null`, qui ne se
distingue pas d'un job qui n'a jamais tourné.

> ⚠️ **Corrigé le 02/09.** Ce champ était déclaré, lu par le worker et compté par
> le bilan — mais **aucun transport ne le posait** : trois consommateurs, zéro
> producteur, `null` sur 100 % des jobs. Le défaut s'est vu quand la question
> « quelles lignes viennent de quel modèle ? » a été posée à froid sur une
> campagne réelle, et qu'il a fallu passer par l'horodatage du journal des
> écritures pour y répondre.

Le verdict de faux départ (réserver une ligne sans rien écrire) appartient au
worker, qui a vu les appels — un résultat qui ne le porte pas vient d'un worker
trop ancien, et la flotte lève plutôt que de le redeviner.

⚠️ **Un claim à VIDE n'est pas un faux départ.** Une réservation qui ne rend
aucune ligne (`row: null`) n'a rien réservé, et en fin de file il y a **toujours
plus d'agents que de lignes** : compter ces jobs a fait échouer une campagne
ABOUTIE (28/08 — 18 lignes sur 20, les 2 dernières sous bail chez des pairs
encore en vol ⟹ 5 jobs à un seul appel ⟹ borne mordue, `exit 1`), et aurait
arrêté à tort une montée par paliers. Le worker déclare donc `claim_vide` et le
driver l'**ignore** : ni +1 au compteur de faux départs (la borne du 28/08), ni
remise à zéro (elle rendrait la borne contournable par alternance), ni point de
rendement — aucune écriture n'était attendue de ce job.

Deux règles selon le chemin, parce que le worker n'y voit pas la même chose :
la **boucle locale** lit la sortie du claim et s'y fie ; en **Conversations** la
boucle tourne chez le fournisseur et aucune sortie ne remonte — la règle de
repli porte alors sur la NATURE des appels, *un job dont tous les appels sont
des gestes de tenue* (`data_claim_next`, `data_release`, `run_start`,
`run_finish`) *n'a fait aucun travail, donc il n'a rien réservé* ; un seul appel
d'outil métier, et il compte. ⚠️ Compter les appels ne suffit pas : sur une file
vide l'agent en fait **deux** — il réserve, reçoit `row: null`, relâche, puis
conclut proprement. Le **bilan** applique la même règle que la borne, sur la
même liste importée : les deux parlent du même job, ils ne peuvent pas se
contredire.

### Le bilan de la campagne

Le driver rend lui-même le **bilan** de sa flotte : à la cadence
`bilan_periode_s` (600 s par défaut) pendant qu'elle tourne, et une fois à la
fin **quelle que soit la borne** — panne et interruption comprises, parce que
c'est précisément ces jours-là qu'on le lit.

```
bilan flotte prospects-demo : sorties 12/30 · statut final : enrichi 10 · echec 2 · abouties 10 · 1,8 M jetons · 180,0 k/aboutie · data_write 14 appels, 2 refusés · Error calling tool 'data_write': écriture refusée par le schéma : `qualification_piece` = … ×2 · détail complet : prospects-demo.bilan.json
```

Le même bilan se pose en JSON à côté de la déclaration (`flotte.yaml` →
`flotte.bilan.json`, réécrit atomiquement à chaque fois) : lignes (départ /
restantes / **sorties** / **par_statut** / **abouties**), jobs (terminés /
échoués), jetons (total, par job, par sortie, par aboutie), et refus d'écriture
(comptes, **motifs** = textes serveur entiers groupés par texte identique, et
**detail** : quand, run, job, journal JSONL relu, texte complet).

⚠️ **La sortie de la file n'est pas l'issue.** « Sortie » = une ligne qui ne
correspond PLUS au `filter` de réservation — autant une ligne enrichie qu'une
ligne basculée `echec` par la plateforme après trois réservations sans écriture.
Le 06/09/2026 ce poste s'appelait « abouties » et disait « 3/3 » sur deux
abandons. Le bilan lit désormais la **colonne de statut au schéma du tableau**
(le field `role="status"`, avec son `lifecycle`) et ventile les lignes du
périmètre — le filtre sans sa clause de statut — par valeur finale ;
« abouties » ne compte que les états **terminaux hors `abandon_state`**. Trois
états, jamais deux : `abouties` vaut `null` **avec sa raison** (`abouties_omis`)
quand le schéma ne déclare pas de statut, quand le filtre ne borne que le
statut (la ventilation porte alors sur tout le tableau), ou quand le périmètre
porte plus de lignes terminales que de sorties (un lot rejoué). Le coût par
ligne vaut `null` tant que rien n'est sorti — jamais un chiffre calculé sur zéro.

Les refus d'écriture (`data_write` refusé par RBAC, quota ou schéma : le job
conclut « done » sans une ligne écrite) se lisent au journal des appels de
l'org, ce qui demande un `org:` dans la déclaration ; sans lui le poste est omis,
et le bilan dit pourquoi. ⚠️ **Le bilan n'interprète plus un refus.** Il y avait
un classificateur (« création refusée par le cran », « ligne inconnue ») : relu
contre les journaux serveur le 06/09, il inventait — et il a été cru. Les textes
sont conservés entiers, groupés par texte identique ; le journal de flotte les
abrège à l'affichage et pointe vers le JSON, et au bilan de fin chaque refus
est écrit entier sur sa ligne, avec le travail et le journal JSONL qui le portent.

## Tests

```bash
pip install -e .[dev] && pytest -q
```

La boucle se teste à sec (faux provider, faux transport) : allowlist fail-closed,
troncature marquée, plafond de tours, refus terminal, fil apposé en double étage
(projection neutre + brut provider), continuation sans tour inventé.
