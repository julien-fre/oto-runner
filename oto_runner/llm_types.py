"""Les types partagés du seam provider — et rien d'autre.

`Turn.raw_content` est OPAQUE pour la boucle : sa forme appartient au provider
(liste de blocs chez Anthropic, message assistant complet chez OpenAI-compat) et
il se rejoue tel quel — c'est aussi ce que le fil du backend stocke en
`provider_raw`. Conséquence assumée en V1 : **un run se continue sur le provider
qui l'a commencé** (les formes ne se traduisent pas entre elles).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ToolCall:
    """Un appel d'outil demandé par le modèle."""
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Turn:
    """Un tour de modèle. `raw_content` = la matière brute à réémettre telle quelle."""
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"       # end_turn | refusal | …
    raw_content: object = None
    usage: dict = field(default_factory=dict)
    # ⚠️ Ce que le FOURNISSEUR dit avoir servi, pas ce qu'on a demandé : un alias
    # flottant (`mistral-large-latest`) ne se date pas après coup. Sans cette
    # estampille, « quelles lignes viennent de quel modèle » n'a plus de réponse
    # dès que la question se pose à froid — et elle s'est posée le 02/09.
    model: Optional[str] = None
    # La température EFFECTIVEMENT envoyée au fournisseur sur ce tour — celle du
    # passage, ou celle de l'hôte à défaut, ou rien. Portée par le tour pour que
    # le journal la LISE au lieu de la déduire d'un `.env` relu après coup.
    temperature: Optional[float] = None
    # Un défaut de FORME constaté dans la réponse du fournisseur — aujourd'hui un
    # seul : un appel d'outil rendu en texte au lieu d'un `tool_calls`
    # (`{"forme": "reference+texte_json", "outil": <nom>}`). Décrit, jamais
    # exécuté ; `None` = tour bien formé.
    defaut: Optional[dict] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LlmUnavailable(RuntimeError):
    """Le substrat LLM n'est pas configuré (lib absente / clé absente)."""
