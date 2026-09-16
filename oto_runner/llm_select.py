"""Le choix du provider — une variable d'env, une erreur franche.

`OTO_RUNNER_PROVIDER` ∈ {anthropic, openai} (défaut anthropic). Conséquence de
la forme de fil opaque (llm_types) : **un run se continue sur le provider qui
l'a commencé** — un worker `openai` qui reprendrait un fil Anthropic rejouerait
des blocs que l'API ne comprend pas. V1 : un worker = un provider, et les
flottes mixtes attendent que le fil porte son provider.
"""
from __future__ import annotations

import os


def get_provider():
    nom = (os.environ.get("OTO_RUNNER_PROVIDER") or "anthropic").strip().lower()
    if nom == "anthropic":
        from . import agent_llm
        return agent_llm
    if nom == "openai":
        from . import agent_llm_openai
        return agent_llm_openai
    if nom == "conversations":
        from . import agent_conversations
        return agent_conversations
    if nom == "claude-code":
        from . import agent_claude_code
        return agent_claude_code
    raise SystemExit(
        f"OTO_RUNNER_PROVIDER inconnu : `{nom}` "
        "(anthropic | openai | conversations | claude-code)")
