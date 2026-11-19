"""La deadline HORLOGE-MURALE des requêtes sortantes — le seul timeout qui coupe.

Le read timeout d'urllib3 se RÉARME à chaque octet reçu, et un handshake SSL
peut pendre au-delà du connect timeout : un serveur qui goutte tient la
connexion indéfiniment (vécu deux fois en une nuit — un tour de modèle figé
35 min, puis deux workers pendus en SSL sur d'autres cibles pendant une
heure). SIGALRM (les workers sont mono-thread) : au-delà du plafond, une
exception NETTE — le retry de job et la reprise du fil font le reste.
"""
from __future__ import annotations

import signal

import requests

_DEFAULT_WALL_S = 420


class DeadlineExceeded(RuntimeError):
    pass


def post_with_deadline(url: str, *, wall_s: int = _DEFAULT_WALL_S, **kwargs):
    return _with_deadline(requests.post, url, wall_s=wall_s, **kwargs)


def get_with_deadline(url: str, *, wall_s: int = _DEFAULT_WALL_S, **kwargs):
    return _with_deadline(requests.get, url, wall_s=wall_s, **kwargs)


def delete_with_deadline(url: str, *, wall_s: int = _DEFAULT_WALL_S, **kwargs):
    return _with_deadline(requests.delete, url, wall_s=wall_s, **kwargs)


def _with_deadline(fn, url, *, wall_s, **kwargs):
    def _coupe(signum, frame):
        raise DeadlineExceeded(
            f"requête > {wall_s}s wall-clock vers {url.split('?')[0][:80]} — "
            "le serveur goutte sans conclure")

    ancien = signal.signal(signal.SIGALRM, _coupe)
    signal.alarm(wall_s)
    try:
        return fn(url, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, ancien)
