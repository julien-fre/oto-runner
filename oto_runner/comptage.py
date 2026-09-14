"""Le COMPTAGE des usages d'un déroulé : ce qui est déclaré, et ce qui ne l'est pas.

Un usage que le fournisseur ne déclare pas n'est pas zéro. Il devenait 0 à chaque
étage — extraction du fournisseur, cumul de la boucle, résultat déclaré, bilans —
si bien qu'un tour muet coûtait « 0 jeton » partout où on le lisait (constat du
13/09/2026 ; cf. « trois états, jamais un zéro qui ressemble à un succès »).

Trois états par poste, jamais deux :
- **déclaré à chaque tour** : la somme ;
- **déclaré zéro** : zéro — posé par le fournisseur lui-même, ou par son contrat
  documenté quand une PHRASE définit l'absence comme zéro (une annotation de schéma
  `default: 0` n'y suffit pas, arbitrage du 13/09/2026). C'est le transport du
  fournisseur qui le pose, jamais ce module ;
- **non déclaré sur au moins un tour** : `None`, et la couverture dit combien de
  tours l'ont déclaré et ce qu'ils ont déclaré.

⚠️ Deux entrées, jamais confondues. `input_tokens` est le NON CACHÉ exact : il n'est
posé que si le cache lu est connu. `input_total_tokens` est l'entrée totale DÉCLARÉE,
cache compris. Quand le cache est inconnu, seule la seconde est connue : elle majore
le non-caché, et une borne peut la compter — elle ne se publie jamais comme un
montant facturé exact.

Un poste manquant n'efface pas les autres : des caches inconnus laissent l'entrée
totale et la sortie connues quand elles sont déclarées.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

POSTES = ("input_tokens", "input_total_tokens", "output_tokens",
          "cache_creation_input_tokens", "cache_read_input_tokens")


def _zeros() -> dict:
    return dict.fromkeys(POSTES, 0)


def _pour_la_borne(entree_non_cachee, entree_totale, sortie, ecriture_cache) -> Optional[int]:
    """Ce qu'une borne de jetons compte pour un tour, ou `None` si elle ne peut plus
    le suivre. L'entrée NON cachée si elle est connue, sinon l'entrée totale (un
    majorant) ; la sortie ; l'écriture de cache si elle est connue. Les jetons LUS
    en cache n'y entrent pas quand on sait les séparer : ils coûtent une fraction du
    tarif, et les compter couperait le passage bien caché qu'on ne veut pas couper."""
    entree = entree_non_cachee if entree_non_cachee is not None else entree_totale
    if entree is None or sortie is None:
        return None
    return int(entree) + int(sortie) + int(ecriture_cache or 0)


def jetons_du_budget(resultat: dict) -> Optional[int]:
    """Ce qu'un budget de FLOTTE compte pour un travail conclu : `usage_tokens` (entrée
    non cachée + sortie) quand il est connu ; sinon l'entrée TOTALE déclarée + la
    sortie, qui le majorent ; sinon `None` — et le budget ne se suit plus. Un
    majorant sert à borner, jamais à publier un montant exact."""
    if resultat.get("usage_tokens") is not None:
        return int(resultat["usage_tokens"])
    total, sortie = resultat.get("usage_input_total"), resultat.get("usage_output")
    return None if total is None or sortie is None else int(total) + int(sortie)


@dataclass
class Compteur:
    """Les tours d'un déroulé, et pour chaque poste : combien l'ont déclaré, et quoi."""
    tours: int = 0
    declares: dict = field(default_factory=_zeros)
    sommes: dict = field(default_factory=_zeros)
    # Ce que la borne a compté sur les tours qu'elle a pu suivre (cf. `_pour_la_borne`).
    borne: int = 0

    def ajouter(self, usage: Optional[dict]) -> list:
        """Compte UN tour ; rend ce qui manque à la borne pour le suivre (vide = suivi)."""
        usage = usage or {}
        self.tours += 1
        for poste in POSTES:
            if usage.get(poste) is not None:
                self.declares[poste] += 1
                self.sommes[poste] += int(usage[poste])
        compte = _pour_la_borne(usage.get("input_tokens"), usage.get("input_total_tokens"),
                                usage.get("output_tokens"), usage.get("cache_creation_input_tokens"))
        if compte is not None:
            self.borne += compte
            return []
        manque = []
        if usage.get("input_tokens") is None and usage.get("input_total_tokens") is None:
            manque.append("entrée")
        if usage.get("output_tokens") is None:
            manque.append("sortie")
        return manque

    def usage(self) -> dict:
        """Par poste : la somme si CHAQUE tour l'a déclaré, sinon `None`."""
        return {poste: (self.sommes[poste]
                        if self.tours and self.declares[poste] == self.tours else None)
                for poste in POSTES}

    def couverture(self) -> dict:
        """Ce qui fonde `usage()` : les tours, et par poste les déclarants et leur somme."""
        return {"tours": self.tours, "declares": dict(self.declares),
                "sommes": dict(self.sommes)}

    @classmethod
    def depuis(cls, couverture: Optional[dict]) -> "Compteur":
        """Le compteur qu'une couverture décrit — pour cumuler des passes déjà closes."""
        c = couverture or {}
        return cls(tours=int(c.get("tours") or 0),
                   declares={p: int((c.get("declares") or {}).get(p) or 0) for p in POSTES},
                   sommes={p: int((c.get("sommes") or {}).get(p) or 0) for p in POSTES})

    def fusion(self, autre: "Compteur") -> "Compteur":
        """Deux passes d'un même travail : leurs tours, déclarants et sommes s'ajoutent."""
        return Compteur(tours=self.tours + autre.tours,
                        declares={p: self.declares[p] + autre.declares[p] for p in POSTES},
                        sommes={p: self.sommes[p] + autre.sommes[p] for p in POSTES},
                        borne=self.borne + autre.borne)
