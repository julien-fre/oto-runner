"""Le CONTENU d'un résultat d'outil MCP : du texte, et des images.

Un outil MCP rend une liste de blocs (`content`). Le runner n'en gardait que le
PREMIER bloc texte : un outil qui montre une image (`web_image`) arrivait au
modèle réduit à sa ligne de description, et l'agent croyait avoir vu l'image.

Ce module lit TOUS les blocs, et nomme ce qu'il ne sait pas transporter
(`BlocNonGere`) au lieu de l'ignorer. Il porte aussi les bornes qu'une image
rencontre sur le chemin du modèle, et les phrases qui disent au modèle ce qu'il
n'a PAS reçu — une image écartée en silence est exactement le défaut corrigé ici.

Les bornes viennent du contrat de l'API Anthropic (vision, octobre 2026) :
- formats acceptés : jpeg, png, gif, webp ;
- 10 Mo par image, en base64 (API directe) ;
- 32 Mo par requête, et le fil ENTIER se renvoie à chaque tour ;
- au-delà de 20 images dans une requête, une limite de dimension plus stricte
  (2000 px) s'applique à TOUTES, résultats d'outils compris : une capture banale
  ferait alors refuser le tour entier.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Les formats que l'API accepte dans un bloc image.
FORMATS_IMAGE = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
#: Une image, en caractères base64 (= octets de la requête).
IMAGE_MAX_B64 = 10_000_000
#: Ce qu'un déroulé fait entrer d'images dans son fil, au total. La moitié des
#: 32 Mo d'une requête : le reste du fil (texte, schémas) doit encore tenir.
DEROULE_MAX_B64 = 16_000_000
#: Le nombre d'images d'un déroulé — au-delà de 20, la limite des requêtes
#: « multi-images » rejetterait des images ordinaires.
DEROULE_MAX_IMAGES = 20


class BlocNonGere(RuntimeError):
    """Un bloc de résultat d'outil que le runner ne sait pas transporter au modèle
    (audio, ressource, image d'un format refusé par l'API…)."""


@dataclass(frozen=True)
class Image:
    """Une image rendue par un outil, en base64, telle que MCP la porte."""
    media_type: str
    data: str

    @property
    def taille(self) -> int:
        return len(self.data)

    def decrire(self) -> str:
        return f"{self.media_type}, {self.taille} caractères base64"


def lire(blocs) -> tuple[str, tuple[Image, ...]]:
    """Les blocs `content` d'un résultat MCP → (texte, images).

    Les blocs texte se joignent dans l'ordre ; les images se gardent à part, la
    forme où elles entrent dans le fil appartenant au provider. Tout autre bloc
    lève `BlocNonGere` : le modèle lit alors l'erreur, au lieu d'un résultat
    amputé qu'il prendrait pour complet."""
    textes: list[str] = []
    images: list[Image] = []
    for bloc in blocs or []:
        sorte = bloc.get("type") if isinstance(bloc, dict) else None
        if sorte == "text":
            textes.append(bloc.get("text") or "")
        elif sorte == "image":
            media_type, data = bloc.get("mimeType"), bloc.get("data")
            if media_type not in FORMATS_IMAGE:
                raise BlocNonGere(
                    f"image au format `{media_type}` : le modèle n'accepte que "
                    f"{', '.join(sorted(FORMATS_IMAGE))}")
            if not isinstance(data, str) or not data:
                raise BlocNonGere(f"image `{media_type}` sans données base64")
            images.append(Image(media_type, data))
        else:
            raise BlocNonGere(
                f"bloc de résultat `{sorte}` : le runner ne transporte au modèle "
                "que du texte et des images")
    return "\n".join(textes), tuple(images)


def admettre(images: tuple[Image, ...], deja_n: int,
             deja_b64: int) -> tuple[tuple[Image, ...], list[str]]:
    """Les images qui entrent dans le fil du déroulé, et une phrase par image
    écartée — jamais un écart muet. `deja_*` = ce que le déroulé a déjà admis."""
    admises: list[Image] = []
    refus: list[str] = []
    for image in images:
        if image.taille > IMAGE_MAX_B64:
            refus.append(f"[Image NON transmise ({image.decrire()}) : au-delà des "
                         f"{IMAGE_MAX_B64} caractères base64 qu'accepte le modèle.]")
        elif (deja_n + len(admises) >= DEROULE_MAX_IMAGES
              or deja_b64 + sum(i.taille for i in admises) + image.taille > DEROULE_MAX_B64):
            refus.append(f"[Image NON transmise ({image.decrire()}) : ce déroulé a "
                         f"atteint son plafond de {DEROULE_MAX_IMAGES} images ou "
                         f"{DEROULE_MAX_B64} caractères base64. Ne conclus rien de "
                         "son contenu.]")
        else:
            admises.append(image)
    return tuple(admises), refus


def non_recues(images: tuple[Image, ...], motif: str) -> str:
    """La phrase qui dit au modèle qu'un résultat portait des images qu'il ne reçoit pas."""
    return (f"[Ce résultat contient {len(images)} image(s) "
            f"({'; '.join(i.decrire() for i in images)}) que tu ne reçois pas : "
            f"{motif}. Ne conclus rien de leur contenu.]")


def substitut(media_type: str, taille: int) -> dict:
    """Le bloc texte qui remplace une image dans le fil PERSISTÉ : le backend
    plafonne un tour à 256 000 caractères, qu'une seule image dépasse. Le modèle
    d'une reprise lit donc qu'une image était là, et qu'il ne la voit plus."""
    return {"type": "text",
            "text": f"[Image {media_type} ({taille} caractères base64) vue par le "
                    "modèle à ce tour, non conservée dans le fil persisté.]"}
