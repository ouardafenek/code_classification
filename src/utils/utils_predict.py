
import json
from pathlib import Path



# =====================
# Load the input sample 
# =====================
def load_input_text(json_path: Path) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    description = data.get("prob_desc_description", "")
    source_code = data.get("source_code", "")

    ground_tags = data.get("tags", [])

    if not description or not source_code:
        raise ValueError(
            "JSON must contain 'prob_desc_description' and 'source_code'"
        )

    # Appliquer les mêmes transformations que lors de l'entraînement pour optimiser les résultats
    cleaned_description = _clean_description(description)
    cleaned_code = _clean_code(source_code)


    return f"{cleaned_description}  {cleaned_code}", ground_tags

# =========================================================================================================
# The cleaning and preprocessing functions used to prepare training data (We are applying them during inference)
# =========================================================================================================

import re
from typing import Optional
import nltk
from nltk.corpus import stopwords

# --- Configuration ---
LATEX_PATTERN = r'\$\$?.*?\$\$?'
STOP_WORDS = set(stopwords.words('english'))

def _clean_description(text: Optional[str]) -> str:
    """
    Nettoie un texte :
    - supprime les expressions LaTeX
    - passe en minuscules
    - supprime les stop words anglais

    Args:
        text (Optional[str]): Texte brut

    Returns:
        str: Texte nettoyé
    """
    if not isinstance(text, str):
        return ""

    # Supprimer le LaTeX
    text = re.sub(LATEX_PATTERN, " ", text)

    # Lowercase
    text = text.lower()

    # Tokenisation simple + suppression stop words
    tokens = [
        token for token in text.split()
        if token not in STOP_WORDS
    ]

    return " ".join(tokens)


# Liste des mots-clés à python qui seraient communs dans tous les codes
PYTHON_STOP_WORDS = {
    'def', 'return', 'if', 'else', 'while', 'import', 'from',
    'print', 'continue', 'elif', 'sys', 'break', 'pass', 'for',
    'in', 'range'
}

def _clean_code(code_str: str) -> str:
    """
    Retire les mots-clés Python d'une chaîne de caractères.
    """
    if not isinstance(code_str, str):
        return ""

    # 1. Construire le pattern regex : \b(def|return|if|...)\b
    # \b signifie "boundary" (frontière de mot), pour ne pas couper "definition" en "inition"
    pattern = r'\b(' + '|'.join(re.escape(word) for word in PYTHON_STOP_WORDS) + r')\b'

    # 2. Remplacer par un espace
    cleaned_code = re.sub(pattern, ' ', code_str)
    # 3. Suppression des parenthèses
    cleaned_code = cleaned_code.replace('(', ' ').replace(')', ' ')
    # 4. Nettoyer les espaces multiples et sauts de ligne
    return " ".join(cleaned_code.split())



