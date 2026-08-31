"""
Text cleaning and normalization utilities for Arabic survey inputs and numerical fields.
"""
import re
import numpy as np


def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic text by unifying letter variants, removing diacritics,
    converting Arabic numerals to English digits, and cleaning punctuation.
    """
    if text is None:
        return ""
    s = str(text).strip().lower()

    # Remove emojis and non-alphanumeric/non-Arabic noise punctuation
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)

    # Unify common Arabic letter variants
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")

    # Arabic to Western digits
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    for i, d in enumerate(arabic_digits):
        s = s.replace(d, str(i))

    # Remove tashkeel (diacritics)
    s = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", s)

    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_gpa(gpa):
    """
    Cleans raw Tawjihi GPA values, handles Arabic numerals, commas, decimal formatting,
    and removes invalid range outliers (< 60 or > 100).
    """
    if isinstance(gpa, str):
        gpa = gpa.strip()
        # Convert Arabic numerals to English numerals
        gpa = re.sub(r'[٠-٩]', lambda x: str(ord(x.group(0)) - ord('٠')), gpa)
        gpa = re.sub(r'[٫,]', '.', gpa)
        gpa = re.sub(r'[^\d.]', '', gpa)
        full, frac = gpa.split('.', maxsplit=1) if '.' in gpa else (gpa, '0')
        if len(frac) > 2:
            frac = frac[0]
        gpa = f"{full}.{frac}"
        try:
            gpa = float(gpa)
        except ValueError:
            return np.nan

    try:
        gpa = float(gpa)
    except (ValueError, TypeError):
        return np.nan

    # Validate reasonable Tawjihi GPA range (60.0 to 100.0)
    if gpa < 60.0 or gpa > 100.0:
        return np.nan

    return gpa


def is_garbage(s: str) -> bool:
    """
    Checks if a normalized raw string represents non-useful or invalid text responses.
    """
    if not s:
        return True
    s = s.strip().lower()
    garbage_tokens = {
        "nan", "لا يوجد", "لسا", "مافي", "فصل اول", "م درست جامعه",
        "لم انقبل مع الاسف", "٣.٨٠", "٣.٦٥", "٧٤.٤", "٣٨٠", "3025",
        "85", "٠", "مستقبل", "بالنسبه"
    }
    if s in garbage_tokens:
        return True
    if re.fullmatch(r"[\d\.,\s]+", s):
        return True
    return False
