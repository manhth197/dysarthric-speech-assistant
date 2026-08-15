import os
import random
import re
import yaml
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import logging
from torch.utils.data import Dataset, DataLoader

from .utils import normalize_vietnamese_text

logger = logging.getLogger(__name__)


# ============================================================
# PHẦN 0: TỪ ĐIỂN ÂM VỊ TIẾNG VIỆT
# ============================================================

INITIAL_CONSONANTS = sorted([
    "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh",
    "l", "m", "n", "ng", "ngh", "nh", "ph", "qu", "r", "s",
    "t", "th", "tr", "v", "x"
], key=len, reverse=True)

FINAL_CONSONANTS = sorted([
    "c", "ch", "m", "n", "ng", "nh", "p", "t"
], key=len, reverse=True)

ALL_VOWELS = {
    'a', 'e', 'i', 'o', 'u', 'y', 'ă', 'â', 'ê', 'ô', 'ơ', 'ư',
    'à', 'á', 'ạ', 'ả', 'ã',
    'ằ', 'ắ', 'ặ', 'ẳ', 'ẵ',
    'ầ', 'ấ', 'ậ', 'ẩ', 'ẫ',
    'è', 'é', 'ẹ', 'ẻ', 'ẽ',
    'ề', 'ế', 'ệ', 'ể', 'ễ',
    'ì', 'í', 'ị', 'ỉ', 'ĩ',
    'ò', 'ó', 'ọ', 'ỏ', 'õ',
    'ồ', 'ố', 'ộ', 'ổ', 'ỗ',
    'ờ', 'ớ', 'ợ', 'ở', 'ỡ',
    'ù', 'ú', 'ụ', 'ủ', 'ũ',
    'ừ', 'ứ', 'ự', 'ử', 'ữ',
    'ỳ', 'ý', 'ỵ', 'ỷ', 'ỹ',
}


# ============================================================
# PHẦN 1: MA TRẬN NHẦM LẪN
# ============================================================

INITIAL_CONFUSION_MAP = {
    "b":   ["m", "p", "đ", "v", "w"], 
    "m":   ["b", "p", "n"],
    "ph":  ["h", "v", "b", "p", "f"],
    "v":   ["b", "ph", "d", "th", "w"],
    "t":   ["đ", "c", "k", "th", "n", "s"],
    "đ":   ["d", "n", "t", "l", "z"],
    "n":   ["l", "đ", "nh", "t", "m"],
    "l":   ["n", "h", "đ", "d", "r"],
    "d":   ["gi", "v", "r", "l", "n", "z"],
    "th":  ["h", "t", "v", "ph", "s"],
    "s":   ["x", "th", "t", "h"],
    "x":   ["s", "t", "h"],
    "tr":  ["t", "ch", "gi", "d", "l"],
    "ch":  ["tr", "t", "x", "s"],
    "r":   ["d", "gi", "g", "l", "h"],
    "gi":  ["d", "r", "i", "z"],
    "k":   ["t", "c", "g", "kh", "h"],
    "c":   ["t", "k", "g", "h"],
    "g":   ["h", "k", "d", "ng", "kh"],
    "gh":  ["g", "h", "kh"],
    "kh":  ["h", "k", "g", "th"],
    "qu":  ["c", "k", "kh", "h", "w", "u"],
    "ng":  ["n", "m", "g", "nh"],
    "ngh": ["n", "nh", "ng", "h"],
    "nh":  ["n", "ng", "l", "m"],
    "h":   ["", "l", "g", "kh"],
}

FINAL_CONFUSION_MAP = {
    "n":  ["ng", "m", "nh", "t", ""],
    "ng": ["n", "nh", "m", "c", ""],
    "nh": ["n", "ng", "ch", ""],
    "m":  ["n", "p", "ng", ""],
    "t":  ["c", "p", "n", ""],
    "c":  ["t", "p", "ng", ""],
    "ch": ["t", "c", "nh", "n", ""],
    "p":  ["m", "t", "c", ""],
}

VOWEL_CONFUSION_MAP = {
    "a":  ["ă", "â", "ơ", "e"],
    "ă":  ["a", "â", "e"],
    "â":  ["a", "ơ", "ă", "u"],
    "e":  ["ê", "a", "i"],
    "ê":  ["e", "i", "ơ", "â"],
    "i":  ["ê", "y", "e", "ư"],
    "y":  ["i", "ê"],
    "o":  ["ô", "a", "u", "ơ"],
    "ô":  ["o", "u", "ơ"],
    "ơ":  ["a", "â", "ư", "ô"],
    "u":  ["ư", "o", "ô"],
    "ư":  ["u", "ơ", "i"],
    "ươ": ["ưa", "ơ", "ua", "ư"],
    "ưa": ["ươ", "a", "ua", "ơ"],
    "uô": ["ua", "ô", "ươ", "o"],
    "ua": ["uô", "a", "ưa", "o"],
    "iê": ["ia", "ê", "yê", "i"],
    "ia": ["iê", "a", "ê"],
    "yê": ["iê", "ê", "ia", "i"],
    "uơ": ["ươ", "ơ"],
    "uy": ["i", "ui", "y"],
    "ui": ["uy", "ôi", "i"],
    "ôi": ["ơi", "ui", "oi", "ô"],
    "ơi": ["ôi", "ai", "ơ"],
    "ai": ["ay", "ơi", "a"],
    "ay": ["ai", "ây", "a"],
    "ây": ["ay", "âu", "â"],
    "ao": ["au", "a", "o"],
    "au": ["ao", "âu", "a"],
    "âu": ["au", "ơu", "â"],
    "oi": ["ôi", "o"],
    "ưu": ["iu", "ư"],
    "iu": ["ưu", "iêu", "i"],
}

TONE_GROUPS = {
    "a": ["à", "á", "ạ", "ả", "ã"],
    "ă": ["ằ", "ắ", "ặ", "ẳ", "ẵ"],
    "â": ["ầ", "ấ", "ậ", "ẩ", "ẫ"],
    "e": ["è", "é", "ẹ", "ẻ", "ẽ"],
    "ê": ["ề", "ế", "ệ", "ể", "ễ"],
    "i": ["ì", "í", "ị", "ỉ", "ĩ"],
    "o": ["ò", "ó", "ọ", "ỏ", "õ"],
    "ô": ["ồ", "ố", "ộ", "ổ", "ỗ"],
    "ơ": ["ờ", "ớ", "ợ", "ở", "ỡ"],
    "u": ["ù", "ú", "ụ", "ủ", "ũ"],
    "ư": ["ừ", "ứ", "ự", "ử", "ữ"],
    "y": ["ỳ", "ý", "ỵ", "ỷ", "ỹ"],
}

TONE_REMOVE_MAP = {
    "à":"a","á":"a","ạ":"a","ả":"a","ã":"a",
    "ằ":"ă","ắ":"ă","ặ":"ă","ẳ":"ă","ẵ":"ă",
    "ầ":"â","ấ":"â","ậ":"â","ẩ":"â","ẫ":"â",
    "è":"e","é":"e","ẹ":"e","ẻ":"e","ẽ":"e",
    "ề":"ê","ế":"ê","ệ":"ê","ể":"ê","ễ":"ê",
    "ì":"i","í":"i","ị":"i","ỉ":"i","ĩ":"i",
    "ò":"o","ó":"o","ọ":"o","ỏ":"o","õ":"o",
    "ồ":"ô","ố":"ô","ộ":"ô","ổ":"ô","ỗ":"ô",
    "ờ":"ơ","ớ":"ơ","ợ":"ơ","ở":"ơ","ỡ":"ơ",
    "ù":"u","ú":"u","ụ":"u","ủ":"u","ũ":"u",
    "ừ":"ư","ứ":"ư","ự":"ư","ử":"ư","ữ":"ư",
    "ỳ":"y","ý":"y","ỵ":"y","ỷ":"y","ỹ":"y",
}

CHAR_TO_BASE = {}
for _base, _variants in TONE_GROUPS.items():
    for _v in _variants:
        CHAR_TO_BASE[_v] = _base
    CHAR_TO_BASE[_base] = _base

HOMOPHONE_MAP = {
    "bạn": ["bàn", "bán", "bản"],
    "bàn": ["bạn", "bán", "bản"],
    "con": ["cơn", "còn", "cón"],
    "còn": ["con", "cơn"],
    "được": ["đước", "dược"],
    "muốn": ["muôn", "muộn"],
    "muộn": ["muốn", "muôn"],
    "không": ["khong", "khống"],
    "ăn":  ["ân", "anh"],
    "anh": ["ăn", "an"],
    "an":  ["anh", "ăn"],
    "nói": ["nỏi", "nòi"],
    "nước": ["nướt", "nức"],
    "người": ["ngươi", "ngừi"],
    "đi":  ["đì", "di"],
    "về":  ["vè", "vê"],
    "với": ["vợi", "vơi", "vời"],
    "cho": ["chò", "chô"],
    "này": ["nay", "nài"],
    "nay": ["này", "nài"],
    "có":  ["cỏ", "cò", "co"],
    "là":  ["la", "lã", "lả"],
    "và":  ["va", "vả", "vã"],
    "một": ["mốt", "mọt", "môt"],
    "hai": ["hay", "hài"],
    "hay": ["hai", "hài"],
    "tôi": ["tối", "tơi", "tòi"],
    "thì": ["thị", "thi"],
    "mà":  ["ma", "mả", "mã"],
    "rồi": ["rỗi", "rời", "dồi"],
    "đây": ["đay", "đấy"],
    "đó":  ["dó", "đố"],
    "tại": ["tải", "tai"],
    "sao": ["sau", "xao"],
    "sau": ["sao", "xau"],
    "cũng": ["cúng", "cùng"],
    "cùng": ["cũng", "cúng"],
    "đến": ["đền", "đên"],
    "nhà": ["nha", "nhả", "nhã"],
    "bao": ["bau", "báo"],
    "báo": ["bao", "bào"],
    "mình": ["minh", "mỉnh"],
    "biết": ["biệt", "biêt"],
    "thấy": ["thầy", "thay"],
    "thầy": ["thấy", "thay"],
    "thay": ["thầy", "thấy"],
    "như": ["nhu", "nhừ"],
    "lại": ["lai", "lải", "lãi"],
    "vào": ["vao", "váo"],
    "ra":  ["da", "rà", "rả"],
    "lên": ["lênh", "len"],
    "xuống": ["suống", "xuông"],
    "trên": ["trênh", "tren"],
    "dưới": ["dười", "gưới"],
    "trong": ["tron", "chong"],
    "ngoài": ["ngoai", "ngòai"],
    "giờ": ["dờ", "gờ"],
    "phải": ["phái", "fải"],
    "cần": ["cân", "cầng"],
    "rất":  ["rấc", "rắt", "dất"],
    "nhiều": ["nhều", "nhiêu"],
    "đang": ["đan", "đàng"],
    "sẽ":  ["sẻ", "xẽ"],
}

HALLUCINATION_INSERTS = [
    "là", "và", "của", "có", "được", "cho", "với",
    "một", "các", "những", "này", "đó", "thì", "mà",
    "cũng", "để", "trong", "về", "ra", "lên",
    "đã", "sẽ", "đang", "rồi", "vẫn", "còn",
    "nhưng", "nếu", "khi", "vì",
]


# ============================================================
# PHẦN 2: HÀM TÁCH ÂM TIẾT
# ============================================================

def split_syllable(word: str):
    """Tách âm tiết → (phụ_âm_đầu, nguyên_âm, phụ_âm_cuối)"""
    w = word.lower()
    initial = ""
    final = ""

    for c in INITIAL_CONSONANTS:
        if w.startswith(c):
            initial = c
            w = w[len(c):]
            break

    for c in FINAL_CONSONANTS:
        if w.endswith(c):
            remaining = w[:-len(c)]
            if remaining:
                final = c
                w = remaining
                break

    return initial, w, final


# ============================================================
# PHẦN 3: HELPER — giữ thanh điệu khi thay nguyên âm
# ============================================================

def _replace_vowel_keep_tone(old_char: str, new_base: str) -> str:
    """Thay nguyên âm nhưng giữ thanh điệu gốc."""
    base = CHAR_TO_BASE.get(old_char, old_char)
    if old_char not in TONE_REMOVE_MAP:
        return new_base
    if base not in TONE_GROUPS or old_char not in TONE_GROUPS[base]:
        return new_base
    tone_idx = TONE_GROUPS[base].index(old_char)
    if new_base in TONE_GROUPS and tone_idx < len(TONE_GROUPS[new_base]):
        return TONE_GROUPS[new_base][tone_idx]
    return new_base


def _strip_tone_from_vowel_part(vowel_part: str) -> str:
    """Bóc toàn bộ thanh điệu khỏi chuỗi nguyên âm → base form."""
    return "".join(TONE_REMOVE_MAP.get(c, c) for c in vowel_part)


def _transfer_tone_pattern(old_vowel: str, new_vowel_base: str) -> str:
    """
    Chuyển thanh điệu từ chuỗi nguyên âm cũ sang chuỗi mới.
    Tìm ký tự mang dấu trong old_vowel, áp tone đó vào ký tự
    tương ứng (vị trí gần nhất) trong new_vowel_base.
    """
    # Tìm tone_idx và vị trí mang dấu trong old_vowel
    tone_idx = None
    tone_position = None
    for i, ch in enumerate(old_vowel):
        if ch in TONE_REMOVE_MAP:
            base = CHAR_TO_BASE[ch]
            if base in TONE_GROUPS and ch in TONE_GROUPS[base]:
                tone_idx = TONE_GROUPS[base].index(ch)
                tone_position = i
                break

    if tone_idx is None:
        return new_vowel_base  # old_vowel không có dấu

    # Áp tone vào ký tự có thể mang dấu trong new_vowel_base
    # Ưu tiên vị trí tương ứng, nếu không thì ký tự cuối cùng có thể mang dấu
    new_chars = list(new_vowel_base)
    applied = False

    # Thử áp vào vị trí tương ứng
    if tone_position is not None and tone_position < len(new_chars):
        ch = new_chars[tone_position]
        base = CHAR_TO_BASE.get(ch, ch)
        if base in TONE_GROUPS and tone_idx < len(TONE_GROUPS[base]):
            new_chars[tone_position] = TONE_GROUPS[base][tone_idx]
            applied = True

    # Fallback: áp vào ký tự cuối có thể mang dấu
    if not applied:
        for j in range(len(new_chars) - 1, -1, -1):
            ch = new_chars[j]
            base = CHAR_TO_BASE.get(ch, ch)
            if base in TONE_GROUPS and tone_idx < len(TONE_GROUPS[base]):
                new_chars[j] = TONE_GROUPS[base][tone_idx]
                break

    return "".join(new_chars)


# ============================================================
# PHẦN 4A: CÁC HÀM TẠO NHIỄU CHUNG (SHARED)
# ============================================================

def drop_initial(word: str) -> str:
    w_lower = word.lower()
    for c in INITIAL_CONSONANTS:
        if w_lower.startswith(c):
            result = word[len(c):]
            return result if result else word
    return word


def drop_final(word: str) -> str:
    w_lower = word.lower()
    for c in FINAL_CONSONANTS:
        if w_lower.endswith(c):
            remaining = word[:-len(c)]
            if remaining:
                return remaining
    return word


def drop_both(word: str) -> str:
    result = drop_initial(word)
    result = drop_final(result)
    return result if result.strip() else word


def confuse_initial(word: str) -> str:
    w_lower = word.lower()
    for c in sorted(INITIAL_CONFUSION_MAP.keys(), key=len, reverse=True):
        if w_lower.startswith(c):
            replacement = random.choice(INITIAL_CONFUSION_MAP[c])
            return replacement + word[len(c):]
    return word


def confuse_final(word: str) -> str:
    w_lower = word.lower()
    for c in sorted(FINAL_CONFUSION_MAP.keys(), key=len, reverse=True):
        if w_lower.endswith(c):
            replacement = random.choice(FINAL_CONFUSION_MAP[c])
            return word[:-len(c)] + replacement
    return word


def confuse_vowel(word: str) -> str:
    """Nhầm nguyên âm — ưu tiên match đôi/ba âm trước."""
    initial, vowel_part, final = split_syllable(word)
    if not vowel_part:
        return word

    vowel_base = _strip_tone_from_vowel_part(vowel_part)

    for vow in sorted(VOWEL_CONFUSION_MAP.keys(), key=len, reverse=True):
        if vow in vowel_base:
            replacement = random.choice(VOWEL_CONFUSION_MAP[vow])
            new_vowel_base = vowel_base.replace(vow, replacement, 1)
            new_vowel = _transfer_tone_pattern(vowel_part, new_vowel_base)
            return initial + new_vowel + final

    # Fallback: thay nguyên âm đơn, giữ thanh điệu
    new_vowel = []
    changed = False
    for ch in vowel_part:
        base = CHAR_TO_BASE.get(ch)
        if base and base in VOWEL_CONFUSION_MAP and not changed:
            new_base = random.choice(VOWEL_CONFUSION_MAP[base])
            new_vowel.append(_replace_vowel_keep_tone(ch, new_base))
            changed = True
        else:
            new_vowel.append(ch)
    return initial + "".join(new_vowel) + final


def remove_tone(word: str) -> str:
    return "".join(TONE_REMOVE_MAP.get(c, c) for c in word)


def wrong_tone(word: str) -> str:
    chars = list(word)
    tone_positions = [(i, ch) for i, ch in enumerate(chars)
                      if ch in CHAR_TO_BASE]
    if not tone_positions:
        return word
    idx, ch = random.choice(tone_positions)
    base = CHAR_TO_BASE[ch]
    if base in TONE_GROUPS:
        all_tones = TONE_GROUPS[base] + [base]
        candidates = [t for t in all_tones if t != ch]
        if candidates:
            chars[idx] = random.choice(candidates)
    return "".join(chars)


def nasalize_stop(word: str) -> str:
    nasal_map = {
        "b": "m", "đ": "n", "d": "n",
        "g": "ng", "k": "ng", "c": "ng", "t": "n",
    }
    w_lower = word.lower()
    for stop, nasal in sorted(nasal_map.items(),
                               key=lambda x: len(x[0]), reverse=True):
        if w_lower.startswith(stop):
            return nasal + word[len(stop):]
    return word


def weaken_consonant(word: str) -> str:
    weaken_map = {
        "b": "w", "đ": "z", "d": "z",
        "g": "h", "k": "h", "c": "h",
        "t": "h", "p": "h",
    }
    w_lower = word.lower()
    for strong, weak in sorted(weaken_map.items(),
                                key=lambda x: len(x[0]), reverse=True):
        if w_lower.startswith(strong):
            return weak + word[len(strong):]
    return word


def repeat_syllable(word: str) -> str:
    if len(word) <= 1:
        return word
    stutter_len = random.randint(1, min(2, len(word) - 1))
    stutter = word[:stutter_len]
    return stutter + " " + word


def insert_filler(word: str) -> str:
    fillers = ["ừ", "à", "ờ", "uh", "ơ", "a", "hừ", "ừm"]
    filler = random.choice(fillers)
    if random.random() < 0.5:
        return filler + " " + word
    return word + " " + filler


def reduce_to_vowel(word: str) -> str:
    _, vowel, _ = split_syllable(word)
    return vowel if vowel else word


def slow_stretch(word: str) -> str:
    chars = list(word)
    vowel_indices = [i for i, c in enumerate(chars) if c in ALL_VOWELS]
    if vowel_indices:
        idx = random.choice(vowel_indices)
        n_repeat = random.randint(1, 2)
        for _ in range(n_repeat):
            chars.insert(idx + 1, chars[idx])
    return "".join(chars)


def homophone_replace(word: str) -> str:
    w_lower = word.lower()
    if w_lower in HOMOPHONE_MAP:
        return random.choice(HOMOPHONE_MAP[w_lower])
    return word


def truncate_syllable(word: str) -> str:
    initial, vowel, final = split_syllable(word)
    if not vowel:
        return word
    if len(vowel) > 1:
        vowel = vowel[0]
    return initial + vowel


# ============================================================
# PHẦN 4B: SPASTIC-SPECIFIC
# ============================================================

# ── Bảng centralize cho diphthong/triphthong ──
_CENTRALIZE_DIPHTHONG_MAP = {
    "iê": "ơ",   "ia": "ơ",   "yê": "ơ",
    "uô": "â",   "ua": "â",
    "ươ": "ơ",   "ưa": "ơ",
    "uy": "ơ",   "ui": "ơ",
    "ôi": "âi",  "ơi": "âi",
    "ai": "ơi",  "ay": "ơi",  "ây": "âi",
    "ao": "â",   "au": "â",   "âu": "â",
    "oi": "âi",
    "ưu": "ơ",   "iu": "ơ",
}

# ── Bảng centralize cho nguyên âm đơn ──
_CENTRALIZE_MONOPHTHONG_MAP = {
    "i": "ơ", "ư": "ơ", "u": "ơ",
    "ê": "â", "ô": "â",
    "e": "ơ", "o": "â",
}


def centralize_vowel(word: str) -> str:
    """
    Trung hòa hóa nguyên âm — spastic dysarthria.

    ✅ FIX #6: Xử lý diphthong/triphthong trước, rồi mới fallback
    sang nguyên âm đơn. Tránh sinh ra tổ hợp "ngoài hành tinh"
    như "ơê", "âư" mà Whisper không bao giờ nhả ra.
    """
    initial, vowel_part, final = split_syllable(word)
    if not vowel_part:
        return word

    vowel_base = _strip_tone_from_vowel_part(vowel_part)

    # Bước 1: Thử match cụm nguyên âm đôi/ba (dài → ngắn)
    for diph in sorted(_CENTRALIZE_DIPHTHONG_MAP.keys(),
                        key=len, reverse=True):
        if diph in vowel_base:
            replacement = _CENTRALIZE_DIPHTHONG_MAP[diph]
            new_vowel_base = vowel_base.replace(diph, replacement, 1)
            new_vowel = _transfer_tone_pattern(vowel_part, new_vowel_base)
            return initial + new_vowel + final

    # Bước 2: Fallback — nguyên âm đơn
    for ch_base in _CENTRALIZE_MONOPHTHONG_MAP:
        if ch_base in vowel_base:
            replacement = _CENTRALIZE_MONOPHTHONG_MAP[ch_base]
            new_vowel_base = vowel_base.replace(ch_base, replacement, 1)
            new_vowel = _transfer_tone_pattern(vowel_part, new_vowel_base)
            return initial + new_vowel + final

    return word


def monophthongize(word: str) -> str:
    """Đơn âm hóa đôi/ba âm — spastic dysarthria."""
    monophthong_map = {
        "iê": "i", "ia": "i", "yê": "i",
        "uô": "ô", "ua": "a",
        "ươ": "ư", "ưa": "ư",
        "uy": "i", "ui": "i",
        "ôi": "ô", "ơi": "ơ",
        "ai": "a", "ay": "a", "ây": "â",
        "ao": "a", "au": "a", "âu": "â",
    }
    initial, vowel_part, final = split_syllable(word)
    vowel_base = _strip_tone_from_vowel_part(vowel_part)

    for diph, mono in sorted(monophthong_map.items(), key=lambda x: len(x[0]),
                              reverse=True):
        if diph in vowel_base:
            new_vowel_base = vowel_base.replace(diph, mono, 1)
            new_vowel = _transfer_tone_pattern(vowel_part, new_vowel_base)
            return initial + new_vowel + final
    return word


def clinical_tone_error(word: str) -> str:
    """
    Lỗi thanh điệu kiểu lâm sàng — spastic dysarthria.
    ✅ FIX #3: 80% lâm sàng, 20% ngẫu nhiên.
    """
    clinical_map = {
        3: 2,  # hỏi → nặng
        4: 1,  # ngã → sắc
        0: 2,  # huyền → nặng
    }
    chars = list(word)
    for i, ch in enumerate(chars):
        base = CHAR_TO_BASE.get(ch)
        if base and base in TONE_GROUPS and ch in TONE_GROUPS[base]:
            tone_idx = TONE_GROUPS[base].index(ch)
            if tone_idx in clinical_map:
                if random.random() < 0.80:
                    new_idx = clinical_map[tone_idx]
                else:
                    candidates = [j for j in range(5) if j != tone_idx]
                    new_idx = random.choice(candidates)
                chars[i] = TONE_GROUPS[base][new_idx]
                break
    return "".join(chars)


def spastic_nasalize_final(word: str) -> str:
    nasal_final_map = {
        "p": "m", "t": "n", "c": "ng", "ch": "nh",
    }
    w_lower = word.lower()
    for stop, nasal in sorted(nasal_final_map.items(),
                               key=lambda x: len(x[0]), reverse=True):
        if w_lower.endswith(stop):
            return word[:-len(stop)] + nasal
    return word


# ============================================================
# PHẦN 4C: ATHETOID-SPECIFIC
# ============================================================

# ── Bảng athetoid cho diphthong ──
_ATHETOID_DIPHTHONG_MAP = {
    "iê": ["uô", "ươ", "a"],
    "ia": ["ua", "ưa", "ô"],
    "yê": ["uô", "ươ", "a"],
    "uô": ["iê", "ươ", "a"],
    "ua": ["ia", "ưa", "ê"],
    "ươ": ["iê", "uô", "a"],
    "ưa": ["ia", "ua", "ô"],
    "uy": ["ôi", "ai", "ơi"],
    "ui": ["uy", "ôi", "ai"],
    "ôi": ["ai", "ơi", "ui"],
    "ơi": ["ôi", "ai", "ui"],
    "ai": ["ôi", "ơi", "au"],
    "ay": ["ây", "ôi", "au"],
    "ây": ["ay", "âu", "ôi"],
    "ao": ["au", "ôi", "ơi"],
    "au": ["âu", "ao", "ôi"],
    "âu": ["au", "ây", "ôi"],
}

_ATHETOID_MONOPHTHONG_MAP = {
    "a": ["o", "e", "ô", "ơ", "u", "i"],
    "ă": ["â", "a", "e", "ơ"],
    "â": ["ă", "a", "ơ", "ô"],
    "e": ["a", "i", "ê", "o"],
    "ê": ["e", "i", "ơ", "a"],
    "i": ["ê", "u", "ư", "a"],
    "o": ["a", "u", "ô", "e"],
    "ô": ["o", "ơ", "u", "a"],
    "ơ": ["â", "a", "ô", "e"],
    "u": ["o", "ư", "i", "a"],
    "ư": ["u", "ơ", "i", "o"],
    "y": ["i", "ê", "u"],
}


def athetoid_burst(word: str) -> str:
    burst_map = {
        "đ": ["b", "g", "k"], "n": ["t", "đ", "g"],
        "l": ["t", "đ", "k"], "m": ["b", "p", "g"],
        "d": ["b", "g", "t"], "v": ["b", "g", "đ"],
        "h": ["k", "g", "t"], "s": ["t", "ch", "k"],
        "x": ["t", "ch", "k"], "b": ["đ", "g", "m"],
        "t": ["k", "đ", "b"], "k": ["t", "g", "b"],
        "c": ["t", "g", "k"], "g": ["k", "b", "đ"],
        "th": ["k", "t", "kh"], "kh": ["th", "g", "k"],
        "ch": ["t", "k", "tr"], "tr": ["ch", "t", "k"],
        "ph": ["b", "v", "t"],
    }
    w_lower = word.lower()
    for c in sorted(burst_map.keys(), key=len, reverse=True):
        if w_lower.startswith(c):
            replacement = random.choice(burst_map[c])
            return replacement + word[len(c):]
    return word


def athetoid_vowel_shift(word: str) -> str:
    """
    Biến thiên nguyên âm ngẫu nhiên — athetoid dysarthria.

    ✅ FIX #6: Xử lý diphthong trước, fallback sang đơn.
    Tránh "tơên", "mâư" — âm tiết alien.
    """
    initial, vowel_part, final = split_syllable(word)
    if not vowel_part:
        return word

    vowel_base = _strip_tone_from_vowel_part(vowel_part)

    # Bước 1: Match diphthong
    for diph in sorted(_ATHETOID_DIPHTHONG_MAP.keys(),
                        key=len, reverse=True):
        if diph in vowel_base:
            replacement = random.choice(_ATHETOID_DIPHTHONG_MAP[diph])
            new_vowel_base = vowel_base.replace(diph, replacement, 1)
            new_vowel = _transfer_tone_pattern(vowel_part, new_vowel_base)
            return initial + new_vowel + final

    # Bước 2: Fallback — nguyên âm đơn
    for ch_base in sorted(_ATHETOID_MONOPHTHONG_MAP.keys(),
                           key=len, reverse=True):
        if ch_base in vowel_base:
            replacement = random.choice(
                _ATHETOID_MONOPHTHONG_MAP[ch_base]
            )
            new_vowel_base = vowel_base.replace(ch_base, replacement, 1)
            new_vowel = _transfer_tone_pattern(vowel_part, new_vowel_base)
            return initial + new_vowel + final

    return word


def athetoid_tone_fluctuation(word: str) -> str:
    chars = list(word)
    tone_positions = [(i, ch) for i, ch in enumerate(chars)
                      if ch in CHAR_TO_BASE and ch != CHAR_TO_BASE.get(ch)]
    if not tone_positions:
        no_tone_pos = [(i, ch) for i, ch in enumerate(chars)
                       if ch in CHAR_TO_BASE and ch == CHAR_TO_BASE.get(ch)
                       and ch in TONE_GROUPS]
        if no_tone_pos:
            idx, ch = random.choice(no_tone_pos)
            chars[idx] = random.choice(TONE_GROUPS[ch])
        return "".join(chars)
    idx, ch = random.choice(tone_positions)
    base = CHAR_TO_BASE[ch]
    if base in TONE_GROUPS:
        all_tones = TONE_GROUPS[base] + [base]
        chars[idx] = random.choice(all_tones)
    return "".join(chars)


def athetoid_extra_sound(word: str) -> str:
    if len(word) <= 1:
        return word
    extra_sounds = ["ơ", "â", "ư", "a", "h", "n"]
    pos = random.randint(1, len(word) - 1)
    extra = random.choice(extra_sounds)
    return word[:pos] + extra + word[pos:]


# ============================================================
# PHẦN 4D: ATAXIC-SPECIFIC
# ============================================================

def ataxic_overshoot(word: str) -> str:
    overshoot_map = {
        "t": "tr", "đ": "tr", "d": "gi",
        "s": "tr", "x": "ch",
        "n": "ng", "m": "ng",
        "c": "k", "g": "gh",
        "b": "ph", "v": "ph",
    }
    w_lower = word.lower()
    for c in sorted(overshoot_map.keys(), key=len, reverse=True):
        if w_lower.startswith(c):
            rest = w_lower[len(c):]
            if c + rest[:1] in INITIAL_CONSONANTS and len(c) < len(c + rest[:1]):
                continue
            replacement = overshoot_map[c]
            return replacement + word[len(c):]
    return word


def ataxic_undershoot(word: str) -> str:
    undershoot_map = {
        "tr": "t", "ch": "t", "kh": "k",
        "ng": "n", "ngh": "n", "nh": "n",
        "gh": "g", "gi": "d", "ph": "p",
        "th": "t", "qu": "k",
    }
    w_lower = word.lower()
    for c in sorted(undershoot_map.keys(), key=len, reverse=True):
        if w_lower.startswith(c):
            replacement = undershoot_map[c]
            return replacement + word[len(c):]
    return word


def ataxic_irregular_stress(word: str) -> str:
    target_tone_idx = random.choice([1, -1])
    chars = list(word)
    for i, ch in enumerate(chars):
        base = CHAR_TO_BASE.get(ch)
        if base and base in TONE_GROUPS and ch != base:
            if target_tone_idx == -1:
                chars[i] = base
            elif target_tone_idx < len(TONE_GROUPS[base]):
                chars[i] = TONE_GROUPS[base][target_tone_idx]
            break
    return "".join(chars)


def ataxic_decompose_cluster(word: str) -> str:
    decompose_map = {
        "tr": ("tờ", "r"), "ch": ("cờ", "h"), "kh": ("cờ", "h"),
        "nh": ("nờ", "h"), "ng": ("nờ", "g"), "ngh": ("nờ", "g"),
        "gh": ("gờ", "h"), "ph": ("pờ", "h"), "th": ("tờ", "h"),
        "gi": ("gờ", "i"), "qu": ("cờ", "u"),
    }
    w_lower = word.lower()
    for c in sorted(decompose_map.keys(), key=len, reverse=True):
        if w_lower.startswith(c):
            part1, part2 = decompose_map[c]
            return part1 + " " + part2 + word[len(c):]
    return word


# ============================================================
# PHẦN 4E: DISPATCH TABLE
# ============================================================

_DISPATCH = {
    "drop_initial":              drop_initial,
    "drop_final":                drop_final,
    "drop_both":                 drop_both,
    "reduce_to_vowel":           reduce_to_vowel,
    "confuse_initial":           confuse_initial,
    "confuse_final":             confuse_final,
    "confuse_vowel":             confuse_vowel,
    "remove_tone":               remove_tone,
    "wrong_tone":                wrong_tone,
    "nasalize_stop":             nasalize_stop,
    "weaken_consonant":          weaken_consonant,
    "repeat_syllable":           repeat_syllable,
    "insert_filler":             insert_filler,
    "slow_stretch":              slow_stretch,
    "homophone_replace":         homophone_replace,
    "truncate_syllable":         truncate_syllable,
    "keep":                      lambda w: w,
    "centralize_vowel":          centralize_vowel,
    "monophthongize":            monophthongize,
    "clinical_tone_error":       clinical_tone_error,
    "spastic_nasalize_final":    spastic_nasalize_final,
    "athetoid_burst":            athetoid_burst,
    "athetoid_vowel_shift":      athetoid_vowel_shift,
    "athetoid_tone_fluctuation": athetoid_tone_fluctuation,
    "athetoid_extra_sound":      athetoid_extra_sound,
    "ataxic_overshoot":          ataxic_overshoot,
    "ataxic_undershoot":         ataxic_undershoot,
    "ataxic_irregular_stress":   ataxic_irregular_stress,
    "ataxic_decompose_cluster":  ataxic_decompose_cluster,
}


# ============================================================
# PHẦN 5: NOISE PROFILES
# ============================================================

SPASTIC_PROFILES = {
    "mild": {
        "confuse_initial": 0.15, "confuse_final": 0.10, "confuse_vowel": 0.05,
        "centralize_vowel": 0.08, "monophthongize": 0.04,
        "wrong_tone": 0.08, "clinical_tone_error": 0.06,
        "nasalize_stop": 0.04, "spastic_nasalize_final": 0.03,
        "drop_initial": 0.03, "drop_final": 0.04, "drop_both": 0.01,
        "slow_stretch": 0.05, "repeat_syllable": 0.03,
        "homophone_replace": 0.06, "keep": 0.15,
    },
    "moderate": {
        "confuse_initial": 0.12, "confuse_final": 0.08, "confuse_vowel": 0.05,
        "centralize_vowel": 0.10, "monophthongize": 0.06,
        "wrong_tone": 0.05, "clinical_tone_error": 0.08,
        "nasalize_stop": 0.06, "spastic_nasalize_final": 0.05,
        "drop_initial": 0.08, "drop_final": 0.06, "drop_both": 0.04,
        "slow_stretch": 0.04, "repeat_syllable": 0.04,
        "truncate_syllable": 0.03, "homophone_replace": 0.04,
        "insert_filler": 0.02,
    },
    "severe": {
        "confuse_initial": 0.08, "confuse_final": 0.05, "confuse_vowel": 0.04,
        "centralize_vowel": 0.12, "monophthongize": 0.08,
        "remove_tone": 0.05, "clinical_tone_error": 0.08,
        "nasalize_stop": 0.07, "spastic_nasalize_final": 0.06,
        "drop_initial": 0.08, "drop_final": 0.06, "drop_both": 0.06,
        "reduce_to_vowel": 0.05, "truncate_syllable": 0.04,
        "repeat_syllable": 0.04, "weaken_consonant": 0.02,
        "homophone_replace": 0.02,
    },
}

ATHETOID_PROFILES = {
    "mild": {
        "confuse_initial": 0.08, "confuse_final": 0.06,
        "athetoid_burst": 0.10, "confuse_vowel": 0.05,
        "athetoid_vowel_shift": 0.08,
        "wrong_tone": 0.05, "athetoid_tone_fluctuation": 0.10,
        "drop_initial": 0.04, "drop_final": 0.04,
        "athetoid_extra_sound": 0.05, "insert_filler": 0.04,
        "repeat_syllable": 0.04, "homophone_replace": 0.06,
        "slow_stretch": 0.05, "keep": 0.16,
    },
    "moderate": {
        "confuse_initial": 0.06, "confuse_final": 0.05,
        "athetoid_burst": 0.14, "confuse_vowel": 0.04,
        "athetoid_vowel_shift": 0.10,
        "wrong_tone": 0.03, "athetoid_tone_fluctuation": 0.12,
        "drop_initial": 0.06, "drop_final": 0.05, "drop_both": 0.03,
        "athetoid_extra_sound": 0.07, "insert_filler": 0.05,
        "repeat_syllable": 0.05, "nasalize_stop": 0.03,
        "homophone_replace": 0.04, "slow_stretch": 0.03, "keep": 0.05,
    },
    "severe": {
        "athetoid_burst": 0.15, "athetoid_vowel_shift": 0.12,
        "athetoid_tone_fluctuation": 0.10, "athetoid_extra_sound": 0.08,
        "confuse_initial": 0.05, "confuse_final": 0.04,
        "drop_initial": 0.08, "drop_final": 0.06, "drop_both": 0.05,
        "reduce_to_vowel": 0.05, "nasalize_stop": 0.05,
        "repeat_syllable": 0.05, "insert_filler": 0.04,
        "weaken_consonant": 0.03, "remove_tone": 0.03, "keep": 0.02,
    },
}

ATAXIC_PROFILES = {
    "mild": {
        "confuse_initial": 0.08, "confuse_final": 0.06,
        "ataxic_overshoot": 0.10, "ataxic_undershoot": 0.10,
        "confuse_vowel": 0.06, "wrong_tone": 0.05,
        "ataxic_irregular_stress": 0.08,
        "drop_initial": 0.03, "drop_final": 0.03,
        "slow_stretch": 0.08, "homophone_replace": 0.06,
        "repeat_syllable": 0.03, "keep": 0.14,
    },
    "moderate": {
        "ataxic_overshoot": 0.12, "ataxic_undershoot": 0.10,
        "ataxic_irregular_stress": 0.10, "ataxic_decompose_cluster": 0.06,
        "confuse_initial": 0.06, "confuse_final": 0.05,
        "confuse_vowel": 0.05, "wrong_tone": 0.04,
        "drop_initial": 0.06, "drop_final": 0.05, "drop_both": 0.03,
        "slow_stretch": 0.08, "repeat_syllable": 0.04,
        "insert_filler": 0.04, "homophone_replace": 0.04,
        "truncate_syllable": 0.03, "keep": 0.05,
    },
    "severe": {
        "ataxic_overshoot": 0.10, "ataxic_undershoot": 0.10,
        "ataxic_irregular_stress": 0.10, "ataxic_decompose_cluster": 0.08,
        "confuse_initial": 0.05, "confuse_final": 0.04,
        "confuse_vowel": 0.04,
        "drop_initial": 0.08, "drop_final": 0.06, "drop_both": 0.05,
        "reduce_to_vowel": 0.04, "slow_stretch": 0.06,
        "repeat_syllable": 0.05, "insert_filler": 0.04,
        "remove_tone": 0.04, "weaken_consonant": 0.03,
        "truncate_syllable": 0.02, "keep": 0.02,
    },
}

SENTENCE_NOISE_PROFILES = {
    ("spastic", "mild"):     {"drop_word": 0.05, "drop_consecutive": 0.02, "swap_words": 0.02, "merge_words": 0.03, "split_word": 0.02, "repeat_word": 0.03, "hallucinate": 0.02},
    ("spastic", "moderate"): {"drop_word": 0.10, "drop_consecutive": 0.05, "swap_words": 0.03, "merge_words": 0.05, "split_word": 0.04, "repeat_word": 0.05, "hallucinate": 0.03},
    ("spastic", "severe"):   {"drop_word": 0.15, "drop_consecutive": 0.10, "swap_words": 0.04, "merge_words": 0.08, "split_word": 0.06, "repeat_word": 0.08, "hallucinate": 0.04},
    ("athetoid", "mild"):     {"drop_word": 0.04, "drop_consecutive": 0.02, "swap_words": 0.03, "merge_words": 0.02, "split_word": 0.02, "repeat_word": 0.05, "hallucinate": 0.03},
    ("athetoid", "moderate"): {"drop_word": 0.08, "drop_consecutive": 0.04, "swap_words": 0.05, "merge_words": 0.04, "split_word": 0.03, "repeat_word": 0.08, "hallucinate": 0.05},
    ("athetoid", "severe"):   {"drop_word": 0.12, "drop_consecutive": 0.08, "swap_words": 0.06, "merge_words": 0.06, "split_word": 0.05, "repeat_word": 0.10, "hallucinate": 0.06},
    ("ataxic", "mild"):       {"drop_word": 0.03, "drop_consecutive": 0.01, "swap_words": 0.02, "merge_words": 0.04, "split_word": 0.05, "repeat_word": 0.03, "hallucinate": 0.02},
    ("ataxic", "moderate"):   {"drop_word": 0.06, "drop_consecutive": 0.03, "swap_words": 0.04, "merge_words": 0.06, "split_word": 0.08, "repeat_word": 0.05, "hallucinate": 0.03},
    ("ataxic", "severe"):     {"drop_word": 0.10, "drop_consecutive": 0.06, "swap_words": 0.05, "merge_words": 0.08, "split_word": 0.10, "repeat_word": 0.06, "hallucinate": 0.05},
}

CP_TYPE_PROFILES = {
    "spastic":  SPASTIC_PROFILES,
    "athetoid": ATHETOID_PROFILES,
    "ataxic":   ATAXIC_PROFILES,
}

DEFAULT_CP_TYPE_WEIGHTS = {
    "spastic": 0.70, "athetoid": 0.20, "ataxic": 0.10,
}


# ============================================================
# PHẦN 6: HÀM ĐIỀU PHỐI NHIỄU
# ============================================================

def noise_word(
    word: str,
    cp_type: str = "spastic",
    severity: str = "moderate",
    allow_chain: bool = True,
) -> str:
    if not word.strip():
        return word

    profiles = CP_TYPE_PROFILES.get(cp_type, SPASTIC_PROFILES)
    profile = profiles.get(severity, profiles["moderate"])

    funcs = list(profile.keys())
    weights = list(profile.values())

    n_noise = 1
    if allow_chain:
        if severity == "severe" and random.random() < 0.30:
            n_noise = 2
        elif severity == "moderate" and random.random() < 0.10:
            n_noise = 2

    for _ in range(n_noise):
        noise_type = random.choices(population=funcs, weights=weights, k=1)[0]
        fn = _DISPATCH.get(noise_type)
        if fn:
            candidate = fn(word)
            if candidate.strip():
                word = candidate # ✅ Fix lỗi logic gán sai

    return word


def noise_sentence(
    sentence: str,
    prob: float = 0.45,
    cp_type: str = "spastic",
    severity: str = "moderate",
) -> str:
    """
    Thêm nhiễu vào toàn bộ câu.

    ✅ FIX #2: Pipeline xử lý trên list[str] xuyên suốt.
    ✅ FIX #5: Nới điều kiện cho câu ngắn (≥2 từ).
    """
    words = sentence.split()
    if not words:
        return sentence

    # ── Giai đoạn 1: Word-level ──
    new_words = []
    for w in words:
        if random.random() < prob:
            noised = noise_word(w, cp_type=cp_type, severity=severity)
            new_words.extend(noised.split())
        else:
            new_words.append(w)

    # ── Giai đoạn 2: Sentence-level ──
    sent_profile = SENTENCE_NOISE_PROFILES.get(
        (cp_type, severity),
        SENTENCE_NOISE_PROFILES[("spastic", "moderate")]
    )

    # ✅ FIX #5: Nới ngưỡng từ > 3 xuống > 2 (tức ≥ 3 từ mới drop 1)
    #            Nới > 2 xuống >= 2 cho swap/merge
    #            Câu 2 từ hoàn toàn có thể bị swap, merge, split

    # Nuốt 1 từ — cần ≥ 3 từ để sau khi drop còn ≥ 2
    if (random.random() < sent_profile.get("drop_word", 0)
            and len(new_words) >= 3):
        idx = random.randint(0, len(new_words) - 1)
        new_words.pop(idx)

    # Nuốt cụm 2-3 từ liên tiếp — cần ≥ 4 để còn tối thiểu 1
    if (random.random() < sent_profile.get("drop_consecutive", 0)
            and len(new_words) >= 4):
        n_drop = random.randint(2, min(3, len(new_words) - 1))
        start = random.randint(0, len(new_words) - n_drop)
        new_words = new_words[:start] + new_words[start + n_drop:]

    # Đảo 2 từ liền kề — ✅ chỉ cần ≥ 2 từ
    if (random.random() < sent_profile.get("swap_words", 0)
            and len(new_words) >= 2):
        idx = random.randint(0, len(new_words) - 2)
        new_words[idx], new_words[idx + 1] = (
            new_words[idx + 1], new_words[idx]
        )

    # Gộp 2 từ — ✅ chỉ cần ≥ 2 từ
    if (random.random() < sent_profile.get("merge_words", 0)
            and len(new_words) >= 2):
        idx = random.randint(0, len(new_words) - 2)
        merged = new_words[idx] + new_words[idx + 1]
        new_words = new_words[:idx] + [merged] + new_words[idx + 2:]

    # Tách từ dài — ✅ chỉ cần ≥ 1 từ có len > 2
    if random.random() < sent_profile.get("split_word", 0):
        long_indices = [i for i, w in enumerate(new_words) if len(w) > 2]
        if long_indices:
            idx = random.choice(long_indices)
            w = new_words[idx]
            split_pos = random.randint(1, len(w) - 1)
            part1, part2 = w[:split_pos], w[split_pos:]
            new_words = (new_words[:idx] + [part1, part2]
                         + new_words[idx + 1:])

    # Lặp nguyên từ — ✅ chỉ cần ≥ 1 từ
    if (random.random() < sent_profile.get("repeat_word", 0)
            and len(new_words) >= 1):
        idx = random.randint(0, len(new_words) - 1)
        n_repeat = random.randint(1, 2)
        for _ in range(n_repeat):
            new_words.insert(idx, new_words[idx])

    # Hallucination — ✅ chỉ cần ≥ 1 từ
    if (random.random() < sent_profile.get("hallucinate", 0)
            and len(new_words) >= 1):
        insert_word = random.choice(HALLUCINATION_INSERTS)
        pos = random.randint(0, len(new_words))
        new_words.insert(pos, insert_word)

    result = " ".join(new_words)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# ============================================================
# PHẦN 7: INFERENCE TẦNG 1 — BATCH PROCESSING
# ============================================================

def _load_audio_fast(audio_path: str, target_sr: int,
                     max_samples: int) -> np.ndarray:
    """
    Sử dụng soundfile để đọc raw PCM cực nhanh (C backend).
    Sử dụng librosa để resample (Tránh dùng torchaudio gây lỗi môi trường CUDA).
    """
    import librosa
    try:
        audio, sr = sf.read(audio_path, dtype='float32')
    except Exception:
        # Fallback cho format không phải WAV/FLAC/OGG
        audio, sr = librosa.load(audio_path, sr=target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        return audio

    # Nếu file là stereo (2 kênh) → gộp thành mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample nếu sample rate của file audio khác với yêu cầu của Whisper (thường là 16000)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Cắt ngắn nếu audio vượt quá độ dài tối đa (30s)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio


class AudioDataset(Dataset):
    """Dataset cho DataLoader, hỗ trợ batch inference."""

    def __init__(self, df: pd.DataFrame, audio_dir: str,
                 sample_rate: int, max_duration: float = 30.0):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['file_name'])
        is_error = False
        try:
            audio = _load_audio_fast(
                audio_path, self.sample_rate, self.max_samples
            )
        except Exception as e:
            logger.warning(f"⚠️  Error loading {audio_path}: {e}")
            audio = np.zeros(self.sample_rate, dtype=np.float32)
            is_error = True
        return {
            'audio': audio,
            'file_name': row['file_name'],
            'ground_truth': normalize_vietnamese_text(row['transcript']),
            'severity': row.get('severity', 'unknown'),
            'cp_type': row.get('cp_type', 'unknown'),
            'is_error': is_error,
        }


def collate_audio(batch, processor, sample_rate):
    """Custom collate: dùng processor để pad/stack features."""
    audios = [item['audio'] for item in batch]
    features = processor(
        audios, sampling_rate=sample_rate,
        return_tensors="pt", padding="max_length"
    )
    return {
        'input_features': features.input_features,
        'file_names': [item['file_name'] for item in batch],
        'ground_truths': [item['ground_truth'] for item in batch],
        'severities': [item['severity'] for item in batch],
        'cp_types': [item['cp_type'] for item in batch],
        'is_errors': [item['is_error'] for item in batch],
    }


def generate_asr_predictions(
    config_path: str = "./config/config.yaml",
    output_csv: str = "./outputs/correction_data_real.csv",
    batch_size: int = 16,
) -> pd.DataFrame:
    """Chạy inference Tầng 1 (PhoWhisper + LoRA) — batch mode."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    asr_output_dir = config['asr']['output_dir']
    model_name     = config['asr']['model_name']
    audio_dir      = config['dataset']['audio_dir']
    metadata_path  = config['dataset']['metadata_path']
    sample_rate    = config['dataset']['sample_rate']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  Device: {device}")

    logger.info("🔧 Loading processor...")
    processor = WhisperProcessor.from_pretrained(model_name)

    logger.info("🔧 Loading PhoWhisper + LoRA weights...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=False,
    )
    model = PeftModel.from_pretrained(base_model, asr_output_dir)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    df = pd.read_csv(metadata_path)
    df = df[df['transcript'].notna()].reset_index(drop=True)

    dataset = AudioDataset(df, audio_dir, sample_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=lambda batch: collate_audio(batch, processor, sample_rate),
    )

    results = []
    logger.info(f"🚀 Running batch inference on {len(df)} samples "
                f"(batch_size={batch_size})...")

    for batch in tqdm(dataloader, desc="ASR Batch Inference"):
        input_features = batch['input_features'].to(device)
        if device == "cuda":
            input_features = input_features.half()

        try:
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    max_length=225,
                    num_beams=2,
                    language="vi",
                    task="transcribe",
                )

            predicted_texts = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            for i in range(len(predicted_texts)):
                if batch['is_errors'][i]: # ✅ FIX: Đã thêm chữ 's'
                    logger.warning(f"   Skipped error audio: {batch['file_names'][i]}")
                    continue
                pred = normalize_vietnamese_text(predicted_texts[i])
                results.append({
                    'file_name':      batch['file_names'][i],
                    'predicted_text': pred,
                    'ground_truth':   batch['ground_truths'][i],
                    'severity':       batch['severities'][i],
                    'cp_type':        batch['cp_types'][i],
                    'is_real':        True,
                })
        except Exception as e:
            logger.warning(f"⚠️  Error in batch: {e}")
            for i in range(len(batch['file_names'])):
                logger.warning(f"   Skipped: {batch['file_names'][i]}")
            continue

    result_df = pd.DataFrame(results)
    
    # ✅ FIX: Bảo vệ DataFrame rỗng
    if result_df.empty:
        logger.error("❌ LỖI NGHIÊM TRỌNG: Không có bất kỳ audio nào được xử lý thành công!")
        raise ValueError("ASR sinh ra kết quả rỗng. Vui lòng kiểm tra lại data đầu vào.")

    diff_df = result_df[
        result_df['predicted_text'] != result_df['ground_truth']
    ]

    logger.info(f"📊 Tổng samples           : {len(result_df)}")
    logger.info(f"📊 Samples có lỗi (train) : {len(diff_df)}")
    logger.info(f"📊 Accuracy Tầng 1        : "
                f"{1 - len(diff_df) / max(len(result_df), 1):.2%}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"💾 Saved → {output_csv}")

    return result_df


# ============================================================
# PHẦN 8: TẠO DỮ LIỆU TỔNG HỢP (SYNTHETIC NOISE)
# ============================================================

def generate_synthetic_data(
    config_path: str = "./config/config.yaml",
    sentences_file: str = "sentences.txt",
    augment_per_sentence: int = 5,
    output_csv: str = "./outputs/correction_data_synthetic.csv",
    severity_weights: tuple = (0.30, 0.40, 0.30),
    cp_type_weights: dict = None,
) -> pd.DataFrame:
    if cp_type_weights is None:
        cp_type_weights = DEFAULT_CP_TYPE_WEIGHTS

    severities = ["mild", "moderate", "severe"]
    cp_types   = list(cp_type_weights.keys())
    cp_weights = list(cp_type_weights.values())
    prob_map = {"mild": 0.25, "moderate": 0.45, "severe": 0.65}

    all_sentences = {}
    
    # 1️⃣ Đọc từ sentences.txt
    if os.path.exists(sentences_file):
        try:
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences_from_file = [
                    normalize_vietnamese_text(line.strip()) 
                    for line in f if line.strip()
                ]
            for sent in sentences_from_file:
                all_sentences[sent] = "file"
            logger.info(f"📄 Loaded {len(sentences_from_file)} sentences from '{sentences_file}'")
        except FileNotFoundError:
            logger.warning(f"⚠️  File not found: '{sentences_file}'")
        except Exception as e:
            logger.warning(f"⚠️  Error reading '{sentences_file}': {e}")
    else:
        logger.warning(f"⚠️  File not found: '{sentences_file}'")

    # 2️⃣ Đọc từ metadata (OPTIONAL - có thể lỗi mà vẫn chạy)
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            metadata_path = config.get('dataset', {}).get('metadata_path')
            
            if not metadata_path:
                logger.warning("⚠️  'metadata_path' not set in config (Optional)")
            elif not os.path.exists(metadata_path):
                logger.warning(f"⚠️  Metadata file not found: {metadata_path} (Optional)")
            else:
                try:
                    df_meta = pd.read_csv(metadata_path)
                    df_meta = df_meta[df_meta['transcript'].notna()]
                    
                    sentences_from_meta = [
                        normalize_vietnamese_text(t)
                        for t in df_meta['transcript'].tolist()
                    ]
                    
                    new_meta_count = 0
                    for sent in sentences_from_meta:
                        if sent not in all_sentences:
                            all_sentences[sent] = "metadata"
                            new_meta_count += 1
                    
                    logger.info(
                        f"📄 Loaded {len(sentences_from_meta)} sentences from metadata "
                        f"(Added {new_meta_count} new)"
                    )
                except pd.errors.ParserError as e:
                    logger.warning(f"⚠️  Error parsing metadata CSV: {e}")
                except Exception as e:
                    logger.warning(f"⚠️  Error processing metadata: {e}")
        else:
             logger.warning(f"⚠️  Config file not found: {config_path}")
             
    except Exception as e:
        logger.warning(f"⚠️  Error reading metadata: {e}")

    # 3️⃣ Filter & Shuffle
    sentences = [
        s for s in all_sentences.keys() 
        if len(s.strip()) > 5
    ]
    random.shuffle(sentences)
    
    if not sentences:
        raise ValueError(
            "❌ Không tìm thấy câu hợp lệ nào từ cả hai nguồn! "
            "Vui lòng kiểm tra lại 'sentences.txt' và cấu hình metadata."
        )

    logger.info(f"📄 Total unique sentences: {len(sentences)}")
    logger.info(f"   From file    : {sum(1 for s in all_sentences.values() if s == 'file')}")
    logger.info(f"   From metadata: {sum(1 for s in all_sentences.values() if s == 'metadata')}")

    # 4️⃣ Tạo synthetic data
    results = []
    logger.info(f"🔧 Generating synthetic data (×{augment_per_sentence} per sentence)")

    for clean_text in tqdm(sentences, desc="Generating noise"):
        for _ in range(augment_per_sentence):
            cp_type = random.choices(cp_types, weights=cp_weights, k=1)[0]
            sev = random.choices(severities, weights=severity_weights, k=1)[0]
            prob = prob_map[sev]

            noisy = noise_sentence(
                clean_text, prob=prob,
                cp_type=cp_type, severity=sev,
            )

            if (noisy.strip() and 
                noisy != clean_text and 
                len(noisy.strip()) > 5):
                results.append({
                    'predicted_text': noisy,
                    'ground_truth':   clean_text,
                    'severity':       sev,
                    'cp_type':        cp_type,
                    'is_real':        False,
                    'source':         all_sentences[clean_text],
                })

    result_df = pd.DataFrame(results)
    
    if result_df.empty:
        raise ValueError("❌ No synthetic data generated!")
    
    result_df = result_df[
        result_df['predicted_text'].str.strip().astype(bool)
    ].reset_index(drop=True)

    # 5️⃣ Deduplication
    before_len = len(result_df)
    result_df = result_df.drop_duplicates(
        subset=['predicted_text', 'ground_truth'],
        keep='first'
    ).reset_index(drop=True)
    
    dropped = before_len - len(result_df)
    if dropped > 0:
        logger.info(f"📊 Dropped {dropped} duplicate pairs")

    # 6️⃣ Lưu File
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 7️⃣ Báo cáo
    logger.info(f"📊 Generated {len(result_df)} unique synthetic pairs")
    logger.info(f"📊 CP type dist : {result_df['cp_type'].value_counts().to_dict()}")
    logger.info(f"📊 Severity dist: {result_df['severity'].value_counts().to_dict()}")
    logger.info(f"📊 Source dist  : {result_df['source'].value_counts().to_dict()}")
    logger.info(f"💾 Saved → {output_csv}")

    return result_df

# ============================================================
# PHẦN 9: KẾT HỢP DỮ LIỆU THỰC + SYNTHETIC
# ============================================================

def combine_correction_data(
    real_csv: str      = "./outputs/correction_data_real.csv",
    synthetic_csv: str = "./outputs/correction_data_synthetic.csv",
    output_csv: str    = "./outputs/correction_data_combined.csv",
    real_weight: int   = 3,
) -> pd.DataFrame:
    dfs = []

    if os.path.exists(real_csv):
        real_df = pd.read_csv(real_csv)
        real_df = real_df[
            real_df['predicted_text'] != real_df['ground_truth']
        ].copy()
        for _ in range(real_weight):
            dfs.append(real_df)
        logger.info(f"📊 Real data     : {len(real_df)} samples "
                    f"× {real_weight} = {len(real_df) * real_weight}")
    else:
        logger.warning(f"⚠️  Real data not found: {real_csv}")

    if os.path.exists(synthetic_csv):
        syn_df = pd.read_csv(synthetic_csv)
        syn_df = syn_df[
            syn_df['predicted_text'].str.strip().astype(bool)
        ].copy()
        dfs.append(syn_df)
        logger.info(f"📊 Synthetic data: {len(syn_df)} samples")
    else:
        logger.warning(f"⚠️  Synthetic data not found: {synthetic_csv}")

    if not dfs:
        raise ValueError(
            "Không có dữ liệu nào! "
            "Chạy generate_asr_predictions() hoặc "
            "generate_synthetic_data() trước."
        )

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    for col in ['severity', 'cp_type']:
        if col not in combined.columns:
            combined[col] = 'unknown'
        combined[col] = combined[col].fillna('unknown')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined.to_csv(output_csv, index=False, encoding='utf-8-sig')

    n_real = combined['is_real'].sum() if 'is_real' in combined.columns else 0
    n_syn  = len(combined) - n_real

    logger.info(f"📊 Combined total : {len(combined)} samples")
    logger.info(f"📊 Real/Synthetic : {n_real} / {n_syn}")
    logger.info(f"📊 CP type dist   : "
                f"{combined['cp_type'].value_counts().to_dict()}")
    logger.info(f"📊 Severity dist  : "
                f"{combined['severity'].value_counts().to_dict()}")
    logger.info(f"💾 Saved → {output_csv}")

    return combined


# ============================================================
# PHẦN 10: ENTRY POINT
# ============================================================

def run_full_pipeline(
    config_path: str     = "./config/config.yaml",
    sentences_file: str  = "sentences.txt",
    augment_per_sentence: int = 5,
    real_weight: int     = 3,
    severity_weights: tuple = (0.30, 0.40, 0.30),
    cp_type_weights: dict = None,
    skip_real: bool      = False,
    batch_size: int      = 16,
) -> pd.DataFrame:
    real_csv      = "./outputs/correction_data_real.csv"
    synthetic_csv = "./outputs/correction_data_synthetic.csv"
    combined_csv  = "./outputs/correction_data_combined.csv"

    if not skip_real:
        logger.info("=" * 60)
        logger.info("BƯỚC 1: Inference Tầng 1 (PhoWhisper + LoRA) — Batch")
        logger.info("=" * 60)
        generate_asr_predictions(
            config_path=config_path,
            output_csv=real_csv,
            batch_size=batch_size,
        )
    else:
        logger.info("⏭️  Bỏ qua Bước 1 (skip_real=True)")

    logger.info("=" * 60)
    logger.info("BƯỚC 2: Tạo synthetic noise (dysarthria-specific)")
    logger.info("=" * 60)
    generate_synthetic_data(
        config_path=config_path,
        sentences_file=sentences_file,
        augment_per_sentence=augment_per_sentence,
        output_csv=synthetic_csv,
        severity_weights=severity_weights,
        cp_type_weights=cp_type_weights,
    )

    logger.info("=" * 60)
    logger.info("BƯỚC 3: Kết hợp dữ liệu")
    logger.info("=" * 60)
    combined = combine_correction_data(
        real_csv=real_csv,
        synthetic_csv=synthetic_csv,
        output_csv=combined_csv,
        real_weight=real_weight,
    )

    logger.info("✅ Pipeline hoàn tất!")
    logger.info(f"   → File sẵn sàng train ViT5: {combined_csv}")
    return combined


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Tạo dữ liệu training cho ViT5 Correction (Tầng 2)"
    )
    parser.add_argument("--config",       default="./config/config.yaml")
    parser.add_argument("--sentences",    default="sentences.txt", help="Đường dẫn tới file txt chứa câu chuẩn")
    parser.add_argument("--augment",      type=int, default=30)
    parser.add_argument("--real_weight",  type=int, default=3)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--skip_real",    action="store_true")
    parser.add_argument("--spastic_pct",  type=float, default=0.70)
    parser.add_argument("--athetoid_pct", type=float, default=0.20)
    parser.add_argument("--ataxic_pct",   type=float, default=0.10)
    args = parser.parse_args()

    cp_type_weights = {
        "spastic":  args.spastic_pct,
        "athetoid": args.athetoid_pct,
        "ataxic":   args.ataxic_pct,
    }

    run_full_pipeline(
        config_path=args.config,
        sentences_file=args.sentences,
        augment_per_sentence=args.augment,
        real_weight=args.real_weight,
        skip_real=args.skip_real,
        cp_type_weights=cp_type_weights,
        batch_size=args.batch_size,
    )