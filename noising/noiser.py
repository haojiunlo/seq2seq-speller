import random
import re
import numpy as np

from tqdm import tqdm
from collections import defaultdict


def get_keyboard_mappings():
    keyboard_mappings = defaultdict(lambda: [])
    keyboard = ["qwertyuiop", "asdfghjkl*", "zxcvbnm***"]
    row = len(keyboard)
    col = len(keyboard[0])

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    for i in range(row):
        for j in range(col):
            for k in range(4):
                x_, y_ = i + dx[k], j + dy[k]
                if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                    if keyboard[x_][y_] == '*':
                        continue
                    if keyboard[i][j] == '*':
                        continue
                    keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])
    return keyboard_mappings


keyboard_mappings = get_keyboard_mappings()


def _get_index(length):
    return int(np.floor(np.random.beta(3, 2) * length))


def _get_random_char():
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]
    return np.random.choice(alphabets, 1)[0]


def _get_keyboard_neighbor(ch):
    if ch not in keyboard_mappings:
        return ch
    return np.random.choice(keyboard_mappings[ch], 1)[0]


def _get_swap_word_representation(word):
    if len(word) >= 3:
        offset = np.random.choice([1, 2], 1, p=[0.95, 0.05])[0]
        max_idx = len(word) - 2 if offset == 1 else len(word) - 3
        idx = min(_get_index(len(word)), max_idx)

        word_ls = list(word)
        w_ = word_ls[idx]
        if word_ls[idx].isnumeric() or word_ls[idx + offset].isnumeric():
            return word
        word_ls[idx] = word_ls[idx + offset]
        word_ls[idx + offset] = w_
        word = "".join(word_ls)
    return word


def _get_drop_word_representation(word):
    if len(word) > 3:
        idx = _get_index(len(word))
        if word[idx].isnumeric():
            return word
        word = word[:idx] + word[idx + 1:]
    return word


def _get_add_word_representation(word):
    if len(word) >= 3:
        idx = _get_index(len(word))
        if word[idx].isnumeric():
            return word
        op = np.random.choice(
            ["repeat", "random", "keyboard", "space"],
            1, p=[0.3, 0.25, 0.4, 0.05])
        if op == "repeat":
            char = word[idx]
        elif op == "random":
            char = _get_random_char()
        elif op == "keyboard":
            char = _get_keyboard_neighbor(word[idx])
        elif op == "space":
            char = " "
        word = word[:idx] + char + word[idx:]

    return word


def _get_keyboard_word_representation(word):
    if len(word) > 3:
        idx = _get_index(len(word))
        if word[idx].isnumeric():
            return word
        keyboard_neighbor = _get_keyboard_neighbor(word[idx])
        word = word[:idx] + keyboard_neighbor + word[idx + 1:]
    return word


def _noise(line, rep_list, typo_probs, op_probs):
    n_corrupted_word = 0
    modified_words = []
    line = re.sub(' +', ' ', line)
    words = line.split(" ")
    for word in words:

        if random.random() < typo_probs:
            rep_type = np.random.choice(rep_list, 1, p=op_probs)[0]
            if rep_type == 'swap':
                new_word = _get_swap_word_representation(word)
            elif rep_type == 'drop':
                new_word = _get_drop_word_representation(word)
            elif rep_type == 'add':
                new_word = _get_add_word_representation(word)
            elif rep_type == 'key':
                new_word = _get_keyboard_word_representation(word)
            elif rep_type == "suffix":
                new_word = word + _get_keyboard_neighbor(word[-1])
            else:
                raise NotImplementedError
            if len(words) < 2:
                new_word = word
        else:
            new_word = word

        if new_word != word:
            n_corrupted_word += 1
        modified_words.append(new_word)
    corrupt = " ".join(modified_words)

    if random.random() < 0.15 and len(modified_words) > 1:
        spaces = list(re.finditer(r"\s", corrupt))
        space_start = np.random.choice(spaces, 1)[0].start()
        corrupt = corrupt[:space_start] + corrupt[space_start + 1:]

    return corrupt


def noise(lines,
          rep_list=['swap', 'drop', 'add', 'key', 'suffix'],
          typo_probs=0.95,
          op_probs=[0.25, 0.15, 0.25, 0.3, 0.05],
          disable_tqdm=True,
          ):
    modified_lines = []
    for line in tqdm(lines, disable=disable_tqdm):
        if isinstance(line, str):
            line = line.strip()
            modified_lines.append(_noise(line, rep_list, typo_probs, op_probs))
        else:
            modified_lines.append(line)

    return modified_lines
