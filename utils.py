import pathlib

import pandas
import numpy
try:
    import cupy
except:
    pass
import librosa


def str_to_int(str_value):
    bytes_value = str_value.encode('utf-8')
    int_value = int(bytes_value.hex(), 16)
    int_value -= 14910113  # magic numbers for katakana
    if int_value > 30:
        int_value -= 190
    return int_value


def make_dict_from_sentence_id_to_char_ids(txt):
    sentence_id_to_char_ids_dict = {}
    df = pandas.read_csv(txt, delimiter='\t')
    for (sentence_id, yomi) in zip(df['sentence_id'], df['yomi']):
        sentence_id = '{0:03d}'.format(sentence_id)
        chars = []
        for char in yomi:
            chars += [str_to_int(char)]
        sentence_id_to_char_ids_dict[sentence_id] = chars
    return sentence_id_to_char_ids_dict


def make_dict_from_char_id_to_str(txt):
    char_id_to_str_dict = {}
    df = pandas.read_csv(txt, delimiter='\t')
    for yomi in df['yomi']:
        for char in yomi:
            char_id_to_str_dict[str_to_int(char)] = char
    return char_id_to_str_dict


def make_dict_from_str_to_char_id(txt):
    str_to_char_id_dict = {}
    df = pandas.read_csv(txt, delimiter='\t')
    for yomi in df['yomi']:
        for char in yomi:
            str_to_char_id_dict[str_to_int(char)] = char
    return str_to_char_id_dict


def path_to_sentence_id(path):
    if isinstance(path, str):
        path = pathlib.Path(path)
    path = path.stem  # remove suffix(.wav)
    return path[-3:]


class Preprocess(object):
    def __init__(self, txt, sr, length):
        self.sentence_id_to_char_ids_dict = \
            make_dict_from_sentence_id_to_char_ids(txt)
        self.sr = sr
        self.length = length

    def __call__(self, path):
        raw, _ = librosa.load(path, self.sr)
        trimed_raw, _ = librosa.effects.trim(raw, top_db=20)

        pad_l = numpy.zeros((self.length - len(trimed_raw)) // 2)
        pad_r = numpy.zeros(self.length - len(trimed_raw) - len(pad_l))
        padded_raw = numpy.concatenate((pad_l, trimed_raw, pad_r))

        input_raw = numpy.expand_dims(padded_raw, 0)

        phonemes = self.sentence_id_to_char_ids_dict[path_to_sentence_id(path)]
        phonemes_length = len(phonemes)

        return (input_raw.astype(numpy.float32),
                numpy.array(phonemes, dtype=numpy.int32),
                numpy.array(phonemes_length, dtype=numpy.int32))

    def convert(self, batch, device):
        if device is None or device < 0:
            raws = numpy.array([raw for raw, _, _ in batch])
            phonemes = [
                numpy.array(phoneme, dtype=numpy.int32) for
                _, phoneme, _ in batch]
            lengths = numpy.array([length for _, _, length in batch])
        else:
            with cupy.cuda.Device(device):
                raws = cupy.array([raw for raw, _, _ in batch])
                phonemes = [
                    cupy.array(phoneme, dtype=cupy.int32) for
                    _, phoneme, _ in batch]
                lengths = cupy.array([length for _, _, length in batch])
        return raws, phonemes, lengths
