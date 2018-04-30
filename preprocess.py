import pathlib

import tqdm
import pandas
import librosa

import params

SAMPLING_RATE = 16000


def convert_to_hiraganas(katakanas):
    hiraganas = ''
    for katakana in katakanas:
        if ord(katakana) == 12540:  # 12540 means "-"
            hiraganas += katakana
        else:
            hiraganas += chr(ord(katakana) - 0x60)
    return hiraganas


def convert_to_preprocessed_path(preprocessed_dir, path):
    return preprocessed_dir.joinpath(*path.parts[-2:])


def get_sentence_id(path, output_cast=int):
    return output_cast(path.stem[-3:])

root_dir = pathlib.Path(params.root)
preprocessed_dir = root_dir.parent.joinpath('preprocessed')
preprocessed_dir.mkdir(mode=0o0755, exist_ok=True)

wav_paths = root_dir.glob('*/*.wav')

df = pandas.read_csv(
    'https://raw.githubusercontent.com/voice-statistics/'
    'voice-statistics.github.com/master/assets/doc/balance_sentences.txt',
    delimiter='\t')
sentence_id_to_hiragana_dic = {
    sentence_id: convert_to_hiraganas(katakanas)
    for sentence_id, katakanas in zip(df['sentence_id'], df['yomi'])}

for wav_path in tqdm.tqdm(wav_paths, total=900):
    preprocessed_path = convert_to_preprocessed_path(
        preprocessed_dir, wav_path)
    if not preprocessed_path.exists():
        preprocessed, _ = librosa.load(wav_path, SAMPLING_RATE)
        preprocessed_path.parent.mkdir(mode=0o0755, exist_ok=True)
        librosa.output.write_wav(
            preprocessed_path, preprocessed, SAMPLING_RATE)

    yomi_path = preprocessed_path.with_name(preprocessed_path.stem + '.txt')
    if not yomi_path.exists():
        yomi = sentence_id_to_hiragana_dic[get_sentence_id(wav_path)]
        with open(yomi_path, 'w') as f:
            f.write(yomi)
