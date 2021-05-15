import re
from os import listdir
from mido import MidiFile
from mido.midifiles.meta import KeySignatureError
import pandas as pd


def parse(file,numerator,denominator,clocksPerTick,demisemiquaverPer24Clocks,key,mode,tempo):
    mid = MidiFile(file, clip=True)
    # for track in mid.tracks:
    # print(track)
    for msg in mid.tracks[0]:
        # print(msg.dict())
        if msg.dict()['type'] == 'key_signature':
            mode = 'minor' if msg.dict()['key'][-1] == 'm' else 'major'
            key = msg.dict()['key'] if mode == 'major' else msg.dict()['key'][:-1]
        if msg.dict()['type'] == 'set_tempo':
            tempo = msg.dict()['tempo']/1000
        if msg.dict()['type'] == 'time_signature':
            numerator = msg.dict()['numerator']
            denominator = msg.dict()['denominator']
            clocksPerTick = msg.dict()['clocks_per_click']
            demisemiquaverPer24Clocks = msg.dict()['notated_32nd_notes_per_beat']
    return numerator,denominator,clocksPerTick,demisemiquaverPer24Clocks,key,mode,tempo


def load_midi(midi_folder: str) -> pd.DataFrame:
    files_in_dir = [f for f in listdir(midi_folder) if re.match('.*\.mid', f)]
    df_midi = pd.DataFrame(columns=['filename', 'numerator', 'denominator', 'clocksPerTick', 'demisemiquaverPer24Clocks', 'key', 'mode', 'tempo'])

    for i, f in enumerate(files_in_dir):
        numerator = denominator = clocksPerTick = demisemiquaverPer24Clocks = -1
        key = mode = 'unknown'
        tempo = -1
        try:
            numerator,denominator,clocksPerTick,demisemiquaverPer24Clocks,key,mode,tempo = parse(midi_folder + f,numerator,denominator,clocksPerTick,demisemiquaverPer24Clocks,key,mode,tempo)
        except (UnicodeDecodeError, KeySignatureError, EOFError) as e:
            continue

        print(f'{numerator},{denominator},{clocksPerTick},{demisemiquaverPer24Clocks},{key},{mode},{tempo}')
        df_tmp = pd.DataFrame([[f.lower(),numerator,denominator,clocksPerTick,demisemiquaverPer24Clocks,key,mode,tempo ]], columns=['filename', 'numerator', 'denominator', 'clocksPerTick', 'demisemiquaverPer24Clocks', 'key', 'mode', 'tempo'])
        df_midi = pd.concat([df_midi, df_tmp])
    return df_midi


def get_scales_one_hots(df_merged: pd.DataFrame):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder

    scalers = []
    columns = ['numerator', 'denominator','clocksPerTick', 'demisemiquaverPer24Clocks', 'tempo']
    for col in columns:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_merged[col].values.reshape(-1, 1))
        tmp_df = pd.DataFrame(scaled_features, index=df_merged.index, columns=[col])
        df_merged[col] = tmp_df[col]
        scalers.append(scaler)


    onehot_encoders = []
    columns = ['key', 'mode']
    for col in columns:
        onehot_encoder = OneHotEncoder(sparse=False)
        scaled_features = onehot_encoder.fit_transform(df_merged[col].values.reshape(-1, 1))
        tmp_df = pd.DataFrame(scaled_features, index=df_merged.index)
        tmp_df.columns = onehot_encoder.get_feature_names([col])
        df_merged.drop([col] ,axis=1, inplace=True)
        df_merged = pd.concat([df_merged, tmp_df ], axis=1)
        onehot_encoders.append(onehot_encoder)
    return scalers, onehot_encoders
