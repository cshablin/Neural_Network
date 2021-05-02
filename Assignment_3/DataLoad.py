import pandas as pd


def load_lyrics(csv_path: str):
    df = pd.read_csv(csv_path, sep='\n', header=None)
    res = df.iloc[:, 0].str.rstrip(r'&, ').str.extract(r'([^,]+),([^,]+),(.+)')
    res.columns = ['artist', 'title', 'lyrics']
    return res
