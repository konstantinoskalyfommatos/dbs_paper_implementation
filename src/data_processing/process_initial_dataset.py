import os
import json
import pandas as pd
import numpy as np


def create_new_train_data_point(orig_data: pd.Series, _synthetic_name: str, ) -> dict:
    res = {}
    for name in orig_data.index:
        res[name] = orig_data[name]

    res['Synthetic Name'] = _synthetic_name
    res['synthetic ref'] = res['ref'].replace(res['Original Name'], res['Synthetic Name'])
    res['synthetic orig_mr'] = res['orig_mr'].replace(res['Original Name'], res['Synthetic Name'])
    return res

def create_new_test_data_point(orig_data: pd.Series, _synthetic_name: str, ) -> dict:
    res = {}
    for name in orig_data.index:
        res[name] = orig_data[name]

    res['Synthetic Name'] = _synthetic_name
    res['synthetic ref'] = res['references'].replace(res['Original Name'], res['Synthetic Name'])
    res['synthetic orig_mr'] = res['meaning_representation'].replace(res['Original Name'], res['Synthetic Name'])
    return res

DATA_DIR = os.path.join(
    os.getcwd(),
    "data/"
)
DATASET_DIR = os.path.join(
    os.getcwd(),
    "data/downloaded_e2e_dataset"
)

with open(os.path.join(DATA_DIR, "synthetic_names.txt"), "r") as f:
    synthetic_names = f.read().splitlines()


def process_dataset(train_or_test: str, data_df: pd.DataFrame, synthetic_names: list,  _size = 5) -> None:
    if train_or_test == "train":
        data_df.drop(["mr"], inplace=True, axis=1)
        data_df['Original Name'] = data_df['orig_mr'].apply(lambda x: x.split('name[')[1].split(']')[0])

        data_df = data_df[
            data_df[['Original Name', 'ref']].apply(lambda row: row['Original Name'] in row['ref'], axis=1)].reset_index(
            drop=True)
        names_dict = {}

        unique_names = np.unique(data_df['Original Name'])
        for unique_name in unique_names:
            names_dict[unique_name] = list(data_df[data_df['Original Name'] == unique_name].index)

        new_dataset = []
        for idx, synthetic_name in enumerate(synthetic_names):
            try:
                name_choices = [_name for _name in names_dict.keys() if len(names_dict[_name]) >= _size]
                if len(name_choices) == 0:
                    _size = _size - 1
                    name_choices = [_name for _name in names_dict.keys() if len(names_dict[_name]) == _size]

                name = np.random.choice(name_choices)
                indices = np.random.choice(names_dict[name], size=_size, replace=False)
                for index in indices:
                    new_dataset.append(create_new_train_data_point(orig_data=data_df.iloc[index],
                                                             _synthetic_name=synthetic_name,
                                                             ))

                    names_dict[name].remove(index)
                # if idx % 10 == 0:
                #     print(len(new_dataset) / len(data_df))
            except Exception as e:
                print(e)
                pass

        pd.DataFrame(new_dataset).to_csv(os.path.join(DATA_DIR, f"new_{train_or_test}.csv"), index=False)

    if train_or_test == "test":
        data_df = data_df[data_df['meaning_representation'].apply(lambda x: 'name' in str(x))].reset_index(
            drop=True)
        data_df['Original Name'] = data_df['meaning_representation'].apply(lambda x: x.split('name[')[1].split(']')[0])

        data_df = data_df.explode('references').reset_index(drop=True)
        data_df = data_df[
            data_df[['Original Name', 'references']].apply(lambda row: row['Original Name'] in row['references'], axis=1)].reset_index(
            drop=True)
        names_dict = {}

        unique_names = np.unique(data_df['Original Name'])
        for unique_name in unique_names:
            names_dict[unique_name] = list(data_df[data_df['Original Name'] == unique_name].index)

        new_dataset = []
        for idx, synthetic_name in enumerate(synthetic_names):
            try:
                name_choices = [_name for _name in names_dict.keys() if len(names_dict[_name]) >= _size]
                if len(name_choices) == 0:
                    _size = _size - 1
                    name_choices = [_name for _name in names_dict.keys() if len(names_dict[_name]) == _size]

                name = np.random.choice(name_choices)
                indices = np.random.choice(names_dict[name], size=_size, replace=False)
                for index in indices:
                    new_dataset.append(create_new_test_data_point(orig_data=data_df.iloc[index],
                                                             _synthetic_name=synthetic_name,
                                                             ))

                    names_dict[name].remove(index)
            except Exception as e:
                print(e)
                pass

        pd.DataFrame(new_dataset).to_csv(os.path.join(DATA_DIR, f"new_{train_or_test}.csv"), index=False)

        unique_names = np.unique(data_df['Original Name'])
        for unique_name in unique_names:
            names_dict[unique_name] = list(data_df[data_df['Original Name'] == unique_name].index)



if __name__ == "__main__":
    train_data_df = None
    test_data_df = None

    for filename in os.listdir(DATASET_DIR):
        filepath = os.path.join(DATASET_DIR, filename)
        if not filename.endswith(".json"):
            continue
        with open(filepath, "r") as f:
            json_file = json.load(f)
        if json_file.get(
                "url") == "https://github.com/tuetschek/e2e-cleaning/raw/master/cleaned-data/train-fixed.no-ol.csv" :
            train_data_df = pd.read_csv(filepath.replace(".json", ""))

        if json_file.get(
                "url") == "https://raw.githubusercontent.com/jordiclive/GEM_datasets/main/e2e/test.json":
            with open(filepath.replace(".json", ""), "r") as f:
                test_json_file = json.load(f)
            test_data_df = pd.DataFrame(test_json_file['test'])

    test_train_split_idx = int((len(test_data_df)/len(train_data_df))*len(synthetic_names))
    process_dataset(train_or_test='train', data_df=train_data_df, synthetic_names = synthetic_names[test_train_split_idx:])
    process_dataset(train_or_test='test', data_df=test_data_df, synthetic_names = synthetic_names[:test_train_split_idx])
