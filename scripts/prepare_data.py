import os
import argparse
import logging
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

random_seed = 42  # for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)

_TRAIN_OUTPUT_FILENAME = "train.csv"
_VAL_OUTPUT_FILENAME = "val.csv"
_TST_OUTPUT_FILENAME = "tst.csv"


def is_folder(path: str) -> None:
    if not os.path.exists(path) or not os.path.isdir(path):
        raise Exception(f"Provided path: {path} is not valid. Exiting...")
        import sys

        sys.exit()


def main(args: argparse.Namespace) -> None:
    is_folder(args.output)

    logging.info("Preparing dataset...")

    data = pd.read_csv(args.data_path)

    # remove duplicates
    data = data.drop_duplicates(subset=["abstractText"], keep="first")

    assert data["abstractText"].duplicated().sum() == 0, "Duplicate are still present"
    logging.debug("Removed duplicates.")

    # from the initial analysis (see notebook), 15 labels were found.
    # Therefore, one-hot needs to be performed again
    data["meshroot_labels"] = data["meshroot"].apply(lambda x: eval(x))

    unique_labels = sorted(
        set(
            [
                single_label
                for row_labels in data["meshroot_labels"]
                for single_label in row_labels
            ]
        )
    )

    data = data.drop(columns=data.columns[6:-1])  # drop prev one-hot
    # perform one-hot again
    col_names = {label.split("[")[-1][:-1]: label for label in unique_labels}

    for col in sorted(col_names.keys()):
        data[col] = 0

    # one-hot
    for index, row in data.iterrows():
        for category in row["meshroot_labels"]:
            label = category.split("[")[-1][:-1]
            if label in data.columns:
                data.at[index, label] = 1

    logging.debug("One-hot performed again.")

    # Shufffle
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    logging.debug("Dataset shuffled.")

    X = data.iloc[:, :7]
    y = data.iloc[:, 7:]

    # Split into train/val/test
    temp_pct = args.vp + args.tp

    # handle imbalancement better than sklearn stratify
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=temp_pct, random_state=random_seed
    )

    # Perform the first split
    for train_index, test_index in msss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # split again between val and test

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args.tp / temp_pct, random_state=random_seed
    )

    for val_index, test_index in msss.split(X_test, y_test):
        X_val, X_test = X_test.iloc[val_index], X_test.iloc[test_index]
        y_val, y_test = y_test.iloc[val_index], y_test.iloc[test_index]

    train_combined = pd.concat([X_train, y_train], axis=1)
    val_combined = pd.concat([X_val, y_val], axis=1)
    tst_combined = pd.concat([X_test, y_test], axis=1)

    #  Create a new column in which all the separate one-hot columns get combined.
    #  This will help later when loading the data.
    train_combined["labels"] = list(train_combined.iloc[:, 7:].values)
    val_combined["labels"] = list(val_combined.iloc[:, 7:].values)
    tst_combined["labels"] = list(tst_combined.iloc[:, 7:].values)

    logging.debug("Individual binary label columns combined into list.")

    train_combined.to_csv(
        os.path.join(args.output, _TRAIN_OUTPUT_FILENAME), index=False
    )
    val_combined.to_csv(os.path.join(args.output, _VAL_OUTPUT_FILENAME), index=False)
    tst_combined.to_csv(os.path.join(args.output, _TST_OUTPUT_FILENAME), index=False)
    logging.debug(f"Split into train/val/test and saved to folder: {args.output}")

    logging.info("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train/val/test split")

    parser.add_argument(
        "-d",
        "--data_path",
        dest="data_path",
        type=str,
        default="hf://datasets/owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH/PubMed Multi Label Text Classification Dataset Processed.csv",
        help="Specify the url fo the pandas dataframe.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="Specify the path to the folder for the train/val/tst splits.",
    )
    parser.add_argument(
        "-vp",
        "--val_pct",
        dest="vp",
        required=False,
        default=0.1,
        type=float,
        help="validation percentage, default is 10%",
    )
    parser.add_argument(
        "-tp",
        "--tst_pct",
        dest="tp",
        required=False,
        default=0.05,
        type=float,
        help="test percentage, default is 5%",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        required=False,
        default=False,
        action="store_true",
        help="increase output verbosity",
    )
    args = parser.parse_args()

    level = "DEBUG" if args.verbose else "INFO"
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    main(args)
