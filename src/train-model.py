"""Treniraj i sačuvaj finalni model za klasifikaciju proizvoda."""

import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


# Putanje se računaju u odnosu na projekat, pa skripta radi iz bilo kog direktorijuma.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "IMLP6_TASK_03-products.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "product-category-linear-svc.joblib"
METRICS_PATH = MODEL_DIR / "product-category-linear-svc-metrics.json"
RANDOM_SEED = 42

# Ove kolone se čuvaju u train/test fajlovima za kasnije modeliranje.
FEATURE_COLUMNS = [
    "product_title_clean",
    "merchant_id",
    "number_of_views",
    "merchant_rating",
    "listing_year",
    "listing_month",
    "listing_dayofweek",
    "number_of_views_missing",
    "merchant_rating_missing",
    "listing_date_missing",
]
TARGET_COLUMN = "category_label"
NUMERIC_COLUMNS = [
    "number_of_views",
    "merchant_rating",
    "listing_year",
    "listing_month",
    "listing_dayofweek",
    "number_of_views_missing",
    "merchant_rating_missing",
    "listing_date_missing",
]
CATEGORICAL_COLUMNS = ["merchant_id"]


def normalize_column_name(column_name: str) -> str:
    """Pretvori naziv kolone u stabilan snake_case oblik."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", column_name.strip().lower())
    return normalized.strip("_")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Učitaj sirovi CSV i standardizuj nazive kolona."""
    if not path.exists():
        raise FileNotFoundError(f"CSV fajl nije pronađen: {path}")

    data = pd.read_csv(path)
    data.columns = [normalize_column_name(column) for column in data.columns]
    return data


def clean_data(data_raw: pd.DataFrame) -> pd.DataFrame:
    """Očisti tekst, brojeve, datume i ponovljene ili konfliktne proizvode."""
    data = data_raw.copy()

    # Standardizujemo tekst i prazne stringove pretvaramo u nedostajuće vrednosti.
    text_columns = ["product_title", "category_label", "product_code"]
    for column in text_columns:
        data[column] = (
            data[column]
            .astype("string")
            .str.normalize("NFKC")
            .str.strip()
            .replace("", pd.NA)
        )

    # Redovi bez naslova ili ciljne kategorije ne mogu se koristiti za učenje.
    data = data.dropna(subset=["product_title", TARGET_COLUMN]).copy()

    # Spajamo različite zapise koji predstavljaju istu kategoriju.
    category_aliases = {
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones",
        "fridge": "Fridges",
    }
    data[TARGET_COLUMN] = data[TARGET_COLUMN].replace(category_aliases)

    # Normalizovan naslov služi kao tekstualna karakteristika i ključ za duplikate.
    data["product_title_clean"] = (
        data["product_title"]
        .str.lower()
        .str.replace(r"[\W_]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    data = data.loc[data["product_title_clean"].ne("")].copy()

    # Numeričke kolone pretvaramo u brojeve, označavamo praznine i popunjavamo ih.
    data["number_of_views"] = pd.to_numeric(data["number_of_views"], errors="coerce")
    data["merchant_rating"] = pd.to_numeric(data["merchant_rating"], errors="coerce")
    data["number_of_views"] = data["number_of_views"].clip(lower=0)
    data.loc[~data["merchant_rating"].between(1, 5), "merchant_rating"] = np.nan
    data["number_of_views_missing"] = data["number_of_views"].isna().astype("int8")
    data["merchant_rating_missing"] = data["merchant_rating"].isna().astype("int8")
    data["number_of_views"] = data["number_of_views"].fillna(
        data["number_of_views"].median()
    )
    data["merchant_rating"] = data["merchant_rating"].fillna(
        data["merchant_rating"].median()
    )

    # Datum pretvaramo u vremenske karakteristike, uz indikator neuspelog parsiranja.
    data["listing_date"] = pd.to_datetime(
        data["listing_date"], format="%m/%d/%Y", errors="coerce"
    )
    data["listing_date_missing"] = data["listing_date"].isna().astype("int8")
    median_listing_date = data["listing_date"].dropna().median()
    data["listing_date"] = data["listing_date"].fillna(median_listing_date)
    data["listing_year"] = data["listing_date"].dt.year
    data["listing_month"] = data["listing_date"].dt.month
    data["listing_dayofweek"] = data["listing_date"].dt.dayofweek
    data["product_code"] = data["product_code"].fillna("unknown")

    # Uklanjamo naslove povezane sa više kategorija jer imaju konfliktne oznake.
    labels_per_title = data.groupby("product_title_clean")[TARGET_COLUMN].nunique()
    ambiguous_titles = labels_per_title[labels_per_title.gt(1)].index
    data = data.loc[~data["product_title_clean"].isin(ambiguous_titles)].copy()

    # Od ponovljenih naslova zadržavamo zapis sa najvećim brojem pregleda.
    data_clean = (
        data.sort_values("number_of_views", ascending=False)
        .drop_duplicates("product_title_clean", keep="first")
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    if data_clean.isna().sum().sum() != 0:
        raise ValueError("Očišćeni podaci i dalje sadrže nedostajuće vrednosti.")
    if not data_clean["product_title_clean"].is_unique:
        raise ValueError("Očišćeni podaci i dalje sadrže duplirane naslove.")

    return data_clean


def split_data(data_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Napravi stratifikovane train/test skupove bez preklapanja naslova."""
    model_data = data_clean[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    model_data["merchant_id"] = model_data["merchant_id"].astype("string")

    train_data, test_data = train_test_split(
        model_data,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=model_data[TARGET_COLUMN],
    )

    train_titles = set(train_data["product_title_clean"])
    test_titles = set(test_data["product_title_clean"])
    if not train_titles.isdisjoint(test_titles):
        raise ValueError("Train i test skup sadrže iste normalizovane naslove.")

    return train_data, test_data


def save_datasets(
    data_clean: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Sačuvaj očišćene podatke i pripremljene skupove kao CSV fajlove."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_clean.to_csv(output_dir / "products_clean.csv", index=False)
    train_data.to_csv(output_dir / "train.csv", index=False)
    test_data.to_csv(output_dir / "test.csv", index=False)


def load_prepared_datasets(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Učitaj ranije pripremljene train i test skupove."""
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    missing_paths = [path for path in (train_path, test_path) if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Nedostaju pripremljeni skupovi: {missing}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    for path, dataset in ((train_path, train_data), (test_path, test_data)):
        missing_columns = required_columns.difference(dataset.columns)
        if missing_columns:
            raise ValueError(
                f"{path} nema obavezne kolone: {sorted(missing_columns)}"
            )
        dataset["merchant_id"] = pd.to_numeric(
            dataset["merchant_id"], errors="raise"
        )

    return train_data, test_data


def build_model() -> Pipeline:
    """Napravi najbolje rešenje iz notebook eksperimenata."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "title_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=30_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
                "product_title_clean",
            ),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            (
                "merchant",
                OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC(C=1.0, tol=1e-3)),
        ]
    )


def evaluate_model(
    model: Pipeline, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> dict[str, object]:
    """Treniraj na train skupu i izmeri rezultat na sačuvanom test skupu."""
    model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])
    predictions = model.predict(test_data[FEATURE_COLUMNS])
    metrics = {
        "model": "LinearSVC",
        "accuracy": float(accuracy_score(test_data[TARGET_COLUMN], predictions)),
        "macro_f1": float(
            f1_score(test_data[TARGET_COLUMN], predictions, average="macro")
        ),
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "categories": int(train_data[TARGET_COLUMN].nunique()),
    }
    print(classification_report(test_data[TARGET_COLUMN], predictions, zero_division=0))
    return metrics


def save_final_model(
    model: Pipeline,
    all_data: pd.DataFrame,
    metrics: dict[str, object],
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Ponovo treniraj na svim podacima i sačuvaj model i evaluacione metrike."""
    model.fit(all_data[FEATURE_COLUMNS], all_data[TARGET_COLUMN])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """Evaluiraj najbolje rešenje, treniraj ga na svim podacima i sačuvaj."""
    train_data, test_data = load_prepared_datasets(OUTPUT_DIR)
    evaluation_model = build_model()
    metrics = evaluate_model(evaluation_model, train_data, test_data)

    final_model = build_model()
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    save_final_model(final_model, all_data, metrics, MODEL_PATH, METRICS_PATH)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Finalni model: {MODEL_PATH.resolve()}")
    print(f"Metrike: {METRICS_PATH.resolve()}")


if __name__ == "__main__":
    main()
