"""Interaktivno testiraj sačuvani model za klasifikaciju proizvoda."""

from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd


# Putanja je relativna u odnosu na projekat, pa skripta radi iz bilo kog foldera.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "product-category-linear-svc.joblib"
EXIT_COMMAND = "exit"


def normalize_title(title: str) -> str:
    """Normalizuj naslov na isti način kao tokom pripreme podataka."""
    normalized = title.strip().lower()
    normalized = re.sub(r"[\W_]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def parse_optional_number(
    value: str, field_name: str, minimum: float, maximum: float | None = None
) -> tuple[float, int]:
    """Pretvori opcioni unos u broj i vrati indikator nedostajuće vrednosti."""
    if not value.strip():
        return np.nan, 1

    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{field_name} mora biti broj ili ostavljen prazan.") from error

    if number < minimum or (maximum is not None and number > maximum):
        allowed_range = f"{minimum}-{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{field_name} mora biti u opsegu {allowed_range}.")

    return number, 0


def create_product_input(
    title: str,
    number_of_views: str,
    merchant_rating: str,
) -> pd.DataFrame:
    """Napravi jedan red sa svim karakteristikama koje model očekuje."""
    clean_title = normalize_title(title)
    if not clean_title:
        raise ValueError("Naslov proizvoda ne sme biti prazan.")

    views, views_missing = parse_optional_number(
        number_of_views, "Broj pregleda", minimum=0
    )
    rating, rating_missing = parse_optional_number(
        merchant_rating, "Ocena trgovca", minimum=1, maximum=5
    )

    # Nepoznat ID trgovca One-Hot encoder bezbedno ignoriše, a datum popunjava imputer.
    return pd.DataFrame(
        [
            {
                "product_title_clean": clean_title,
                "merchant_id": -1,
                "number_of_views": views,
                "merchant_rating": rating,
                "listing_year": np.nan,
                "listing_month": np.nan,
                "listing_dayofweek": np.nan,
                "number_of_views_missing": views_missing,
                "merchant_rating_missing": rating_missing,
                "listing_date_missing": 1,
            }
        ]
    )


def prompt_value(message: str) -> str:
    """Učitaj korisnički unos i zaustavi program kada korisnik unese 'exit'."""
    value = input(message).strip()
    if value.lower() == EXIT_COMMAND:
        print("Izlazak iz programa...")
        raise SystemExit(0)
    return value


def main() -> None:
    """Učitaj model i omogućavaj predikcije dok korisnik ne prekine program."""
    if not MODEL_PATH.exists():
        print(f"Model nije pronađen: {MODEL_PATH}")
        print("Prvo pokrenite: python src/train-model.py")
        raise SystemExit(1)

    # Joblib fajl sadrži i preprocessing i istrenirani LinearSVC klasifikator.
    model = joblib.load(MODEL_PATH)
    print("Model je uspesno ucitan!")
    print("Unesite 'exit' u bilo kom trenutku za prekid.\n")

    while True:
        try:
            title = prompt_value("Naslov proizvoda: ")
            views = prompt_value("Broj pregleda (Enter ako nije poznat): ")
            rating = prompt_value("Ocena trgovca 1-5 (Enter ako nije poznata): ")

            # Pipeline sam primenjuje TF-IDF, skaliranje i One-Hot kodiranje.
            product = create_product_input(title, views, rating)
            prediction = model.predict(product)[0]
            print(f"\nPredvidjena kategorija: {prediction}")
            print("-" * 50)
        except ValueError as error:
            print(f"Neispravan unos: {error}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nIzlazak iz programa...")
            return


if __name__ == "__main__":
    try:
        main()
    except SystemExit as error:
        sys.exit(error.code)