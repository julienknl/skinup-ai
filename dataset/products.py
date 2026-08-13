from pathlib import Path
import pandas as pd
from collections import Counter
import json

with open("config/config.json", "r") as config:
    CONFIG = json.load(config)

def start_preprocessing():
    folder = Path(CONFIG.get("data_path"))

    dataframes = []

    for file in folder.glob("*.csv"):
        df = pd.read_csv(file)
        dataframes.append(df)

    products = pd.concat(dataframes, ignore_index=True)
    products.to_csv("data/skincare_products/products.csv", index=False)

def __get_common_ingredients(dataset):
    ingredients = dataset["ingredients"]
    ing_array = []

    for ingredient in ingredients:
        ing = ingredient.split(",")
        ing_array.extend(ing)

    cleaned_data = [ingredient.strip().lower() for ingredient in ing_array]
    ing_counts = Counter(cleaned_data)
    most_commons_ing = ing_counts.most_common(CONFIG.get("most_common_ing_count"))

    df = pd.DataFrame(most_commons_ing, columns=["ingredients", "count"])
    df.to_csv(CONFIG.get("data_path") + "common_ingredients.csv", index=False)

def __cleaning_common_ingredients(dataset):
    aliases = {
        "water\\aqua\\eau" : "water",
        "aqua/water/eau" : "water",
        "aqua" : "water",
        "aqua (water)" : "water",
        "eau)" : "water",
        "water (aqua)" : "water",
        "water/aqua/eau" : "water",
        "fragrance (parfum)" : "fragrance",
        "parfum (fragrance)" : "fragrance",
        "aloe barbadensis (aloe vera) leaf juice" : "aloe vera",
        "aloe barbadensis leaf juice" : "aloe vera",
        "african shea butter": "shea butter",
        "butyrospermum parkii (shea) butter": "shea butter",
        "titanium dioxide (ci 77891)": "titanium dioxide",
        "tocopherol (multisource)": "tocopherol",
    }

    # Rename some ingredients to represent only one ingredient
    dataset["ingredients"] = dataset["ingredients"].replace(aliases)
    dataset = dataset.groupby("ingredients", as_index=False)["count"].sum()

    # Remove unknown ingredients
    int_values = pd.to_numeric(dataset["ingredients"], errors="coerce").notna()

    dataset = dataset[~int_values]
    dataset.to_csv(CONFIG.get("data_path") + "cleaned_common_ingredients.csv", index=False)


if __name__ == "__main__":

    products_path = Path(CONFIG.get("product_path"))

    print("Verifying if product file already exist..")
    if not products_path.exists():
        print("Product file does not exists..Creating the product file from multiple datasets..")
        start_preprocessing()
        print("Product file created.")

    if products_path.exists():
        print("Product file already exists..Finding the most common ingredient used to build ingredient knowledge base..")
        products = pd.read_csv(products_path)
        __get_common_ingredients(products)
        print("Raw knowledge base created..Cleaning the data..")
        __cleaning_common_ingredients(pd.read_csv(CONFIG.get("data_path") + "common_ingredients.csv"))
        print("Ingredient knowledge base created.")