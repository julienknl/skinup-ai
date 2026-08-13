from pathlib import Path
import pandas as pd
import json

CONFIG = {
    "product_path" : "data/skincare_products/products.csv",
    "ingredient_knowledge_path" : "data/skincare_products/cleaned_common_ingredients_knowledge_v3.csv",
    "aliases" : "data/skincare_products/aliases.json",
    "recommended_role_scores" : {
        "beneficial" : 2,
        "supportive" : 1,
        "neutral" : 0,
        "caution" : -1,
        "avoid" : -2
    }
}

def __clean_ingredients(ingredients):
    with open(CONFIG.get("aliases"), "r", encoding="utf-8") as file:
        aliases = json.load(file)

    for key in aliases.keys():
        ingredients = ingredients.lower().replace(key, aliases.get(key))
    
    return ingredients

def get_top_products(skin_condition):
    products = pd.read_csv(CONFIG.get("product_path"))
    ingredient_knowledge = pd.read_csv(CONFIG.get("ingredient_knowledge_path"))

    # Select ingredients useful for the identified skin condition
    selected_ingredients = ingredient_knowledge[ingredient_knowledge["condition"] == skin_condition]
    selected_ingredients["scores"] = selected_ingredients["recommendation_role"].map(CONFIG.get("recommended_role_scores"))

    score_results = []

    for _, product in products.iterrows():
        clean_ingredients = __clean_ingredients(product["ingredients"])
        ing_array = clean_ingredients.split(",")
        ing_array = [ingredient.strip() for ingredient in ing_array]
        selected_ingredients_set = set(selected_ingredients["ingredient"].str.strip().str.lower())
        matched_ing = [ingredient for ingredient in ing_array if ingredient in selected_ingredients_set]

        matched_rows = selected_ingredients[selected_ingredients["ingredient"].isin(matched_ing)]
        score_results.append({"product_name" : product["product_name"],
                              "score" : matched_rows["scores"].sum()})

    top_products = sorted(score_results, 
                          key=lambda product: product["score"],
                          reverse=True)[:5]

    return top_products


if __name__ == "__main__":
    get_top_products("oily")
