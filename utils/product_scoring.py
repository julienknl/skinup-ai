from pathlib import Path
import pandas as pd
import json

with open("config/config.json", "r") as config:
    CONFIG = json.load(config)

products = pd.read_csv(CONFIG.get("product_path"))
ingredient_knowledge = pd.read_csv(CONFIG.get("ingredient_knowledge_path"))

def __clean_ingredients(ingredients):
    with open(CONFIG.get("aliases"), "r", encoding="utf-8") as file:
        aliases = json.load(file)

    for key in aliases.keys():
        ingredients = ingredients.lower().replace(key, aliases.get(key))
    
    return ingredients

def get_top_products(skin_condition):
    
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
                              "matched_ingredients" : matched_ing,
                              "score" : matched_rows["scores"].sum()})

    top_products = sorted(score_results, 
                          key=lambda product: product["score"],
                          reverse=True)[:CONFIG.get("num_top_products")]

    unique_products = []

    for product in top_products:
        if not any(p["product_name"] == product['product_name'] for p in unique_products):
            unique_products.append(product)
    
    return unique_products


if __name__ == "__main__":
    top_products = get_top_products("acne")
    print(top_products)
