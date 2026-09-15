from model.skin_analyser.model import SkinAnalyser
from utils.image_verifier import is_validate
from utils.product_scoring import get_top_products
from rag.vector_database import Database
from rag.retrieve import RetrieveKnowledge
from model.llm.model import LLM
import json
import logging

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

with open("config/config.json", "r") as config:
    CONFIG = json.load(config)

def recommend_products(image):
    
    print("Validating clarity of the image...")
    if is_validate(raw_img=image):

        # Get skin condition
        skin_condition = _get_skin_condition(image=image)
        
        # Get top products recommendation based from top scoring products
        top_products_dict, ingredients, products_context = _get_top_products(skin_condition=skin_condition)

        # Retrieve the purpose of each ingredients identified for skin care
        retrieve_results = _retrieve_ingredients_purpose(skin_condition=skin_condition, ingredients=ingredients)

        # Build the context
        print("Building context..")
        context = {
            "skin_condition" : skin_condition,
            "products_context" : products_context,
            "total_products_count" : len(top_products_dict),
            "documents" : retrieve_results
        }

        print("Context build. Give a few momemnts to the large language model to write the explanation in user friendly..")
        llm = LLM()
        result = llm.provide_report(context=context)

        return result[0]["generated_text"]


def _get_skin_condition(image):
    print("Image clear.. Ready to analyse")
    skin_analyser = SkinAnalyser()
    results = skin_analyser.analyse(image=image)

    skin_condition_high = max(
        results,
        key=lambda result: result["score"]
    )

    skin_condition = skin_condition_high["label"] if skin_condition_high["score"] >= CONFIG.get("skin_condition_identifier_threshold") else "normal"

    print(f"Skin analysis completed with skin condition {skin_condition}. Getting top products recommendation based on the analysed skin condition..")

    return skin_condition

def _get_top_products(skin_condition):
    top_products_dict = get_top_products(skin_condition=skin_condition)
    ingredients = []
    
    for product in top_products_dict:
        ingredients.extend(product["matched_ingredients"])
    
        ingredients = list(set(ingredients))
    
        print(f"Top recommended products retrieved.")
    
        products_context = ""
    
        for index, product in enumerate(top_products_dict, start=1):
            products_context += f"""
            Product {index}: {product["product_name"]}
            Score: {product["score"]}
            Matched ingredients: {", ".join(product["matched_ingredients"])}
            """

    return top_products_dict, ingredients, products_context

def _retrieve_ingredients_purpose(skin_condition, ingredients):
    print("Now retrieving the context of the ingredients found on those products..")
    query = f"Find ingredients that help with skin conditions {skin_condition}" if skin_condition != "normal" else f"Find ingredients that help to maintain normal skin"
    database = Database()
    retrieve = RetrieveKnowledge(database=database)
    retrieve_results = retrieve.retrieve(queries=[query], n_results=len(ingredients), where={"$and" : [
        {"ingredient" : {"$in" : ingredients}},
        {"condition" : skin_condition}
    ]})["documents"][0]

    print(f"Retrieved context of the identified ingredients.")

    return retrieve_results

