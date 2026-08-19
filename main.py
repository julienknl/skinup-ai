from model.skin_analyser.model import SkinAnalyser
from utils.image_verifier import is_validate
from utils.product_scoring import get_top_products
from rag.vector_database import Database
from rag.retrieve import RetrieveKnowledge
from model.llm.model import LLM
import json
import logging

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def main():
    with open("config/config.json", "r") as config:
        CONFIG = json.load(config)

    print("""**************************************************
            *                                                *
            *                 S K I N U P  A I               *
            *                                                *
            *        Smart Skin Analysis & Recommendation    *
            *                                                *
            *                       Welcome                  *
            **************************************************""")

    # testing/tmp_img/acne3.png
    image_path = input("Enter your image path: ")

    print("Validating clarity of the image...")
    if is_validate(image_path=image_path):
        print("Image clear.. Ready to analyse")
        skin_analyser = SkinAnalyser()
        results = skin_analyser.analyse(image=image_path)

        skin_condition_high = max(
            results,
            key=lambda result: result["score"]
        )

        skin_condition = skin_condition_high["label"] if skin_condition_high["score"] >= CONFIG.get("skin_condition_identifier_threshold") else "normal"

        print(f"Skin analysis completed with skin condition {skin_condition}. Getting top products recommendation based on the analysed skin condition..")
        # Get top products recommendation based from top scoring products
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

        print("Now retrieving the context of the ingredients found on those products..")
        query = f"Find ingredients that help with skin conditions {skin_condition}" if skin_condition != "normal" else f"Find ingredients that help to maintain normal skin"
        database = Database()
        retrieve = RetrieveKnowledge(database=database)
        retrieve_results = retrieve.retrieve(queries=[query], n_results=len(ingredients), where={"$and" : [
            {"ingredient" : {"$in" : ingredients}},
            {"condition" : skin_condition}
        ]})["documents"][0]

        print(f"Retrieved context of the identified ingredients.")
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
        print(result[0]["generated_text"])

        

if __name__ == "__main__":
    main()
