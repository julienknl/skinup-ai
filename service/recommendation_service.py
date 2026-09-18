from model.skin_analyser.model import SkinAnalyser
from utils.image_verifier import is_validate
from utils.product_scoring import get_top_products
from rag.vector_database import Database
from rag.retrieve import RetrieveKnowledge
from model.llm.model import LLM
import json
import logging
import time
import uuid

logger = logging.getLogger(__name__)

with open("config/config.json", "r") as config:
    CONFIG = json.load(config)

def recommend_products(image):
    
    request_id = uuid.uuid4()
    is_validated, validate_compute_time = _validate_image(image=image, request_id=request_id)

    if is_validated:

        # Get skin condition
        skin_condition, skin_condition_compute_time = _get_skin_condition(image=image, request_id=request_id)
        
        # Get top products recommendation based from top scoring products
        top_products_dict, ingredients, products_context, top_products_compute_time = _get_top_products(skin_condition=skin_condition, request_id=request_id)

        # Retrieve the purpose of each ingredients identified for skin care
        retrieve_results, retrieval_compute_time = _retrieve_ingredients_purpose(skin_condition=skin_condition, ingredients=ingredients, request_id=request_id)

        # Build the context
        print("Building context..")
        context = {
            "skin_condition" : skin_condition,
            "products_context" : products_context,
            "total_products_count" : len(top_products_dict),
            "documents" : retrieve_results
        }
        result, llm_inference_time = _llm_response(context=context, request_id=request_id)

        _log(
            request_id=request_id,
            validate_time=validate_compute_time,
            analyser_inference_time=skin_condition_compute_time,
            top_products_time=top_products_compute_time,
            retrieval_time=retrieval_compute_time,
            llm_inference_time=llm_inference_time
        )

        return result[0]["generated_text"]

def _validate_image(image, request_id):
    print("Validating clarity of the image...")
    try:
        logger.info("[%s] Request received for skin analysis and recommendation", request_id)
        is_validated, validate_compute_time = is_validate(raw_img=image)
        return is_validated, validate_compute_time
    except ValueError as e:
        logger.warning("[%s] Validation error: %s", request_id, e)
        raise


def _get_skin_condition(image, request_id):
    try:
        logger.info("[%s] Starting skin analysis", request_id)
        start_time = time.perf_counter()
        skin_analyser = SkinAnalyser()
        results = skin_analyser.analyse(image=image)

        skin_condition_high = max(
            results,
            key=lambda result: result["score"]
        )

        skin_condition = skin_condition_high["label"] if skin_condition_high["score"] >= CONFIG.get("skin_condition_identifier_threshold") else "normal"

        end_time = time.perf_counter()
        compute_time = end_time - start_time

        return skin_condition, compute_time
    except Exception as e:
        logger.warning("[%s] Skin condition analysis error: %s", request_id, e)
        raise

def _get_top_products(skin_condition, request_id):

    try:
        logger.info("[%s] Getting top products from skin condition", request_id)
        start_time = time.perf_counter()
        top_products_dict = get_top_products(skin_condition=skin_condition)
        ingredients = []
        
        for product in top_products_dict:
            ingredients.extend(product["matched_ingredients"])
        
            ingredients = list(set(ingredients))
        
            products_context = ""
        
            for index, product in enumerate(top_products_dict, start=1):
                products_context += f"""
                Product {index}: {product["product_name"]}
                Score: {product["score"]}
                Matched ingredients: {", ".join(product["matched_ingredients"])}
                """
        end_time = time.perf_counter()
        total_time = end_time - start_time

        return top_products_dict, ingredients, products_context, total_time
    
    except Exception as e:
        logger.warning("[%s] Top products retrieval error: %s", request_id, e)
        raise
    

def _retrieve_ingredients_purpose(skin_condition, ingredients, request_id):
    try:
        logger.info("[%s] Retrieval ingredient knowledge started from skin condition", request_id)
        start_time = time.perf_counter()
        query = f"Find ingredients that help with skin conditions {skin_condition}" if skin_condition != "normal" else f"Find ingredients that help to maintain normal skin"
        database = Database()
        retrieve = RetrieveKnowledge(database=database)
        retrieve_results = retrieve.retrieve(queries=[query], n_results=len(ingredients), where={"$and" : [
            {"ingredient" : {"$in" : ingredients}},
            {"condition" : skin_condition}
        ]})["documents"][0]

        print(f"Retrieved context of the identified ingredients.")
        end_time = time.perf_counter()
        total_time = end_time - start_time

        return retrieve_results, total_time
    except ValueError as e:
        logger.warning("[%s] Knowledge retrieval error: %s", request_id, e)
        raise

    

def _llm_response(context, request_id):
    try:
        logger.info("[%s] LLM start building user friendly response from raw context", request_id)
        start_time = time.perf_counter()
        print("Context build. Give a few momemnts to the large language model to write the explanation in user friendly..")
        llm = LLM()
        result = llm.provide_report(context=context)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        return result, total_time
    except Exception as e:
        logger.warning("[%s] LLM inference error: %s", request_id, e)
        raise

def _log(request_id, validate_time, analyser_inference_time, top_products_time, retrieval_time, llm_inference_time):
    logger.info("""[%s] Skin analysis and product recommendation completed | 
    Image validation time %.2f s | 
    Skin analysis inference time %.2f s | 
    Top products fetch time %.2f s | 
    Ingredient knowledge retrieval time %.2f s | 
    LLM inference time %.2f s""", request_id, validate_time, analyser_inference_time, top_products_time, retrieval_time, llm_inference_time)

