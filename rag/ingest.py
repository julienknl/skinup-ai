import chromadb
import uuid
import pandas as pd
import json

with open("config/config.json", "r") as config:
    CONFIG = json.load(config)

class IngestKnowledge:

    def __init__(self, ingredients):
        self.ingredients = ingredients
        # self.client = chromadb.PersistentClient(path="./chroma_db")

    def start_ingestion(self):
        ingredients = self.ingredients
        documents = []

        for _, ingredient in ingredients.iterrows():

            text_to_embed = f"""
            Ingredient: {ingredient["ingredient"]}
            Condition: {ingredient["condition"]}
            Role: {ingredient["recommendation_role"]}
            Effect: {ingredient["effect"]}
            Evidence: {ingredient["evidence_text"]}
            """

            metadata = {
                "ingredient": ingredient["ingredient"],
                "condition": ingredient["condition"],
                "recommendation_role": ingredient["recommendation_role"],
                "evidence_strength": ingredient["evidence_strength"],
                "evidence_type": ingredient["evidence_type"],
                "source_title": ingredient["source_title"],
                "source_url": ingredient["source_url"]
            }

            documents.append({
                "id": uuid.uuid4(),
                "content" : text_to_embed,
                "metadata" : metadata
            })

        print(documents[:2])

        # collection = self.client.get_or_create_collection(name="ingredient_knowledge")

# Testing purposes
if __name__ == "__main__":

    ingredients = pd.read_csv(CONFIG.get("ingredient_knowledge_path"))
    ingestion = IngestKnowledge(ingredients)
    ingestion.start_ingestion()
