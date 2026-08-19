import os
from dotenv import load_dotenv
from transformers import pipeline
load_dotenv()

class LLM:
    def __init__(self, type="text-generation", model=os.getenv("LLM")):
        self.model = model
        self.llm = pipeline(
            type,
            model=self.model,
            device_map="auto"
        )

    def provide_report(self, context):
        prompt = f"""
        Skin condition:
        {context["skin_condition"]}

        Recommended products:
        {context["products_context"]}

        Ingredient evidence:
        {context["documents"]}
        """

        messages = [
            {
                "role" : "system",
                "content" : f"""
                You are a helpful skincare assistant.

                Give a short, user-friendly explanation for each recommended product but before providing the explanation for each
                recommended products, start with a 1 or 2 sentences introduction, introducing the recommended products starting from
                the highest score product to the low one.
                
                For each product:
                - State the product name.
                - Explain in 1-2 sentences why it may be suitable for the identified skin condition based on the evidence.
                - Mention only the most relevant 2-3 ingredients.
                - Use only the provided evidence.
                - Do not invent ingredients or effects.
                - Mention a caution only if it is important.
                - Do not skip any product.
                
                Provide an explanation for all {context["total_products_count"]} products within the range of 1000 tokens or less.
                """
            },
            {
                "role" : "user",
                "content" : prompt
            }
        ]

        response = self.llm(
            messages,
            max_new_tokens=1000,
            do_sample=False,
            return_full_text=False
        )

        return response
