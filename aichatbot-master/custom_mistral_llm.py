from langchain_core.language_models import LLM
from typing import List
import requests
import os
from typing import ClassVar

class MistralChatLLM(LLM):
    model_name: ClassVar[str] = "mistralai/Mistral-7B-Instruct-v0.3"
    api_url: ClassVar[str] = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    # Replace hardcoded token with environment variable
    token: ClassVar[str] = os.getenv("HF_TOKEN")


    def _call(self, prompt: str, stop: List[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512}
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()[0]["generated_text"]

    @property
    def _llm_type(self) -> str:
        return "custom_mistral"
