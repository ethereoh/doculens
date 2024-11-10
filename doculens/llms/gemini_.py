"Using Gemini's API"

import os

import google.generativeai as genai

from doculens.base.agent import BaseAgent


class GeminiAgent(BaseAgent): 
    
    def __init__(self, config): 
        self.config = config

        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model_name = self.config.model_name or 'gemini-1.5-flash'

        self.model = genai.GenerativeModel(model_name=self.model_namemodel_name, 
                                           system_instruction=self._set_system_prompt(self.config.system_prompt))
    def invoke(self, payload):
        return super().invoke(payload)
    
    def prettify(self, payload) -> str:
        return payload.text
    

    def _setup_resp(self): 
        return genai.types.GenerationConfig(
            temperature=self.config.temperature, 
            top_k=self.config.top_k
        )
    
    def _set_system_prompt(self, payload):
        return (
            payload
            or "Hãy đóng vai là một cố vấn đề về luật, nhiệm vụ của bạn là trả lời cho người dùng về các vấn đề liên quan đến luật hình sự, dân sự, lao động, v.v. Bạn có thể hỏi về các vấn đề như quy định về tội phạm, quy định về hợp đồng, quy định về lao động, v.v. Hãy trả lời các câu hỏi một cách rõ ràng và cụ thể."
        )
