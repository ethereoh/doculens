"Using SambaNova endpoint"

import os
import openai
from doculens.base.agent import BaseAgent

SAMBANOVA_ENDPOINT = "https://api.sambanova.ai/v1"

class OpenAIAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.model_name or "Meta-Llama-3.1-8B-Instruct"
        self.client = openai.OpenAI(
            api_key=os.getenv("SAMBANOVA_API_KEY"), base_url=SAMBANOVA_ENDPOINT
        )
        self.system_prompt = self._set_system_prompt(self.config.system_prompt)

    def invoke(self, payload, prettify: bool = True):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": payload},
            ],
            temperature=0.1,
            top_p=0.1,
        )

        if prettify:
            response = self.prettify(response)

        return response

    def prettify(self, payload):
        response = payload.choices[0].message.content
        return response

    def _set_system_prompt(self, payload):
        return (
            payload
            or "Hãy đóng vai là một cố vấn đề về luật, nhiệm vụ của bạn là trả lời cho người dùng về các vấn đề liên quan đến luật hình sự, dân sự, lao động, v.v. Bạn có thể hỏi về các vấn đề như quy định về tội phạm, quy định về hợp đồng, quy định về lao động, v.v. Hãy trả lời các câu hỏi một cách rõ ràng và cụ thể."
        )
