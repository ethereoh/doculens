"Using Ollama Agent"

import ollama

from doculens.base.agent import BaseAgent


class OllamaAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self.username = "minhleduc0210"
        self.model_name = config.model_name or "qwen2.5:3b"
        self.system_prompt = self._set_system_prompt(self.config.system_prompt)

    def _init_model(self):
        "If model exist, use, otherwise pull"
        ...

    def invoke(self, payload, prettify: bool = True):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": payload},
            ],
        )

        if prettify:
            response = self.prettify(response)

        return response

    def prettify(self, payload):
        response = payload["message"]["content"]
        return response

    def _check_model(self, model_name):
        try:
            "Check if model exists"
            return ollama.show(model=model_name)
        except:
            return

    def show_model(self) -> dict:
        return ollama.list()

    def _set_system_prompt(self, payload):
        return (
            payload
            or "Hãy đóng vai là một cố vấn đề về luật, nhiệm vụ của bạn là trả lời cho người dùng về các vấn đề liên quan đến luật hình sự, dân sự, lao động, v.v. Bạn có thể hỏi về các vấn đề như quy định về tội phạm, quy định về hợp đồng, quy định về lao động, v.v. Hãy trả lời các câu hỏi một cách rõ ràng và cụ thể."
        )
