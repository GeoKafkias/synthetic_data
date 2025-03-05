from dataclasses import dataclass, field

from adalflow import (
    DataClass,
    DataClassParser,
    Generator,
    ModelClient,
    Parameter,
    ParameterType,
)

simple_template = """
<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""

BASE_PROMPT = """"""


@dataclass
class ArtificialData(DataClass):
    sentence: str = field(
        metadata={
            "desc": "A sentence that refers to a person's feedback after a visit to the branch of a bank in the UK."
        }
    )
    answer: int = field(metadata={"desc": "The answer to the question"})

    __output_fields__ = ["sentence", "answer"]


output_parser = DataClassParser(
    data_class=ArtificialData, return_data_class=True, format_type="json"
)


class LLMGenerator:
    def __init__(self, model_client: ModelClient, model_kwargs: dict, prompt: str = ""):
        super().__init__()

        system_prompt = Parameter(
            data=prompt,
            role_desc="To give task instructions to the language model",
            param_type=ParameterType.PROMPT,
            requires_opt=False,
        )

        prompt_kwargs = {
            "system_prompt": system_prompt,
            "output_format_str": output_parser.get_output_format_str(),
        }

        self.llm_producer = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=simple_template,
            prompt_kwargs=prompt_kwargs,
            use_cache=False,
            output_processors=output_parser,
        )


if __name__ == "__main__":
    import os

    from adalflow import GoogleGenAIClient, setup_env
    from adalflow.core.model_client import ModelType
    from google import genai

    class GoogleGenAINewClient(ModelClient):
        def __init__(self):
            super().__init__()
            self.sync_client = self.init_sync_client()

        def init_sync_client(self) -> genai.Client:
            return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        def convert_inputs_to_api_kwargs(
            self, input=None, model_kwargs=..., model_type=ModelType.UNDEFINED
        ):
            return super().convert_inputs_to_api_kwargs(input, model_kwargs, model_type)

        def call(self, api_kwargs=..., model_type=ModelType.LLM):
            return self.sync_client.models.generate_content()

    setup_env()
    google_api = {
        "model_client": GoogleGenAIClient(),
        "model_kwargs": {"model": "gemini-2.0-flash", "temperature": 0.7, "top_p": 1},
    }
