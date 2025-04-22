from typing import Any, Dict, List, Optional
import google.generativeai as genai
import typing_extensions as typing
import json

class GeminiConnector:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def typeddict_to_json_schema(self, schema_class):
        properties = {}
        for field_name, field_type in schema_class.__annotations__.items():
            if field_type == bool:
                field_schema = {"type": "boolean"}
            elif field_type == str:
                field_schema = {"type": "string"}
            elif field_type == int:
                field_schema = {"type": "integer"}
            elif field_type == list:
                field_schema = {"type": "array", "items": {"type": "string"}}
            else:
                raise ValueError(f"Unsupported type: {field_type}")
            properties[field_name] = field_schema

        return {
            "type": "object",
            "required": list(properties.keys()),
            "properties": properties
        }

    def call_with_structured_output(
            self,
            prompt: str,
            schema,
            images: Optional[List[str]] = None,
            system_prompt: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Call Gemini with function calling capabilities
            """
            if system_prompt:
                self.model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system_prompt)
            
            if isinstance(schema, dict):
                json_schema = schema
            else:
                json_schema = self.typeddict_to_json_schema(schema)
            
            if images:
                input = [{'mime_type':'image/jpeg', 'data': image} for image in images] + [prompt]
            else:
                input = prompt

            res = self.model.generate_content(
                input,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", 
                    response_schema=json_schema,
                )
            )

            return json.loads(res.candidates[0].content.parts[0].text)