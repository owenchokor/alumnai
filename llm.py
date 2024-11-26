import json
import os
import boto3
from tqdm import tqdm

class LLM:
    def __init__(self, llm='llama'):
        self.llm = llm
        if llm == 'llama':
            import boto3
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-west-2'
            )
            self.model = "us.meta.llama3-2-11b-instruct-v1:0"
        elif llm == 'gpt':
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self.client = OpenAI(api_key = api_key)
            self.model = "gpt-4o-mini"
        else:
            raise ValueError(f"Unsupported LLM type: {llm}")

    def setResponse(self, prompt):
        if self.llm == 'llama':
            self.prompt = prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": self.prompt},
                    ],
                }
            ]

            try:
                response = self.bedrock.converse(
                    modelId=self.model,
                    messages=messages,
                )
                self.response = response["output"]["message"]["content"][0]["text"]
            except Exception as e:
                raise RuntimeError(f"Error during Llama API call: {e}")
            
        elif self.llm == 'gpt':
            messages = [{
                "role": "system",
                "content": "you are a helpful assistant."
            }, {
                "role": "user",
                "content": prompt
            }]

            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
                self.response = response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"Error during GPT API call: {e}")

    def getResponse(self):
        if not hasattr(self, 'response'):
            raise RuntimeError("Response not set. Call setResponse() first.")
        return self.response

class Embedder:
    def __init__(self):
        self.runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'
        )

    def getRequest(self, request_body):
        response = self.runtime.invoke_model(
            modelId='amazon.titan-embed-text-v1',  # Example Titan embedding model ID; replace with the correct one
            accept='application/json',
            contentType='application/json',
            body=request_body
        )
        response_body = json.loads(response['body'].read())
        embedding_vector = response_body['embedding']
        return embedding_vector