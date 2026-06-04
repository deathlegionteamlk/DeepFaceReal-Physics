
import os
import json
import requests
import time
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime

CREDITS = 'Powered By DeathLegionTeamLK'

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
FALLBACK_MODEL = "microsoft/phi-3-mini-4k-instruct:free"

class ConversationHistory:

    def __init__(self, max_history: int = 20):
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def add_system_prompt(self, prompt: str):

        self.messages = [m for m in self.messages if m['role'] != 'system']
        self.messages.insert(0, {"role": "system", "content": prompt})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def clear(self):
        system_msgs = [m for m in self.messages if m['role'] == 'system']
        self.messages = system_msgs

    def to_dict(self) -> List[Dict[str, str]]:
        return self.messages

class LLMCharacter:

    def __init__(
        self,
        api_key: str = None,
        model: str = DEFAULT_MODEL,
        max_history: int = 20
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model
        self.conversation = ConversationHistory(max_history=max_history)
        self.system_prompt: str = ""
        self.character_name: str = "Assistant"
        self.temperature: float = 0.7
        self.max_tokens: int = 256

    def set_character(self, name: str, system_prompt: str = None):
        self.character_name = name
        if system_prompt:
            self.system_prompt = system_prompt
            self.conversation.add_system_prompt(system_prompt)

    def _make_request(self, messages: List[Dict[str, str]], stream: bool = False) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8081",
            "X-Title": "Real-Time Deepfake System"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }

        response = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            stream=stream,
            timeout=30
        )
        return response

    def send_message(self, user_message: str) -> str:
        self.conversation.add_message("user", user_message)
        messages = self.conversation.get_messages()

        try:
            response = self._make_request(messages, stream=False)
            response.raise_for_status()
            data = response.json()

            content = data['choices'][0]['message']['content']
            self.conversation.add_message("assistant", content)
            return content

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return f"[API Key Error] OpenRouter authentication failed. Please set OPENROUTER_API_KEY."
            elif response.status_code == 429:
                return f"[Rate Limited] Too many requests. Please wait."
            else:
                return f"[Error {response.status_code}] {str(e)}"

        except requests.exceptions.ConnectionError:
            return "[Connection Error] Cannot reach OpenRouter API. Check internet connection."

        except Exception as e:
            return f"[Error] {str(e)}"

    def send_message_stream(self, user_message: str) -> Generator[str, None, None]:
        self.conversation.add_message("user", user_message)
        messages = self.conversation.get_messages()

        try:
            response = self._make_request(messages, stream=True)
            response.raise_for_status()

            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content_piece = delta['content']
                                full_content += content_piece
                                yield content_piece
                        except json.JSONDecodeError:
                            continue

            self.conversation.add_message("assistant", full_content)

        except requests.exceptions.HTTPError as e:
            error_msg = f"[API Error] {str(e)}"
            yield error_msg

        except requests.exceptions.ConnectionError:
            yield "[Connection Error] Cannot reach OpenRouter API."

        except Exception as e:
            yield f"[Error] {str(e)}"

    def clear_history(self):
        self.conversation.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation.to_dict()

class DemoLLMCharacter(LLMCharacter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._canned_responses = {
            "default": [
                "That's an interesting question! Let me think about it...",
                "I see what you mean. Here's my take on that.",
                "Great point! I'd love to explore that further.",
                "That reminds me of something... Let me share my thoughts.",
                "Hmm, let me consider that from a different angle.",
            ],
        }
        self._response_idx = 0

    def send_message(self, user_message: str) -> str:
        if self.api_key:
            return super().send_message(user_message)
        self.conversation.add_message("user", user_message)
        response = self._canned_responses["default"][self._response_idx % len(self._canned_responses["default"])]
        self._response_idx += 1
        self.conversation.add_message("assistant", response)
        return response

    def send_message_stream(self, user_message: str) -> Generator[str, None, None]:
        if self.api_key:
            yield from super().send_message_stream(user_message)
            return
        self.conversation.add_message("user", user_message)
        response = self._canned_responses["default"][self._response_idx % len(self._canned_responses["default"])]
        self._response_idx += 1
        self.conversation.add_message("assistant", response)
        yield response

def create_llm_character(api_key: str = None) -> LLMCharacter:
    api_key = api_key or OPENROUTER_API_KEY
    if api_key:
        return LLMCharacter(api_key=api_key)
    else:
        print("[LLMCharacter] No OPENROUTER_API_KEY set. Using demo mode with canned responses.")
        print("[LLMCharacter] Set OPENROUTER_API_KEY environment variable for real AI responses.")
        return DemoLLMCharacter()