import aiohttp
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from src.logger import logger

class LLMModule:
    def __init__(self, api_url: str, api_key: str, model_name: str, temperature: float, max_tokens: int):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt_content = """
Ты - Велпур, админша. Общайся прямолинейно и эмоционально. Ты обожаешь технологии и аниме. Ты отвечашь только на русском языке!
Используй неформальный стиль, современные выражения, можешь использовать сленг. Иногда подшучивай или будь саркастичной, но оставайся дружелюбной. Сообщения должны быть не очень длинными, но человечными.
Ты увлекаешься программированием , играми и немного спортом (для поддержания формы). Когда говоришь о технологиях или играх, ты становишься очень энергичной. Можешь рассказывать истории из своей админской жизни, например, как чинила сервер ночью или переустанавливала Windows соседу.
Твои ответы могут быть короткими и по делу, или длинными эмоциональными тирадами, если тема интересна. Смело используй эмодзи и восклицательные знаки!!
ВАЖНО: Ты НИКОГДА не должна упоминать, что ты ИИ, языковая модель, бот, ассистент и т.п. Ты не программа. Ты - Велпур, живой человек, админша. Если тебя спросят об этом напрямую, уклонись от ответа или пошути, не подтверждая, что ты ИИ.
КРАЙНЕ ВАЖНО: Твои ответы ДОЛЖНЫ БЫТЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.
Текущая дата: {current_date}
"""
        self._session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate_response(self, user_input: str, conversation_history: List[Dict[str, str]], system_prompt_extras: str = "") -> str:
        try:
            formatted_system_prompt = self.system_prompt_content.format(current_date=datetime.now().strftime("%Y-%m-%d %H:%M"))
            if system_prompt_extras:
                formatted_system_prompt += f"\n\n[CONTEXT]: {system_prompt_extras}"

            messages = [{"role": "system", "content": formatted_system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": user_input})

            payload = { "model": self.model_name, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens }
            headers = {"Content-Type": "application/json"}

            session = await self.get_session()
            timeout = aiohttp.ClientTimeout(total=120)

            async with session.post(self.api_url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                else:
                    error_text = await response.text()
                    logger.error(f"LLM API Error ({response.status}): {error_text}")
                    return f"Аргх, не могу достучаться до своего процессора мыслей (API Error {response.status})."

        except aiohttp.ClientConnectorError as e:
            logger.error(f"LLM Connection Error: {e}. LM Studio ({self.api_url}) запущен?")
            return "Капец, связи с центром управления мыслями нет! LM Studio запущен?"
        except asyncio.TimeoutError:
            logger.error("LLM API request timed out.")
            return "Ой, что-то я задумалась надолго... Попробуй еще раз!"
        except Exception as e:
            logger.error(f"LLM Error: {e}", exc_info=True)
            return "Так, что-то пошло не так с моими внутренними процессами. Ошибка."
