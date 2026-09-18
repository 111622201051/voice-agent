# agent/llm_client.py
import re
import ollama
import logging

logger = logging.getLogger(__name__)

# Per-turn language steering added on top of the system prompt so replies
# reliably mirror the detected language of the user's voice.
LANGUAGE_INSTRUCTION = {
    "ta": (
        "Reply entirely in Tamil, using Tamil script (தமிழ்). "
        "Keep it brief and natural for spoken voice. Do not use English."
    ),
    "en": "Reply entirely in English. Keep it brief and natural for spoken voice.",
    "tanglish": (
        "Reply naturally in Tanglish: Tamil words written in English letters "
        "(e.g. 'naan sollunga', 'appdi iruku') mixed with English, matching "
        "the user's exact mixed style. Keep it brief and natural for spoken voice."
    ),
}


class LLMClient:
    def __init__(self, model_name="qwen2.5:3b"):
        """
        Initialize the Ollama client.
        Make sure Ollama is running in the background!
        """
        self.model_name = model_name
        logger.info(f"LLM Client initialized with model: {model_name}")

    def chat(self, messages: list, language_style: str = None) -> str:
        """
        Send a list of messages to the LLM and return clean spoken response.
        messages format: [{"role": "user", "content": "Hello"}, ...]
        language_style: 'ta' (Tamil), 'en' (English), 'tanglish' (Mixed).
        """
        try:
            logger.info(f"Sending request to Ollama ({self.model_name})... [Lang: {language_style}]")
            
            # Prepare messages copy
            prepared_messages = list(messages)

            # Strengthen language mirroring with a per-turn steering instruction
            if language_style and language_style in LANGUAGE_INSTRUCTION:
                prepared_messages.insert(
                    0, {"role": "system", "content": LANGUAGE_INSTRUCTION[language_style]}
                )
            
            response = ollama.chat(
                model=self.model_name,
                messages=prepared_messages,
                options={
                    "num_predict": 90,
                    "temperature": 0.4
                }
            )
            assistant_text = response['message']['content'].strip()

            # Clean markdown asterisks, hashes, backticks and bullet points for clean voice reading
            assistant_text = re.sub(r'[*_#`~>]', '', assistant_text)
            assistant_text = re.sub(r'^\s*[-•]\s+', '', assistant_text, flags=re.MULTILINE)
            assistant_text = re.sub(r'\s{2,}', ' ', assistant_text).strip()
            
            return assistant_text
        except Exception as e:
            logger.error(f"Ollama error: {e}. Is the Ollama app running?")
            if language_style == "ta":
                return "மன்னிக்கவும், எனது அமைப்பில் சிறிய பிழை ஏற்பட்டுள்ளது."
            return "I am sorry, I am having trouble connecting to my local model."