from deep_translator import GoogleTranslator


class Translation:
    def __init__(self, from_lang="vi", to_lang="en"):
        # Wrapper for translation; uses deep_translator (Google) to avoid
        # conflicts with httpx/httpcore used by transformers/huggingface_hub.
        self.__from_lang = from_lang
        self.__to_lang = to_lang
        self.translator = GoogleTranslator(source=from_lang, target=to_lang)

    def preprocessing(self, text):
        return text.lower()

    def __call__(self, text):
        """
        Preprocesses text then translates it.

        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text)
