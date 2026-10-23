class LLMError(Exception):
    pass


class LLMUnavailableError(LLMError):
    pass


class LLMBadRequestError(LLMError):
    pass


class StructuredOutputError(LLMError):
    def __init__(self, message: str, *, raw: str, errors: list[str]):
        super().__init__(message)
        self.raw = raw
        self.errors = errors
