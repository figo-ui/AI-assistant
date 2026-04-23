from .services.language_support import detect_language


class RequestLanguageMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        header = str(request.META.get("HTTP_ACCEPT_LANGUAGE", "")).split(",")[0].strip().lower()
        request.preferred_language = detect_language("", preferred=header[:2] if header else None)
        return self.get_response(request)
