from fastapi import Request
from fastapi.responses import JSONResponse


class OcrGatewayError(Exception):
    def __init__(self, error_code: str, detail: str, status_code: int = 400):
        self.error_code = error_code
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def error_response(error_code: str, detail: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": error_code,
            "detail": detail,
            "status_code": status_code,
        },
    )


async def gateway_error_handler(_request: Request, exc: OcrGatewayError) -> JSONResponse:
    return error_response(exc.error_code, exc.detail, exc.status_code)
