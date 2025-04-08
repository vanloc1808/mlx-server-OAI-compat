from http import HTTPStatus
from typing import Union

from app.schemas.openai import ErrorResponse


def create_error_response(
    message: str,
    err_type: str = "internal_error",
    status_code: Union[int, HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR
):
    return ErrorResponse(
        message=message,
        type=err_type,
        code=status_code.value if isinstance(status_code, HTTPStatus) else status_code
    ).model_dump()