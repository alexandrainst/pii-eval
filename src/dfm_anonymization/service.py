import os
from functools import lru_cache

import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from dfm_anonymization.anonymize import Anonymizer


class AnonymizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to anonymize.")
    whitelist: list[str] | None = Field(
        default=None, description="Strings to avoid anonymizing."
    )


class AnonymizeResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str


@lru_cache
def get_anonymizer() -> Anonymizer:
    return Anonymizer()


app = FastAPI(title="DFM Anonymization", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/anonymize", response_model=AnonymizeResponse)
def anonymize(
    payload: AnonymizeRequest,
    anonymizer: Anonymizer = Depends(get_anonymizer),
) -> AnonymizeResponse:
    result = anonymizer(payload.text, payload.whitelist)
    return AnonymizeResponse(text=result.text)


def main(host: str = None, port: int = None) -> None:
    if not host:
        host = os.getenv("DFM_ANON_API_HOST", "0.0.0.0")
    if not port:
        port = int(os.getenv("DFM_ANON_API_PORT", "8000"))
    uvicorn.run("dfm_anonymization.service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
