from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(5, ge=1, le=10)
    stream: bool = False
    mode: Literal["dense", "hybrid", "hybrid_rerank"] | None = None


class Source(BaseModel):
    id: int
    source: str
    page: int
    heading: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class PlacesRequest(BaseModel):
    text: str = Field(min_length=3, max_length=4000)


class Place(BaseModel):
    name: str
    lat: float
    lon: float


class PlacesResponse(BaseModel):
    places: list[Place]
