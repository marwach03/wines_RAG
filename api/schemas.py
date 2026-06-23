"""
Pydantic request/response schemas for the Wines Retrieval + Generation API.
No business logic here — just the data shapes shared by main.py, retriever.py
and generator.py.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_LIMIT = 5


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=50, description="Number of results to return")

    project:    Optional[str] = Field(None, description="Filter: 'stilema' or 'mastroberardino'")
    category:   Optional[str] = Field(None, description="Filter: wine category (e.g. 'ICON', 'Taurasi')")
    wine_type:  Optional[str] = Field(None, description="Filter: 'red', 'white', 'rosato', 'passito', 'oil'")
    language:   Optional[str] = Field(None, description="Filter: 'it' or 'en'")
    chunk_type: Optional[str] = Field(None, description="Filter: 'wine_sheet', 'history', or 'chronology'")


class RetrievedChunk(BaseModel):
    chunk_id:   str
    score:      float
    title:      Optional[str] = None
    text:       Optional[str] = None
    project:    Optional[str] = None
    category:   Optional[str] = None
    wine_type:  Optional[str] = None
    language:   Optional[str] = None
    chunk_type: Optional[str] = None
    wine_name:  Optional[str] = None
    vintage:    Optional[int] = None


class RetrieveResponse(BaseModel):
    query:              str
    detected_language:    Optional[str] = None
    results:              list[RetrievedChunk]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question, in any language")
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=20, description="Number of chunks to retrieve as context")

    project:    Optional[str] = Field(None, description="Filter: 'stilema' or 'mastroberardino'")
    category:   Optional[str] = Field(None, description="Filter: wine category (e.g. 'ICON', 'Taurasi')")
    wine_type:  Optional[str] = Field(None, description="Filter: 'red', 'white', 'rosato', 'passito', 'oil'")
    language:   Optional[str] = Field(None, description="Filter: 'it' or 'en' (filters SOURCE chunks, not the answer's language)")
    chunk_type: Optional[str] = Field(None, description="Filter: 'wine_sheet', 'history', or 'chronology'")


class AskResponse(BaseModel):
    query:              str
    detected_language:    Optional[str] = None
    answer:               str
    sources:              list[RetrievedChunk]