from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from search.service import answer_regulatory_question

app = FastAPI(
    title="ReguLens API",
    description="AI-powered regulatory disclosure analysis using SEC climate rules",
    version="1.0.0"
)


class DisclosureRequest(BaseModel):
    query: str
    version: Optional[str] = None  # "2024_final" | "2022_proposed"


@app.post("/disclosure-analysis")
def disclosure_analysis(req: DisclosureRequest):
    """
    Analyze a regulatory disclosure question using SEC climate rules.
    """
    return answer_regulatory_question(
        query = req.query,
        version=req.version
    )
    
@app.get("/health")
def health():
    return {"status": "ok"}

