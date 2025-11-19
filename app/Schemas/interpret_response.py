from pydantic import BaseModel
from typing import List, Dict, Any

class ClusterAnalysis(BaseModel):
    cluster_id: int
    top_phrases: List[str]
    emotion_valence: float
    intent_score: float
    friction_score: float
    representative_text: str

class CampaignInsight(BaseModel):
    channel: str
    creative_message: str
    resonance_score: float
    improvement_direction: str

class InterpretResponse(BaseModel):
    clusters: List[ClusterAnalysis]
    campaign_insights: List[CampaignInsight]
    brand_alignment_score: float
    meta: Dict[str, Any]
