from pydantic import BaseModel
from typing import List, Optional

class Signal(BaseModel):
    source: str
    text: str
    timestamp: Optional[str] = None

class AudienceProfile(BaseModel):
    name: str
    age_range: str
    location: str
    psychographics: List[str]
    primary_goal: str
    primary_pain: str
    signal_sources: List[str]

class CampaignInput(BaseModel):
    channel: str
    creative_message: str
    CTR: float
    CVR: float
    spend: float

class InterpretRequest(BaseModel):
    brand_dna: str
    banned_phrases: List[str]
    audience_profile: AudienceProfile
    campaign_inputs: List[CampaignInput]
    signals: List[Signal]
