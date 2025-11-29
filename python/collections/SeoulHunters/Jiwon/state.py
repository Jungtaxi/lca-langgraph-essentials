from typing import List, Optional, Any, TypedDict
from pydantic import BaseModel, Field, field_validator
 
class TravelPreference(BaseModel):
    target_area: List[str] = Field(default_factory=list)
    duration: Optional[int] = None
    themes: List[str] = Field(default_factory=list)
    intensity: int = 50
    companions: List[str] = Field(default_factory=list)
    transport: List[str] = Field(default_factory=list)

    @field_validator("target_area", mode="before")
    @classmethod
    def to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class Place(BaseModel):
    name: str
    category: Optional[str] = None
    address: Optional[str] = None
    road_address: Optional[str] = None
    mapx: Optional[str] = None
    mapy: Optional[str] = None
    link: Optional[str] = None
    telephone: Optional[str] = None
    theme: str
    area: str
    source: str = "naver_local"


class AgentState(TypedDict, total=False):
    user_input: str
    prefs: TravelPreference
    tag_plan: Any
    place_pool: List[Place]
    routes: Any
    main_place_candidates: Optional[List[Place]]
    selected_main_places: Optional[List[Place]]
