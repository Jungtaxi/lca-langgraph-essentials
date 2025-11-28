from pydantic import BaseModel, field_validator, Field
from typing import List, Optional

class TravelPreference(BaseModel):
    # 여행 장소 (여러 개일 수도 있으니까 리스트로 정의)
    target_area: List[str] = Field(default_factory=list)

    # 체류 기간 (일수)/ 테마 / 강도 / 동행 / 이동수단
    duration: Optional[int] = None
    themes: List[str] = Field(default_factory=list)
    intensity: int = 50
    companions: List[str] = Field(default_factory=list)
    transport: List[str] = Field(default_factory=list)

    # 🔥 "서울" 같이 문자열 하나만 와도 ["서울"] 리스트로 바꿔주는 전처리기
    @field_validator("target_area", mode="before")
    @classmethod
    def target_area_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        # 이미 리스트거나 다른 iterable이면 리스트로 한 번 감싸줌
        return list(v)
