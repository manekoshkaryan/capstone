from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ObstacleInfo(BaseModel):
    label: str
    distance_m: float
    bearing: Literal["far_left", "left", "center", "right", "far_right"]
    is_moving: bool = False


class FreeSectors(BaseModel):
    far_left: float = 0.0
    left: float = 0.0
    center: float = 0.0
    right: float = 0.0
    far_right: float = 0.0


class SceneContext(BaseModel):
    obstacles: list[ObstacleInfo] = Field(default_factory=list)
    free_sectors: FreeSectors = Field(default_factory=FreeSectors)
    best_direction: Literal["far_left", "left", "center", "right", "far_right"] = "center"
    center_clear_m: float = 0.0
    trigger: Literal[
        "new_obstacle",
        "path_blocked",
        "direction_changed",
        "periodic_update",
        "path_cleared",
        "user_question",
    ] = "periodic_update"
    last_guidance: str = ""
    user_query: Optional[str] = None


class GuidanceResponse(BaseModel):
    action: Literal[
        "STOP", "GO_STRAIGHT", "BEAR_LEFT", "BEAR_RIGHT",
        "GO_LEFT", "GO_RIGHT", "WAIT",
    ]
    speech: str = Field(
        description="Spoken aloud to user. Imperative. Max 10 words. No filler."
    )
    urgency: Literal["critical", "warn", "info"]
    reasoning: str = Field(
        description="Internal reasoning. Not spoken. Why this action was chosen."
    )
