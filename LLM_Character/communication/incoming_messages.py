from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Optional
from enum import Enum


class MessageTypes(Enum):
    STARTMESSAGE = "StartMessage"
    PROMPTMESSAGE = "PromptMessage"
    MOVEMESSAGE = "MoveMessage"


class BaseMessage(BaseModel):
    type: MessageTypes
    data: Any

# ---------------------------------------------------------------------------
# PUTTERS/ POSTERS
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# class data sent from unity to python endpoint for sending chat messages.
class PromptData(BaseModel):
    persona_name: str
    user_name: str
    message: str


class PromptMessage(BaseMessage):
    data: PromptData

# ---------------------------------------------------------------------------
# class data sent from unity to python endpoint for sending update data.


class OneLocationData(BaseModel):
    world: str
    sector: str
    arena: Optional[str] = None
    obj: Optional[str] = None


class EventData(BaseModel):
    # subject should be of format "{world}:{sector}:{arena}:{obj}"
    action_event_subject: Optional[str] = None
    action_event_predicate: Optional[str] = None
    action_event_object: Optional[str] = None
    action_event_description: Optional[str] = None


class PerceivingData(BaseModel):
    name: str
    curr_loc: OneLocationData
    events: list[EventData]


class MoveMessage(BaseMessage):
    data: list[PerceivingData]

# ---------------------------------------------------------------------------
# class data sent from unity to python endpoint for sending updated data.


class PersonaScratchData(BaseModel):
    curr_location: Optional[OneLocationData] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    innate: Optional[str] = None
    learned: Optional[str] = None
    currently: Optional[str] = None
    lifestyle: Optional[str] = None
    living_area: Optional[OneLocationData] = None

    recency_w: Optional[int] = None
    relevance_w: Optional[int] = None
    importance_w: Optional[int] = None
    recency_decay: Optional[int] = None
    importance_trigger_max: Optional[int] = None
    importance_trigger_curr: Optional[int] = None
    importance_ele_n: Optional[int] = None


class PersonaData(BaseModel):
    name: str
    scratch_data: Optional[PersonaScratchData] = None
    spatial_data: Dict[str, Dict[str, Dict[str, List[str]]]]


class UserData(BaseModel):
    old_name: str
    name: str


class MetaData(BaseModel):
    curr_time: Optional[str] = None
    sec_per_step: Optional[int] = None


class UpdateMetaMessage(BaseMessage):
    data: MetaData


class UpdatePersonaMessage(BaseMessage):
    data: PersonaData


class UpdateUserMessage(BaseMessage):
    data: UserData

# ---------------------------------------------------------------------------
# class data sent from unity to python endpoint for sending intial setup data.


class StartData(BaseModel):
    fork_sim_code: Optional[str]
    sim_code: str


class StartMessage(BaseMessage):
    data: StartData

# ---------------------------------------------------------------------------
# https://stackoverflow.com/questions/67699451/make-every-field-as-optional-with-pydantic


class FullPersonaScratchData(BaseModel):
    curr_location: OneLocationData
    first_name: str
    last_name: str
    age: int
    innate: str
    learned: str
    currently: str
    lifestyle: str
    living_area: OneLocationData

    recency_w: int
    relevance_w: int
    importance_w: int
    recency_decay: int
    importance_trigger_max: int
    importance_trigger_curr: int
    importance_ele_n: int


class FullPersonaData(BaseModel):
    name: str
    scratch_data: FullPersonaScratchData
    spatial_data: Dict[str, Dict[str, Dict[str, List[str]]]]


class AddPersonaMessage(BaseMessage):
    data: FullPersonaData


# ---------------------------------------------------------------------------
# GETTERS
# ---------------------------------------------------------------------------


class GetPersonasMessage(BaseMessage):
    data: None


class GetUsersMessage(BaseMessage):
    data: None


class PersonID(BaseModel):
    name: str


class GetPersonaDetailsMessage(BaseMessage):
    data: PersonID


class GetSavedGamesMessage(BaseMessage):
    data: None


class GetMetaDataMessage(BaseMessage):
    data: None
