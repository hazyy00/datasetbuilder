from typing import TypedDict


class PipelineData(TypedDict):
    chat_id: str
    file_name: str
    q: list[str]
    p: list[list[str]]
    a: list[str]
    e: list[str]
