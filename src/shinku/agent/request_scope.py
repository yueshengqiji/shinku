"""Host-owned intent boundary. Background messages are evidence, not requests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestScope:
    requester: str
    session: str
    primary_message_id: str
    requests: tuple[tuple[str, str], ...]

    def user_messages(self) -> list[str]:
        return [text for _, text in self.requests if text.strip()]
