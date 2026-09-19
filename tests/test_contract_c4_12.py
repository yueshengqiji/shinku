from __future__ import annotations

import json
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shinku.qq.host import NapCatHost
from shinku.qq.webhook import create_napcat_webhook_router


class NapCatWebhookTests(unittest.TestCase):
    def _event(self) -> dict:
        return {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-1",
            "message_id": "m-12",
            "message": [{"type": "text", "data": {"text": "真红"}}],
        }

    def test_webhook_accepts_json_and_returns_scheduling_summary(self) -> None:
        app = FastAPI()
        app.include_router(create_napcat_webhook_router(NapCatHost()))
        response = TestClient(app).post("/events/napcat", json=self._event())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "scheduled": True, "visual_ready": False, "visual_pending": False})

    def test_webhook_can_require_bearer_token_without_echoing_it(self) -> None:
        app = FastAPI()
        app.include_router(create_napcat_webhook_router(NapCatHost(), event_token="event-secret"))
        client = TestClient(app)
        self.assertEqual(client.post("/events/napcat", json=self._event()).status_code, 401)
        response = client.post("/events/napcat", json=self._event(), headers={"Authorization": "Bearer event-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("event-secret", response.text)

    def test_webhook_rejects_invalid_json_and_non_message_event(self) -> None:
        app = FastAPI()
        app.include_router(create_napcat_webhook_router(NapCatHost()))
        client = TestClient(app)
        self.assertEqual(client.post("/events/napcat", content=b"not-json").status_code, 400)
        notice = {"post_type": "notice", "notice_type": "group_upload"}
        self.assertEqual(client.post("/events/napcat", content=json.dumps(notice)).status_code, 400)

    def test_router_rejects_bad_path_configuration(self) -> None:
        with self.assertRaises(ValueError):
            create_napcat_webhook_router(NapCatHost(), path="events/napcat")


if __name__ == "__main__":
    unittest.main()
