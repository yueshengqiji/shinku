from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from shinku.api.app import create_app
from shinku.qq.host import NapCatHost


class NapCatBackendMountTests(unittest.TestCase):
    def _event(self) -> dict:
        return {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-13",
            "message_id": "m-13",
            "message": [{"type": "text", "data": {"text": "真红"}}],
        }

    def test_backend_does_not_expose_napcat_route_by_default(self) -> None:
        response = TestClient(create_app()).post("/events/napcat", json=self._event())
        self.assertEqual(response.status_code, 404)

    def test_backend_mounts_napcat_route_only_with_injected_host(self) -> None:
        host = NapCatHost()
        app = create_app(napcat_host=host, napcat_webhook_path="/hooks/qq", napcat_event_token="hook-secret")
        client = TestClient(app)
        self.assertIs(app.state.napcat_host, host)
        self.assertEqual(client.post("/hooks/qq", json=self._event()).status_code, 401)
        response = client.post(
            "/hooks/qq",
            json=self._event(),
            headers={"Authorization": "Bearer hook-secret"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["scheduled"])


if __name__ == "__main__":
    unittest.main()
