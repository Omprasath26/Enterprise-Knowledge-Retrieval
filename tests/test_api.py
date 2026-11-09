from dataclasses import replace

from fastapi.testclient import TestClient

import backend.api as api


def test_local_api_health_and_server_owned_access_scope():
    with TestClient(api.app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/health").json()["status"] == "healthy"
        config = client.get("/api/v1/access-tags").json()
        assert config["access_tags"] == list(api.settings.access_tags)
        assert config["max_upload_size_mb"] == api.settings.max_upload_size_mb


def test_query_rejects_client_supplied_permissions():
    with TestClient(api.app, client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the policy?", "access_tags": ["admin"]},
        )
    assert response.status_code == 422


def test_api_refuses_non_loopback_clients():
    with TestClient(api.app, client=("203.0.113.10", 50000)) as client:
        response = client.get("/health")
    assert response.status_code == 403


def test_invalid_upload_returns_its_document_id(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "settings", replace(api.settings, data_dir=tmp_path))
    api.get_metadata.cache_clear()
    try:
        with TestClient(api.app, client=("127.0.0.1", 50000)) as client:
            response = client.post(
                "/api/v1/documents/upload",
                data={"department": api.settings.access_tags[0]},
                files={"file": ("invalid.exe", b"bad content")},
            )
        assert response.status_code == 400
        detail = response.json()["detail"]
        record = api.get_metadata().get_document(detail["document_id"])
        assert detail["processing_status"] == "failed"
        assert record["processing_status"] == "failed"
        assert not (tmp_path / record["object_path"]).exists()
    finally:
        api.get_metadata.cache_clear()
