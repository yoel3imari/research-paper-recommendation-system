"""Tests the static dashboard served by the FastAPI application.

The dashboard is a single HTML file (`dashboard/index.html`) with no build
step. These tests keep the root route wired up and verify that the page is
served from the same origin as the API.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_dashboard_is_served_at_root():
    client = TestClient(create_app())
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "PaperScout" in response.text
    assert "tailwindcss" in response.text
    assert "alpinejs" in response.text


def test_dashboard_is_not_in_openapi_schema():
    client = TestClient(create_app())
    paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
    assert "/api/v1/health" in paths
