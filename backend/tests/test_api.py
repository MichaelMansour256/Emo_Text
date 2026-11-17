from conftest import requires_ffmpeg
from fastapi.testclient import TestClient

from emotext.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_inspect_upload(sample_pdf):
    with sample_pdf.open("rb") as fh:
        resp = client.post("/v1/books/inspect", files={"file": ("book.pdf", fh)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "The Silent Road"
    assert [c["title"] for c in body["chapters"]] == ["Chapter 1: The Letter", "Chapter 2: The Road"]


def test_rejects_unsupported_type():
    resp = client.post("/v1/books/inspect", files={"file": ("x.exe", b"MZ")})
    assert resp.status_code == 415


def test_unknown_job_is_404():
    assert client.get("/v1/jobs/doesnotexist").status_code == 404


@requires_ffmpeg
def test_job_lifecycle(sample_txt):
    with sample_txt.open("rb") as fh:
        resp = client.post("/v1/jobs", files={"file": ("book.txt", fh)},
                           data={"provider": "tone", "max_chapters": "1"})
    assert resp.status_code == 202
    job = resp.json()
    # Celery runs eagerly in tests, so the job is already finished.
    assert job["status"] == "success", job
    chapter = job["result"]["chapters"][0]["file"]

    audio = client.get(f"/v1/jobs/{job['job_id']}/files/{chapter}")
    assert audio.status_code == 200 and len(audio.content) > 1000
    assert client.get(f"/v1/jobs/{job['job_id']}/files/../../secrets").status_code == 404
    assert client.get(f"/v1/jobs/{job['job_id']}/files/work/chapters.ffmeta").status_code == 404

    assert client.delete(f"/v1/jobs/{job['job_id']}").status_code == 204
    assert client.get(f"/v1/jobs/{job['job_id']}").status_code == 404
