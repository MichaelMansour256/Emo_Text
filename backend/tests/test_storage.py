import boto3
import pytest
from conftest import requires_ffmpeg
from fastapi.testclient import TestClient
from moto import mock_aws

from emotext import storage
from emotext.api import app
from emotext.settings import get_settings

client = TestClient(app, follow_redirects=False)


def test_local_storage_keeps_keys_inside_the_root(tmp_path):
    store = storage.LocalStorage(tmp_path)
    src = tmp_path / "in.txt"
    src.write_text("hi")
    store.put_file("uploads/a.txt", src)
    assert store.read_bytes("uploads/a.txt") == b"hi" and not src.exists()
    with pytest.raises(ValueError):
        store.local_file("../outside.txt")


@pytest.fixture
def s3(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="books")
        s = get_settings()
        for name, value in {"storage_backend": "s3", "s3_bucket": "books",
                            "s3_region": "us-east-1", "s3_access_key_id": "test",
                            "s3_secret_access_key": "test"}.items():
            monkeypatch.setattr(s, name, value)
        storage.get_storage.cache_clear()
        yield boto3.client("s3", region_name="us-east-1")
    storage.get_storage.cache_clear()


@requires_ffmpeg
def test_bucket_storage_end_to_end(s3, sample_txt):
    user = {"X-Dev-User": "s3user@test"}
    with sample_txt.open("rb") as fh:
        project = client.post("/v1/projects", files={"file": ("book.txt", fh)},
                              data={"rights_confirmed": "true"}, headers=user).json()
    pid = project["id"]
    keys = lambda: {o["Key"] for o in s3.list_objects_v2(Bucket="books").get("Contents", [])}
    assert f"uploads/{pid}.txt" in keys()

    done = client.post(f"/v1/projects/{pid}/render", headers=user,
                       json={"provider": "tone", "max_chapters": 1}).json()
    assert done["status"] == "done", done
    rendered = keys()
    assert f"jobs/{pid}/script.json" in rendered and f"jobs/{pid}/manifest.json" in rendered
    assert any(k.startswith(f"jobs/{pid}/chapters/") and k.endswith(".mp3") for k in rendered)
    assert not any("/work/" in k for k in rendered)

    url = done["result"]["chapters"][0]["url"]
    assert url.startswith("https://books.s3") and "X-Amz-Signature" in url
    assert client.get(f"/v1/projects/{pid}/script", headers=user).json()[0]["lines"]

    assert client.delete(f"/v1/projects/{pid}", headers=user).status_code == 204
    assert not {k for k in keys() if pid in k}
