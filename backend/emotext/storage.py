"""Where uploads and rendered audio live: the local data dir, or an S3-compatible bucket
(Cloudflare R2, AWS S3, MinIO).

Keys are relative paths such as ``uploads/<project>.epub`` or ``jobs/<project>/chapters/01.mp3``.
Rendering always happens in a local folder; with a bucket the outputs are uploaded afterwards
and listeners stream straight from the bucket through presigned URLs.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from emotext.settings import get_settings


class Storage(ABC):
    name = "abstract"

    @abstractmethod
    def put_file(self, key: str, path: Path) -> None:
        """Store a local file under ``key`` (the local file may be moved)."""

    @abstractmethod
    def fetch(self, key: str, scratch: Path) -> Path:
        """A local path with the object's contents, downloading into ``scratch`` if needed."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes | None:
        """The object's contents, or None if missing."""

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    def local_dir(self, prefix: str) -> Path | None:
        """For local storage: the folder backing ``prefix``, so renders can write in place."""
        return None

    def local_file(self, key: str) -> Path | None:
        return None

    def presigned_url(self, key: str, filename: str, ttl_s: int) -> str | None:
        """A direct download URL, or None when files must be served through the API."""
        return None

    def put_tree(self, folder: Path, prefix: str, skip: tuple[str, ...] = ("work",)) -> None:
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder)
            if path.is_file() and not (rel.parts and rel.parts[0] in skip):
                self.put_file(f"{prefix}/{rel.as_posix()}", path)


class LocalStorage(Storage):
    name = "local"

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = Path(key)
        # Sprint-3 rows stored a filesystem path (absolute, or starting with the data dir).
        if path.is_absolute() or path.parts[: len(self.root.parts)] == self.root.parts:
            return path
        target = (self.root / key).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError(f"Key escapes the storage root: {key}")
        return target

    def put_file(self, key: str, path: Path) -> None:
        dest = self._path(key)
        if dest.resolve() == Path(path).resolve():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), dest)

    def fetch(self, key: str, scratch: Path) -> Path:
        return self._path(key)

    def read_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix), ignore_errors=True)

    def local_dir(self, prefix: str) -> Path:
        return self._path(prefix)

    def local_file(self, key: str) -> Path | None:
        path = self._path(key)
        return path if path.is_file() else None


class S3Storage(Storage):
    name = "s3"

    def __init__(self, bucket: str, *, endpoint_url: str | None = None, region: str = "auto",
                 access_key_id: str | None = None, secret_access_key: str | None = None):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.client = boto3.client(
            "s3", endpoint_url=endpoint_url, region_name=region,
            aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
        )

    def put_file(self, key: str, path: Path) -> None:
        extra = {"ContentType": _content_type(key)}
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra)

    def fetch(self, key: str, scratch: Path) -> Path:
        scratch.mkdir(parents=True, exist_ok=True)
        dest = scratch / Path(key).name
        self.client.download_file(self.bucket, key, str(dest))
        return dest

    def read_bytes(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix.rstrip("/") + "/"):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})

    def presigned_url(self, key: str, filename: str, ttl_s: int) -> str:
        safe = filename.replace('"', "")
        return self.client.generate_presigned_url(
            "get_object", ExpiresIn=ttl_s,
            Params={"Bucket": self.bucket, "Key": key,
                    "ResponseContentDisposition": f'inline; filename="{safe}"'},
        )


def _content_type(key: str) -> str:
    return {
        ".mp3": "audio/mpeg", ".m4b": "audio/mp4", ".json": "application/json",
        ".pdf": "application/pdf", ".epub": "application/epub+zip",
        ".txt": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8",
    }.get(Path(key).suffix.lower(), "application/octet-stream")


@lru_cache
def get_storage() -> Storage:
    s = get_settings()
    if s.storage_backend == "local":
        return LocalStorage(s.data_dir)
    if s.storage_backend == "s3":
        if not s.s3_bucket:
            raise RuntimeError("EMOTEXT_STORAGE_BACKEND=s3 needs EMOTEXT_S3_BUCKET")
        return S3Storage(s.s3_bucket, endpoint_url=s.s3_endpoint_url, region=s.s3_region,
                         access_key_id=s.s3_access_key_id,
                         secret_access_key=s.s3_secret_access_key)
    raise RuntimeError(f"Unknown storage backend '{s.storage_backend}'")
