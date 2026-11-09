import io
import zipfile

import pytest

from backend.uploads import UploadValidationError, validate_upload


def test_accepts_nonempty_utf8_text():
    assert validate_upload("policy.txt", b"Keep records for seven years.", 100) == ".txt"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("empty.txt", b""),
        ("unsupported.exe", b"payload"),
        ("policy.pdf", b"not a pdf"),
        ("policy.txt", b"\xff"),
        ("table.csv", b",,\n"),
    ],
)
def test_rejects_invalid_uploads(filename, content):
    with pytest.raises(UploadValidationError):
        validate_upload(filename, content, 100)


def test_rejects_oversized_upload():
    with pytest.raises(UploadValidationError, match="maximum upload size"):
        validate_upload("policy.txt", b"12345", 4)


def test_accepts_structurally_valid_docx():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
    assert validate_upload("policy.docx", stream.getvalue(), 4096) == ".docx"
