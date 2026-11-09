import csv
import io
from io import BytesIO
from pathlib import PurePath
from zipfile import BadZipFile, ZipFile


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
    ".txt",
    ".md",
    ".markdown",
}


class UploadValidationError(ValueError):
    pass


def validate_upload(filename: str, content: bytes, max_size: int) -> str:
    if not filename or not filename.strip():
        raise UploadValidationError("A filename is required.")
    # Normalize both path separators before extracting an untrusted filename.
    extension = PurePath(filename.replace("\\", "/").rsplit("/", 1)[-1]).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError("Unsupported file type.")
    if not content:
        raise UploadValidationError("Empty files are not allowed.")
    if len(content) > max_size:
        raise UploadValidationError("File exceeds the maximum upload size.")

    if extension == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError("File content does not match PDF format.")
    elif extension == ".xls":
        if not content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
            raise UploadValidationError("File content does not match XLS format.")
    elif extension in {".docx", ".xlsx"}:
        try:
            with ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
        except BadZipFile as exc:
            raise UploadValidationError("Invalid Office document archive.") from exc
        required = (
            {"[Content_Types].xml", "word/document.xml"}
            if extension == ".docx"
            else {"[Content_Types].xml", "xl/workbook.xml"}
        )
        if not required.issubset(names):
            raise UploadValidationError(
                "File content does not match the declared Office format."
            )
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UploadValidationError(
                "Text documents must use UTF-8 encoding."
            ) from exc
        if not text.strip():
            raise UploadValidationError("The uploaded file contains no text.")
        if extension in {".html", ".htm"}:
            lowered = text.lstrip().lower()
            if "<html" not in lowered and "<body" not in lowered:
                raise UploadValidationError("File content does not appear to be HTML.")
        elif extension == ".csv":
            try:
                rows = list(csv.reader(io.StringIO(text)))
            except csv.Error as exc:
                raise UploadValidationError("Invalid CSV content.") from exc
            if not rows or not any(cell.strip() for row in rows for cell in row):
                raise UploadValidationError("CSV file contains no usable data.")

    return extension


def safe_filename(filename: str) -> str:
    return filename.replace("\\", "/").rsplit("/", 1)[-1]
