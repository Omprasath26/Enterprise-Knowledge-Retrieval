from backend.metadata import MetadataStore


def test_only_current_completed_version_is_retrievable(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    first_version = "a" * 64
    second_version = "b" * 64
    common = {
        "document_id": "doc-1",
        "original_filename": "policy.txt",
        "department": "hr",
        "file_type": "txt",
        "file_size": 10,
    }
    store.record_upload(
        **common,
        version_hash=first_version,
        object_path="documents/doc-1/first.txt",
    )
    store.set_status("doc-1", first_version, "completed")
    assert store.current_completed_versions(("hr",)) == {
        ("doc-1", first_version)
    }

    store.record_upload(
        **common,
        version_hash=second_version,
        object_path="documents/doc-1/second.txt",
    )
    assert store.current_completed_versions(("hr",)) == set()
    store.set_status("doc-1", second_version, "completed")
    assert store.current_completed_versions(("hr",)) == {
        ("doc-1", second_version)
    }
    store.set_parser_warnings("doc-1", second_version, ["ConversionWarning"])
    record = store.get_document("doc-1")
    assert record["object_path"] == "documents/doc-1/second.txt"
    assert record["parser_warnings"] == ["ConversionWarning"]


def test_empty_access_scope_fails_closed(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    assert store.current_completed_versions(()) == set()
