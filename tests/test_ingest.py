import pytest

from app import ingest


def test_txt_roundtrip():
    assert "hello world" in ingest.extract("a.txt", b"hello world")


def test_csv_becomes_readable_rows():
    text = ingest.extract("data.csv", b"name,role\nAda,engineer\n")
    assert "Ada" in text and "engineer" in text


def test_json_is_pretty_printed():
    text = ingest.extract("d.json", b'{"limit": 55, "unit": "credits"}')
    assert "limit" in text and "55" in text


def test_html_strips_scripts_and_tags():
    html = b"<html><body><p>Visible</p><script>hidden()</script></body></html>"
    text = ingest.extract("page.html", html)
    assert "Visible" in text
    assert "hidden" not in text


def test_unknown_extension_raises():
    with pytest.raises(ingest.UnsupportedFile):
        ingest.extract("archive.zip", b"PK\x03\x04")


def test_empty_file_raises():
    with pytest.raises(ingest.ExtractionError):
        ingest.extract("empty.txt", b"   ")


def test_xml_parser_rejects_external_entities():
    # A billion-laughs / XXE style payload must not be expanded or fetched.
    payload = (
        b'<?xml version="1.0"?>'
        b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        b"<root>&xxe;</root>"
    )
    with pytest.raises(ingest.ExtractionError):
        ingest.extract("evil.xml", payload)
