from __future__ import annotations

from drb_qwen.url_fetcher import extract_text_from_html, normalize_visit_endpoint, should_try_fetch_url


def main() -> None:
    html = """
    <html>
      <head><title>Ignored title</title><style>.x{color:red}</style></head>
      <body>
        <script>window.secret = "bad";</script>
        <article>
          <h1>Research title</h1>
          <p>First paragraph with useful evidence.</p>
          <p>Second paragraph with 2026 data.</p>
        </article>
      </body>
    </html>
    """
    text = extract_text_from_html(html)
    assert "Research title" in text
    assert "First paragraph with useful evidence." in text
    assert "Second paragraph with 2026 data." in text
    assert "window.secret" not in text
    assert should_try_fetch_url("https://example.com/a")
    assert not should_try_fetch_url("ftp://example.com/a")
    assert normalize_visit_endpoint("http://localhost:8765") == "http://localhost:8765/visit"
    assert normalize_visit_endpoint("http://localhost:8765/visit") == "http://localhost:8765/visit"
    print("smoke_test_url_fetcher passed")


if __name__ == "__main__":
    main()
