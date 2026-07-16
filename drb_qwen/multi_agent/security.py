from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


BLOCKED_HOSTS = {
    "localhost",
    "metadata.google.internal",
    "metadata.azure.internal",
    "instance-data.ec2.internal",
}

COMMON_MULTI_LABEL_SUFFIXES = {
    "ac.uk",
    "co.jp",
    "co.kr",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.sg",
    "edu.au",
    "gov.au",
    "gov.cn",
    "gov.uk",
    "net.au",
    "net.cn",
    "org.au",
    "org.cn",
    "org.uk",
}

COMMUNITY_HOSTS = {
    "baijiahao.baidu.com",
    "zhihu.com",
    "zhuanlan.zhihu.com",
    "xueqiu.com",
    "guba.eastmoney.com",
    "caifuhao.eastmoney.com",
    "toutiao.com",
}
INDEPENDENT_MEDIA_HOSTS = {
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
    "bbc.com",
    "ft.com",
    "xinhuanet.com",
    "chinanews.com.cn",
    "thepaper.cn",
    "finance.sina.com.cn",
}
PRIMARY_ORGANIZATION_HOSTS = {
    "imf.org",
    "worldbank.org",
    "oecd.org",
    "who.int",
    "un.org",
    "wgc.org",
}


def validate_external_url(url: str) -> tuple[bool, str]:
    """Reject obvious SSRF targets before URL fetch or visit dispatch.

    Production deployments should additionally enforce the same rule in an
    egress proxy after DNS resolution, because application-only validation
    cannot fully prevent DNS rebinding.
    """

    try:
        parsed = urlparse(str(url).strip())
    except Exception:
        return False, "invalid URL"
    if parsed.scheme not in {"http", "https"}:
        return False, "unsupported URL scheme"
    if parsed.username or parsed.password:
        return False, "embedded URL credentials are not allowed"
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False, "URL host is missing"
    if host in BLOCKED_HOSTS or host.endswith((".localhost", ".local", ".internal")):
        return False, "local or metadata host is not allowed"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True, ""
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        return False, "non-public IP address is not allowed"
    return True, ""


def source_independence_group(url: str) -> str:
    try:
        host = (urlparse(str(url)).hostname or "").lower().rstrip(".")
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return host
    suffix = ".".join(labels[-2:])
    if suffix in COMMON_MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix


def classify_source_authority(url: str, title: str = "") -> tuple[str, float]:
    """Classify publisher authority separately from extraction completeness."""

    try:
        parsed = urlparse(str(url or ""))
        host = (parsed.hostname or "").lower().rstrip(".")
        path = parsed.path.lower()
    except Exception:
        return "unknown", 0.4
    label = str(title or "").casefold()

    if host.endswith((".gov", ".gov.cn", ".gov.uk", ".gov.au")) or host in {
        "gov.cn",
        "gov.uk",
    }:
        return "official", 0.95
    if any(host == value or host.endswith(f".{value}") for value in PRIMARY_ORGANIZATION_HOSTS):
        return "primary", 0.9
    if host in COMMUNITY_HOSTS or any(host.endswith(f".{value}") for value in COMMUNITY_HOSTS):
        return "community", 0.3
    if host == "www.163.com" and path.startswith("/dy/"):
        return "aggregator", 0.35
    if host in INDEPENDENT_MEDIA_HOSTS or any(host.endswith(f".{value}") for value in INDEPENDENT_MEDIA_HOSTS):
        return "independent_media", 0.7
    if host.endswith((".edu", ".edu.cn", ".ac.uk")):
        return "institutional", 0.8
    # Document hints can identify primary material only after known community,
    # aggregator, and media publishers have been classified. A news headline
    # mentioning an annual report must not inherit the report's authority.
    if host.startswith("ir.") or any(
        marker in path or marker in label
        for marker in ("annual-report", "annual_report", "investor-relations", "年报", "年度报告")
    ):
        return "primary", 0.85
    return "unknown", 0.5
