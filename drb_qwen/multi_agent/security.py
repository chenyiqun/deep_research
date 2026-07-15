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
