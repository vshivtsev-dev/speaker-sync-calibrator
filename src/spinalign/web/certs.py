"""Self-signed TLS for the local UI.

Browsers only hand out a microphone in a secure context, and a LAN address
like ``http://192.168.1.20:8443`` is not one. Nothing about this app wants TLS
otherwise — it is purely the price of ``getUserMedia`` on a phone.

The certificate is generated once and reused, so the browser warning is
accepted once rather than on every restart. Every local address we can find
goes into the SAN list, because the phone will reach the machine by IP.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import socket
import ssl
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_VALIDITY_DAYS = 3650


def local_addresses() -> list[str]:
    """Best-effort list of addresses this machine answers on."""
    found = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        found.add(socket.gethostbyname(hostname))
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass

    # Ask the routing table which interface would reach the outside world; the
    # address it picks is the one a phone on the same network can use.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))
        found.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()

    return sorted(found)


def ensure_certificate(directory: Path) -> tuple[Path, Path]:
    """Return paths to a certificate and key, generating them if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    cert_path = directory / "spinalign-cert.pem"
    key_path = directory / "spinalign-key.pem"

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SpinAlign")])

    alt_names: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for address in local_addresses():
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(address)))
        except ValueError:
            continue

    now = dt.datetime.now(dt.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return cert_path, key_path


def ssl_context(directory: Path) -> ssl.SSLContext:
    cert_path, key_path = ensure_certificate(directory)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context
