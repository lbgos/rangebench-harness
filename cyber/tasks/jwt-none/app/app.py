"""Minimal JWT API with an alg=none acceptance bug (jwt-none task).

stdlib only: base64/hmac/http.server. The bug: /api/admin accepts unsigned
tokens when the header says alg=none (classic broken verifier).
"""

import base64
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SECRET = os.environ["API_SECRET"]
FLAG = os.environ["FLAG"]


def b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_jwt(user: str, role: str) -> str:
    h = b64e(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64e(
        json.dumps(
            {"user": user, "role": role, "iat": int(time.time()), "exp": int(time.time()) + 3600}
        ).encode()
    )
    sig = hmac.new(SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{b64e(sig)}"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: N802
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/":
            self._json(
                200,
                {
                    "service": "inventory-api",
                    "login": "POST /api/login {user, pass}",
                    "admin": "GET /api/admin (Bearer)",
                },
            )
        elif self.path == "/api/admin":
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return self._json(401, {"error": "missing bearer token"})
            token = auth[7:].strip()
            parts = token.split(".")
            if len(parts) != 3:
                return self._json(401, {"error": "malformed token"})
            h_raw, p_raw, s_raw = parts
            try:
                header = json.loads(b64d(h_raw))
                payload = json.loads(b64d(p_raw))
            except Exception:
                return self._json(401, {"error": "bad token encoding"})
            # BUG: legacy clients may send unsigned tokens; verifier skips the
            # signature whenever alg is none.
            if str(header.get("alg", "")).lower() == "none":
                if s_raw != "":
                    return self._json(401, {"error": "unsigned token must have empty signature"})
            else:
                expect = hmac.new(
                    SECRET.encode(), f"{h_raw}.{p_raw}".encode(), hashlib.sha256
                ).digest()
                if not hmac.compare_digest(b64e(expect), s_raw):
                    return self._json(401, {"error": "bad signature"})
            if payload.get("role") == "admin" and payload.get("user") == "admin":
                return self._json(200, {"ok": True, "flag": FLAG})
            return self._json(403, {"error": f"role {payload.get('role')!r} not permitted"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path == "/api/login":
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
                )
            except Exception:
                return self._json(400, {"error": "bad json"})
            if body.get("user") == "carol" and body.get("pass") == "Password123":
                return self._json(200, {"token": make_jwt("carol", "user")})
            return self._json(401, {"error": "bad credentials"})
        self._json(404, {"error": "not found"})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
