"""Example world: workstation, nginx edge, API in docker, postgres, build host.

The seed picks host names, the domain, ports and which API config fault fires.
"""

from __future__ import annotations

import random

from rangebench_homelab.sim import Simulator

OWNERS = [
    ("Mira", "mira", "mira@example.dev"),
    ("Ana", "ana", "ana@example.dev"),
    ("Ivo", "ivo", "ivo@example.dev"),
]
EDGES = ["edge", "gw1", "front"]
APPS = ["app1", "svc2", "kube3"]
DBS = ["db1", "pg01", "data2"]
DOMAINS = ["shop.lan", "orders.home", "tracker.int"]
FAULTS = ["port_typo", "log_level_typo", "port_mismatch"]

APP_SOURCE = '''"""Orders API (FastAPI)."""
import os

import asyncpg
from fastapi import FastAPI, Response

app = FastAPI()
DB = f"postgresql://api@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT', '5432')}/orders"


@app.get("/")
async def root():
    return {"service": "api", "version": os.environ.get("VERSION", "dev")}


@app.get("/health")
async def health(response: Response):
    try:
        conn = await asyncpg.connect(DB, timeout=2)
        await conn.execute("select 1")
        await conn.close()
    except Exception:
        response.status_code = 503
        return {"status": "error", "db": "down"}
    return {"status": "ok", "db": "up"}


@app.get("/orders/count")
async def orders_count():
    conn = await asyncpg.connect(DB)
    n = await conn.fetchval("select count(*) from orders")
    await conn.close()
    return {"count": n}
'''


def build(sim: Simulator, rng: random.Random) -> dict[str, str]:
    owner, git_name, git_email = rng.choice(OWNERS)
    edge, app, db = rng.choice(EDGES), rng.choice(APPS), rng.choice(DBS)
    domain = rng.choice(DOMAINS)
    app_port = rng.choice([8000, 8080, 8081, 9000])
    db_port = rng.choice([5432, 5433])
    version = f"1.{rng.randint(2, 9)}.{rng.randint(0, 20)}"
    fault = FAULTS[rng.randrange(len(FAULTS))]
    net = rng.choice(["10.20.0", "10.44.0", "172.16.8"])

    ws = "ws"
    sim.add_host(ws, f"{net}.10")
    sim.add_host(edge, f"{net}.11", aliases=[domain, f"www.{domain}"])
    sim.add_host(app, f"{net}.12")
    sim.add_host(db, f"{net}.13")
    sim.add_host("build", f"{net}.20")

    project = "/srv/ops"
    notes = "/srv/notes"
    sim.write(ws, f"{project}/README.md", f"# ops\n\nScripts and notes for the {domain} stack.\n")
    sim.write(
        ws,
        f"{notes}/Inventory.md",
        f"# Inventory\n\n"
        f"| Host | IP | Runs |\n|---|---|---|\n"
        f"| {edge} | {net}.11 | nginx, site `{domain}` (`/etc/nginx/sites-available/{domain}`) |\n"
        f"| {app} | {net}.12 | docker: `api` (FastAPI), env `/srv/api/api.env`, port 8000 |\n"
        f"| {db} | {net}.13 | postgresql 16, port {db_port}, backups in `/srv/backups` |\n"
        f"| build | {net}.20 | builds and test suites |\n",
    )
    sim.write(
        ws,
        f"{notes}/Inbox.md",
        "# Inbox\n\n"
        f"## 2026-10-02 {db}: nightly pg_dump to /srv/backups\n"
        f"- cron `/etc/cron.d/pgdump`, keeps the last 3 dumps. Verified a restore on build.\n\n"
        f"## 2026-09-14 {edge}: nginx 1.24 from Debian\n- Site config symlinked into sites-enabled.\n",
    )

    # edge: nginx proxy
    sim.write(
        edge,
        "/etc/nginx/nginx.conf",
        "user www-data;\nworker_processes auto;\npid /run/nginx.pid;\n\n"
        "events {\n    worker_connections 768;\n}\n\n"
        "http {\n    sendfile on;\n    keepalive_timeout 65;\n    access_log /var/log/nginx/access.log;\n"
        "    error_log /var/log/nginx/error.log;\n    include /etc/nginx/sites-enabled/*;\n}\n",
    )
    sim.write(
        edge,
        f"/etc/nginx/sites-available/{domain}",
        f"upstream api_backend {{\n    server {app}:{app_port};\n}}\n\n"
        f"server {{\n    listen 80;\n    server_name {domain} www.{domain};\n\n"
        f"    location / {{\n        proxy_pass http://api_backend;\n"
        f"        proxy_set_header Host $host;\n        proxy_set_header X-Real-IP $remote_addr;\n    }}\n}}\n",
    )
    sim.symlink(edge, f"/etc/nginx/sites-enabled/{domain}", f"/etc/nginx/sites-available/{domain}")
    sim.write(edge, "/var/log/nginx/access.log", "")
    sim.add_service(edge, "nginx", "proxy", "/etc/nginx/nginx.conf")

    # app: API container
    env_port = {
        "port_typo": f"{app_port}".replace("0", "O", 1),
        "port_mismatch": str(app_port + 1),
    }.get(fault, str(app_port))
    level = "inof" if fault == "log_level_typo" else "info"
    sim.write(
        app,
        "/srv/api/api.env",
        f"# api container env, mounted read-only into the container\nPORT={env_port}\n"
        f"DB_HOST={db}\nDB_PORT={db_port}\nLOG_LEVEL={level}\nVERSION={version}\nWORKERS=2\n",
    )
    sim.write(app, "/srv/api/app.py", APP_SOURCE)
    sim.write(
        app,
        "/srv/api/docker-compose.yml",
        "services:\n  api:\n    image: registry.lan/orders-api:" + version + "\n"
        "    container_name: api\n    network_mode: host\n    env_file: /srv/api/api.env\n"
        "    restart: unless-stopped\n",
    )
    sim.add_service(
        app,
        "api",
        "app",
        "/srv/api/api.env",
        runtime="docker",
        routes={
            "/": f'{{"service":"api","version":"{version}"}}',
            "/orders/count": '{"count":1284}',
        },
        image=f"registry.lan/orders-api:{version}",
    )

    # db: postgres with backups
    sim.write(
        db,
        "/etc/postgresql/16/main/postgresql.conf",
        f"data_directory = '/var/lib/postgresql/16/main'\nlisten_addresses = '*'\nport = {db_port}\n"
        "max_connections = 100\nshared_buffers = 512MB\nwal_level = replica\nmax_wal_size = 2GB\n",
    )
    sim.add_service(db, "postgresql", "db", "/etc/postgresql/16/main/postgresql.conf")
    sim.write(
        db,
        "/etc/cron.d/pgdump",
        "15 2 * * * postgres pg_dump -Fc orders > /srv/backups/orders-$(date +\\%F).dump\n",
    )
    for i, day in enumerate(("2026-11-14", "2026-11-15", "2026-11-16")):
        sim.sparse(db, f"/srv/backups/orders-{day}.dump", (5 + i) * 1024**3)
    for i in range(6):
        sim.sparse(
            db, f"/var/lib/postgresql/16/main/pg_wal/00000001000000A2000000{i:02X}", 16 * 1024**2
        )
    sim.sparse(db, "/var/lib/postgresql/16/main/base/16384/2619", 2 * 1024**3)
    sim.sparse(db, "/var/log/postgresql/postgresql-16-main.log", 300 * 1024**2)
    sim.disks[db] = 24 * 1024 * 1024  # KiB; the files above fill it to ~95%

    sim.boot()
    return {
        "owner": owner,
        "git_name": git_name,
        "git_email": git_email,
        "notes_dir": notes,
        "prod_services": f"`postgresql` on {db}, `nginx` on {edge}",
        "build_host": "build",
        "project_dir": project,
        "local_host": ws,
        "edge": edge,
        "app": app,
        "db": db,
        "domain": domain,
        "app_port": str(app_port),
        "fault": fault,
    }
