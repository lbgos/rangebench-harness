"""Simulator: services follow their config files on disk; stubs answer from live state."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from rangebench_homelab.sim import (
    GAP_EXIT,
    Gap,
    Request,
    Response,
    Result,
    Service,
    Simulator,
    go_template,
)

NGINX = """events {}
http {
    server {
        listen 80;
        server_name site.lan;
        location / { proxy_pass http://app:8000; }
    }
}
"""


def world(root: Path, api_port: str = "8000") -> Simulator:
    sim = Simulator(root)
    sim.add_host("ws", "10.0.0.1")
    sim.add_host("edge", "10.0.0.2", aliases=["site.lan"])
    sim.add_host("app", "10.0.0.3")
    sim.write("edge", "/etc/nginx/nginx.conf", NGINX)
    sim.add_service("edge", "nginx", "proxy", "/etc/nginx/nginx.conf")
    sim.write("app", "/srv/api.env", f"PORT={api_port}\n")
    sim.add_service("app", "api", "app", "/srv/api.env", routes={"/": '{"ok":true}'})
    sim.boot()
    sim.local_host = "ws"
    return sim


class SimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_restart_reevaluates_config(self) -> None:
        sim = world(self.root, api_port="80OO")
        api = sim.service("app", "api")
        self.assertEqual(api.state, "failed")
        self.assertIn(
            "502 Bad Gateway", sim.handle("ws", ["curl", "-s", "http://site.lan/"]).stdout
        )
        sim.write("app", "/srv/api.env", "PORT=8000\n")
        self.assertEqual(api.state, "failed", "editing the file alone changes nothing")
        res = sim.handle("app", ["systemctl", "restart", "api"])
        self.assertEqual(res.exit, 0, res.stderr)
        self.assertEqual(api.state, "active")
        self.assertEqual(sim.handle("ws", ["curl", "-s", "http://site.lan/"]).stdout, '{"ok":true}')

    def test_bad_config_and_restart_fails_with_log(self) -> None:
        sim = world(self.root)
        sim.write("app", "/srv/api.env", "PORT=8000\nLOG_LEVEL=loud\n")
        res = sim.handle("app", ["systemctl", "restart", "api"])
        self.assertEqual(res.exit, 1)
        self.assertEqual(sim.service("app", "api").state, "failed")
        log = sim.handle("app", ["journalctl", "-u", "api", "--no-pager"]).stdout
        self.assertIn("invalid LOG_LEVEL 'loud'", log)
        self.assertEqual(sim.handle("app", ["systemctl", "is-active", "api"]).exit, 3)

    def test_nginx_reload_keeps_old_config_on_error(self) -> None:
        sim = world(self.root)
        sim.write("edge", "/etc/nginx/nginx.conf", NGINX.replace("listen 80;", "listen 80"))
        self.assertEqual(sim.handle("edge", ["nginx", "-t"]).exit, 1)
        self.assertEqual(sim.handle("edge", ["systemctl", "reload", "nginx"]).exit, 1)
        self.assertEqual(sim.service("edge", "nginx").state, "active")
        self.assertEqual(sim.handle("ws", ["curl", "-s", "http://site.lan/"]).stdout, '{"ok":true}')

    def test_curl_follows_service_state(self) -> None:
        sim = world(self.root)
        sim.handle("app", ["systemctl", "stop", "api"])
        res = sim.handle(
            "ws", ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://site.lan/"]
        )
        self.assertEqual(res.stdout, "502")
        res = sim.handle("ws", ["curl", "-sS", "http://app:8000/"])
        self.assertEqual(res.exit, 7)
        self.assertIn("Couldn't connect", res.stderr)

    def test_unsupported_flag_is_uniform_gap(self) -> None:
        sim = world(self.root)
        res = sim.handle("app", ["systemctl", "--frobnicate", "status", "api"])
        self.assertEqual(res.exit, GAP_EXIT)
        self.assertTrue(res.stderr.startswith("systemctl: unsupported here:"))
        self.assertEqual(len(sim.gaps), 1)
        self.assertIn("--frobnicate", sim.gaps[0]["argv"])

    def test_transitions_and_probes_are_logged(self) -> None:
        sim = world(self.root)
        sim.step = 1
        sim.handle("app", ["systemctl", "restart", "api"])
        sim.handle("ws", ["curl", "-s", "http://site.lan/"])
        kinds = [(e.kind, e.action, e.probe) for e in sim.events]
        self.assertIn(("transition", "restart", False), kinds)
        self.assertEqual(sim.events[-1].kind, "stub")
        self.assertTrue(sim.events[-1].probe)

    def test_df_tracks_deleted_files(self) -> None:
        sim = world(self.root)
        sim.sparse("app", "/srv/dump.bin", 4 * 1024**3)
        before = sim.disk_used("app")
        self.assertGreater(before, 4 * 1024 * 1024)
        sim.real_path("app", "/srv/dump.bin").unlink()
        self.assertEqual(before - sim.disk_used("app"), 4 * 1024 * 1024)
        du = sim.handle("app", ["du", "-sh", "/", "/usr"]).stdout.splitlines()
        self.assertEqual([line.split("\t")[1] for line in du], ["/", "/usr"])

    def docker_world(self) -> Simulator:
        sim = world(self.root)
        sim.write("app", "/srv/stack/compose.yaml", "services: {}\n")
        sim.write("app", "/srv/stack/web.env", "PORT=9000\n")
        sim.add_service(
            "app",
            "web",
            "app",
            "/srv/stack/web.env",
            runtime="docker",
            image="acme/web:1.2",
            compose="/srv/stack",
        )
        sim.boot()
        return sim

    def test_docker_run_rm_and_create(self) -> None:
        sim = self.docker_world()
        self.assertIn("compose", sim.handle("app", ["docker", "--help"]).stdout)
        res = sim.handle("app", ["docker", "run", "-d", "--name", "web", "acme/web:1.2"])
        self.assertEqual(res.exit, 125, "name in use")
        self.assertEqual(sim.handle("app", ["docker", "rm", "web"]).exit, 1, "running needs -f")
        self.assertEqual(sim.handle("app", ["docker", "rm", "-f", "web"]).exit, 0)
        self.assertNotIn("web", sim.handle("app", ["docker", "ps", "-a"]).stdout)
        self.assertEqual(sim.find("app", "web").present, False)
        res = sim.handle(
            "app",
            [
                "docker",
                "run",
                "-d",
                "--name",
                "web2",
                "-p",
                "9100:9000",
                "-e",
                "PORT=9000",
                "acme/web:1.2",
            ],
        )
        self.assertEqual(res.exit, 0, res.stderr)
        self.assertEqual(
            sim.handle(
                "ws",
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://app:9100/health"],
            ).stdout,
            "200",
        )
        self.assertEqual(
            sim.handle("app", ["docker", "create", "--name", "web3", "acme/web:1.2"]).exit, 0
        )
        rows = sim.handle(
            "app", ["docker", "ps", "-a", "--format", "{{.Names}} {{.State}}"]
        ).stdout.splitlines()
        self.assertEqual(sorted(rows), ["web2 running", "web3 exited"])

    def test_compose_down_up_ps_logs(self) -> None:
        sim = self.docker_world()
        cwd = "/srv/stack"
        self.assertIn("web", sim.handle("app", ["docker", "compose", "ps"], cwd=cwd).stdout)
        sim.handle("app", ["docker", "compose", "down"], cwd=cwd)
        self.assertEqual(sim.handle("app", ["docker", "ps", "-q"]).stdout, "")
        res = sim.handle("app", ["docker", "compose", "-f", "/srv/stack/compose.yaml", "up", "-d"])
        self.assertEqual(res.exit, 0, res.stderr)
        self.assertEqual(sim.find("app", "web").state, "active")
        self.assertIn(
            "Uvicorn running",
            sim.handle("app", ["docker", "compose", "logs", "--tail", "5"], cwd=cwd).stdout,
        )
        actions = [e.action for e in sim.events if e.kind == "transition"]
        self.assertEqual(actions, ["stop", "remove", "create", "start"])

    def test_docker_exec_runs_with_container_env(self) -> None:
        sim = self.docker_world()
        seen: list[str] = []
        sim.exec_hook = lambda host, command, stdin: (seen.append(command), Result("hi\n"))[1]
        self.assertEqual(
            sim.handle("app", ["docker", "exec", "web", "printenv", "PORT"]).stdout, "hi\n"
        )
        self.assertIn("PORT=9000", seen[0])
        sim.handle("app", ["docker", "stop", "web"])
        self.assertEqual(sim.handle("app", ["docker", "exec", "web", "true"]).exit, 1)

    def test_docker_inspect_env_is_the_env_at_last_start(self) -> None:
        sim = self.docker_world()
        env = ["docker", "inspect", "web", "--format", "{{json .Config.Env}}"]
        sim.write("app", "/srv/stack/web.env", "PORT=9000\nVERSION=2\n")
        self.assertEqual(sim.handle("app", env).stdout, '["PORT=9000"]\n')
        self.assertNotIn("VERSION=2", sim.handle("app", ["docker", "inspect", "web"]).stdout)
        sim.handle("app", ["docker", "restart", "web"])
        self.assertEqual(sim.handle("app", env).stdout, '["PORT=9000","VERSION=2"]\n')

    def test_journalctl_since_until(self) -> None:
        sim = world(self.root)
        api = sim.service("app", "api")
        now = sim.now()
        sim.log(api, "old crash", at=now - timedelta(hours=5))
        sim.log(api, "mid warning", at=now - timedelta(hours=2))
        out = sim.handle(
            "app", ["journalctl", "-u", "api", "--since", "3 hours ago", "--no-pager"]
        ).stdout
        self.assertIn("mid warning", out)
        self.assertNotIn("old crash", out)
        until = (now - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        out = sim.handle("app", ["journalctl", "-u", "api", "-S", "-6h", "-U", until]).stdout
        self.assertEqual([line.split(": ", 1)[1] for line in out.splitlines()], ["old crash"])
        self.assertEqual(
            sim.handle("app", ["journalctl", "--since", "last tuesday"]).exit, GAP_EXIT
        )

    def test_systemctl_show_properties_version_and_timers(self) -> None:
        sim = world(self.root)
        props = [
            "ExecStart",
            "ExecReload",
            "ExecMainStartTimestamp",
            "ActiveEnterTimestamp",
            "Environment",
            "EnvironmentFiles",
            "WorkingDirectory",
        ]
        res = sim.handle(
            "app", ["systemctl", "show", "api", *[a for p in props for a in ("-p", p)]]
        )
        self.assertEqual(res.exit, 0, res.stderr)
        self.assertRegex(
            res.stdout, r"ExecMainStartTimestamp=\w{3} \d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC\n"
        )
        self.assertIn("EnvironmentFiles=\n", res.stdout)
        reload = sim.handle(
            "edge", ["systemctl", "show", "nginx", "-p", "ExecReload", "--value"]
        ).stdout
        self.assertIn("/usr/sbin/nginx -s reload", reload)
        self.assertTrue(
            sim.handle("ws", ["systemctl", "--version"]).stdout.startswith("systemd 255")
        )
        self.assertIn("0 timers listed.", sim.handle("ws", ["systemctl", "list-timers"]).stdout)

    def test_ps_fields_and_pid_selection(self) -> None:
        sim = world(self.root)
        api = sim.service("app", "api")
        api.since = sim.now() - timedelta(hours=2, minutes=5)
        res = sim.handle("app", ["ps", "-o", "pid,ppid,etime,cmd", "-p", str(api.pid)])
        self.assertEqual(res.exit, 0, res.stderr)
        header, row = res.stdout.splitlines()
        self.assertEqual(header.split(), ["PID", "PPID", "ELAPSED", "CMD"])
        self.assertEqual(row.split()[:3], [str(api.pid), "1", "02:05:00"])
        self.assertIn("--env-file /srv/api.env", row)
        self.assertEqual(
            sim.handle("app", ["ps", "-o", "etimes=", "-p", str(api.pid)]).stdout.strip(), "7500"
        )
        self.assertEqual(sim.handle("app", ["ps", "-p", "99999"]).exit, 1)
        self.assertIn(str(api.pid), sim.handle("app", ["ps", "-eo", "pid,comm"]).stdout)
        self.assertEqual(sim.handle("app", ["ps", "-o", "pid"]).exit, GAP_EXIT)

    def test_curl_output_and_dump_header(self) -> None:
        sim = world(self.root)
        self.assertEqual(
            sim.handle("ws", ["curl", "-s", "-o", "-", "http://site.lan/"]).stdout, '{"ok":true}'
        )
        res = sim.handle(
            "ws",
            ["curl", "-s", "-o", "body.json", "-w", "%{http_code}", "http://site.lan/"],
            cwd="/root",
        )
        self.assertEqual(res.stdout, "200")
        self.assertEqual(sim.read("ws", "/root/body.json"), '{"ok":true}')
        res = sim.handle("ws", ["curl", "-s", "-D", "-", "http://site.lan/"])
        self.assertTrue(res.stdout.startswith("HTTP/1.1 200 OK\r\n"), res.stdout)
        self.assertTrue(res.stdout.endswith('{"ok":true}'))
        self.assertEqual(
            sim.handle("ws", ["curl", "-s", "-o", "/nope/x", "http://site.lan/"]).exit, 23
        )
        sim.disks["ws"] = sim.disk_used("ws")  # df at 100%
        full = sim.handle("ws", ["curl", "-sS", "-o", "/root/more.json", "http://site.lan/"])
        self.assertEqual(
            (full.exit, full.stderr), (23, "curl: (23) Failure writing output to destination\n")
        )

    def test_app_handler_sees_body_and_headers_through_the_proxy(self) -> None:
        sim = world(self.root)

        def chat(sim: Simulator, svc: Service, req: Request) -> Response | None:
            if req.path != "/v1/chat/completions":
                return None
            model = json.loads(req.body or "{}").get("model")
            if req.headers.get("Authorization") != "Bearer k1":
                return Response(401, '{"error":"invalid_key"}')
            return (
                Response(200, json.dumps({"model": model}))
                if model == "fast"
                else Response(404, '{"error":"model_not_found"}')
            )

        sim.service("app", "api").handler = chat
        url = "http://site.lan/v1/chat/completions"
        auth = ["-H", "Authorization: Bearer k1"]
        ok = sim.handle("ws", ["curl", "-s", *auth, "-d", '{"model":"fast"}', url])
        self.assertEqual(ok.stdout, '{"model": "fast"}')
        self.assertIn(
            "model_not_found",
            sim.handle("ws", ["curl", "-s", *auth, "--json", '{"model":"x"}', url]).stdout,
        )
        self.assertIn(
            "invalid_key", sim.handle("ws", ["curl", "-s", "-d", '{"model":"fast"}', url]).stdout
        )
        sim.write("ws", "/root/req.json", '{"model":"fast"}')
        self.assertEqual(
            sim.handle("ws", ["curl", "-s", *auth, "-d", "@req.json", url], cwd="/root").stdout,
            '{"model": "fast"}',
        )
        self.assertEqual(
            sim.handle("ws", ["curl", "-s", "http://site.lan/"]).stdout,
            '{"ok":true}',
            "None falls through to routes",
        )

    def test_ssh_version_df_block_size_and_du_files0(self) -> None:
        sim = world(self.root)
        self.assertIn("OpenSSH_", sim.handle("ws", ["ssh", "-V"]).stderr)
        sim.sparse("app", "/srv/big.img", 3 * 2**30)
        bytes_ = sim.handle("app", ["df", "-B1", "/"]).stdout.splitlines()
        self.assertIn("1B-blocks", bytes_[0])
        self.assertEqual(int(bytes_[1].split()[1]), sim.disks["app"] * 1024)
        self.assertRegex(sim.handle("app", ["df", "-BG", "/"]).stdout, r"1G-blocks[\s\S]*\s20G\s")
        sim.write("app", "/root/list", "/srv/big.img\0/srv/api.env\0")
        res = sim.handle("app", ["du", "-b", "--files0-from=/root/list"])
        self.assertEqual(res.stdout, f"{3 * 2**30}\t/srv/big.img\n4096\t/srv/api.env\n")
        piped = sim.handle("app", ["du", "-sh", "--files0-from=-"], stdin="/srv/big.img\0")
        self.assertEqual(piped.stdout, "3.0G\t/srv/big.img\n")

    def test_go_template_subset(self) -> None:
        doc = {
            "State": {"Running": True},
            "Config": {"Env": ["A=1", "B=2"], "Labels": {"b": "2", "a": "1"}, "Cmd": None},
            "Mounts": [{"Source": "/s", "Destination": "/d"}],
        }
        self.assertEqual(
            go_template("{{.State.Running}} {{json .Config.Env}}", doc), 'true ["A=1","B=2"]'
        )
        self.assertEqual(
            go_template("{{range .Config.Env}}{{println .}}{{end}}", doc), "A=1\nB=2\n"
        )
        self.assertEqual(
            go_template('{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\\n"}}{{end}}', doc),
            "/s -> /d\n",
        )
        self.assertEqual(
            go_template("{{range $k, $v := .Config.Labels}}{{$k}}={{$v}} {{end}}", doc), "a=1 b=2 "
        )
        self.assertEqual(
            go_template(
                '{{if .Config.Cmd}}x{{else}}none{{end}} {{index .Config.Labels "a"}} {{len .Mounts}}',
                doc,
            ),
            "none 1 1",
        )
        self.assertEqual(
            go_template("{{.Config.Labels}} {{.State.Running | json}}", doc), "map[a:1 b:2] true"
        )
        with self.assertRaises(Gap):
            go_template("{{.Config.Nope}}", doc)

    def test_docker_formats_filters_volumes_networks(self) -> None:
        sim = self.docker_world()
        web = sim.find("app", "web")
        assert web
        web.volumes = ["/srv/stack/web.env:/app/.env:ro", "webdata:/data"]
        fmt = '{{json .Config.Cmd}} {{.State.StartedAt}} {{.LogPath}} {{index .Config.Labels "com.docker.compose.project"}}'
        out = sim.handle("app", ["docker", "inspect", "web", "--format", fmt]).stdout
        self.assertRegex(
            out,
            r'^\["uvicorn","app:app","--host","0.0.0.0","--port","9000"\] \d{4}-\d\d-\d\dT[\d:.]+Z /var/lib/docker/containers/\w{64}/\w{64}-json.log stack\n$',
        )
        mounts = sim.handle(
            "app",
            [
                "docker",
                "inspect",
                "-f",
                "{{range .Mounts}}{{.Type}} {{.Source}} {{.RW}};{{end}}",
                "web",
            ],
        ).stdout
        self.assertEqual(
            mounts,
            "bind /srv/stack/web.env false;volume /var/lib/docker/volumes/webdata/_data true;\n",
        )
        self.assertEqual(
            sim.handle("app", ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]).stdout,
            "acme/web:1.2\n",
        )
        table = sim.handle(
            "app", ["docker", "images", "--format", "table {{.Repository}}\\t{{.Size}}"]
        ).stdout.splitlines()
        self.assertEqual(table[0].split(), ["REPOSITORY", "SIZE"])
        self.assertEqual(
            sim.handle(
                "app", ["docker", "ps", "--filter", "name=^web$", "--format", "{{.Names}}"]
            ).stdout,
            "web\n",
        )
        self.assertEqual(
            sim.handle("app", ["docker", "ps", "-a", "--filter", "status=exited", "-q"]).stdout, ""
        )
        self.assertEqual(sim.handle("app", ["docker", "volume", "ls", "-q"]).stdout, "webdata\n")
        self.assertIn(
            '"Mountpoint": "/var/lib/docker/volumes/webdata/_data"',
            sim.handle("app", ["docker", "volume", "inspect", "webdata"]).stdout,
        )
        self.assertEqual(sim.handle("app", ["docker", "volume", "inspect", "nope"]).exit, 1)
        nets = sim.handle(
            "app", ["docker", "network", "ls", "--format", "{{.Name}} {{.Driver}}"]
        ).stdout.splitlines()
        self.assertEqual(nets, ["bridge bridge", "host host", "none null"])
        self.assertIn("Images", sim.handle("app", ["docker", "system", "df"]).stdout)
        self.assertEqual(
            sim.handle(
                "app", ["docker", "info", "--format", "{{.ContainersRunning}} {{.Driver}}"]
            ).stdout,
            "1 overlay2\n",
        )
        self.assertEqual(
            sim.handle(
                "app", ["docker", "image", "inspect", "acme/web:1.2", "-f", "{{index .RepoTags 0}}"]
            ).stdout,
            "acme/web:1.2\n",
        )

    def test_curl_write_out_several_urls_and_forms(self) -> None:
        sim = world(self.root)
        seen: list[Request] = []

        def echo(sim: Simulator, svc: Service, req: Request) -> Response | None:
            seen.append(req)
            return Response(302, "", [("Location", "/next")]) if req.path == "/old" else None

        sim.service("app", "api").handler = echo
        fmt = "%{http_code} %{content_type} %{size_download} %{remote_ip}:%{remote_port} %{num_redirects} %{time_total} %header{server}\\n"
        out = sim.handle(
            "ws", ["curl", "-s", "-o", "/dev/null", "-w", fmt, "http://site.lan/"]
        ).stdout
        self.assertRegex(
            out, r"^200 application/json 11 10\.0\.0\.2:80 0 \d\.\d{6} nginx/1\.24\.0\n$"
        )
        out = sim.handle(
            "ws",
            [
                "curl",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code} %{redirect_url}",
                "http://site.lan/old",
            ],
        ).stdout
        self.assertEqual(out, "302 http://site.lan/next")
        down = sim.handle(
            "ws",
            ["curl", "-s", "-w", "%{http_code} %{remote_ip} %{exitcode}", "http://nowhere.lan/"],
        )
        self.assertEqual((down.stdout, down.exit), ("000  6", 6))
        two = sim.handle(
            "ws",
            ["curl", "-s", "-w", " %{http_code}\\n", "http://site.lan/", "http://app:8000/health"],
        ).stdout
        self.assertEqual(two, '{"ok":true} 200\n{"status":"ok"} 200\n')
        sim.write("ws", "/root/up.txt", "hello")
        sim.handle(
            "ws",
            ["curl", "-s", "-F", "f=@up.txt", "-F", "note=hi", "http://site.lan/upload"],
            cwd="/root",
        )
        form = seen[-1]
        self.assertTrue(form.headers["Content-Type"].startswith("multipart/form-data; boundary="))
        self.assertIn('name="f"; filename="up.txt"', form.body)
        self.assertIn("hello", form.body)
        sim.handle("ws", ["curl", "-s", "-r", "0-9", "--http1.1", "http://site.lan/x/../y"])
        self.assertEqual((seen[-1].path, seen[-1].headers["Range"]), ("/y", "bytes=0-9"))
        sim.handle("ws", ["curl", "-s", "--path-as-is", "http://site.lan/x/../y"])
        self.assertEqual(seen[-1].path, "/x/../y")
        self.assertIn("curl 8.", sim.handle("ws", ["curl", "--version"]).stdout)

    def test_systemctl_mask_kill_unit_files_and_show(self) -> None:
        sim = world(self.root)
        api = sim.service("app", "api")
        res = sim.handle("app", ["systemctl", "mask", "api"])
        self.assertIn("Created symlink /etc/systemd/system/api.service → /dev/null", res.stderr)
        self.assertEqual(sim.handle("app", ["systemctl", "restart", "api"]).exit, 1)
        self.assertEqual(sim.handle("app", ["systemctl", "is-enabled", "api"]).stdout, "masked\n")
        self.assertIn(
            "api.service masked",
            " ".join(sim.handle("app", ["systemctl", "list-unit-files", "api*"]).stdout.split()),
        )
        self.assertEqual(
            sim.handle("app", ["systemctl", "show", "api", "-p", "LoadState", "--value"]).stdout,
            "masked\n",
        )
        self.assertEqual(api.state, "active")
        self.assertEqual(
            sim.handle("app", ["systemctl", "mask", "--now", "api"]).exit,
            0,
            "already masked; --now still stops it",
        )
        self.assertEqual(api.state, "inactive")
        self.assertEqual(
            sim.handle("app", ["systemctl", "show", "api", "-p", "Loaded", "-p", "User"]).stdout,
            "User=\n",
        )
        sim.handle("app", ["systemctl", "unmask", "api"])
        sim.handle("app", ["systemctl", "start", "api"])
        self.assertEqual(sim.handle("app", ["systemctl", "is-enabled", "api"]).stdout, "enabled\n")
        pid = api.pid
        sim.handle("app", ["systemctl", "kill", "-s", "KILL", "api"])
        self.assertEqual(
            (api.state, api.pid != pid), ("active", True), "Restart=on-failure brings it back"
        )
        sim.handle("app", ["systemctl", "kill", "api"])
        self.assertEqual(api.state, "inactive", "SIGTERM is a clean exit")
        self.assertEqual(
            [e.action for e in sim.events if e.kind == "transition"],
            ["stop", "start", "restart", "stop"],
        )
        show = sim.handle(
            "app",
            [
                "systemctl",
                "show",
                "api",
                "-p",
                "TriggeredBy,Wants,User,RootDirectory,EnvironmentFile",
            ],
        ).stdout
        self.assertEqual(
            show,
            "RootDirectory=\nUser=\nWants=\nTriggeredBy=\n",
            "systemd's order: service, then unit properties",
        )
        two = sim.handle(
            "app", ["systemctl", "show", "api", "ghost", "-p", "LoadState", "-p", "ExecStart"]
        ).stdout
        self.assertRegex(
            two,
            r"^ExecStart=\{ path=/usr/bin/python3 .*\}\nLoadState=loaded\n\nExecStart=\nLoadState=not-found\n$",
        )
        self.assertEqual(sim.handle("app", ["systemctl", "is-system-running"]).stdout, "running\n")

    def test_ps_sort_cwd_user_and_errors(self) -> None:
        sim = world(self.root)
        sim.write("app", "/srv/w.env", "PORT=8100\n")
        w = sim.add_service("app", "worker", "app", "/srv/w.env", workdir="/opt/worker", user="svc")
        sim.boot()
        rows = sim.handle("app", ["ps", "-eo", "pid,user,cwd", "--sort=-pid"]).stdout.splitlines()
        self.assertEqual(rows[1].split(), [str(w.pid), "svc", "/opt/worker"])
        self.assertEqual(rows[-1].split(), ["1", "root", "/"])
        self.assertEqual(
            sim.handle("app", ["ps", "-o", "pid,comm,args=", "-p", "1"])
            .stdout.splitlines()[0]
            .split(),
            ["PID", "COMMAND"],
        )
        self.assertEqual(
            sim.handle("app", ["ps", "-o", "pgid=", "-p", str(w.pid)]).stdout.strip(), str(w.pid)
        )
        self.assertEqual(
            sim.handle("app", ["ps", "-o", "pid=", "1", str(w.pid)]).stdout.split(),
            ["1", str(w.pid)],
        )
        missing = sim.handle("app", ["ps", "-o", "lstart=", "-p"])
        self.assertEqual(missing.exit, 1)
        self.assertIn("list of process IDs must follow -p", missing.stderr)
        self.assertIn(
            "unknown user-defined format specifier",
            sim.handle("app", ["ps", "-o", "bogus", "-p", "1"]).stderr,
        )
        self.assertTrue(sim.handle("app", ["ps", "aux", "--sort=-%cpu"]).stdout.startswith("USER"))
        self.assertIn(
            "WorkingDirectory=/opt/worker", sim.handle("app", ["systemctl", "cat", "worker"]).stdout
        )

    def test_ss_unix_sockets_and_filters(self) -> None:
        sim = world(self.root)
        self.assertIn("/run/systemd/journal/stdout", sim.handle("app", ["ss", "-lx"]).stdout)
        est = sim.handle("app", ["ss", "-tn", "state", "established"]).stdout
        self.assertEqual(est, "Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n")
        port = sim.handle("app", ["ss", "-tlnH", "( sport = :8000 )"]).stdout.splitlines()
        self.assertEqual(len(port), 1)
        self.assertIn(":8000", port[0])
        self.assertEqual(sim.handle("app", ["ss", "-tlnH", "dport = :8000"]).stdout.strip(), "")

    def test_du_df_flags_and_journal_priority(self) -> None:
        sim = world(self.root)
        sim.sparse("app", "/srv/d/dump.bin", 5 * 2**20)
        sim.write("app", "/srv/d/keep.txt", "x")
        self.assertEqual(
            sim.handle("app", ["du", "-B1", "--apparent-size", "/srv/d/dump.bin"]).stdout,
            f"{5 * 2**20}\t/srv/d/dump.bin\n",
        )
        self.assertEqual(
            sim.handle("app", ["du", "-h", "--block-size=1K", "/srv/d/dump.bin"]).stdout,
            "5120\t/srv/d/dump.bin\n",
        )
        self.assertEqual(
            sim.handle("app", ["du", "-sb", "--exclude=*.bin", "/srv/d"]).stdout, "8192\t/srv/d\n"
        )
        self.assertIn("GNU coreutils", sim.handle("app", ["du", "--version"]).stdout)
        avail = sim.handle("app", ["df", "-B1", "--output=avail", "/"]).stdout.splitlines()
        self.assertEqual(
            (avail[0].strip(), int(avail[1])),
            ("Avail", (sim.disks["app"] - sim.disk_used("app")) * 1024),
        )
        self.assertIn("1M-blocks", sim.handle("app", ["df", "-m", "/"]).stdout)
        api = sim.service("app", "api")
        sim.log(api, "ERROR: db timeout")
        sim.log(api, "WARNING: slow query")
        errs = sim.handle("app", ["journalctl", "-u", "api", "-p", "err", "--no-pager"]).stdout
        self.assertIn("db timeout", errs)
        self.assertNotIn("slow query", errs)
        self.assertIn(
            "slow query", sim.handle("app", ["journalctl", "-u", "api", "-p", "warning"]).stdout
        )
        self.assertEqual(
            sim.handle("app", ["journalctl", "-g", "slow", "-o", "cat"]).stdout,
            "WARNING: slow query\n",
        )
        self.assertEqual(sim.handle("app", ["journalctl", "-b", "-1", "-u", "api"]).exit, 1)
        self.assertIn("IDX BOOT ID", sim.handle("app", ["journalctl", "--list-boots"]).stdout)

    def test_nginx_c_start_and_ssh_g(self) -> None:
        sim = world(self.root)
        sim.write("edge", "/tmp/bad.conf", "events {}\nhttp { server { listen 80 }\n")
        bad = sim.handle("edge", ["nginx", "-t", "-c", "/tmp/bad.conf"])
        self.assertEqual(bad.exit, 1)
        self.assertIn("configuration file /tmp/bad.conf test failed", bad.stderr)
        self.assertEqual(sim.handle("edge", ["nginx", "-t", "-c", "/etc/nginx/nginx.conf"]).exit, 0)
        self.assertIn("still could not bind()", sim.handle("edge", ["nginx"]).stderr)
        sim.handle("edge", ["nginx", "-s", "stop"])
        self.assertEqual(sim.handle("edge", ["nginx"]).exit, 0)
        self.assertEqual(sim.service("edge", "nginx").state, "active")
        cfg = sim.handle("ws", ["ssh", "-G", "deploy@app"]).stdout
        self.assertIn("user deploy\nhostname app\nport 22\n", cfg)


if __name__ == "__main__":
    unittest.main()
