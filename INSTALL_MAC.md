# MoCoLUS — Install on macOS

Runs entirely on your Mac. No internet needed after first download. ~5 minutes total.

## 1. Install Docker Desktop

Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).

- **Apple Silicon** (M1/M2/M3/M4) — pick the "Apple Silicon" download
- **Intel Mac** — pick the "Intel Chip" download

Open Docker Desktop from Applications after install. Wait for the whale icon in your menu bar to stop animating (means it's ready).

## 2. Open Terminal

Press **⌘ + Space**, type `Terminal`, press Enter.

## 3. Run MoCoLUS

Paste this single command and hit Enter:

```bash
docker run -d --name moculus -p 8080:8000 --restart unless-stopped ahastava/moculus:latest
```

First run takes ~2 min to download the image (~800MB). After that it starts instantly.

## 4. Open the simulator

Go to **[http://localhost:8080](http://localhost:8080)** in Safari or Chrome.

Done! You should see the BLUE protocol zones, pathology presets, and clinical scenarios.

---

## Common commands

| What | Command |
|---|---|
| Stop simulator | `docker stop moculus` |
| Start again | `docker start moculus` |
| Check if running | `docker ps` |
| View logs | `docker logs moculus` |
| Update to latest | `docker pull ahastava/moculus:latest && docker rm -f moculus && docker run -d --name moculus -p 8080:8000 --restart unless-stopped ahastava/moculus:latest` |
| Uninstall | `docker rm -f moculus && docker rmi ahastava/moculus:latest` |

## Troubleshooting

**"Cannot connect to the Docker daemon"** — Docker Desktop isn't running. Open it from Applications.

**Page doesn't load at localhost:8080** — Wait ~30 seconds after starting (container boot time), then refresh.

**Port 8080 already in use** — Change `8080:8000` to `9090:8000` in the run command, then open `http://localhost:9090`.

**Still stuck?** Email Alex with the output of `docker logs moculus`.
