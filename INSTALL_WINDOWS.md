# MoCoLUS — Install on Windows

Runs entirely on your PC. No internet needed after first download. ~5-10 minutes total.

## 1. Install Docker Desktop

Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/). Pick the **Windows** download.

During install, leave the default options checked (including "Use WSL 2 instead of Hyper-V"). You may need to restart your computer after install.

Open Docker Desktop from the Start menu. Wait for the whale icon in your system tray (bottom-right) to stop animating (means it's ready). First launch can take 1-2 minutes.

## 2. Open PowerShell

Press **Windows key**, type `PowerShell`, press Enter.

## 3. Run MoCoLUS

Paste this single command and hit Enter:

```powershell
docker run -d --name moculus -p 8080:8000 --restart unless-stopped ahastava/moculus:latest
```

First run takes ~2 min to download the image (~800MB). After that it starts instantly.

## 4. Open the simulator

Go to **[http://localhost:8080](http://localhost:8080)** in Chrome or Edge.

Done! You should see the BLUE protocol zones, pathology presets, and clinical scenarios.

---

## Common commands

Run these in PowerShell:

| What | Command |
|---|---|
| Stop simulator | `docker stop moculus` |
| Start again | `docker start moculus` |
| Check if running | `docker ps` |
| View logs | `docker logs moculus` |
| Update to latest | `docker pull ahastava/moculus:latest; docker rm -f moculus; docker run -d --name moculus -p 8080:8000 --restart unless-stopped ahastava/moculus:latest` |
| Uninstall | `docker rm -f moculus; docker rmi ahastava/moculus:latest` |

## Troubleshooting

**"Docker Desktop is not running"** — Open Docker Desktop from the Start menu and wait for the whale icon.

**"WSL 2 installation is incomplete"** — Open PowerShell as Administrator and run `wsl --install`, then restart your PC.

**Page doesn't load at localhost:8080** — Wait ~30 seconds after starting (container boot time), then refresh.

**Port 8080 already in use** — Change `8080:8000` to `9090:8000` in the run command, then open `http://localhost:9090`.

**Still stuck?** Email Alex with the output of `docker logs moculus`.
