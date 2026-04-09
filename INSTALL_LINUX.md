# MoCoLUS — Install on Linux

Runs entirely on your machine. No internet needed after first download. ~5 minutes total.

## 1. Install Docker

**Ubuntu / Debian:**
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```
Log out and back in (so your user can run Docker without `sudo`).

**Fedora / RHEL:**
```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```
Log out and back in.

**Arch:**
```bash
sudo pacman -S docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```
Log out and back in.

## 2. Run MoCoLUS

Paste this single command in your terminal:

```bash
docker run -d --name moculus -p 8080:8000 --restart unless-stopped ahastava/moculus:latest
```

First run takes ~2 min to download the image (~800MB). After that it starts instantly.

## 3. Open the simulator

Go to **[http://localhost:8080](http://localhost:8080)** in Firefox or Chrome.

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

**"permission denied while trying to connect to the Docker daemon socket"** — You haven't logged out and back in after adding your user to the `docker` group. Either do that, or prefix commands with `sudo`.

**"Cannot connect to the Docker daemon"** — Start the Docker service: `sudo systemctl start docker`.

**Page doesn't load at localhost:8080** — Wait ~30 seconds after starting (container boot time), then refresh.

**Port 8080 already in use** — Change `8080:8000` to `9090:8000` in the run command, then open `http://localhost:9090`.

**Still stuck?** Email Alex with the output of `docker logs moculus`.
