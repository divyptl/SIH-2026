# Deployment

| Part | Where | Updated by |
|---|---|---|
| Web app (`frontend/`) | Cloudflare Pages: `satquerymindcraft.pages.dev` | `.github/workflows/deploy-frontend.yml` on pushes to `main` |
| API image | GitHub Container Registry: `ghcr.io/divyptl/satquery-api` (linux/arm64) | `.github/workflows/deploy-backend.yml` on pushes to `main` |
| API (`backend/` + `ml/`) | Raspberry Pi 5 (8 GB), Docker Compose, public HTTPS URL through Tailscale Funnel | `deploy/pi/update.sh`, run by cron on the Pi every 5 min |
| Fine-tuned checkpoints (~2.3 GB) | Private Hugging Face model repo `prayag17/satquery-weights` | `deploy/upload_weights.py`, by hand |

After a push to `main`, GitHub builds the image on a native ARM runner and pushes
it as `:latest`. Within 5 minutes the Pi pulls it and swaps the container. The
API is back in seconds and answers slowly for about a minute while the models load.

The API runs on CPU and holds about 3.1 GB of RAM with every model loaded.

## Files

- `Dockerfile`: CPU-only image (x86-64 or ARM64). Swaps the lockfile's CUDA torch for
  the CPU build of the same version, and bakes in GroundingDINO's base weights.
- `start.sh`, `fetch_weights.py`: container entrypoint. Downloads `WEIGHTS_REPO` into
  `checkpoints/`, then starts uvicorn with one worker.
- `upload_weights.py`: pushes the checkpoints the API serves to the weights repo.
- `pi/compose.yaml`: runs the published image on `127.0.0.1:7860`, with the downloads
  kept in Docker volumes.
- `pi/update.sh`: pulls a new image and restarts the container only when it changed.
- `pi/.env.example`: settings and secrets for the Pi.

## One-time Pi setup

**1. Check the OS.** `uname -m` must print `aarch64` (64-bit Raspberry Pi OS). About
10 GB of free disk is needed; an SSD loads the models much faster than an SD card.

**2. Docker.**

```sh
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER      # log out and back in afterwards
```

**3. Swap**, as headroom above the ~3 GB the API uses. On Raspberry Pi OS Bookworm:

```sh
sudo dphys-swapfile swapoff
sudo sed -i 's/^#\?CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/; s/^#\?CONF_MAXSWAP=.*/CONF_MAXSWAP=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```

**4. The deploy folder.** The Pi needs only these three files, not the whole repo:

```sh
mkdir -p ~/satquery && cd ~/satquery
base=https://raw.githubusercontent.com/divyptl/SIH-2026/main/deploy/pi
curl -fsSL -O $base/compose.yaml -O $base/update.sh -O $base/.env.example
chmod +x update.sh
cp .env.example .env && nano .env   # fill in HF_TOKEN and OPENROUTER_API_KEY
```

The values in `backend/.env` work for both keys.

**5. Access to the image.** GitHub makes new packages private. Use one of these:

- *Public image (simplest):* the package owner (divyptl) opens the package on GitHub,
  goes to **Package settings → Change visibility → Public**. The image holds only code;
  weights and keys stay outside it.
- *Private image:* log the Pi in once with a classic personal access token that has the
  `read:packages` scope:
  `echo <token> | docker login ghcr.io -u <github-user> --password-stdin`

**6. First start.** The first image must already exist, so run **Deploy backend** from
the repo's Actions tab first (or push to `main`). Then, on the Pi:

```sh
~/satquery/update.sh
docker compose -f ~/satquery/compose.yaml logs -f api   # first start downloads ~2.3 GB of checkpoints
curl http://127.0.0.1:7860/health
```

**7. Automatic updates** every 5 minutes. Run `crontab -e` and add:

```
*/5 * * * * $HOME/satquery/update.sh >> $HOME/satquery/update.log 2>&1
```

**8. Public HTTPS URL** with Tailscale Funnel (free, and the URL doesn't change):

```sh
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale funnel --bg 7860    # the first time, it prints a link to enable Funnel
tailscale funnel status            # shows https://<pi-name>.<tailnet>.ts.net
```

Check `https://<pi-name>.<tailnet>.ts.net/health` from another device.

## Web app on Cloudflare Pages

The API URL is baked in at build time. Set these in the GitHub repo under
**Settings → Secrets and variables → Actions**:

- secrets: `CLOUDFLARE_API_TOKEN` (permission *Cloudflare Pages: Edit*) and `CLOUDFLARE_ACCOUNT_ID`
- variables: `CF_PAGES_PROJECT` = `satquerymindcraft`, `VITE_API_BASE_URL` = the Funnel URL

Then run **Deploy frontend** from the Actions tab once. After that, both halves
redeploy on every push to `main` that touches them.

## Operating it

```sh
docker compose -f ~/satquery/compose.yaml logs -f api
docker stats satquery-api-1
tail ~/satquery/update.log
```

Don't push to `main` during a demo: the Pi picks the change up within 5 minutes and
the API is slow for about a minute while it restarts.

**Rolling back:** each build is also tagged `sha-<short-commit>`. To pin one, add
`SATQUERY_IMAGE=ghcr.io/divyptl/satquery-api:sha-<short-commit>` to `~/satquery/.env`
and run `update.sh`. Remove the line to follow `:latest` again.

**Changing checkpoints:** upload the new file to the weights repo, and update both the
path in the Dockerfile's `ENV` block and the `FILES` list in `upload_weights.py`.
To make the Pi download them again, run
`cd ~/satquery && docker compose down && docker volume rm satquery_checkpoints && ./update.sh`.
