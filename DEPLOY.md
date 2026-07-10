# Deploying IL Profiler to Fly.io

A single always-on container with a persistent volume, behind a password (and
optionally Cloudflare Access). The cloud instance runs **profiles** against a
**prebuilt vector index**; it does *not* ingest (the raw, copyrighted corpus
stays on your machine). `IL_PROFILER_CLOUD=1` hides the ingest UI automatically.

## Prerequisites

- A locally built index (`data/chroma/` populated) — you already have this.
- [flyctl](https://fly.io/docs/flyctl/install/) installed and `fly auth login` done.
- Your `TOGETHER_API_KEY`.

## 1. Create the app and volume

```bash
cd il_profiler

# Create the app WITHOUT deploying yet (reads fly.toml; pick a unique name).
fly apps create il-profiler        # or: fly launch --no-deploy --copy-config

# Persistent volume the container mounts at /app/data (1–2 GB is plenty).
fly volumes create il_data --region iad --size 2
```

If you change the app name or region, update `fly.toml` to match.

## 2. Set secrets

```bash
fly secrets set TOGETHER_API_KEY=sk-...        # your Together key
fly secrets set APP_PASSWORD='choose-a-strong-shared-password'
```

`APP_PASSWORD` turns on the built-in login gate. (Leave it unset only if you put
Cloudflare Access in front instead — see step 5.)

## 3. Deploy

```bash
fly deploy
```

First deploy builds the image and boots one machine. The app will be up at
`https://<app>.fly.dev`, but the volume is empty — seed it next.

## 4. Seed the volume with your local index

Copy the locally built index (and any run snapshots you want reviewers to see)
onto the volume, one time:

```bash
# Bundle the local data dir (index + run snapshots).
tar czf il_data.tgz -C data .

# Push it to the running machine and unpack into the mounted volume.
fly sftp put il_data.tgz /app/data/il_data.tgz
fly ssh console -C "sh -c 'cd /app/data && tar xzf il_data.tgz && rm il_data.tgz'"

# Restart so the app picks up the seeded index.
fly apps restart il-profiler
```

Verify: open the app, log in, and the sidebar should show a green **Vector
index (… chunks)**. The **Run** tab shows the profiles questionnaire (no ingest
controls). The raw `il_data.tgz` can be deleted locally afterward.

## 5. (Optional, recommended for reviewers) Cloudflare Access

The `APP_PASSWORD` gate is a shared password. For per-reviewer identity (invite
by email, revoke individually, no shared secret):

1. Point a custom domain at the app: `fly certs add app.yourdomain.com`, then in
   Cloudflare add a **proxied** CNAME `app -> <app>.fly.dev`.
2. Cloudflare → Zero Trust → Access → Add a **self-hosted** application for
   `app.yourdomain.com`.
3. Add a policy allowing specific reviewer emails (one-time email PIN or Google).
4. Once Access is enforcing, you can `fly secrets unset APP_PASSWORD` to drop the
   redundant password gate.

## Updating the app

Code changes: `git push` then `fly deploy`. The volume (index + runs) persists
across deploys — you only re-seed if you rebuild the index locally.

## Cost / safety notes

- Every profile run spends Together credits on **your** key. Access is gated, so
  only your team/reviewers can trigger runs — keep the password/Access list tight.
- One `shared-cpu-1x` / 1 GB machine + a 2 GB volume is a few dollars a month.
  Bump `memory` in `fly.toml` to `2gb` if large runs OOM.
- The instance stays always-on (`min_machines_running = 1`) so long runs and the
  on-disk index are never interrupted by scale-to-zero.
