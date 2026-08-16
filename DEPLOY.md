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

# Persistent volume the container mounts at /app/data. The index is ~0.4 GB,
# so 1 GB is plenty (volume storage is ~$0.15/GB/month).
fly volumes create il_data --region iad --size 1
```

### Cap Fly spending at ~$10

In the [Fly dashboard](https://fly.io/dashboard) → your org → **Billing**:

1. Add **$10 prepaid credit**.
2. Under **Spending management**, set a **monthly spending limit** (e.g. $10) —
   Fly suspends resources if you hit it.

With scale-to-zero (set in `fly.toml`) the machine only bills compute while in
use, so a small-team tool stays far under $10/month — mostly just the ~$0.15/mo
volume.

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

## Shipping the topic layer (optional)

The inductive topic layer (`il_rag/topics.py`) is fitted **locally** — BERTopic
pulls UMAP/HDBSCAN/scikit-learn, and UMAP on 15.5k × 1024 vectors peaks well
above this machine's 1 GB. Only the small JSON results are shipped; the app
reads them and never imports BERTopic.

```bash
# local, one-off (needs: pip install -r requirements-topics.txt)
python scripts/07_run_topics.py fit
python scripts/07_run_topics.py crosstab --run <run_id>

# ship the results (~75 KB) to the volume
RUN=<run_id>
tar czf /tmp/topics_payload.tgz data/topics "data/profiles/runs/$RUN/topics"
fly sftp put /tmp/topics_payload.tgz /app/topics_payload.tgz
fly ssh console -C "sh -c 'cd /app && tar xzf topics_payload.tgz && rm topics_payload.tgz'"
fly apps restart il-profiler
```

The payload contains chunk **ids**, topic keywords and aggregate percentages —
no article text — so it carries none of the corpus's copyright exposure.
Re-fit and re-ship whenever the index is rebuilt, since topics are keyed to
chunk ids.

## Source PDF links — deliberately local-only

The Audit tab can turn each `[excerpt N]` citation into a link to the PDF it
names (see the README). That feature is **off in the cloud, and must stay off.**

Streamlit serves `<app dir>/static` at `/app/static/...` over plain HTTP with
`Access-Control-Allow-Origin: *`, and that route is handled *below* the Python
app — the password gate in `app.py` does not cover it. Anything in `static/` is
readable by anyone who guesses the URL. Publishing the corpus there would
publish copyrighted PDFs.

Two independent things prevent that, and neither is redundant:

1. `IL_PROFILER_CLOUD=1` (set in `fly.toml`) makes `pdf_url()` in `app.py`
   return before it touches the filesystem, so a cloud instance never publishes
   a PDF. Note this stops *publishing*, not *serving* — a file already sitting
   in `static/` would still be served, which is why (2) matters.
2. `static/` is in `.dockerignore`, so locally-published PDFs are never baked
   into the image. The cloud filesystem has no `static/` directory at all.

Also leave `IL_PROFILER_DATASET_ROOT` unset in the cloud (it is not a Fly
secret, and there is no dataset there to point at). If you ever add a corpus to
the volume, do not point this at it.

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
