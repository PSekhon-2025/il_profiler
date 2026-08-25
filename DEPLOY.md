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

The Topics tab's **keyword retention** section (§9.6) adds a second local-only
artifact, `data/lexicon/`: the cached word vectors and the word-pair
calibration that turns a raw cosine into a readable percentile. It is built from
the Chroma index and costs a few hundred embedding calls once. It is *read-only*
in the container — the app never embeds on demand — so the neighborhood explorer
works there with no API key at all.

```bash
# local, one-off (needs: pip install -r requirements-topics.txt)
python scripts/07_run_topics.py fit
python scripts/07_run_topics.py crosstab --run <run_id>

# local, one-off: the word-pair calibration behind the keyword-retention scores
python scripts/13_run_topic_keywords.py calibrate
python scripts/13_run_topic_keywords.py score --run <run_id>

# ship the results (~8 MB, nearly all of it word_vectors.npz) to the volume
RUN=<run_id>
tar czf /tmp/topics_payload.tgz data/topics data/lexicon     "data/profiles/runs/$RUN/topics" "data/profiles/runs/$RUN/topic_keywords"
fly sftp put /tmp/topics_payload.tgz /app/topics_payload.tgz
fly ssh console -C "sh -c 'cd /app && tar xzf topics_payload.tgz && rm topics_payload.tgz'"
fly apps restart il-profiler
```

The payload contains chunk **ids**, topic keywords, single-word vectors and
aggregate percentages — no article text — so it carries none of the corpus's
copyright exposure. Re-fit and re-ship whenever the index is rebuilt, since
topics are keyed to chunk ids; re-run `calibrate` too, since the background
distribution is a property of the corpus vocabulary. If `data/lexicon/` is
absent the app degrades cleanly: the semantic rung is disabled and the page says
so, rather than falling back to an uncalibrated cosine.

## Shipping the source documents (so `[excerpt N]` links work)

The Audit tab turns each `[excerpt N]` citation into a link to the document it
names. That only works where the documents actually exist, and the image ships
neither the corpus nor the generated press-record PDFs (`data/` and `static/`
are both `.dockerignore`d). So they travel to the volume separately, like the
prebuilt index and the topic layer.

**Read this first.** Streamlit serves `<app dir>/static` at `/app/static/...`
with `Access-Control-Allow-Origin: *`, and that route is handled *below* the
Python app, so **`APP_PASSWORD` does not cover it**. Once the corpus is on the
machine, any PDF a viewer has caused to be published is fetchable by anyone who
reaches the hostname and guesses the path. What does cover it is **Cloudflare
Access** (§5), which gates the whole hostname at the edge, static route
included. If this instance is not behind Access, treat the corpus as reachable
by anyone who knows the URL.

`IL_PROFILER_DISABLE_SOURCE_LINKS=1` turns the links off again without a code
change.

### 1. Build the archive locally

```bash
# needs IL_PROFILER_DATASET_ROOT set and, for press links, stage 10 already run
.venv/Scripts/python scripts/11_bundle_cloud_sources.py --dry-run   # check the size
.venv/Scripts/python scripts/11_bundle_cloud_sources.py
```

Roughly **500 MB**: 98 corpus PDFs (489 MB — one of them, `oversight of ai.pdf`,
is 161 MB on its own), 3,000 press-record PDFs (14 MB), and the two maps (1 MB).
Only PDFs a chunk can actually cite are included.

### 2. Make room on the volume

The index is ~0.4 GB, so a 1 GB volume will not hold both. Check and extend:

```bash
fly volumes list
fly volumes extend <volume-id> --size 3
```

### 3. Ship and unpack

```bash
fly sftp put dist/cloud_sources.tar /app/data/cloud_sources.tar
fly ssh console -C "sh -c 'cd /app/data && tar xf cloud_sources.tar && rm cloud_sources.tar'"
fly apps restart il-profiler
```

`fly.toml` already sets `IL_PROFILER_DATASET_ROOT=/app/data/corpus`; the article
PDFs and both maps need no setting, because their defaults
(`/app/data/articles`, `/app/data/pdf_pages.json`,
`/app/data/article_sources.json`) are exactly where the archive unpacks them.

### 4. Verify

```bash
fly ssh console -C "ls /app/data/corpus/OpenAI/Essays and Statements/"
```

Then open the Audit tab on a run with quotes: the excerpt numbers should be
links. Cited PDFs are copied into `/app/static/` lazily, on first reference —
that copy lives on the machine's ephemeral rootfs, not the volume, so it is
rebuilt after each deploy and costs no volume space.

**The ids must match.** The links resolve published documents by the filename in
the chunk metadata, and press records by chunk id, so the maps are only correct
if the deployed index was built from the same `pdf_corpus.txt` and the same
RTFs. If you rebuild the index from a differently-ordered corpus, re-run stages
10 and 11 and re-ship. `scripts/10_build_article_pdfs.py --check-ids` compares
the press map against a local index.

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
