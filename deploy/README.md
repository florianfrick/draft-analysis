# Hosted web app (AWS Lambda, near-$0 at low traffic)

This is the deployment path for the **live demo embedded in the portfolio
page**. It keeps the exact PyTorch heads and the real LangGraph tool loop; the
only intentional differences from `bash webapp/serve.sh` are:

* the process is stateless — `webapp/lambda_handler.py` rebuilds `DraftState`
  from the posted board on every request, so sandboxes do not survive between
  requests (the browser already re-sends the board every time);
* `sql` and `soloq_matchup` are ablated and removed from the system prompt —
  neither corpus ships in the slim image;
* the API is public (an iframe cannot sign IAM requests). LLM spend is bounded
  by the Cloudflare Worker's `AGENT_DAILY_LIMIT` turn cap; raw request volume
  relies on the Lambda free tier, with no per-browser auth or rate limit.

The portfolio page stays on GitHub Pages, but the demo itself is served by the
Function URL, which also serves `index.html`, `app.js` and `style.css`. Embed
that URL directly:

```html
<iframe src="https://<function-url>/" ...></iframe>
```

`webapp/static/index.html` uses relative asset paths, and `app.js` / `agent.js`
still read `?api=<FunctionURL>` (or `window.API_BASE`) before every `/api/...`
call, so the same files can also be hosted on GitHub Pages and pointed at the
API.

## One-time console work

Do this in the AWS console before using the CLI:

1. Region: **us-west-2 (Oregon)**.
2. Enable MFA on the root account and on the IAM user you will use.
3. Set a **$1 billing budget / alarm** with an email notification. This is the
   safety net if traffic or a misconfiguration ever leaves the free tier.
4. Create an access key for that IAM user, then:

```bash
aws configure                 # region: us-west-2
aws sts get-caller-identity   # confirm the right account
```

AWS CLI v2 is on the user `PATH` here (`aws --version` confirms 2.42.2), and
Git Bash gets a `~/bin/aws` shim. On a machine without it, install AWS CLI v2
or call `C:\Program Files\Amazon\AWSCLIV2\aws.exe` explicitly.

The CLI identity needs IAM, ECR, Lambda, CloudWatch Logs and `iam:PassRole`.

## LLM token

The DeepSeek key stays in Cloudflare. The webapp uses the separate
`WEBAPP_LLM_TOKEN` bearer token you generated in `.env`. Copy the **exact**
current `LLM_API_KEY` value from `.env` into
`deploy/terraform/terraform.tfvars` as `llm_token` — do not paste it into the
tracked `terraform.tfvars.example` or the shell history. A different token
produces a clean 401 from the Worker even though the deployment itself is
healthy; compare SHA-256 prefixes (never the values) to confirm:

```bash
python - <<'PY'
import hashlib, re
from pathlib import Path
def key(path, name):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if re.match(rf"^{name}\s*=", line):
            return line.partition("=")[2].strip().strip('"')
env = key(".env", "LLM_API_KEY")
tf = key("deploy/terraform/terraform.tfvars", "llm_token")
print(bool(env and tf and env == tf))
PY
```

The Worker must already have the `/v1/chat/completions` route deployed with
`MAX_TOKENS = "3000"` and `WEBAPP_LLM_TOKEN` set as a secret.

## Build and push the slim image

The image is CPU-only PyTorch plus the two shipped heads: **2.25 GB
uncompressed, 496 MiB compressed** — about a fifth of the dev image's ~12 GB
by either measure. The CPU torch wheel dominates; Lambda needs an AWS base
image, so `Dockerfile.web` extends `public.ecr.aws/lambda/python:3.13`.

Create the tfvars file first:

```bash
cp deploy/terraform/terraform.tfvars.example deploy/terraform/terraform.tfvars
# edit terraform.tfvars: llm_token = <WEBAPP_LLM_TOKEN>
```

Two-phase apply is necessary because Lambda cannot be created before the image
exists in ECR:

```bash
cd deploy/terraform
terraform init
terraform apply \
  -target=aws_ecr_repository.web \
  -target=aws_ecr_lifecycle_policy.web \
  -target=aws_iam_role.web \
  -target=aws_iam_role_policy_attachment.logs \
  -target=aws_cloudwatch_log_group.web
cd ../..
```

Then, from the repo root:

```bash
TAG=latest
docker build --platform linux/amd64 --provenance=false \
  -f Dockerfile.web -t "draft-web:$TAG" .

REPO=$(terraform -chdir=deploy/terraform output -raw ecr_repository_uri)
REGISTRY=${REPO%%/*}

aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin "$REGISTRY"

docker tag "draft-web:$TAG" "$REPO:$TAG"
docker push "$REPO:$TAG"
```

Local smoke check of the image itself (no AWS):

```bash
docker run --rm -p 9000:8080 -e WEB_ORIGIN=https://florianfrick.github.io \
  draft-web:$TAG
curl -sS -X POST \
  'http://localhost:9000/2015-03-31/functions/function/invocations' \
  -H 'Content-Type: application/json' \
  -d '{"httpMethod":"GET","path":"/api/champions","headers":{}}'
```

That should return a Lambda-shaped response with `statusCode: 200` and the
champion/patch/model payload in `body`.

## Create the function and embed it

```bash
terraform -chdir=deploy/terraform apply
terraform -chdir=deploy/terraform output -raw function_url
```

Paste `function_url` into the portfolio page's iframe `src`. `embed_url` is an
alias for the same value. No `?api=` is needed because the iframe document and
its `/api/...` calls share the Function URL origin. Give the iframe a generous
height; the agent panels sit below the board.

To update the app later: rebuild/push a new tag, set `image_tag` in
`terraform.tfvars`, and `terraform apply`. The ECR lifecycle policy keeps only
the **current** image, so storage does not drift upward with each deploy — push
first, then apply, or the old tag is expired while the function still points at
it. Re-pushing the same mutable tag does **not** make Terraform update the
function; use a fresh tag.

## Live deployment notes

The deployed function is `loldraft-web` in `us-west-2`, currently on image tag
`v8-heading`, with a 300-second timeout. The first invocation after an idle period
takes about **50 seconds** to import CPU PyTorch and load both heads; warm
requests are around 0.2 seconds. A 60-second timeout fails cold-start agent
turns, so keep the 300-second setting.

Two Cloudflare details are load-bearing:

* `llm_client.py` sends an explicit `User-Agent: DraftAssistantWeb/1.0`.
  urllib's default `Python-urllib/3.x` UA is rejected at the Cloudflare edge
  with **error 1010**, which looks like an auth failure from inside the loop.
* The Function URL CORS list is `GET, POST` only. AWS answers the `OPTIONS`
  preflight itself and rejects an explicit `OPTIONS` entry.
* `lambda_handler.py` forwards the static file's MIME type. Without that,
  `app.js` and `style.css` are served as `application/octet-stream`, which can
  stop the iframe's scripts from running.
* A Function URL event splits the path and query into `rawPath` and
  `rawQueryString`. `_request_path()` rejoins them before handing the event to
  `server.Handler.do_GET`; passing only `rawPath` silently drops every `?i=`
  and `?q=`, so picking a real draft returned **"no draft #-1"** and the search
  filter showed all 46,939 games.

The live smoke checks were `/api/champions` (200), `/api/next-move` (200),
`/api/draft?i=0` (200, real game), `/api/drafts?q=WLDs` (624 hits, not 46,939),
an `OPTIONS` preflight (200 with `Access-Control-Allow-Origin` set), and
`/api/agent` (200, `deepseek-flash` answer after two tool-calling LLM requests).

## Cost truth — $0 at this traffic

The image is **2.25 GB uncompressed** but **496 MiB compressed** as stored in
ECR, and ECR bills the compressed blobs (`imageSizeInBytes` = 520 MB). Always
quote the compressed figure when reasoning about storage cost.

| component | cost at this traffic |
|---|---|
| Lambda + Function URL | inside the permanent free tier (1 M requests / 400k GB-s per month) |
| CloudWatch Logs | inside the permanent 5 GB ingest/storage allowance at demo traffic |
| Data transfer out | inside the permanent 100 GB/month allowance |
| Public ECR | always-free 50 GB, **but Lambda cannot use a global public repo in us-east-1 as the image source for a function in us-west-2** |
| **Private ECR in us-west-2** | 500 MB-month free for 12 months, then **$0.10 per GB-month** |

One 496 MiB image is **0.48 GiB — inside the 500 MiB allowance**, and the
lifecycle rule keeps exactly one, so storage costs nothing. Lambda pulls from
ECR in the same Region at no charge; pushes are inbound and free. The $1 billing
alarm is the guardrail.

Two changes bought that margin, and both were worth more than they look:

* the checkpoint `COPY`s take only `best.pt` from each run, not the whole
  directory — `latest.pt` was 321 MB of last-epoch state that `server.py`
  skips by name;
* the shipped next-move head is the 12.8 MB siamese policy rather than the
  97 MB naive transformer. That costs **1.8 pp top-1 / 3.4 pp top-5**
  (18.9/50.8 vs 20.7/54.2, paired p 5.3e-28). If that accuracy is ever worth
  more than 84 MB, swap `POLICY_DEFAULT` back and expect ~573 MiB — still about
  a cent a month.

A genuinely PyTorch-free variant would need a <250 MB Lambda zip (ONNX Runtime
or an ExecuTorch-style export plus JSON metadata). That is a different
inference backend, not a Terraform change; it should be done only after
checking that its logits still match the shipped heads.

## Files

| file | role |
|---|---|
| `Dockerfile.web` | CPU-only Lambda image; copies only the served modules, two checkpoints and small data tables |
| `requirements-web.txt` | matching `+cpu` torch wheel + LangGraph only |
| `webapp/lambda_handler.py` | stateless Function URL adapter; reuse of `server.Handler`, tool ablation, CORS |
| `webapp/static/app.js`, `agent.js` | `apiUrl()` so one static build works locally and behind the iframe |
| `deploy/terraform/` | IAM, logs, private ECR, Lambda image, Function URL and `embed_url` output |
