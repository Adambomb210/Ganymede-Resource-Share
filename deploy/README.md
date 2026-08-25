# Deploying the coordinator

Operational notes only. For the why, see
[`docs/02-architecture-v2.md`](../docs/02-architecture-v2.md) §6.5 (deployment)
and §6.6 (storage, backups, GC).

## Bring it up locally

```
cd deploy
cp ganymede.env.example ganymede.env
# edit ganymede.env: at minimum set real S3_ACCESS_KEY / S3_SECRET_KEY
docker compose up --build
```

This starts MinIO (`http://localhost:9000`, console on `:9001`) and the
coordinator (`http://localhost:8000`). Check `curl http://localhost:8000/healthz`.

## Create the first contributor key

```
docker compose exec coordinator python3 -m scripts.issue_key --name "your-name"
```

The plaintext key is printed once. Save it now -- only its hash is stored, and
there is no recovery path. It's what goes into a worker's `GANYMEDE_KEY`.

## Create the first run

```
docker compose exec coordinator python3 -m scripts.newrun \
    --run-id my-first-run \
    --base-model Qwen/Qwen3-1.7B-Base --base-precision bf16 \
    --dataset dolly15k --num-buckets 64 --target-rounds 20 \
    --target-steps 2000 --min-round-sec 300 --max-round-sec 2400 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --target-modules q_proj,k_proj,v_proj,o_proj
```

Building the seed adapter only fetches the base model's `config.json` -- not
its weights -- but the container does need outbound network access to
HuggingFace for that one request. Use `--dry-run` first to sanity-check the
adapter (tensor count, size) without writing anything.

## What to change for a real deployment

- **Hostnames.** `STORAGE_HOST` and `COORDINATOR_HOST` in `ganymede.env` move
  from `http://localhost:...` to your two real subdomains (§6.5 -- two
  subdomains, not one; path-routing both under one host fights the S3
  client). `MINIO_SERVER_URL` in `docker-compose.yml` reads `STORAGE_HOST`
  automatically, but it must stay byte-identical to it -- that's the
  presigned-URL host footgun (§6.6): sign for the wrong hostname and every
  worker's presigned URL 403s with an error that doesn't say why.
- **TLS termination.** Put a reverse proxy (nginx, Caddy, your LB) in front of
  both subdomains and terminate TLS there; the containers keep speaking plain
  HTTP to the proxy. Set `GANYMEDE_REQUIRE_TLS=true` once that's in place --
  it's the production setting, and the coordinator trusts
  `X-Forwarded-Proto` from the proxy to know a request arrived over HTTPS.
- **The separate volume.** `docker-compose.yml` already gives MinIO's data
  and the coordinator's SQLite file their own named volumes rather than
  writing into a container's own filesystem layer. On a real server, back
  those volumes with a disk or partition that is not the OS disk -- the whole
  point (§6.6) is that losing the OS disk should not also lose every
  checkpoint.
- **Off-box backups.** Point `scripts/backup.py` at a *different* endpoint,
  bucket, and credentials than this deployment's own storage -- it refuses to
  run otherwise, on purpose (§6.6: "a backup that lives on the machine it is
  backing up is not a backup"). Run it on a schedule (cron, systemd timer)
  after every round close, e.g.:

  ```
  python3 -m scripts.backup \
      --dest-endpoint https://backup-storage.example \
      --dest-bucket ganymede-backups \
      --dest-access-key "$BACKUP_ACCESS_KEY" \
      --dest-secret-key "$BACKUP_SECRET_KEY"
  ```

  Run `scripts/gc.py` on the same schedule to reclaim old worker submission
  artifacts (`--dry-run` first to see what it would delete; `--yes` to
  actually delete).
