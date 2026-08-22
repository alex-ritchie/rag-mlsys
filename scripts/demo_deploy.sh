#!/usr/bin/env bash
# Hosted demo deploy (spec §5.11): demo backend on Fly.io + frontend on Cloudflare Pages.
# Prereqs: `flyctl` and `wrangler` logged in; secrets exported: ANTHROPIC_API_KEY, SUPABASE_DB_URL.
set -euo pipefail
cd "$(dirname "$0")/.."
APP=${FLY_APP:-mlsysbook-rag-demo}
PAGES=${CF_PAGES_PROJECT:-mlsysbook-rag}

echo "== backend ($APP)"
flyctl apps create "$APP" --yes 2>/dev/null || true
flyctl secrets set -a "$APP" ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" SUPABASE_DB_URL="$SUPABASE_DB_URL" DEMO_IP_SALT="$(openssl rand -hex 16)"
flyctl deploy -a "$APP" --config docker/fly.toml --dockerfile docker/gateway.Dockerfile --build-arg BASE=mlsysbook-rag/python-base:cpu-models --remote-only
BACKEND_URL="https://$APP.fly.dev"
echo "backend: $BACKEND_URL"

echo "== frontend ($PAGES)"
( cd frontend && VITE_API_BASE="$BACKEND_URL" VITE_PROFILE=demo pnpm build )
wrangler pages deploy frontend/dist --project-name "$PAGES"
echo "done. Test: curl -N $BACKEND_URL/api/ask -H 'content-type: application/json' -d '{\"question\":\"What is quantization?\"}'"
