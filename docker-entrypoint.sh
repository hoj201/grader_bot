#!/bin/sh
set -e

mkdir -p /data

# On a fresh Fly volume there's no local DB yet; restore the latest S3
# replica if one exists so we don't start from an empty gallery. This is a
# safety net (disaster recovery) -- the Fly volume is the primary store,
# litestream replicate below is the offsite backup.
if [ -n "$S3_BUCKET" ] && [ ! -f "$WORKSHEETS_DB_PATH" ]; then
    echo "No local DB at $WORKSHEETS_DB_PATH, attempting litestream restore from S3..."
    litestream restore -if-replica-exists -config /app/litestream.yml "$WORKSHEETS_DB_PATH" || true
fi

if [ -n "$S3_BUCKET" ]; then
    litestream replicate -config /app/litestream.yml &
fi

exec streamlit run app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0
