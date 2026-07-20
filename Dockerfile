FROM python:3.13-slim

# TeX Live packages needed to compile worksheet_template.tex (geometry,
# graphicx, tikz, tikzpagenodes, xcolor, calc) via latexmk. Deliberately
# scoped rather than texlive-full to keep the image size manageable.
RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-pictures \
        texlive-fonts-recommended \
        latexmk \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# litestream for SQLite <-> S3 replication (see litestream.yml)
RUN curl -fsSL -o /tmp/litestream.deb \
        https://github.com/benbjohnson/litestream/releases/download/v0.3.13/litestream-v0.3.13-linux-amd64.deb \
    && dpkg -i /tmp/litestream.deb \
    && rm /tmp/litestream.deb

RUN pip install --no-cache-dir poetry==1.8.3 \
    && poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

COPY . .

ENV WORKSHEETS_DB_PATH=/data/worksheets.sqlite3

EXPOSE 8501

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
