# Description
GraderBot is a code-base for generating arithmetic worksheets that can be auto-graded.  Worksheets are generated in latex in two modes: `cv mode=true` or `cv mode=false`.  See `tex/demo.tex` for reference.

The LaTeX sources (`gbworksheet.sty`, `questions.sty`, the templates, and the
generated `aruco_images/` markers) live in the `tex/` directory. Compile from
there:
```
cd tex && latexmk -pdf demo.tex
```

To clean the directory run
```
cd tex && latexmk -c demo.tex
```

or use `-C` if you also want the `.pdf` file removed.

## CV mode
If you would like to control the value of `cv mode` from the command line at compilation time then do
```shell
cd tex && latexmk -pdf -usepretex='\def\WSCVMode{0}' demo.tex 
```
for `cv mode = false`.  For `cv mode = true` just change the 0 into a 1 in the above command.


## Sythesizing Student Work
The [worksheet_synth](./graderbot/worksheet_synth.py) python module is for making synthetic images of student work.  It can compile latex, fill in answer boxes, and add noise and perspective skewing.  This is primarily useful for making unit-tests for grader bot, which is concerned primarily with inverting the `worksheet_synth` module.  Here is an example code-snippet

```python
import cv2
from graderbot.worksheet_synth import fill_worksheet, perspective_skew_image, add_image_noise
import numpy as np

# fill_worksheet returns one BGR image per page (a worksheet may span
# several pages), so index/iterate the list.
pages = fill_worksheet('tex/demo.tex', {'add001': '12', 'sub001': r'\frac{3}{5}'})
skewed = perspective_skew_image(pages[0], max_skew=0.02, rng=np.random.default_rng(42))
noisy = add_image_noise(skewed, noise_level=0.05, rng=np.random.default_rng(7))
cv2.imwrite('output_filename.png', noisy)
```

## Worksheet storage (S3 + SQLite)
The [worksheetbot](./graderbot/worksheetbot.py) pipeline can upload the generated PDFs
(student/blank, cv, and answer key) to S3 and record them, along with the
`.tex` source and question JSON, in a SQLite database via
[storage.py](./graderbot/storage.py). This is opt-in: pass `--bucket` (or set the
`S3_BUCKET` env var) when running `python -m graderbot.worksheetbot`. If no bucket is
configured, the worksheet is still compiled but nothing is uploaded or
stored.

### System dependencies
Beyond the Python packages (managed with poetry), grading and worksheet
compilation shell out to native binaries that must be on your `PATH`:
- **Tesseract** — `pytesseract` (used by [graderbot/ocr.py](./graderbot/ocr.py)
  to read the student-name box) is only a wrapper around the `tesseract`
  binary. Install it with `brew install tesseract` on macOS or
  `apt-get install tesseract-ocr` on Debian/Ubuntu. The `Dockerfile` installs
  it for the deployed image.
- **TeX Live / latexmk** — for compiling the worksheet templates (see above).

### One-time setup
1. Create an S3 bucket for PDF + database backups.
2. Create an IAM user/role with an S3 policy scoped to that bucket. This
   repo's code only needs `s3:PutObject`/`s3:GetObject`, but litestream also
   needs `s3:GetBucketLocation`/`s3:ListBucket` (bucket-level) and
   `s3:DeleteObject`/`s3:ListMultipartUploadParts`/`s3:AbortMultipartUpload`
   (object-level) for replication and retention. Example policy:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Sid": "BucketLevel",
               "Effect": "Allow",
               "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
               "Resource": "arn:aws:s3:::<your-bucket-name>"
           },
           {
               "Sid": "ObjectLevel",
               "Effect": "Allow",
               "Action": [
                   "s3:PutObject",
                   "s3:GetObject",
                   "s3:DeleteObject",
                   "s3:ListMultipartUploadParts",
                   "s3:AbortMultipartUpload"
               ],
               "Resource": "arn:aws:s3:::<your-bucket-name>/*"
           }
       ]
   }
   ```
3. Add the following to `.env`:
   ```
   S3_BUCKET=<your-bucket-name>
   AWS_ACCESS_KEY_ID=<...>
   AWS_SECRET_ACCESS_KEY=<...>
   AWS_REGION=<...>
   WORKSHEETS_DB_PATH=worksheets.sqlite3
   ```
   `litestream.yml` reads its DB path from `WORKSHEETS_DB_PATH`, so this must
   be set (and match the path used by `graderbot/app.py` / `graderbot.worksheetbot`) for
   replication to point at the right file.
   (The `graderbot` package is imported top-level, so run tests and scripts from
   the repo root — `pytest` picks up `pythonpath = ["."]` automatically, and for
   ad-hoc runs use `python -m graderbot.<module>` or set `PYTHONPATH=.`.)

### Backups
The SQLite database (`worksheets.sqlite3` by default) is meant to be
continuously replicated to S3 with [litestream](https://litestream.io/),
using the config in [litestream.yml](./litestream.yml). Litestream itself
runs as an external process; this repo's code only writes to the local
SQLite file in WAL mode (required for litestream) and does not start or
manage the litestream process.

Install litestream locally with:
```shell
brew install benbjohnson/litestream/litestream
```

`litestream.yml` references `${S3_BUCKET}`, so export `.env` before starting
replication. Start it manually, once per development session, in its own
terminal (or backgrounded), before running `python -m graderbot.worksheetbot` or
`streamlit run graderbot/app.py`:
```shell
set -a; source .env; set +a
litestream replicate -config litestream.yml
```

### Mathpix OCR logging
To build a labelled dataset for a future in-house OCR model (issue #1), every
Mathpix call made by `read_box` can be logged: the exact PNG posted to Mathpix
is stored in S3 (content-addressed as `mathpix/<sha256>.png`) and a
`MATHPIX_CALL` row (image URL, image hash, raw Mathpix response JSON, and parsed
answer text) is written to the SQLite DB. This is opt-in and self-gating: it
does nothing unless a bucket is configured via `MATHPIX_LOG_BUCKET` (falling
back to `S3_BUCKET`), and it reuses `WORKSHEETS_DB_PATH` for the database.
Logging failures are non-fatal — they never interrupt OCR or grading. Set in
`.env`:
```
MATHPIX_LOG_BUCKET=<your-bucket-name>   # optional; defaults to S3_BUCKET
```

## Web frontend
[app.py](./graderbot/app.py) is a Streamlit app with three tabs: **Gallery**, to browse
previously created worksheets and open their student/cv/answer-key PDFs via
presigned S3 links; **Create**, to generate a new worksheet from a
prompt (runs the same pipeline as `graderbot.worksheetbot`, including S3 upload +
DB storage); and **Grade**, to upload a PDF of scanned student work and have
it auto-graded. Each page's QR code is matched to its stored worksheet, graded
against the stored answer key (via `scan_grader.mark_scan`), and returned both
as per-student JSON results and as a single marked-up PDF (a score header on
each page, correct answers written beside the wrong ones). It requires
`S3_BUCKET` and `ANTHROPIC_API_KEY` to be set (see
above); it reads/writes the same `worksheets.sqlite3` database as the CLI by
default (override with the `WORKSHEETS_DB_PATH` env var). Run it from the repo
root with:
```shell
streamlit run graderbot/app.py
```

## Fly.io
We currently deploy to fly.io at the url https://grader-bot.fly.dev

Deployment is automated: every push to the `main` branch triggers a deploy to
fly.io. To deploy, merge/push your changes to `main`. (You can still deploy
manually with `fly deploy` if needed.)