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

### Schema (ERD)
All the tables live in the one SQLite DB created by `init_db` (`storage.py`).
`CLASSROOM`/`STUDENT`/`NAME_IMAGES`/`NAME_EMBEDDINGS`/`PENDING_NAME_LABEL` form
the roster/name-classifier chain (issues #2/#43/#46/#92), though the trained
classifier itself is no longer scoped to a `CLASSROOM` (issue #109 -- see
"Handwriting name classifier" below); `WORKSHEET`, `STY_VERSION`, and
`MATHPIX_CALL` are standalone. The
`WORKSHEET.sty_hash -> STY_VERSION.hash` link is a convention followed in code
(`record_sty_version`), not a SQLite `FOREIGN KEY` constraint.

```mermaid
erDiagram
    CLASSROOM ||--o{ STUDENT : enrolls
    CLASSROOM |o--o{ PENDING_NAME_LABEL : "queues crops for (optional, issue #109)"
    STUDENT ||--o{ NAME_IMAGES : "has samples"
    STUDENT ||--o{ NAME_EMBEDDINGS : "has embeddings"
    NAME_IMAGES ||--o| NAME_EMBEDDINGS : "embedded as"
    STY_VERSION ||--o{ WORKSHEET : "compiled with (by convention)"

    CLASSROOM {
        int id PK
        string label UK
        string created_at
    }
    STUDENT {
        int id PK
        int classroom_id FK
        string first_name
        string last_name
        string nickname
        string created_at
    }
    NAME_IMAGES {
        int id PK
        int student_id FK
        string box_id
        string image_s3url
        string image_sha256
        string created_at
    }
    NAME_EMBEDDINGS {
        int id PK
        int student_id FK
        int name_image_id FK "UNIQUE"
        string embedding_s3url
        string created_at
    }
    PENDING_NAME_LABEL {
        int id PK
        int classroom_id FK "nullable, issue #109"
        string box_id
        string image_s3url
        string image_sha256
        string predicted_name
        float confidence
        string source
        string created_at
    }
    WORKSHEET {
        int id PK
        string prompt
        string tex_source
        string questions_json
        string model
        int num_questions
        string title
        string header
        string public_id
        string boxes_json
        string student_pdf_s3url
        string cv_pdf_s3url
        string answers_pdf_s3url
        string sty_hash "not a DB FK"
        string created_at
    }
    STY_VERSION {
        string hash PK
        string content
        string created_at
    }
    MATHPIX_CALL {
        int id PK
        string image_s3url
        string image_sha256
        string response_json
        string response_text
        string created_at
    }
```

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
   BASE_URL=https://grader-bot.fly.dev
   ```
   `litestream.yml` reads its DB path from `WORKSHEETS_DB_PATH`, so this must
   be set (and match the path used by `graderbot/app.py` / `graderbot.worksheetbot`) for
   replication to point at the right file.

   `BASE_URL` is optional (defaults to `https://grader-bot.fly.dev`) and is only
   used to build the permanent `?dl=<public_id>` download links shown in the
   Gallery tab for each worksheet's student PDF (see `graderbot/app.py`'s
   `build_permanent_download_url`). Set it explicitly in production (already
   done in `fly.toml`) so it stays correct if the app's domain ever changes.
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

### Answer OCR: No OCR vs Mathpix vs Google Cloud Vision
Answer boxes default to **No OCR** (issue #82): it never attempts to
transcribe the box at all. It relies entirely on the existing
blank-detection check every backend already runs before OCR
(`imaging.is_blank`): a blank box is still graded as blank, but a filled-in
box is simply marked wrong (with the correct answer written beside it, same
as any other wrong answer) without ever guessing what was written. Students
still get to see which answers they got right, without OCR ever having a
chance to misread their handwriting. The Grade tab's "Read answers with"
dropdown can switch a run to Mathpix or Google Cloud Vision instead:
- **Mathpix** (`graderbot/ocr.py`) handles handwritten LaTeX fractions well
  but is tuned for college-level math and occasionally misreads a sloppy
  digit (issue #70: "9" as "G", "14" as "1 h"). It's the only backend that
  reads fractions, so switch to it for a worksheet that has them.
- **Google Cloud Vision** uses `DOCUMENT_TEXT_DETECTION` (Google's mode for
  dense/handwritten text) with no allowlist support and no fraction-splitting.

An EasyOCR backend used to sit between these two (issue #70), restricted to
a character allowlist plus an experimental fraction-bar detector. It was
removed (issue #82): it read student handwriting too poorly to be worth the
separate sidecar service (`easyocr_service/`, plus its Docker/Modal
deployment) it required.

`graderbot/answer_reader.py`'s three `AnswerReader`s (`MathpixAnswerReader`,
`GoogleVisionAnswerReader`, `NoOcrAnswerReader`) mirror the existing
`NameReader` pattern used for student identification.

**Google Cloud Vision** needs no sidecar and no new Python dependency —
`GoogleVisionAnswerReader` calls the Vision REST API directly with
`requests`, the same shape as the Mathpix call. Set in `.env`:
```
GOOGLE_VISION_API_KEY=<your-api-key>
```
(from a GCP project with the Cloud Vision API enabled — see
https://cloud.google.com/vision/docs/setup). Works on fly.io as-is, since
it's a plain HTTPS call.

### Answer verification: CNN verifier (experimental, issue #81)
Both backends above transcribe a crop open-vocabulary, with no idea
what a middle-school worksheet's answer is even supposed to look like. The
"CNN verifier (experimental)" option in the Grade tab's "Read answers with"
dropdown instead **verifies** a crop against the known answer plus a
handful of OCR-confusable near-misses (`graderbot/response_candidates.py`),
using a small CRNN trained from scratch on this project's own domain
(digits, `.`, `-`) rather than a general-purpose alphabet — directly
targeting the kind of confusion (e.g. "1" vs "/") a borrowed English-prose
model couldn't resolve (see the paused `handwriting-ctc-match` spike, issue
#73).

- **Scope (v1): plain numeric answers only** — integers, decimals,
  negatives. A `\frac{a}{b}` question on the same worksheet still falls
  back to Mathpix automatically; there's no need to switch dropdowns
  mid-worksheet.
- **Runs in-process** — its only dependency, `onnxruntime`, ships real
  wheels for both Intel macOS and fly.io (unlike torch), so
  `CnnResponseScorer` (`graderbot/response_scorer.py`) just loads
  `models/response_scorer/weights.onnx` off disk, no sidecar, no Modal
  deployment for inference.
- **Training is the part that needs Modal** — the CRNN itself is trained
  with torch (`training/`, isolated from the main project's
  `pyproject.toml` since this dev environment has no torch wheel available
  at all). Generate synthetic training crops with
  `graderbot/answer_glyph_synth.py`, train + export with:
  ```shell
  poetry run modal run training/modal_app.py --steps 20000
  ```
  which writes `models/response_scorer/{weights.onnx,vocab.json}` straight
  into the repo. `training/eval.py` reports per-answer-type accuracy on
  held-out synthetic data.
- **No real-data labeling.** A "copy worksheet" harvesting flow (a
  worksheet that already printed each answer for the student to copy, so
  the printed text was trusted as the ground-truth label) was tried and
  removed: asking students to manually annotate their own handwriting
  doesn't reflect how they actually write on a normal worksheet (students
  write far more neatly when asked to copy a printed answer), so the labels
  it produced weren't representative. A separate Mathpix-seeded manual
  review pass (`scripts/label_handwriting.py`) was removed alongside it.
  Real labels are the ground truth a synthetic-only model needs checked
  against before being trusted (the gap that sank the `pylaia-iam` spike),
  so until a trustworthy real-data source exists, the Grade tab option
  stays labeled "(experimental)" and `training/eval.py` only reports the
  synthetic side.

### Handwriting name classifier
Students are identified on a scanned worksheet either by OCR'ing the name box
or by recognizing their handwriting. The handwriting path (issue #2) runs
end to end through the Streamlit app. Classrooms are still how the **Roster**
and **Name sheets** tabs organize students for ingest, but the trained
classifier itself is a single model spanning every classroom (issue #109) —
grading and the **Name Classifier** tab no longer ask which class you're
working with:

1. **Name sheets** tab — print one name-collection page per student.
2. **Roster** tab — upload the scanned sheets. `ingest_name_sheets` crops each
   handwriting sample to S3 + the `NAME_IMAGES` table, then `vectorize_samples`
   embeds each crop into `NAME_EMBEDDINGS`.
3. **Name Classifier** tab — "Evaluate classifier" runs leave-one-out
   cross-validation over every student across every classroom (worth checking
   before trusting it), and "Train classifier" fits one model on all of them
   and saves it to `name_classifier/global.joblib` in S3. **Retrain after
   ingesting new name sheets** — the saved model does not update on its own.
   A 3D t-SNE projection of every student's handwriting embeddings is also
   available for debugging the classifier, behind a "Load 3D visualization"
   button (issue #113) since it's expensive to compute for a large roster.
4. **Grade** tab — the "Read student names with" dropdown picks between the
   trained classifier and OCR for that run, and the results table shows which
   name was read off each page and how confident the reader was, so a doubtful
   read can be checked by hand (issue #58). Every page whose name-read
   confidence falls below `pending_name_capture.LOW_CONFIDENCE_THRESHOLD`
   (0.5) has its name-box crop queued for the **Label Names** tab below
   (issue #92), instead of that read simply going unchecked. While any
   student (in any classroom) has fewer than
   `pending_name_capture.MIN_NAME_IMAGES_PER_STUDENT` (5) labelled
   `NAME_IMAGES`, confidence is ignored and every name-box crop is queued
   instead (issue #110) — a classifier that has never seen a student can
   still read their crop "confidently" as someone else, so relying on
   confidence alone would never surface that student's crops for labelling.
5. **Label Names** tab — works through that queue one crop at a time (by
   default chosen uniformly at random), showing the reader's own guess as a
   hint. Assigning the right student inserts a `NAME_IMAGES` row and
   immediately embeds it (same as step 2), so it's ready the next time step
   3 retrains; "Discard" drops an unusable crop instead. Because step 4's
   bootstrap capture queues every student's crops indiscriminately while
   *any* student is under quota, one under-represented student's crops can
   be a small fraction of a large queue and take a while to come up by
   chance. A "Student coverage" expander shows each student's progress
   toward the quota and how many queued crops currently guess them, and a
   "Review queue for" filter lets a teacher draw only from one student's
   guessed crops instead of waiting on the random draw — filtered on the
   reader's guess, not verified identity, so it can still miss a crop that
   got misread as someone else.

`vectorize_samples` (see [embedding.py](./graderbot/embedding.py)) chooses its
embedder from the `NAME_EMBEDDER` env var:
- `voyage` (the default) — `RemoteEmbedder` calls the
  [Voyage multimodal-3](https://www.voyageai.com/) embedding API, chosen over
  self-hosting so no heavyweight (torch) service needs deploying. It scored a
  perfect leave-one-out accuracy on a small real roster versus ~0.56 for raw
  pixels (issue #56). Requires `VOYAGE_API_KEY`.
- `local` — `LocalEmbedder`, a lightweight in-process resize+flatten embedder
  with no API key, for offline use.

```
VOYAGE_API_KEY=<your-voyage-api-key>
NAME_EMBEDDER=voyage   # optional; `voyage` (default) or `local`
```

Embedders produce different-sized vectors (Voyage 1024, `LocalEmbedder` 4096),
and a stored vector records no embedder of its own. Training and the Name
Classifier tab therefore keep only the vectors matching the *current* embedder's
dimension and report how many they skipped — so switching `NAME_EMBEDDER`
without re-ingesting silently shrinks the training set rather than mixing
incompatible vectors.

## Web frontend
[app.py](./graderbot/app.py) is a Streamlit app with seven tabs: **Gallery**, to browse
previously created worksheets and open their student/cv/answer-key PDFs via
presigned S3 links; **Create**, to generate a new worksheet from a
prompt (runs the same pipeline as `graderbot.worksheetbot`, including S3 upload +
DB storage); **Grade**, to upload a PDF of scanned student work and have
it auto-graded; **Name sheets**, to paste a class roster (one name per line)
and download a printable PDF of name-collection worksheets — one page per
student (see [name_worksheets.py](./graderbot/name_worksheets.py) and issue #45);
**Roster**, to ingest those sheets back in and manage a class's students;
**Name Classifier**, to inspect, cross-validate, and train the handwriting
name classifier (see above); and **Label Names**, to manually assign a student to
the low/no-confidence name crops grading queues up (issue #92, see above).
Each graded page's QR code is matched to its stored
worksheet, graded
against the stored answer key (via `scan_grader.mark_scan`), and returned both
as a per-worksheet participation report -- title plus each classroom's list
of students (alphabetical by last name) who did it, via
`scan_grader.participation_report` (issue #107) -- and as a single marked-up
PDF (correct answers written beside the wrong ones). It requires
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

### Viewing logs
`fly logs -a grader-bot` streams recent logs. The app's own log lines are
prefixed `graderbot.app`; set `LOG_LEVEL=DEBUG` (fly secrets/env) for more
verbosity. Note the machine auto-suspends when idle (`auto_stop_machines`,
`min_machines_running = 0` in `fly.toml`), so live-tailing shows nothing until
a request wakes it — hit https://grader-bot.fly.dev first, or use `fly logs`
right after a deploy triggers a fresh machine start.