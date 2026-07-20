# Description
GraderBot is a code-base for generating arithmetic worksheets that can be auto-graded.  Worksheets are generated in latex in two modes: `cv mode=true` or `cv mode=false`.  See `demo.tex` for reference.

The compile command is
```
latexmk -pdf demo.tex
```

To clean the directory run
```
latexmk -c demo.tex
```

or use `-C` if you also want the `.pdf` file removed.

## CV mode
If you would like to control the value of `cv mode` from the command line at compilation time then do
```shell
latexmk -pdf -usepretex='\def\WSCVMode{0}' demo.tex 
```
for `cv mode = false`.  For `cv mode = true` just change the 0 into a 1 in the above command.


## Sythesizing Student Work
The [worksheet_synth](./worksheet_synth.py) python module is for making synthetic images of student work.  It can compile latex, fill in answer boxes, and add noise and perspective skewing.  This is primarily useful for making unit-tests for grader bot, which is concerned primarily with inverting the `worksheet_synth` module.  Here is an example code-snippet

```python
import cv2
from worksheet_synth import fill_worksheet, perspective_skew_image, add_image_noise
import numpy as np

filled = fill_worksheet('demo.tex', {'add001': '12', 'sub001': r'\frac{3}{5}'})
skewed = perspective_skew_image(filled, max_skew=0.02, rng=np.random.default_rng(42))
noisy = add_image_noise(skewed, noise_level=0.05, rng=np.random.default_rng(7))
cv2.imwrite('output_filename.png', noisy)
```

## Worksheet storage (S3 + SQLite)
The [worksheetbot](./worksheetbot.py) pipeline can upload the generated PDFs
(student/blank, cv, and answer key) to S3 and record them, along with the
`.tex` source and question JSON, in a SQLite database via
[storage.py](./storage.py). This is opt-in: pass `--bucket` (or set the
`S3_BUCKET` env var) when running `worksheetbot.py`. If no bucket is
configured, the worksheet is still compiled but nothing is uploaded or
stored.

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
   be set (and match the path used by `app.py`/`worksheetbot.py`) for
   replication to point at the right file.

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
terminal (or backgrounded), before running `worksheetbot.py` or
`streamlit run app.py`:
```shell
set -a; source .env; set +a
litestream replicate -config litestream.yml
```

## Web frontend
[app.py](./app.py) is a Streamlit app with two tabs: **Gallery**, to browse
previously created worksheets and open their student/cv/answer-key PDFs via
presigned S3 links, and **Create**, to generate a new worksheet from a
prompt (runs the same pipeline as `worksheetbot.py`, including S3 upload +
DB storage). It requires `S3_BUCKET` and `ANTHROPIC_API_KEY` to be set (see
above); it reads/writes the same `worksheets.sqlite3` database as the CLI by
default (override with the `WORKSHEETS_DB_PATH` env var). Run it with:
```shell
streamlit run app.py
```

## Fly.io
We currently deploy to fly.io at the url https://grader-bot.fly.dev

Deployment is **not** automated on every update to the `main` branch.  It must be initiated manually.