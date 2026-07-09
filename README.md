# Description
PencilBot is a code-base for generating arithmetic worksheets that can be auto-graded.  Worksheets are generated in latex in two modes: `cv mode=true` or `cv mode=false`.  See `demo.tex` for reference.

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
The [worksheet_synth](./worksheet_synth.py) python module is for making synthetic images of student work.  It can compile latex, fill in answer boxes, and add noise and perspective skewing.  This is primarily useful for making unit-tests for pencil bot, which is concerned primarily with inverting the `worksheet_synth` module.  Here is an example code-snippet

```python
import cv2
from worksheet_synth import fill_worksheet, perspective_skew_image, add_image_noise
import numpy as np

filled = fill_worksheet('demo.tex', {'add001': '12', 'sub001': r'\frac{3}{5}'})
skewed = perspective_skew_image(filled, max_skew=0.02, rng=np.random.default_rng(42))
noisy = add_image_noise(skewed, noise_level=0.05, rng=np.random.default_rng(7))
cv2.imwrite('output_filename.png', noisy)
```

# Tasks
Work on tasks in the order given

## Create `read_box` function
We will need a funciton with a signature like
```python
def read_box(image_fn: str, box: Box) -> str:
```
which takes in the filename of an image (or perhaps a more faithful representation like a `fitz.Matrix`) and then reads the hand-written text in the box.  There is currently such a function in `pencilbot.py` with an empty body.  Please fill it in.

Previous attempts to create this function have failed for the following reasons:

 1. We should expect the hand-writing to occasionally bleed outside the box slightly

 2. The text inside the box is exclusively answers to basic arithmetic problems involving rational numbers where all the numerators and denominators are below 1000.  Generic OCR will often mistake a 1 for and l, and can not understand the various ways of writing fractions.

I would recommend using something like `pix2tex`.  Returning raw latex would actually be ideal.

### Unit test
As a test for this function, you should be able to create boxes using `extract_answer_boxes` on the file created by `demo.tex`.  Then read those boxes on the file `demo_answer_key.pdf` using the newly created `read_box` function.  The expected text for question id `add_001` is 12, and the expected text for question `sub001` is 11.

## Create `extract_name` function
Every worksheet has a field at the top where students are to write their name (see `worksheet.sty`).

In this task you are to fill in the body of the `extract_name(worksheet_fn: str)` function which is currently in `pencilbot.py`.

This function reads the name from the name field using either OCR or a neural network (please discuss). If it helps, feel free to alter `worksheet.sty` by making the "name:" text a different color (dark green perhaps), so that you can filter for where to look for the name.  If you go this route, it would be wise to make the color toggleable using the `cv mode` variable (see `demo.tex` for an example).

### Unit Test
See if you can create a version of the worksheet with a name written in the name field in something that resembles hand-writing.  Have me preview your image before hand.

## Create `is_correct` function
For each each answer and response we will have strings of LaTeX code that we need to compare. The function
`is_correct` is responsible for comparing them.  This function body is currently blank in `pencilbot.py`.

### Unit Test
The following pairs should return `True`
```json
{
    "123": "123",
    "\frac{13}{1}": "13",
    "1.234567": "1.23456"
}
```
whereas these pairs should return `False`
```json
{
    "123": "128",
    "\frac{13}{1}": "\frac{13}{2}",
    "1.234567": "1.28456"
}
```

## Reduce the amount of times files are opened
Currently we are using file-names to pass around images.  This is inefficient as we re-open the same file many times (particularly the answer key).  Consider passing around images as `fitz.Matrix` or something.  Discuss formats before proceeding with this task. 