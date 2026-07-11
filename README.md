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

## Reduce the amount of times files are opened
Currently we are using file-names to pass around images.  This is inefficient as we re-open the same file many times (particularly the answer key).  Consider passing around images as `fitz.Matrix` or something.  Discuss formats before proceeding with this task. 

## Log all call-reponses from mathpix
We want to build a labelled training set so that someday we can build our own model, and not rely on mathpix.