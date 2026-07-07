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