# Programming workflow
- Read the README.md before you begin work.

- Before implementing new features ask clarifying questions before presenting your approach.

- Every new python routine should have a new pytest. Consider making the unit-test before making the code.

- Run all pytests before ending a work session to make sure you have not broken anything.

# Viewing generated images
When you generate image files (PNGs from CV/diagnostic work, plots, etc.), open
them with the macOS `open` command so I can see them (I'm on a Mac and can't
otherwise see files you write, especially temp files). If a session produces
many intermediate images, open only the key/final ones rather than every one to
avoid stacking Preview windows.

# Python
We use poetry for dependency management.  Use the associated virtual environment when using python locally.

# Github
For fetching github issues do
```shell
gh issue view 13 --json number,title,body,comments,labels,state -R hoj201/grader_bot
```

if an issue has the label `no claude` or `wontfix` then you should not work on it.



