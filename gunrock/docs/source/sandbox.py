#!/usr/bin/env python3

# run as
# ./hive-year1-summary.py && open darpa.pdf

import os
import tempfile
import subprocess
import re

files = ['sandbox.html.md']

pandoc_cmd = ['pandoc',
              '--template=darpa-template.tex',
              '--variable', 'title=Sandbox',
              '--variable', 'author=Ben Johnson \\and Weitang Liu \\and Agnieszka Łupińska \\and Muhammad Osama \\and John D. Owens \\and Yuechao Pan \\and Leyuan Wang \\and Xiaoyun Wang \\and Carl Yang',
              '--variable', 'postauthor=UC Davis',
              '--variable', 'documentclass=memoir',
              '--variable', 'fontsize=10pt',
              '--variable', 'classoption=oneside',
              # '--variable', 'classoption=article',
              '--variable', 'toc-depth=0',
              '--toc',
              '-o', 'sandbox.pdf',
              # '-o', 'sandbox.tex',
              ]
pandoc_cmd.extend(files)

print(pandoc_cmd)

subprocess.run(pandoc_cmd)
