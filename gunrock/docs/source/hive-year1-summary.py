#!/usr/bin/env python3

# run as
# ./hive-year1-summary.py && open darpa.pdf

import os
import tempfile
import subprocess
import re

files = sorted([f for f in os.listdir('./hive')
                if ((f.startswith('hive_') and
                     f.endswith('.html.md') and
                     f != 'hive_year1_summary.html.md' and
                     f != 'hive_template.html.md' and
                     f != 'hive_scaling.html.md' and
                     f != 'hive_sandbox.html.md'))])

# I put this back since it doesn't get included in the PDF otherwise
files.append('hive_scaling.html.md')

print("""---
title: HIVE Year 1 Report&colon; Executive Summary

toc_footers:
  - <a href='https://github.com/gunrock/gunrock'>Gunrock&colon; GPU Graph Analytics</a>
  - Gunrock &copy; 2018 The Regents of the University of California.

search: true

full_length: true
---

# HIVE Year 1 Report&colon; Executive Summary

This report is located online at the following URL: <https://gunrock.github.io/docs/hive_year1_summary.html>.

Herein UC Davis produces the following three deliverables that it promised to deliver in Year 1:

1. **7--9 kernels running on a single GPU on DGX-1**. The PM had indicated that the application targets are the graph-specific kernels of larger applications, and that our effort should target these kernels. These kernels run on one GPU of the DGX-1. These kernels are in Gunrock's GitHub repository as standalone kernels. While we committed to delivering 7--9 kernels, as of the date of this addendum, we deliver all 11 v0 kernels.
2. **(High-level) performance analysis of these kernels**. In this report we analyze the performance of these kernels.
3. **Separable communication benchmark predicting latency and throughput for a multi-GPU implementation**. This report (and associated code, also in the Gunrock GitHub repository) analyzes the DGX-1's communication capabilities and projects how single-GPU benchmarks will scale on this machine to 8 GPUs.

Specific notes on applications and scaling follow:

""",
      file=open('hive_year1_summary.html.md', 'w'))

with open('hive_year1_summary.html.md', 'a') as dest:
    for f in files:
        fname = f[:-3]
        with open(f) as file:
            contents = file.read()
            title = re.search('\n# (.*)\n', contents).group(1)
            summary = re.search(
                '\n## Summary of Results\n\n([^#]*)\n\n#', contents).group(1)
            dest.write(f'## {title} \n**[{title}](https://gunrock.github.io/docs/{fname})** \n{summary}\n\n')

files.insert(0, 'hive_year1_summary.html.md')

pandoc_cmd = ['pandoc',
              '--template=darpa-template.tex',
              '--variable', 'title=A Commodity Performance Baseline for HIVE Graph Applications:\\\\Year 1 Report',
              '--variable', 'subtitle=(Addendum, 16 November 2018)',
              '--variable', 'author=Ben Johnson \\and Weitang Liu \\and Agnieszka Łupińska \\and Muhammad Osama \\and John D. Owens \\and Yuechao Pan \\and Leyuan Wang \\and Xiaoyun Wang \\and Carl Yang',
              '--variable', 'postauthor=UC Davis',
              '--variable', 'documentclass=memoir',
              '--variable', 'fontsize=10pt',
              '--variable', 'classoption=oneside',
              # '--variable', 'classoption=article',
              '--variable', 'toc-depth=0',
              '--toc',
              '-o', 'darpa.pdf',
              # '-o', 'darpa.tex',
              ]
pandoc_cmd.extend(files)

print(pandoc_cmd)

subprocess.run(pandoc_cmd)
