<p align="center">
  <a href="https://github.com/gunrock/docs/"><img src="https://github.com/gunrock/docs/blob/master/source/images/GunrockLogo150px.png" alt="GunrockLogo"></a>
</p>

## Deploying Gunrock Slate Docs (http://gunrock.github.io/docs)

### Prerequisites

You're going to need:
 - **Linux or OS X** — Windows may work, but is unsupported.
 - **Ruby, version 2.3.1 or newer**
 - **Bundler** — If Ruby is already installed, but the `bundle` command doesn't work, just run `gem install bundler` in a terminal.

### Editing and Publishing
 - [Supported Markdown Syntax](https://github.com/lord/slate/wiki/Markdown-Syntax)
 - [Publishing docs to Github Pages](https://github.com/lord/slate/wiki/Deploying-Slate)


## Slate Header for Gunrock
- Create a markdown page with the extension `.html.md` and add it to the source directory.
- Slate markdowns use the following header for page settings like title, footer, etc.

```
---
# title of the page
title: <Gunrock-Page-Title>

# add a language tab, doesn't work with full_length set to true
# must be one of https://git.io/vQNgJ
language_tabs:
  - shell
  - ruby
  - python
  - javascript

# page footer
toc_footers:
  - <a href='https://github.com/gunrock/gunrock'>Gunrock; GPU Graph Analytics</a>
  - Gunrock &copy; 2018 The Regents of the University of California.

# add markdown files in /source/includes directory to append the page with this header
includes:
  - <filetoinclude>
  - <filetoinclude>

# include search bar in the left menu?
search: true

# full length page or two sections?
full_length: true 
---
```

- Note that vega graphs are just `<div id=""><script> (json dump) </script></div>`.
