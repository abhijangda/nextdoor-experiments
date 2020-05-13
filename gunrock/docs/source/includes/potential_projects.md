# Possible Gunrock projects

Possible projects are in two categories: infrastructure projects that make Gunrock better but have minimal research value, and research projects that are longer-term and hopefully have research implications of use to the community.

For any discussion on these, please use the existing Github issue (or make one).

## Infrastructure projects

- Containerize Gunrock (a Docker container) [[issue](https://github.com/gunrock/gunrock/issues/349)]
- Support a Windows build [[issue](https://github.com/gunrock/gunrock/issues/213)]
- Develop a procedure to go from "How does Gunrock do on dataset X" to actually getting results and the right command lines for dataset X. Right now we do this manually with lots of iterations every time. We can automate and document this much better.
- Many apps have minimal documentation; we need better text when a user runs `./bin/primitive --help`.

## Research projects

- Better defaults and/or decision procedures for setting Gunrock parameters (possibly a machine-learning approach for this)
- How can we preprocess Gunrock input to increase performance? This could be either reordering CSR for better performance (e.g., reverse Cuthill-McKee) or a new format.
- If we had a larger number of X in the hardware&mdash;e.g., more registers, more SMs, more threads/SM, more shared memory, bigger cache---how would it help performance? (Where would we want NVIDIA to spend more transistors to best help our performance?)
- How much locality is there in frontiers with respect to the "active" frontier vs. the entire set of vertices? Interesting visualization project, for instance: Get a list of the active vertices in a frontier as a function of iteration, so iteration 0 is vertex set A, iteration 1 is vertex set B, etc. For one iteration, visualize the vertex set as a color per chunk of vertices, say, 1024 vertices per pixel. If all 1024 vertices are part of that frontier, the pixel is white, if 0 black, and gray in between. Then each iteration makes another row of pixels. This shows three things: (a) how many vertices are in the frontier compared to not; (b) how much spatial locality there is; (c) how the frontier evolves over time. One of the goals of this effort would be to determine how useful it would be to do some reordering of vertices either statically or dynamically, and either locally (within a chunk of vertices) or globally.
