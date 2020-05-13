# Thoughts on performance evaluation for HIVE apps       {#perf_eval}

Ben asked

> It'd be helpful to see an example of a good performance measurement though, since I'm not sure the sorts of things you guys usually produce for that

and

> "metrics for performance identified" --- what's an example of that _not_ being runtime?

These are great questions.

What DARPA is interested in is _understanding_ these applications and their potential performance. They have a high-level understanding of the applications themselves but not so much about their implementation on a GPU and how that might translate to the new architectures that they are targeting. Of course our problem is not figuring out how to map these workflows to these new architectures. But we are implementing them in a parallel way (in many cases, in a way that has not been done before) and the lessons we learn will be applicable to these new architectures. Our goal is to be the state of the art. If we make it harder for the other performers to achieve their 1000x speedups over the state of the art, great, that's our job.

It is important that we be honest. Sometimes when we write papers we tend to try to highlight the best results and perhaps not highlight the worst results. We want to be as honest as we can. If there's a deficiency in the GPU or its programming system that keeps us from being as good as we can be, we need to point that out. If we haven't optimized something, say so frankly. If we think a new better algorithm might improve our performance in some regard, say so frankly. There should be a future-work section if there is any more work at all to be done. (And it's also OK to point out core-Gunrock things we need to improve! e.g., "Needs this other mode in this operator" or "that operator has terrible performance for this particular use case" or "really would benefit from a new operator here".)

So, what kinds of things do we want to measure and report?

Runtime on different datasets, and throughput in MTEPS, are the obvious ones. We should also look at memory usage. What is the largest dataset that fits into GPU memory and why (where does the memory go)? Many of the workloads have straightforward data structures that use a lot of memory but could perhaps be reduced with some interesting data structure work; we should detail this when appropriate.

We don't just want to look at runtime. We also want to contribute to the understanding of the application. These are workflow-dependent; you need to pick what's most appropriate for your workflow; this might include:

- A breakdown of runtime into different components ("turns out we spend all of our time here")
- Bandwidth analysis of memory performance, e.g., what is the fraction of peak that we're sustaining the workflow as a whole or on particular kernels, and what levels of the memory system are we stressing, or are we getting any locality in cache
- A bottleneck analysis of where we're slowest (computation, memory, bus bandwidth, etc.)

This is useful: https://hiveprogram.com/wiki/pages/viewpage.action?spaceKey=WOR&title=Performance+Evaluation+Using+Workflows . It says what exactly is being measured in the OpenMP versions. We need to be this precise. Definitely look at it for your workflow.

Muhammad comment:

> Ben and I were chatting about the performance metrics that make sense to report; I pointed out that there are some applications that might have a metric additional to the performance stuff (runtime, memory usage, etc.), which will determine some sort of characteristic about the app as well. For example, "how close are geolocation's predictions to ground truth, what is the error margin here?" or *SGM* is similar; and some V1 apps also will need this additional metric.
>
> Other than this, runtime, MTEPS (?), memory analysis (usage, bandwidth utilized) seem sufficient for report. Some additional metrics could be determined through profiling as you listed those already.
>
> It all depends on how much time we would like to spend per app, and we have to be a little conservative here because there's a lot of stuff to analyze and not enough time.
