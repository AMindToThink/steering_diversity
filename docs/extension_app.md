Application Questions
Your Research

- Outline the current state of your research and your plan for the extension period.

I am investigating whether activation steering collapses the diversity of LLM outputs. This question has direct implications for the validity of model organisms research in AI safety, and is important for steering research more broadly. In particular, one use-case of steering is to make AIs 'think they are deployed' so we can evaluate them without their awareness of being evaluated influence their behaviors. This is promising, however if steering produces model organisms that are behave in only a small set of the ways they normally would, then safety evaluations built on steering may miss concerning behaviors the model is would actually do when deployed. I have run diversity experiments across two model architectures (Qwen2.5-1.5B, Llama-3-8B) and three steering concepts (happiness, style, creativity), finding that steering does reduce measurable diversity (by 19%–50% depending on concept), though the effect varies substantially across steering vectors and models, and there is more work to be done in validating that the measurements are meaningful.

Concurrently, I conducted a baseline survey of evaluation awareness across 13 frontier models, and I am preparing to perform a steering experiment testing whether the test-awareness probes from one paper work on an eval-awareness benchmark from another paper, and more relevantly to my research, what this steering does to diversity.

I also developed a novel in-context learning (ICL) diversity metric that provides a coherence-weighted, model-based measure of response diversity (really "learnable structure"), validated across GPT-2 and Qwen2.5-32B. This is being written up as a separate short paper.

**Plan for extension:**
I do not know how much of my current plan

1. **Run the eval awareness steering experiment**. This is the key experiment connecting my two research threads. Does steering a model to be evaluation-unaware also collapse the diversity of its outputs?
2. **Validate ICL diversity metric on real data.** Currently, the main results in my ICL Diversity tests were from synthetic data meant to demonstrate particular desirable features of the metric (eg that it increases when there are more types of responses). I have attempted to evaluate it with a diversity measurement system from the literature, but got negative results. I am confident enough that I'm on the right track that I think there was a problem with the experimental structure and the construct validity.
3. **Apply the ICL diversity metric** to all steering experiments for a coherence-aware diversity measure.
4. **Run pass@k experiments for steered models.** Pass@k is a universally respected measure for performance, and it is very connected to diversity. Unlike the previous metrics, this one requires some measure of "success" and a set of problems for the model to solve instead of just comparing sets of responses from one prompt.
5. **Compare elicitation methods** to test which methods best resist diversity collapse. The research is primarily on steering methods, so make an as-fair-as-possible comparison of (at least) SAEs and CAA. Include prompting and (as a stretch goal) fine-tuning. instructed prompting and fine-tuning baselines,
6. **Write up results** for ICML 2026 workshop submission (deadline ~April 24) and NeurIPS 2026 (deadline ~mid-May).
7. **Identify "striking examples".** Finding response types present under one elicitation method but absent under another would make the diversity collapse concrete and impactful.
8. **Extend diversity experiments to safety-relevant concepts** (deception, sycophancy) rather than only benign concepts (happy, style, creativity).
9. **Answering more open questions.** During the course of this research, more and more open questions revealed themselves, and I am keeping track of them. I hope to form a complete picture of what is going on.

Link to research

- Please include a link to your current research deliverable, such as a GoogleDoc or Slides presentation. This document does not need to be in its final form.

<!-- TODO: Insert link to paper.md hosted on GitHub, or a Google Doc export -->

Goals

- What are your goals for this extension? What would success look like?

**Primary goal:** Complete and submit a paper demonstrating how steering affects output diversity. Ideally, I could make some concrete recommendations for safety researchers who use steering to construct model organisms, though more likely I will be issuing a warning instead.

**Success looks like:**

1. A submitted paper (ICML workshop or NeurIPS) characterizing how steering affects output diversity in general, with a focus on safety-relevant concepts (deception, eval awareness).
2. The ICL diversity metric paper submitted separately (again, ICML workshop or NeurIPS).

**Stretch goals:**

Compare the steering diversity with that of post-training.
Inject within-prompt variation via subtle prompt changes/paraphrases, and measure diversity of output in response.
Compare against random vectors as control.

Making the most of the extension

- Describe the specific actions you will take to maximise the impact of the extension. Examples include:

Collaborating with a particular researcher X,
Networking with individuals at a government agency Y or think-tank Z,
Applying for jobs or research positions (provide specific role titles).

1. **Reaching out to the authors of "Steering Evaluation-Aware Models to Act Like They Are Deployed"** (Tim Hua, Andrew Qin, Samuel Marks, Neel Nanda). Their work is a motivation for my eval-awareness experiments, and my results on how steering affects diversity could inform how they interpret steered model behavior.

2. **Engaging with Apollo Research, METR, Redwood Research** who work on model organisms and elicitation methods. My findings have practical implications for how they design evaluation pipelines that rely on steering.

3. **Applying for research positions** at AI safety organizations. I am currently going through MATS stage 3. I will send applications for entry-level researcher positions at Anthropic, Redwood, Apollo, and others. I really feel like if I can turn my current research into a publication, my application to these groups would be much stronger.
<!-- TODO: Add specific role titles and application deadlines -->

If you did not receive an extension, what would your most likely next step be?

- For example: returning to a previous role, applying for other fellowships or positions, continuing the research independently, or something else.

Please be specific about your plans and timeline.

<!-- TODO: Matthew — fill in your specific fallback plan. Some options to consider:
- Continue the research independently while applying for positions
- Apply for MATS or other fellowships
- Return to a previous role
- Apply for PhD programs
-->

April-June continue the research on my own. Hopefully Shi would still like to mentor me even without the ERA extension.
I will apply to LASR labs, go down the list of jobs on 80k and aisafety.com/jobs, and generally search for good AI Safety research jobs.
Backup plan: I have a return offer to an undetermined SWE position at Amazon in Arlington (near DC).

Logistics
Duration

- Please specify your desired duration of the extension. On average, we expect extensions to be 2-4 weeks long, but we will offer extensions of up to 8 weeks in some circumstances.

<!-- TODO: Matthew — choose duration. Given the plan (GPU experiments + paper writing + submission deadlines), 4–6 weeks seems appropriate. The ICML workshop deadline (~April 24) is ~6 weeks away; NeurIPS (~mid-May) is ~9 weeks. -->

I would like to keep working and target my papers for ICML in late April and NeurIPS on May 07, 2026. So my ideal would be for the extension to last for 7 weeks after the fellowship ends, so I can continue working with Shi and David on the projects until NeurIPS.

Time commitment

- How many a hours a week would you commit to the ERA extension? We expect most extensions to be for 10, 20, or 40 hours/week.

<!-- TODO: Matthew — choose hours/week. 40 hours/week would be ideal given the paper deadlines. -->

15 hours per week

Rationale

- Please explain the reasons for your desired duration and time commitment. Remember, extensions are not just extra time but an opportunity to make a long-term impact. Consider this when determining how long and how many hours per week you want to commit to the extension.

The timeline is driven by two submission deadlines:

- **ICML 2026 workshop** (~April 24): A 4-page workshop paper on steering and diversity with the existing experiments plus eval awareness steering results. This may involve a lot of appendices to have content without exceeding the page limit.
- **NeurIPS 2026** (~mid-May): A full paper including additional steering concepts, elicitation method comparisons, and the ICL diversity metric.

Time commitment: 15 hours per week would be best, since I will also be balancing school and other responsibilities.

<!-- TODO: Adjust rationale based on chosen duration/commitment -->

What additional resources would you need for your extension?

- Please include: whether you'd require any travel support (e.g., for conferences), compute, and any other costs beyond the standard stipend. If yes, please also provide a rough budget for this.

1. **GPU compute**: Stronger models are more eval-aware, but also more expensive. Access to GPUs on RunPod will be increasingly important as I exit the small-scale testing phase and scale up my experiments to larger, stronger models. I would very much appreciate if ERA were to continue providing me with RunPod credits. If I get an extension, I would be in touch with David to discuss any particularly expensive experiments I would want to do. At the most extreme end, I may want to do experiments that involve running, steering, and sampling many responses from a 1T parameter model.
<!-- TODO: Estimate specific GPU hours needed and rough cost -->

2. **Potential Travel**: I have a conflict on the weekend of ControlConf (the EA Midwest Retreat), though I would be interested in traveling to a conference to discuss my research and meet other researchers.
<!-- TODO: Add location and estimated travel costs -->

3. I am currently unsure whether I will be getting course credit for doing this research. If I can get course credit, I cannot also get a stipend (Northwestern policy). If I am not getting course credit, I would appreciate a stipend.

Location

- Where would you plan to be based for the duration of your extension? Please include any date(s) by which you would have to leave the UK, if applicable. Note that we cannot provide accommodation for the extension period.

<!-- TODO: Matthew — fill in your location and any UK visa constraints -->

I would be working in Evanston, Illinois at Northwestern University.

Anything else?

<!-- TODO: Optional — anything else to mention? E.g., the ICL diversity metric as a standalone contribution, the EasySteer float16 bug fix contributed upstream, etc. -->

I have done some mini research projects over the last few weeks. In particular, prompt-distillation and forum_persona should be of interest to the AI Safety community. I hope to do more work like this.
https://github.com/AMindToThink?tab=repositories
