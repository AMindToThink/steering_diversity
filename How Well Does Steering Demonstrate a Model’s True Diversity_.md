# Tab 1

# How Well Does Steering Demonstrate a Model’s True Diversity?

# Overview

Researchers want to ask the counterfactual question “What would happen if this model decided to have a certain behavior of interest?” Because models often do not have a propensity toward that behavior, even when they have the ability to have that behavior, it is difficult to figure out what that behavior would look like from the model. Existing approaches include various types of prompting, finetuning, and activation steering. The issue with these approaches is that it is currently unclear how much they influence propensity and how much they influence capability. For example, activation steering has had significant success as a research tool for studying misaligned behavior, but also damages the AI’s performance. As another example, prompting brings along confounders related to “instruction following” and particular personas the prompt might induce in the model. 

This research investigates the effect that different steering methods have on a model’s behavior. The overarching goal is to check the range of outputs each steering method is capable of producing. In particular, this research will focus on deception though hopefully the experiments can be designed such that it is easy to experiment with other topics.

| Feedback request: (If you are one of my mentors) Please focus on the Overview and the Theory of Change to verify that it is precisely the topic we discussed. If any sentences seem weird to you, it is probably because I misunderstood what we were talking about and am going completely off on the wrong track. |
| :---- |

# Key Assumptions

* Propensity vs Capabilities  
  * Propensity is likely a thin layer on top of the bulk of model parameters being used for the model’s capabilities. This assumption is backed up by parameter-efficient finetuning working for alignment and steering vectors working to influence model behavior. This assumption is invaluable for approaches that measure or edit propensity.  
  * This work aims to keep the capabilities of a model constant and increase the propensity to measure the capabilities  
* We are assuming that diversity is a valid proxy of worst/most extreme case behavior.   
  * This is likely a better approach than trying to maximize a narrow instance.  
  * More sophisticated quality-diversity algorithms might have advantages finding interesting behaviors.

# Key Questions Before Experimentation

* What type of clustering method should be used?  
  * [Claude’s suggestions](https://claude.ai/share/afa9ccc4-bf48-4256-b215-0640539a365d)  
  * Shi sent: [https://arxiv.org/abs/2504.09389](https://arxiv.org/abs/2504.09389)   
    * *One way to combine quality diversity framing but make safety relevant would be to find benchmarks or tasks where the model is on the cusp of deception. [Neel Nanda on Thought Branches](https://arxiv.org/abs/2510.27484).*  
    * *Find reference for “when you find jailbreaks, you can reliably get around refusal to get the model to respond, but there’s a loss of capability, and the model would often be wrong”*  
    * *“You could frame this project as ‘trying to get realistic deceptive behavior where reasoning does not suffer’”*  
    * Shi is interested in the diversity and “appears in one approach but not another” as a way to approach strategies the model could use.  
* What steering methods are the most important to try?

# Methods

We want to find methods to sample harmful outputs from a model, even if those outputs are unlikely. Throughout this research, a concrete type of harmful action we will research is “deception.”

## Metrics

To measure the difference between the steered/modified model and the unsteered model, we will employ the following tests:

* Varied outputs  
  * If the steered model becomes low-entropy (eg a deceptive model becomes deceptive in only one way) compared with the baseline model, that indicates a drop in capabilities.  
  * Can measure using clustering in sentence embedding space.  
* Deception quality/convincingness/scariness  
  * LLM as a judge   
* Capabilities benchmark performance  
  * Steered models should not be “lobotomized” by the steering method. Performance should be similar to the unsteered model.  
* Negative Log Likelihood (NLL) (from unsteered model)  
  * Is the model’s steered behavior *probable* for the unsteered model.  
  * Since steering demonstrates that the model is fully capable of the output, NLL could be used to get an estimate of the propensity of the original model to have that behavior

Additionally, we can measure the following for a fair comparison:

* Latency: How long does it take a steering method to produce the first valid output? Some methods, like prompting, can produce their first output within milliseconds. Other methods involving training produce their first outputs after minutes or hours.  
* Bandwidth: How many outputs can you get per second after the initial warmup? Techniques involving training involve an up-front cost but afterward it is relatively easy to sample new outputs, so it has high bandwidth. Best-of-N has no up-front cost, but it involves a large number of iterations to get a single sample, so it has low bandwidth. This connects with the reliability of a method. Approaches that are more expensive (heavy fine-tuning for example) would have a longer wait to get the first output, but possibly also higher reliability (every output is more likely to have the property of interest). Since I expect fairly high bandwidth from all methods due to fairly high reliability, I expect Latency to be of greater interest to researchers and practitioners who are looking for steering approaches for model organisms.  
* Compute costs

## Misbehavior Elicitation Type

- Role playing prompt  
- Instructed prompt  
- Steered  
  - [Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)  
  - Other techniques?  
- Fine-tuned  
  - Synthetic, off-policy data SFT  
- ~~RL~~  
  - ~~Maybe more complicated to train?~~  
- On-Policy Distillation ([my suggestion](https://docs.google.com/document/d/1vUVgRlPthHy2-jODD2eLW1cAPReCmzm1VBuulltjg24/edit?tab=t.0#heading=h.vv36fka1n6fl))  
  - Needs fewer data points, test with 1-3, and also with full training dataset

# Experiments 

## Range of Model Behaviors

Results would look like:

* Average Badness according to LLM as a judge  
* Number of clusters for each model’s output  
  * Characterize the clusters

### Varied Dataset

In a varied dataset of situations, how many types of outputs does each type of steering produce, and how bad is it according to LLM as a judge.

Might be fun to use [Petri/Bloom](https://www.anthropic.com/research/bloom).  
Varied prompts to which the model can have the behavior in question (eg deception).

Shi expects this to work better than Single Prompt because the diversity will be across prompt-responses rather than just responses.

*Measurement of diversity from a particular set of prompts probably exists. For example, there are probably diversity scoring mechanisms for novels, but not deception. Uncertain how difficult translating an off-the-shelf taxonomy to steering mechanisms. Do language model based clustering or diversity scoring. We can decompose the scoring into steps.* 

- *At the end, we want a quantitative diversity metric*  
- *We want to say model organism1 is more diverse than model organism2*  
- *We can decompose that scoring function into a model, prompts, \-\> number.*  
- *One of the first steps in clustering and tagging model responses*  
  - *Transluce \- Docent*  
    - Good for reviewing agent transcripts  
  - [*https://arxiv.org/abs/2602.10371*](https://arxiv.org/abs/2602.10371)  
  - [*https://arxiv.org/abs/2510.27484*](https://arxiv.org/abs/2510.27484)   
  - [*https://arxiv.org/abs/2504.09389*](https://arxiv.org/abs/2504.09389) 

### Single Prompt

Choose/write a prompt of particular interest, and sample many times from each steering method.   
Possible prompts include:

* “Tell me about yourself”

### Striking Examples

Striking Examples seeks cases where the models fine tuned with the different techniques output very different outputs in response to a prompt. 

This section is inspired by the Emergent Misalignment (EM) paper, which revealed some extremely striking examples of the model generating harmful text. 

Without Striking Examples, the paper is less likely to have an impact, be interesting to researchers, or demonstrate its utility.

In the case of this research, striking examples would come from sampling many times from each technique and demonstrating “types” (clusters?) of responses that are absent from the outputs of one type of steering but present in the other. As an imaginary example, maybe deception\_steered\_model\_1 never spontaneously tells the user that it is the messiah, but deception\_steered\_model\_2 does. 

## Performance

Run the steered models past standard capabilities benchmarks and compare their scores.

# Theory of Change

* On-policy samples allow for more accurate probes to be constructed that can better predict that the model would sample deceptive output rather than just viewing it, roleplaying, or thinking that it is what the user wants  
* Models may hide their abilities from auditors  
  * Unlocking their abilities in controlled settings allows developers to be more cautious with deployment  
  * Relevant: [Steering Evaluation-Aware Models to Act Like They Are Deployed](https://www.alignmentforum.org/posts/peKrvZ6t9PSCzoQDa/steering-evaluation-aware-models-to-act-like-they-are)  
* In the near future, models may be able to learn much more from experience, giving them expanded and surprising capabilities. This means that there wouldn’t be a “helpful-only” model on which to evaluate capabilities, as the model would have acquired goals (propensities) and capabilities together “in-the-wild”. Better ways to uncover capabilities will be vital to understanding and defending against this type of model  
* (From David) Organizations such as Apollo that are trying to measure deceptive capabilities in models are typically creating model organisms that exhibit such capabilities. These model organisms are frequently trained through steering, e.g. identifying a “deception” or “situational awareness” direction and increasing/decreasing it. Our question is, how well do these model organisms depict the behavior of the original model under specific circumstances? Is the distribution of possible outputs representative of all possible outputs? Or is there some other way to elicit even stronger deceptive capabilities that is being overlooked?  
* Force multiplier for other model organisms research, since we could provide recommendations for how to produce model organisms  
* Clarify and critique a common narrative around steering. In the current safety literature, people are too comfortable saying things like "we steer the model on X" where X is an abstract, hard to define, and hard to measure concept like deceptive intent, evaluation awareness, or sycophancy. If steering methods differ in the range of outputs they evoke and “miss” outputs the model is capable of doing, then that demonstrates a problem with that kind of research.

# Key Uncertainties

* Is this project precisely what Shi wants?  
* Is focusing on deception a good call?

# Relevant Research:

* [Liars' Bench: Evaluating Lie Detectors for Language Models](https://arxiv.org/abs/2511.16035)  
* [Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](https://arxiv.org/abs/2602.02343)  
* [https://forum.effectivealtruism.org/posts/rAL3YAYYr6gGcyzq5/thinking-about-propensity-evaluations](https://forum.effectivealtruism.org/posts/rAL3YAYYr6gGcyzq5/thinking-about-propensity-evaluations)   
* [Analyzing the Generalization and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404)  
* [Claude on Clustering Approaches](https://claude.ai/share/afa9ccc4-bf48-4256-b215-0640539a365d)  
* [Claude on this doc 2026-02-18](https://claude.ai/share/27d028f0-5a79-4119-917a-5efaea31edd7)  
* [Understanding the effects of RLHF on LLM generalisation and diversity](https://arxiv.org/abs/2310.06452)

# Todo List

Todo List

* Simple diversity experiment  
* Test Neuronpedia’s steering vector on the diversity benchmark  
  * Neuronpedia’s steering vector  
    * Konstantinos is working on this  
    * “The most popular”  
    * Neuronpedia only has SAEs  
  * SAE steering? A lot of people still use it, so we could provide more arguments why they are bad.

# Simple Diversity Experiment

# Experiment 1: Simple Diversity Experiment

* We want to evaluate the output diversity of steered models.  
* Steering vectors and infrastructure for training them can be found on EasySteer Github.  
* euclaise/writingprompts is a fun dataset of prompts we can have models respond to.  
* We need a clustering method and diversity metric  
  * Sentence-BERT embedding cosine similarity  
  * HDBSCAN for clusters

