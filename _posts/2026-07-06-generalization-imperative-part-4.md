---
layout: post
title: "The Generalization Imperative (4/4): Interpretability, Weak-to-Strong, and the Path Forward"
date: 2026-07-06 09:00:00
description: Part 4 of 4 — Sections 10–12. Mechanistic interpretability of OOD generalization, weak-to-strong supervision and trust functions, and a synthesis across all 43 papers with references.
tags: generalization deep-learning machine-learning interpretability alignment weak-to-strong
categories: blog
toc: true
series: generalization-imperative
series_part: 4
---

**📖 Series:** [Part 1](/blog/2026/generalization-imperative-part-1/)  |  [Part 2](/blog/2026/generalization-imperative-part-2/)  |  [← Part 3](/blog/2026/generalization-imperative-part-3/)  |  **Part 4**

---

## 10. Peering Inside: Mechanistic Interpretability of Generalization

> *If we can see what breaks when generalization fails, we can fix it.*

The newest line of attack, barely a year old in its current form, uses tools from mechanistic interpretability to understand generalization from the inside.

**Sparse autoencoders at the edge.** Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, and Danilo Bzdok, in a paper accepted at ICML 2026 <a href="#ref24">[24]</a>, used sparse autoencoders (SAEs) to probe what happens inside language models when they encounter OOD inputs such as typos, unusual phrasings, and jailbreak attempts. Their finding: OOD inputs cause models to activate significantly more fallacious internal concepts. The model doesn't just make mistakes; it *thinks wrong*. They propose a mechanistically grounded fine-tuning strategy that targets these fallacious concept activations, making the model more robust.

This paper represents an important conceptual shift: OOD is not just a property of the input distribution, but of the model's private computational processes. A prompt that looks in-distribution to a human might be OOD to the model's internal concept space, and SAEs give us a tool to detect this.

---

## 11. Weak-to-Strong Generalization: When Supervision is the Bottleneck

> *If you can't reliably evaluate what your model does, how do you know it's generalizing?*

In December 2023, OpenAI's Superalignment team, led by Ilya Sutskever and Jan Leike, published a paper that reframed the generalization problem in a startling way <a href="#ref35">[35]</a>. The central challenge of aligning superhuman AI is that humans will not be able to reliably evaluate superhuman outputs. If you can't tell whether an answer is right, you can't provide a training signal. The team proposed studying an accessible analogy: can a weak model supervise a strong model?

**The experiment.** The setup is elegant. A small pretrained model (GPT-2-level) is fine-tuned on ground-truth labels to serve as the "weak supervisor." A large pretrained model (GPT-4) is then fine-tuned using only the weak supervisor's predictions, never seeing ground truth. The question: can the strong model recover capabilities beyond its weak teacher?

With naive fine-tuning, the answer was disappointing. GPT-4, supervised by a GPT-2-level model, recovered only about half the gap between the weak supervisor and the strong ceiling (GPT-4 fine-tuned on ground truth). The strong model was imitating its teacher's errors.

Then the team discovered a simple technique that changed everything: an auxiliary confidence loss that encourages the strong model to be more confident in its own predictions, including confidently disagreeing with the weak supervisor when its internal representations support a different answer. The intuition: the strong pretrained model already has good representations for the task. The weak supervisor's job is not to *teach* but to *elicit* what the strong model already knows. With the confidence loss, GPT-4 recovered roughly 80% of the gap to ground truth on NLP tasks, performing at approximately GPT-3.5 level despite never seeing a correct label.

**What transfers, and what doesn't.** The results varied dramatically across domains. On NLP tasks (sentiment analysis, natural language inference, reading comprehension), weak-to-strong generalization was robust. On chess puzzles and reward modeling, it was much weaker. And on ChatGPT preference data, the method barely worked at all. The pattern suggests that weak-to-strong generalization succeeds when the strong model's pretraining already encodes task-relevant knowledge; it fails when the task requires capabilities that pretraining doesn't provide.

**Beyond confident disagreement.** Three additional methods showed promise. Bootstrapping, which uses intermediate-sized models as stepping stones (small → medium → large) rather than going directly from very weak to very strong, improved chess performance. Early stopping was critical: without it, the strong student eventually overfits to the weak supervisor's errors, undoing its generalization gains. And unsupervised generative fine-tuning, which improves the strong model's representations on task-relevant data before weak supervision, provided additional benefits.

**Theoretical foundations.** Follow-up work has begun formalizing the phenomenon. Charikar et al. (NeurIPS 2024) proved that the strong model's improvement over the weak model is exactly quantified by the misfit error, the disagreement between their internal representations. Medvedev et al. (2025) extended this to random feature models, showing that the student can achieve quadratically lower error than the teacher, and that early stopping is provably essential.

**Why this matters for generalization.** Weak-to-strong generalization connects two problems that are usually discussed separately: generalization and alignment. The connection is this: if the central challenge of superalignment is that humans can't judge superhuman outputs, then generalization from weak to strong supervision is the core technical problem of AI safety. The confident-disagreement method is, in essence, a technique for eliciting generalization from a model when you can't provide ground-truth feedback, a scenario that will become increasingly common as models surpass human capabilities in more domains.

**Trust Functions: reframing weak-to-strong as data selection.** In 2026, Arda Uzunoglu, Alvin Zhang, and Daniel Khashabi at Johns Hopkins pushed the weak-to-strong program further with *Trust Functions* <a href="#ref43">[43]</a>, reframing the problem as data selection: the key is determining *which* weak labels are trustworthy, not how to globally trade off weak supervision against the strong model's priors. Their method learns a neural module, a trust function, that assigns a scalar trust score to each weak label by analyzing the teacher's last-layer hidden state. Trust-filtered training achieves near-lossless weak-to-strong generalization: student models match or even surpass ground-truth supervised models across world knowledge (ARC, OpenBookQA, SciQ), quantitative reasoning (AIME, MATH), and strategy games (chess puzzles). In chess, trust-filtered training produced a Qwen3-14B student that achieved 44.1% accuracy versus 39.9% from ground-truth training: the student *outperformed* the human-annotated labels, because 32% of selected moves were actually better alternatives that human annotators had missed.

Three mechanistic explanations emerged. First, trust scores create an implicit easy-first curriculum that naturally biases selection toward examples the strong model can learn from. Second, trust functions recover near-optimal alternatives, moves the weak teacher couldn't find but that are valid or even superior. Third, trust-filtered training batches produce coherent gradient updates with a lower-rank, better-aligned gradient structure. The method also compounds gains through iterative weak-to-strong chains: train a student, then use it as the next teacher, with the final model outperforming any single-stage approach. Trust Functions transforms weak-to-strong generalization from a capability-recovery problem into a capability-*amplification* problem, and suggests that the gap between weak supervision and strong performance may be bridgeable with surprisingly simple mechanisms.

---

## 12. Where Do We Go From Here?

Looking across these papers and talks, I see a field that has moved from naming the problem to understanding its mechanisms, and, increasingly, to deploying generalization in the real world and proving when and why it happens. Sutskever and Gu gave us the conceptual language: generalization is compression <a href="#ref25">[25]</a>, data efficiency is the unifying thread connecting interpretability, reasoning, multimodality, and democratization <a href="#ref26">[26]</a>, and the age of scaling has given way to the age of ideas <a href="#ref29">[29]</a>. The early empirical work (2016–2020) established that generalization is a real puzzle, proposed meta-learning as a framework, and demonstrated that architecture matters. The middle period (2020–2023) formalized the problem through Chomsky hierarchies and grokking dynamics, achieved the first human-level systematic generalization result, and reframed generalization as an alignment problem through weak-to-strong supervision <a href="#ref35">[35]</a>. The current wave (2025–2026) is providing the theory, from phase transitions, scaling laws, geometric characterization, measure-theoretic bounds, and energy-landscape analyses to exact high-dimensional characterizations and rigorous sample-complexity separations, that explains *why* earlier approaches worked, while extending these insights from toy settings to real LLM pretraining <a href="#ref27">[27]</a>, practical optimization tools <a href="#ref28">[28]</a>, embodied robotic systems <a href="#ref34">[34]</a>, the causal structure of data itself <a href="#ref31">[31]</a>, <a href="#ref32">[32]</a>, <a href="#ref33">[33]</a>, curriculum-driven compositional generalization <a href="#ref40">[40]</a>, <a href="#ref41">[41]</a>, physical law discovery <a href="#ref42">[42]</a>, recurrent length generalization <a href="#ref36">[36]</a>, formal language acquisition <a href="#ref39">[39]</a>, and near-lossless weak-to-strong supervision <a href="#ref43">[43]</a>.

Several cross-cutting themes emerge:

1. **Generalization is a phase transition, and a recurrent struggle.** Whether in grokking dynamics <a href="#ref12">[12]</a>, <a href="#ref20">[20]</a>, <a href="#ref21">[21]</a>, training data diversity <a href="#ref15">[15]</a>, recurrent depth <a href="#ref17">[17]</a>, real LM pretraining <a href="#ref27">[27]</a>, or diffusion model generation <a href="#ref30">[30]</a>, the transition from memorization to generalization is sudden, not gradual. But Wen et al.'s mode-hopping discovery <a href="#ref27">[27]</a> adds a crucial twist: at scale, the transition is not one-and-done. Models flip back and forth between parrot and intelligence modes throughout pretraining. Generalization is not a destination; it's a fight that never fully ends. Monitoring the right signals (spectral entropy, weight norm, induction head formation, energy landscape curvature) might let us detect and accelerate favorable transitions.
2. **Architecture is destiny, but architecture is subtle.** The difference between a transformer that generalizes and one that memorizes can come down to parameter sharing <a href="#ref10">[10]</a>, adaptive routing <a href="#ref11">[11]</a>, shift-invariant positional encodings <a href="#ref8">[8]</a>, <a href="#ref23">[23]</a>, or the initialization bias of an ACT halting mechanism <a href="#ref19">[19]</a>. The universal transformer / looped model program is maturing from proof-of-concept to practical scaling laws <a href="#ref18">[18]</a>.
3. **Data diversity has a phase diagram.** Kawata et al. <a href="#ref15">[15]</a> showed that data diversity is not "more is better" but a specific condition for inducing generalizable circuits. Combined with the task vector geometry results <a href="#ref22">[22]</a>, Wen et al.'s finding that different pretraining data windows promote parrot or intelligence modes <a href="#ref27">[27]</a>, and π0.5's demonstration that data from other robots and the web transfers across embodiments <a href="#ref34">[34]</a>, we're developing a precise language for what training distributions produce what kinds of generalization.
4. **Causality is the principled path to invariance.** IRM <a href="#ref7">[7]</a> opened the door; causal representation learning <a href="#ref31">[31]</a>, <a href="#ref32">[32]</a>, axiomatic training of transformers for causal reasoning, and causal reinforcement learning <a href="#ref33">[33]</a> are now walking through it. The core insight, that OOD generalization requires learning the causal structure of the data-generating process rather than just its statistical surface, is increasingly supported by both theory and practical results.
5. **OOD** and in-distribution are different computational modes. Task vector geometry <a href="#ref22">[22]</a>, SAE concept analysis <a href="#ref24">[24]</a>, and measure-theoretic bounds <a href="#ref23">[23]</a> all point to the same conclusion: OOD generalization is not just "harder in-distribution"; it relies on different computational mechanisms operating in different representational subspaces. Litman and Guo's signal/reservoir theory <a href="#ref28">[28]</a> gives this a mechanistic foundation: what reaches the signal channel generalizes; what stays in the reservoir doesn't.
6. **The meta-learning bet is paying off.** Lake and Baroni's Nature result <a href="#ref14">[14]</a> showed that meta-learning on compositional tasks can produce human-level systematic generalization. Combined with architectural innovations like looped transformers <a href="#ref17">[17]</a>, <a href="#ref18">[18]</a>, <a href="#ref19">[19]</a>, latent reasoning mechanisms <a href="#ref16">[16]</a>, practical optimization tools like the SNR gate <a href="#ref28">[28]</a>, and embodied systems like π0.5 <a href="#ref34">[34]</a>, the path to building machines that learn like humans is becoming clearer.
7. **Embodied generalization is the ultimate test.** π0.5 <a href="#ref34">[34]</a> demonstrates that the principles discovered in language and vision, such as data diversity, meta-learning, hierarchical reasoning, and world models, transfer to the physical world. But it also reveals a new requirement: generalization across embodiments. A robot that learns to pick up a cup should be able to do so with a different gripper, on a different arm, in a different kitchen. Cross-embodiment generalization is the next frontier.
8. **Generalization and alignment are converging.** Weak-to-strong generalization <a href="#ref35">[35]</a> reveals that the problem of eliciting capabilities from a model you can't fully evaluate is structurally identical to the problem of aligning a model you can't fully supervise. Sutskever's 2025 vision <a href="#ref29">[29]</a> of superintelligence as a continuously learning entity that can amalgamate knowledge across instances raises the stakes: if generalization is what enables a model to learn from weak supervision, then solving generalization is a prerequisite for safe superintelligence.
9. **Theory and practice are converging.** Litman and Guo's SNR preconditioner <a href="#ref28">[28]</a> is the kind of result the field needs more of: a deep theoretical framework that produces a one-line code change with measurable impact. It accelerates grokking by 5×, costs nothing extra, and works across domains from modular arithmetic to preference optimization. The diffusion model energy-landscape theory <a href="#ref30">[30]</a> connects generative AI to decades of associative memory research, revealing that "spurious states," long dismissed as failures, are the birthplace of generalization. The exact high-dimensional theory of single-head attention <a href="#ref37">[37]</a> provides quantitative predictions for weight spectra and scaling laws, not just bounds. This is what progress should look like.
10. **The right inductive bias transforms a curve-fitter into a physicist.** Liu et al.'s Kepler-to-Newton result <a href="#ref42">[42]</a> is the cleanest demonstration yet: two models with identical architectures and identical training data learn radically different world models, one geometric (Kepler) and one mechanistic (Newton), depending on a single hyperparameter (context length). Both achieve high predictive accuracy, but only the mechanistic model generalizes to counterfactual interventions. This result crystallizes the causal program's central claim <a href="#ref7">[7]</a>, <a href="#ref31">[31]</a> and gives the architecture hypothesis <a href="#ref17">[17]</a>, <a href="#ref18">[18]</a>, <a href="#ref19">[19]</a> a new vocabulary: inductive biases are not just about improving accuracy; they determine whether a model learns to *describe* the world or to *explain* it.
11. **Generalization can be unlocked, not just trained.** Two apparently disparate results converge on a hopeful message. Ruiz and Gu <a href="#ref36">[36]</a> showed that length generalization in Mamba requires only ~500 steps of state-diverse post-training, because the capability was latent in the weights all along. Uzunoglu et al. <a href="#ref43">[43]</a> showed that weak-to-strong generalization can be near-lossless with a simple trust-scoring mechanism, since the strong model already knows most of what it needs, and just needs to learn when to trust its teacher. Both results suggest that for certain kinds of generalization, the bottleneck is not capability acquisition but capability *elicitation*. The model already has the circuits; we just need to learn how to activate them.
12. **Curriculum is a first-class generalization primitive.** SOAR <a href="#ref40">[40]</a> and Rajaraman et al.'s Curriculum II <a href="#ref41">[41]</a> demonstrate that the *order* in which examples are presented can reduce sample complexity from exponential to subpolynomial, and can bootstrap a model from zero success to competent performance through self-generated stepping stones. Meta-learning's original promise was "learn to learn"; the 2026 refinement is "learn to *teach*." The right sequence of learning experiences is not just a pedagogical convenience; it is a mathematical determinant of whether generalization is possible at all.
13. **The theoretical foundations are solidifying.** Single-head attention now has a complete, closed-form theory of generalization <a href="#ref37">[37]</a>. The computational advantage of depth for compositional targets has been rigorously proven <a href="#ref38">[38]</a>. The transition from local statistics to hierarchical grammar has been mechanistically explained <a href="#ref39">[39]</a>. The Chomsky-hierarchy program <a href="#ref8">[8]</a>, <a href="#ref13">[13]</a> that began by cataloguing *what* networks can learn is being succeeded by a theory that explains *how* they learn it. This is the maturation of a field: from empirical observation, to formal characterization, to mechanistic understanding.

The challenge articulated across all these works, from Sutskever's compression theory and jagged intelligence diagnosis, Gu's data-efficiency manifesto, Kazuki Irie's architecture program, Lake's cognitive science benchmarks, the causal inference community's insistence on mechanisms over correlations, π0.5's demonstration that robots can generalize across homes, SOAR's proof that models can teach themselves <a href="#ref40">[40]</a>, the Kepler-to-Newton demonstration that inductive biases distinguish curve-fitting from understanding <a href="#ref42">[42]</a>, the rigorous theories of attention generalization <a href="#ref37">[37]</a> and compositional depth separation <a href="#ref38">[38]</a>, to the trust-function discovery that weak supervision can produce stronger models than ground truth <a href="#ref43">[43]</a>, all point to the same thing: current AI needs too much data, and models don't generalize like humans do. But we now have architectural recipes, theoretical frameworks, diagnostic tools, causal methods, embodied testbeds, alignment connections, curriculum primitives, and practical optimizers that didn't exist even two years ago. The pieces are on the table. The synthesis is the work ahead.

---

## References

<a id="ref1"></a>[1] Lake, B. M., Ullman, T. D., Tenenbaum, J. B., & Gershman, S. J. (2017). Building Machines That Learn and Think Like People. *Behavioral and Brain Sciences*, 40, e253. arXiv:1604.00289.

<a id="ref2"></a>[2] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding Deep Learning Requires Rethinking Generalization. *ICLR 2017*. arXiv:1611.03530.

<a id="ref3"></a>[3] Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016). Matching Networks for One Shot Learning. *NeurIPS 2016*. arXiv:1606.04080.

<a id="ref4"></a>[4] Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016). Meta-Learning with Memory-Augmented Neural Networks. *ICML 2016*. arXiv:1605.06065.

<a id="ref5"></a>[5] Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML 2017*. arXiv:1703.03400.

<a id="ref6"></a>[6] Ha, D. & Schmidhuber, J. (2018). World Models. *NeurIPS 2018*. arXiv:1803.10122.

<a id="ref7"></a>[7] Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. arXiv:1907.02893.

<a id="ref8"></a>[8] Bhattamishra, S., Ahuja, K., & Goyal, N. (2020). On the Ability and Limitations of Transformers to Recognize Formal Languages. *EMNLP 2020*. arXiv:2009.11264.

<a id="ref9"></a>[9] Irie, K. (2020). Language Models as Representations with Reduced Self-Attention. *ICASSP 2020*.

<a id="ref10"></a>[10] Csordás, R., Irie, K., & Schmidhuber, J. (2021). The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers. *EMNLP 2021*. arXiv:2108.12284.

<a id="ref11"></a>[11] Csordás, R., Irie, K., & Schmidhuber, J. (2022). The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization. *ICLR 2022*. arXiv:2110.07732.

<a id="ref12"></a>[12] Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

<a id="ref13"></a>[13] Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., & Ortega, P. A. (2023). Neural Networks and the Chomsky Hierarchy. *ICLR 2023*. arXiv:2207.02098.

<a id="ref14"></a>[14] Lake, B. M. & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. *Nature*, 623, 115–121. arXiv:2310.01673.

<a id="ref15"></a>[15] Kawata, R., Song, Y., Bietti, A., Nishikawa, N., Suzuki, T., Vaiter, S., & Wu, D. (2025). From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers. *NeurIPS 2025*. arXiv:2512.18634.

<a id="ref16"></a>[16] Altabaa, A., Chen, S., Lafferty, J., & Yang, Z. (2025). Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning. arXiv:2510.14095.

<a id="ref17"></a>[17] Chen, H.-H. (2026). Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization. arXiv:2603.21676.

<a id="ref18"></a>[18] Schwethelm, K., Rueckert, D., & Kaissis, G. (2026). How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models. arXiv:2604.21106.

<a id="ref19"></a>[19] Sapunov, G. (2026). Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning. arXiv:2604.21999.

<a id="ref20"></a>[20] Truong, X. K. et al. (2026). Spectral Entropy Collapse as a Phase Transition in Delayed Generalisation: An Interventional and Predictive Framework for Grokking. arXiv:2604.13123.

<a id="ref21"></a>[21] Hidajat, K., Stoll, S., & An, J. (2026). Grokking as Structural Inference: Transformers Need Bayesian Lottery Tickets. arXiv:2605.15787.

<a id="ref22"></a>[22] Yan, H., Yang, H., & Zhong, Y. (2026). Task Vector Geometry Underlies Dual Modes of Task Inference in Transformers. arXiv:2605.03780.

<a id="ref23"></a>[23] Zhang, Y., Zhang, Y., Zhou, X., & Chen, X. (2026). A Measure-Theoretic Analysis of Reasoning: Structural Generalization and Approximation Limits. arXiv:2605.19944.

<a id="ref24"></a>[24] Suresh, P., Stanley, J., Joseph, S., Scimeca, L., & Bzdok, D. (2026). At the Edge of Understanding: Sparse Autoencoders Trace The Limits of Transformer Generalization. *ICML 2026*. arXiv:2606.26396.

<a id="ref25"></a>[25] Sutskever, I. (2023). An Observation on Generalization. Talk at Simons Institute, UC Berkeley, August 2023. https://simons.berkeley.edu/news/observation-generalization

<a id="ref26"></a>[26] Gu, A. (2024). Data Efficiency. Essay. Albert Gu is Assistant Professor of Machine Learning at Carnegie Mellon University and Chief Scientist of Cartesia AI. Appeared on *TIME*'s list of the most influential people in AI, 2024.

<a id="ref27"></a>[27] Wen, J., Wu, Z., Song, D., & Chen, L. (2026). Generalization Dynamics of LM Pre-training. Blog post, May 2026. https://jiaxin-wen.github.io/blog/generalization-dynamics

<a id="ref28"></a>[28] Litman, E. & Guo, G. (2026). A Theory of Generalization in Deep Learning. arXiv:2605.01172.

<a id="ref29"></a>[29] Sutskever, I. (2025). Interview with Dwarkesh Patel, November 2025. https://www.dwarkesh.com/p/ilya-sutskever-2

<a id="ref30"></a>[30] Pham, B., Raya, G., Negri, M., Zaki, M. J., Ambrogioni, L., & Krotov, D. (2025). Memorization to Generalization: Emergence of Diffusion Models from Associative Memory. arXiv:2505.21777.

<a id="ref31"></a>[31] Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*, 109(5), 612–634.

<a id="ref32"></a>[32] 郭若城, 程璐, 刘昊, 刘欢. (2023). 《因果推断与机器学习》. 电子工业出版社. (Chapter 3: 因果表征学习与泛化能力)

<a id="ref33"></a>[33] 因果强化学习统一框架综述 (2025). A Unified Framework for Causal Reinforcement Learning: Survey, Taxonomy, Algorithms, and Applications.

<a id="ref34"></a>[34] Physical Intelligence (π). (2025). π0.5: A Vision-Language-Action Model with Open-World Generalization. Blog post, April 2025. https://www.pi.website/blog/pi05. arXiv:2504.16054.

<a id="ref35"></a>[35] Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J., Sutskever, I., & Wu, J. (2023). Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision. *ICML 2024 (Oral)*. arXiv:2312.09390.

<a id="ref36"></a>[36] Ruiz, R. B. & Gu, A. (2025). Understanding and Improving Length Generalization in Recurrent Models. *ICML 2025 (PMLR Vol. 267)*. arXiv:2507.02782.

<a id="ref37"></a>[37] Boncoraglio, F., Erba, V., Troiani, E., Xu, Y., Krzakala, F., & Zdeborová, L. (2026). Single-Head Attention in High Dimensions: A Theory of Generalization, Weights Spectra, and Scaling Laws. *ICML 2026*. arXiv:2509.24914.

<a id="ref38"></a>[38] Tabanelli, H., Dandi, Y., Pesce, L., & Krzakala, F. (2026). Efficient Learning of Compositional Targets with Hierarchical Spectral Methods. *ICML 2026*. arXiv:2602.10867.

<a id="ref39"></a>[39] Parley, J. T., Cagnetta, F., & Wyart, M. (2026). Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics. *ICML 2026*. arXiv:2602.06065.

<a id="ref40"></a>[40] Sundaram, S., Quan, J., Kwiatkowski, A., Ahuja, K., Ollivier, Y., & Kempe, J. (2026). Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability. *ICML 2026*. arXiv:2601.18778.

<a id="ref41"></a>[41] Rajaraman, N., Huang, A., Dudik, M., Schapire, R., Foster, D., & Krishnamurthy, A. (2026). Learning to Reason with Curriculum II: Compositional Generalization. arXiv:2606.27721.

<a id="ref42"></a>[42] Liu, Z., Sanborn, S., Ganguli, S., & Tolias, A. (2026). From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers. *ICML 2026*. arXiv:2602.06923.

<a id="ref43"></a>[43] Uzunoglu, A., Zhang, A., & Khashabi, D. (2026). Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher. *ICML 2026*. arXiv:2606.01000.

---

*This blog post surveys 43 papers and talks collected from Kazuki Irie's X/Twitter recommendations, recent generalization research, vision-setting pieces by Ilya Sutskever and Albert Gu, the causal ML literature, embodied AI from Physical Intelligence, the OpenAI Superalignment team, and the latest ICML 2025–2026 wave of generalization theory. All referenced papers are available in the companion repository at `generalization/ref/`.*
