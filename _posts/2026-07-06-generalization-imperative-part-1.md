---
layout: post
title: "The Generalization Imperative (1/4): From the Puzzle to Computation Theory"
date: 2026-07-06 09:00:00
description: Part 1 of 4 — Sections 1–3. The puzzle of generalization, the architecture hypothesis, and computation-theoretic foundations.
tags: generalization deep-learning machine-learning survey alignment causality
categories: blog
toc: true
series: generalization-imperative
series_part: 1
---

**July 2026**

---

> *In 2025, I worked on estimating human brain microstructure from diffusion MRI. At MICCAI, we presented [ufo-3](https://github.com/tensor2023/ufo-3), a deep learning model that achieved state-of-the-art accuracy, when trained and tested on the same subject. But the moment I trained it across multiple subjects, it struggled. The patterns that worked perfectly for one brain failed for another. I had built a model that could **memorize a brain, not one that understood what a brain is**. That experience sent me down a rabbit hole. I wanted to know: what would it take to build machines that truly generalize? This post is the answer I found: a close reading of 43 papers, from cognitive science manifestos to the bleeding edge of ICML 2026, from robotic kitchens to causal diagrams, from grokking dynamics to the alignment frontier. The pieces, I believe, are finally falling into place.*

---

1. The Puzzle

In 2017, Chiyuan Zhang and colleagues at Cornell published a paper that should have been deeply unsettling <a href="#ref2">[2]</a>. They showed that deep neural networks, the same architectures powering the image recognition revolution, could perfectly fit a training set of randomly labeled images. The network memorized noise, yet still generalized on real data. Standard learning theory (VC dimension, Rademacher complexity) couldn't explain this. The paper's title said it all: *Understanding Deep Learning Requires Rethinking Generalization*.

That same year, Brenden Lake, Tomer Ullman, Josh Tenenbaum, and Samuel Gershwin published a manifesto in *Behavioral and Brain Sciences* titled *Building Machines That Learn and Think Like People* <a href="#ref1">[1]</a>. Their argument was sweeping: despite impressive pattern recognition, deep learning was missing the core ingredients of human intelligence: compositional generalization, causal reasoning, intuitive physics, and learning-to-learn. A child who learns the word "dax" for one strange object immediately knows what "two daxes" means. A neural network that learns to stack red blocks doesn't automatically know how to stack blue ones.

These two papers, published in the same year, defined the twin poles of the generalization problem: (a) we don't understand why deep learning generalizes at all, and (b) when it does generalize, it doesn't do so in the flexible, compositional way that humans do.

### 1.1 Generalization as Compression: Ilya Sutskever's Observation

In August 2023, at a Simons Institute workshop on Large Language Models and Transformers, Ilya Sutskever delivered a talk titled *An Observation on Generalization* <a href="#ref25">[25]</a> that reframed the entire problem in one elegant stroke.

His thesis: unsupervised learning is compression. There exists a one-to-one correspondence between compressors and predictors: every compression algorithm implicitly defines a probability distribution, and every predictor implements a compression scheme. The theoretical ideal is Kolmogorov complexity: the shortest program that reproduces the data. A perfect Kolmogorov compressor, by definition, generalizes optimally, because it has extracted the minimal sufficient description of the data.

Large neural networks trained with SGD, Sutskever argued, are approximating this Kolmogorov compressor. Bigger models have lower "regret", meaning they come closer to the theoretical compression limit. He pointed to OpenAI's iGPT as validation: a transformer trained simply to predict the next pixel learned representations that captured high-level visual concepts, without any visual-specific inductive biases. Compression alone was sufficient.

Sutskever's framing provides a theoretical north star for the generalization problem. It explains *why* scaling works (larger models are better approximators of the Kolmogorov compressor), and it explains *why* scaling hits diminishing returns (the gap between the Kolmogorov limit and the model's approximation shrinks with scale). But it leaves a crucial question open: the Kolmogorov compressor is uncomputable. So what *tractable* learning algorithms, architectures, and objectives approach it most efficiently?

#### 1.1.2 The 2025 Refinement: Jagged Intelligence and the Age of Research

Two years later, in a November 2025 interview with Dwarkesh Patel <a href="#ref29">[29]</a>, Sutskever returned to the generalization problem with sharper observations and a darker diagnosis. The conversation marked a turning point in the field's self-understanding.

**Jagged generalization.** Sutskever identified a paradox at the heart of modern AI: models ace hard benchmarks but fail on trivially simple real-world tasks. His canonical example: a coding model fixes a bug but introduces a second bug; asked to fix *that* bug, it reverts to the *first* bug, cycling indefinitely. He calls this jagged intelligence: performance that is spiky, brittle, and non-monotonic across tasks that differ only slightly. Two explanations: (1) reinforcement learning makes models narrow and single-minded, overly focused on reward at the expense of common sense; (2) researchers unconsciously design RL training environments inspired by public benchmarks, causing models to overfit to test-like tasks rather than learning true generalization.

**The age of scaling is over.** Sutskever offered a periodization of AI history: 2012–2020 was the Age of Research (tinkering, breakthroughs like AlexNet and Transformers); 2020–2025 was the Age of Scaling (just add data and compute, which worked until it didn't); and 2025 onward is the Age of Research *with bigger computers*. Pretraining data is running out. Marginal returns from scaling are diminishing. The bottleneck is no longer compute; it's ideas.

**Emotions as value functions.** The most provocative idea in the interview connects generalization to neuroscience. Sutskever cited a famous case: a brain-damaged patient who lost emotional processing could still reason normally, but became paralyzed at trivial decisions, spending hours choosing socks. Emotions, Sutskever argued, serve as a robust, evolution-hardcoded value function: they provide intermediate feedback ("losing a queen in chess is bad *now*," not just at checkmate), dramatically improving learning efficiency. Current AI lacks this. Existing "LLM-as-a-Judge" approaches are complex, brittle, and task-specific. Human emotions are simple, universal, and robust across novel situations, which are precisely the properties a generalization-enabling value function needs.

**Humans have a better learning algorithm.** While evolutionary priors explain some human advantages (vision, motor skills), humans also learn abstract modern skills such as math and coding, vastly more efficiently than AI. A teenager learns to drive in roughly 10 hours; an autonomous vehicle needs millions of simulated crashes. Language, mathematics, and programming are recent in evolutionary terms, yet humans still far outperform models in learning efficiency per sample. This proves humans possess a fundamentally better learning algorithm, not just better priors, and discovering it is the central research challenge.

**Superintelligence redefined.** Sutskever's final vision: not a model that *knows* every job, but a model that can learn to do every job through continuous learning, like a hyper-efficient human employee who learns on the job and rapidly surpasses human capability. Unlike humans, who cannot directly copy knowledge to one another, millions of AI instances could fuse their collective learnings into one brain, a process Sutskever calls amalgamation.

If Sutskever's 2023 talk gave the generalization problem its theoretical north star, compression, his 2025 interview gave it renewed urgency. The easy gains from scaling are exhausted. The path forward requires new ideas about learning itself.

### 1.2 The Unifying Problem: Albert Gu on Data Efficiency

A year later, Albert Gu, Assistant Professor at CMU, Chief Scientist at Cartesia AI, and creator of the Mamba state-space model, published a short essay that connected generalization to nearly every open problem in AI <a href="#ref26">[26]</a>.

His argument is disarmingly direct:

> *"Building a foundation model takes tremendous amounts of data. In the coming year, I hope we'll enable models to learn more from less data."*
>
> *"Current models consume much more data than humans require for learning. We've known this for a while, but we've ignored it due to the amazing effectiveness of scaling. It takes trillions of tokens to train a model but orders of magnitude less for a human to become a reasonably intelligent being. Human learning shows that there's a learning algorithm, objective function, architecture, or a combination thereof that can learn more sample-efficiently than current models."*

Gu then traces the implications of data efficiency through six interconnected problems:

- **Data curation**: The fact that most work in foundation model training is about data preparation, not architecture, is a symptom. We're doing the model's abstraction work *for* it, ahead of time.
- **Feature engineering**: We congratulated ourselves on removing handcrafted features (edge detectors, n-grams), but we've simply moved that engineering elsewhere; tokenization is just implicit feature engineering. There's still room to make architectures that handle rawer modalities directly.
- **Multimodality**: The key to learning from multiple data types together is finding the core abstractions common to all modalities. This should enable models to learn from *less* data by leveraging all modalities jointly.
- **Interpretability and robustness**: A model that produces higher-level abstractions is inherently more interpretable, since we can track what concepts it captured and how. And such a model should be more robust to noise and require less data.
- **Reasoning**: Extracting higher-level patterns should enable better reasoning over them. Better reasoning should, reciprocally, mean less training data is needed.
- **Democratization**: Only a handful of players can afford the data and compute for state-of-the-art models. More data-efficient models would make AI accessible to domains that lack wealth or massive datasets.

Gu's key insight is that we don't know which of these is cause and which is effect, but they're all the same problem viewed from different angles. Solve data efficiency, and you may solve interpretability, reasoning, and democratization at once.

### 1.3 Where We Stand

These perspectives, from Lake's cognitive science and Zhang's empirical puzzle to Sutskever's evolving theory of compression and jagged intelligence and Gu's data-efficiency unification, define the generalization problem from every angle. The next decade of research has been a sustained assault on this puzzle. In this post, I trace twelve lines of attack across more than forty papers and several vision-setting talks, from foundational classics to the bleeding edge of 2026, reaching into the physical world of robotics, the causal structure of reality, the alignment challenges of superhuman AI, and the theoretical machinery of generalization itself.

---

## 2. The Architecture Hypothesis: Can Structure Unlock Generalization?

> *Maybe generalization isn't something you train for. Maybe it's something you build in.*

The most sustained research program on generalization through architecture design comes, fittingly, from the lineage of Jürgen Schmidhuber's lab. The central bet is simple: recurrence and parameter sharing are the architectural primitives of generalization.

### 2.1 The Early Case for Recurrence

In 2020, Kazuki Irie published a paper at ICASSP with a disarmingly simple recipe <a href="#ref9">[9]</a>: (1) set K=V in self-attention, eliminating the value cache entirely; (2) make feed-forward blocks deeper while using fewer self-attention layers. The result? A NOPE (no positional encoding) transformer that showed a surprising out-of-distribution generalization advantage. This was, to Irie's knowledge, the first paper reporting this NOPE–generalization connection, a finding that would echo through later theoretical work.

The following year, Róbert Csordás and colleagues at IDSIA published *The Devil is in the Detail* at EMNLP 2021 <a href="#ref10">[10]</a>, arguing that parameter sharing and Universal Transformers (UTs) are directly relevant for improved reasoning. Their key insight: the difference between a transformer that generalizes and one that doesn't often comes down to seemingly minor details: shared weights across layers, careful initialization, training tricks. The "devil" was real, and it was hiding in the hyperparameters.

### 2.2 The Neural Data Router

In 2022, Csordás and Schmidhuber took the next logical step with the *Neural Data Router* (NDR) at ICLR <a href="#ref11">[11]</a>. NDR introduced adaptive control flow into transformers: each token, at each layer, dynamically decides its routing path through the network. Rather than processing every token through every layer uniformly, NDR learns to route information selectively. On algorithmic tasks, this adaptive routing yielded significant improvements in systematic generalization, the kind where test examples are genuinely different from training ones, not just held-out samples from the same distribution.

### 2.3 The 2026 Explosion

The architecture hypothesis is having its moment. The first half of 2026 alone saw an extraordinary density of papers advancing recurrent-depth transformers.

**Thinking deeper, not longer.** Hung-Hsuan Chen from National Central University proposed a depth-recurrent Transformer that decouples computational depth from parameter count by iteratively applying a shared-weight Transformer block in latent space <a href="#ref17">[17]</a>. Three stabilization mechanisms: a *silent thinking objective* (supervise only the final output, not intermediate steps), *LayerScale initialization*, and *identity-biased recurrence* enable 20+ stable recurrence steps. The paper's most striking finding is a computational threshold: performance jumps from chance to near-perfect once sufficient loop steps are provided, like a phase transition in reasoning capacity. Crucially, Chen found that intermediate supervision is actually harmful, because it causes models to learn heuristic shortcuts rather than genuine multi-step reasoning.

**What is one recurrence worth?** Kristian Schwethelm, Daniel Rueckert, and Georgios Kaissis at MCML Munich quantified the value of recurrence with a scaling law <a href="#ref18">[18]</a>. They fit a joint scaling law of the form:

$$
L = E + A(N_{\text{once}} + r^{\varphi} N_{\text{rec}})^{-\alpha} + BD^{-\beta}
$$

where the recurrence-equivalence exponent $\varphi = 0.46$ sits between 0 (no benefit from recurrence) and 1 (recurrence is as good as unique layers). In practical terms: at 4 recurrences, a 410M looped model matches a 580M standard model, but costs as much to train as a 1B model. Hyperconnections raise $\varphi$ to 0.65, making recurrence genuinely more parameter-efficient. Based on 116 pretraining runs, this is the most systematic quantification of the recurrence–capacity trade-off to date.

**Universal Transformers need memory.** Grigory Sapunov studied Adaptive Computation Time (ACT) with memory tokens on a single-block Universal Transformer tackling Sudoku <a href="#ref19">[19]</a>. The findings are remarkably practical: memory tokens are empirically necessary (no configuration without them achieves non-trivial performance); they substitute as a resource with ponder depth; and a default ACT initialization fails >70% of the time due to premature halting. The fix is almost too simple: invert the ACT halting bias to −3. The paper also reveals that attention heads spontaneously specialize into three roles: *memory readers*, *constraint propagators*, and *integrators*, across recursive depth.

**Recursive latent space reasoning.** Awni Altabaa, Siyu Chen, John Lafferty, and Zhuoran Yang at Yale introduced four architectural mechanisms for compositional OOD generalization in a unified framework <a href="#ref16">[16]</a>: (1) input-adaptive recurrence, (2) algorithmic supervision, (3) anchored latent representations via a discrete bottleneck, and (4) an explicit error-correction mechanism. Tested on GSM8K-style modular arithmetic tasks, the combination of these mechanisms enables systematic, compositional OOD generalization.

**The unexplored states hypothesis.** A different architecture family, state-space models (SSMs) like Mamba, faces its own generalization challenge: length generalization. Ricardo Buitrago Ruiz and Albert Gu at CMU and Cartesia AI addressed this at ICML 2025 <a href="#ref36">[36]</a> with a disarmingly simple diagnosis: recurrent models fail to generalize to longer sequences because during training they only visit a limited subset of all *attainable* state distributions. They call this the unexplored states hypothesis. The fix is equally simple: three training-free post-training interventions: random noise state initialization, fitted noise matched to training-state statistics, and state passing (using the final state of one sequence as the initial state for another), require only ~500 steps (~0.1% of the pretraining budget) and enable length generalization from 2K to 128K tokens, with state passing reaching up to 256K. On long-context reasoning tasks like BABILong and passkey retrieval, these interventions produce dramatic improvements. The unexplored states hypothesis reframes length generalization as a coverage problem: the model already knows how to process long sequences; it has simply never been asked to do so during training. This is a remarkably hopeful finding. It suggests that at least some forms of OOD generalization can be unlocked with minimal intervention, because the capability is latent in the model's existing parameters, waiting only to be exercised.

### 2.4 What the Architecture Hypothesis Gets Right

Reading across these papers, a coherent thesis emerges: generalization is not a training phenomenon; it is an architectural property. Recurrence provides a computational budget that scales with problem complexity independently of parameter count. Parameter sharing enforces the inductive bias that the same operation applies regardless of position or depth. And adaptive computation allows the model to allocate more "thinking" to harder problems.

The limitation, of course, is that architecture alone cannot teach a model *what* to generalize, only *how* to generalize.

---

## 3. The Computation-Theoretic Lens: What Can Neural Networks Even Learn?

> *Before you ask whether a model generalizes, ask whether it can represent the target function at all.*

A parallel line of work approaches generalization from the opposite direction: not "how do we make models generalize?" but "what are the fundamental limits of what they can represent?"

**Transformers and formal languages.** Satwik Bhattamishra and colleagues at EMNLP 2020 were among the first to systematically study transformers on formal languages <a href="#ref8">[8]</a>. Their central finding was counterintuitive: transformers being bad at *parity* doesn't imply they're bad at "harder" languages. The Chomsky hierarchy, a strict containment of language classes by generative power, doesn't cleanly predict transformer capability. A language that is formally "simpler" might be harder for a transformer to learn than one that is formally "more complex." This paper is, deservedly, one of Kazuki Irie's absolute favorites.

**Neural networks and the Chomsky hierarchy.** Grégoire Delétang and colleagues at ICLR 2023 extended this program <a href="#ref13">[13]</a>, systematically testing neural network architectures across all levels of the Chomsky hierarchy. The emerging picture: architecture matters enormously for *which* formal languages a network can learn, and this doesn't always align with classical complexity classes.

**A measure-theoretic framework.** The most ambitious theoretical treatment to date comes from Yuyang Zhang, Yifu Zhang, Xuehai Zhou, and Xiaoyin Chen in May 2026 <a href="#ref23">[23]</a>. They formalize reasoning via optimal transport theory, projecting discrete reasoning trajectories into a continuous metric space and using the Wasserstein-1 distance to quantify domain shifts. Their findings are both rigorous and practical:

- **Position-dependent attention** (absolute positional encoding) fails to preserve shift invariance, yielding an $\Omega(1)$ Lipschitz constant, meaning it is provably brittle under distribution shift.
- **Shift-invariant mechanisms** (Rotary Position Embeddings, or RoPE) preserve equivariance and bound the error, exactly as Irie's NOPE intuition suggested six years earlier.
- There exists a strict circuit depth lower bound for constant-depth (TC⁰) transformers: scaling physical layer depth is *necessary* to prevent representation collapse, and scaling width alone cannot compensate, due to irreducible approximation bounds in Barron spaces.

Validated across 54 Transformer configurations on combinatorial search, this paper bridges the gap between Irie's empirical NOPE observations and a rigorous mathematical theory of why shift-invariance matters for OOD generalization.

**A complete theory of single-head attention generalization.** Fabrizio Boncoraglio, Vittorio Erba, and colleagues at EPFL delivered perhaps the most complete theoretical treatment of attention generalization to date at ICML 2026 <a href="#ref37">[37]</a>. They studied empirical risk minimization in a single-head tied-attention layer trained on synthetic high-dimensional sequence tasks, using tools from random matrix theory, spin-glass theory, and approximate message passing. The result is an exact high-dimensional characterization of training and test error, interpolation and recovery thresholds, and, most strikingly, the full singular-value distribution of the learned query–key weight matrix, including low-rank structure and isolated spectral outliers that qualitatively match observations in real transformers. For targets with power-law spectra, learning proceeds through sequential spectral recovery: the model learns the strongest spectral modes first, then progressively weaker ones, producing sharp emergence phenomena and power-law scaling laws. The theory also reveals that weight decay on query/key matrices induces an implicit nuclear-norm regularization that favors low-rank solutions, which explains why factorized QK training outperforms direct training of their product. This is the kind of theory the field has been waiting for: not just a bound, but a complete, quantitative, experimentally validated description of how and why attention generalizes.

**Why depth helps generalization.** Hugo Tabanelli, Yatin Dandi, and colleagues at ICML 2026 <a href="#ref38">[38]</a> provided one of the first rigorous proofs that depth yields a genuine computational advantage for generalization to compositional targets. In a controlled high-dimensional Gaussian setting with explicit three-layer networks trained via layer-wise spectral estimators, they showed that the compositional structure of the target allows learning to proceed in stages: an intermediate representation reveals structure inaccessible at the input level, whereas any shallow (two-layer) estimator must resolve all components simultaneously. The result is a sharp separation in sample complexity between two- and three-layer learning strategies. This provides theoretical grounding for the intuition behind looped/recurrent transformers <a href="#ref17">[17]</a>, <a href="#ref18">[18]</a>: depth, when matched to the compositional structure of the target, reduces the sample complexity of generalization.

**From local statistics to grammar.** Jack Parley, Francesco Cagnetta, and Matthieu Wyart at ICML 2026 <a href="#ref39">[39]</a> addressed a question that sits at the intersection of computation theory (Section 3) and representation learning: how do deep networks learn hierarchical language structure from raw sentences? Using probabilistic context-free grammars (PCFGs) as a tractable testbed, they introduced a tunable class of PCFGs where both ambiguity and cross-scale correlation structure can be controlled, then proposed a learning mechanism inspired by deep convolutional networks that links learnability and sample complexity to specific language statistics. The core insight: correlations at different scales lift local ambiguities, enabling the emergence of hierarchical representations from data. Their predictions were validated across both deep convolutional and transformer-based architectures. This work bridges the Chomsky-hierarchy program <a href="#ref8">[8]</a>, <a href="#ref13">[13]</a> with modern representation learning: it shows not just *that* networks can learn formal languages, but *how* they do it, through multi-scale statistical structure that is present in natural language itself.

---


---

**📖 Series:** [Part 2 →](/blog/2026/generalization-imperative-part-2/)  |  [Part 3](/blog/2026/generalization-imperative-part-3/)  |  [Part 4](/blog/2026/generalization-imperative-part-4/)
---

## References

<a id="ref1"></a>[1] Lake, B. M., Ullman, T. D., Tenenbaum, J. B., & Gershman, S. J. (2017). Building Machines That Learn and Think Like People. *Behavioral and Brain Sciences*, 40, e253. arXiv:1604.00289.

<a id="ref2"></a>[2] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding Deep Learning Requires Rethinking Generalization. *ICLR 2017*. arXiv:1611.03530.

<a id="ref8"></a>[8] Bhattamishra, S., Ahuja, K., & Goyal, N. (2020). On the Ability and Limitations of Transformers to Recognize Formal Languages. *EMNLP 2020*. arXiv:2009.11264.

<a id="ref9"></a>[9] Irie, K. (2020). Language Models as Representations with Reduced Self-Attention. *ICASSP 2020*.

<a id="ref10"></a>[10] Csordás, R., Irie, K., & Schmidhuber, J. (2021). The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers. *EMNLP 2021*. arXiv:2108.12284.

<a id="ref11"></a>[11] Csordás, R., Irie, K., & Schmidhuber, J. (2022). The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization. *ICLR 2022*. arXiv:2110.07732.

<a id="ref13"></a>[13] Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., & Ortega, P. A. (2023). Neural Networks and the Chomsky Hierarchy. *ICLR 2023*. arXiv:2207.02098.

<a id="ref16"></a>[16] Altabaa, A., Chen, S., Lafferty, J., & Yang, Z. (2025). Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning. arXiv:2510.14095.

<a id="ref17"></a>[17] Chen, H.-H. (2026). Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization. arXiv:2603.21676.

<a id="ref18"></a>[18] Schwethelm, K., Rueckert, D., & Kaissis, G. (2026). How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models. arXiv:2604.21106.

<a id="ref19"></a>[19] Sapunov, G. (2026). Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning. arXiv:2604.21999.

<a id="ref23"></a>[23] Zhang, Y., Zhang, Y., Zhou, X., & Chen, X. (2026). A Measure-Theoretic Analysis of Reasoning: Structural Generalization and Approximation Limits. arXiv:2605.19944.

<a id="ref25"></a>[25] Sutskever, I. (2023). An Observation on Generalization. Talk at Simons Institute, UC Berkeley, August 2023. https://simons.berkeley.edu/news/observation-generalization

<a id="ref26"></a>[26] Gu, A. (2024). Data Efficiency. Essay. Albert Gu is Assistant Professor of Machine Learning at Carnegie Mellon University and Chief Scientist of Cartesia AI. Appeared on *TIME*'s list of the most influential people in AI, 2024.

<a id="ref29"></a>[29] Sutskever, I. (2025). Interview with Dwarkesh Patel, November 2025. https://www.dwarkesh.com/p/ilya-sutskever-2

<a id="ref36"></a>[36] Ruiz, R. B. & Gu, A. (2025). Understanding and Improving Length Generalization in Recurrent Models. *ICML 2025 (PMLR Vol. 267)*. arXiv:2507.02782.

<a id="ref37"></a>[37] Boncoraglio, F., Erba, V., Troiani, E., Xu, Y., Krzakala, F., & Zdeborová, L. (2026). Single-Head Attention in High Dimensions: A Theory of Generalization, Weights Spectra, and Scaling Laws. *ICML 2026*. arXiv:2509.24914.

<a id="ref38"></a>[38] Tabanelli, H., Dandi, Y., Pesce, L., & Krzakala, F. (2026). Efficient Learning of Compositional Targets with Hierarchical Spectral Methods. *ICML 2026*. arXiv:2602.10867.

<a id="ref39"></a>[39] Parley, J. T., Cagnetta, F., & Wyart, M. (2026). Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics. *ICML 2026*. arXiv:2602.06065.


*This is Part 1 of a 4-part series surveying 43 papers on generalization in deep learning. Full references across all parts are available in [Part 4](/blog/2026/generalization-imperative-part-4/).*