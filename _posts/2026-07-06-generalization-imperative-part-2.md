---
layout: post
title: "The Generalization Imperative (2/4): Grokking, Meta-Learning, and OOD Generalization"
date: 2026-07-06 09:00:00
description: Part 2 of 4 — Sections 4–6. Grokking dynamics and mode-hopping in real LLMs, meta-learning foundations, and OOD generalization through invariance and phase transitions.
tags: generalization deep-learning machine-learning survey grokking meta-learning
categories: blog
toc: true
series: generalization-imperative
series_part: 2
---

**📖 Series:** [← Part 1](/blog/2026/generalization-imperative-part-1/)  |  **Part 2**  |  [Part 3 →](/blog/2026/generalization-imperative-part-3/)  |  [Part 4](/blog/2026/generalization-imperative-part-4/)

---

## 4. The Dynamics of Discovery: Grokking and the Emergence of Generalization

> *Sometimes generalization doesn't happen gradually; it erupts.*

In 2022, Alethea Power and colleagues at OpenAI reported a phenomenon that captured the imagination of the field <a href="#ref12">[12]</a>: on small algorithmic datasets, a transformer's validation accuracy would suddenly jump from chance to near-perfect, long after training accuracy had already saturated. They called this grokking: the model first memorizes, then, mysteriously, "gets it."

Grokking is fascinating because it suggests generalization is not a smooth function of training time but a phase transition. And in 2026, the theoretical understanding of grokking has matured dramatically.

**Structural inference.** Kai Hidajat, Solden Stoll, and Joseph An at the University of Washington proposed a decoupled theory <a href="#ref21">[21]</a>: grokking requires two conditions. First, a Bayesian Structural Condition: attention must place sufficient probability mass on every informative token. Second, a Goldilocks Norm Condition: the MLP's capacity must be in a narrow sweet spot. Their key insight is that MLP memorization starves attention of structural gradient, so the model gets stuck in an "explaining-away plateau" where the MLP has memorized well enough that the attention mechanism receives no signal to learn the generalizable structure. Remarkably, a simple KL-based structural intervention can bypass this delay, with grokking time following an inverse-intervention-strength scaling law.

**Spectral entropy collapse.** Truong Xuan Khanh and colleagues independently discovered a complementary signature <a href="#ref20">[20]</a>: the spectral entropy of the representation covariance matrix collapses in a predictable pattern before grokking. This entropy crosses a task-specific threshold, and the remaining time to grokking can be predicted from the entropy gap, with out-of-sample accuracy. The entropy collapse couples with Fourier-aligned representations in cyclic-group tasks, suggesting a deep connection between representation geometry and generalization.

The practical implication is profound: if grokking is a phase transition with reliable early-warning signals, we might eventually detect impending generalization before it happens and accelerate it.

### 4.3 Mode-Hopping: Grokking at the Scale of Real LLMs

The grokking literature has a glaring limitation: it is almost entirely conducted on small algorithmic tasks such as modular arithmetic, group operations, and synthetic grammars. Do these phenomena survive at the scale of real language model pretraining?

Jiaxin Wen, Zhengxuan Wu, Dawn Song, and Lijie Chen at UC Berkeley and Stanford tackled this question head-on in May 2026 <a href="#ref27">[27]</a>. They studied intermediate checkpoints of OLMo3 (7B, 32B) and Apertus (8B, 70B), fully open models trained well beyond Chinchilla-optimal budgets (9× to 90×). They designed six behavioral probes that test not *what* the model knows, but *how* it knows it:

| Probe                | What it tests                                                |
| -------------------- | ------------------------------------------------------------ |
| Flipped Answer       | Memorized fact patterns vs. genuine in-context learning      |
| Repetitive Answer    | Copying repetitive surface patterns vs. in-context reasoning |
| Successive Answer    | Pattern-matching successive sequences vs. true arithmetic    |
| Truthy Answer        | What*sounds* true vs. what *is* true                     |
| Intuitive Answers    | System 1 heuristics vs. System 2 reasoning                   |
| Multi-hop Persona QA | Disconnected facts vs. coherent personas                     |

The result was startling. Throughout pretraining, models do not smoothly mature from parrots to reasoners. Instead, they undergo mode-hopping: frequent and reversible flips between two distinct computational modes:

- **Parrot mode**: Latching onto shallow patterns, memorized associations, in-context repetition, what "sounds true," System 1 intuition.
- **Intelligence mode**: Genuine in-context learning, System 2 reasoning, distinguishing truth from plausibility, coherent persona construction.

The most dramatic example: on the Successive Answer eval, OLMo3 32B hit 81% accuracy at 2.17T tokens, collapsed to 0% at 2.19T tokens, then rebounded to 81.7% at 2.21T tokens. The model learned, forgot, and relearned the capability in the span of 40B tokens, a rounding error in pretraining terms.

Crucially, standard evals (sentiment, topic, math QA, knowledge benchmarks) showed smooth, monotonic improvement across the same checkpoints. The mode-hopping was invisible to conventional evaluation.

Wen et al. frame this as a capacity allocation problem: in capacity-bounded models, generalizable circuits compete with shallow circuits learned early in training. The data distribution in each pretraining window determines which wins. Scaling helps, and larger models hop less dramatically, but doesn't eliminate the phenomenon. Even 70B models mode-hop on sufficiently hard tasks.

Three practical implications emerged:

1. **Checkpoint selection matters.** Intermediate checkpoints that exhibit generalization behavior often outperform the final pretraining checkpoint on downstream tasks like GPQA and alignment robustness. The "best" model isn't always the last one.
2. **Data selection can stabilize generalization.** By monitoring generalization dynamics, one can identify which pretraining data windows help versus hurt, enabling controlled, stable generalization.
3. **"Simpler solutions generalize better" is false.** The generalizable solutions found by models during intelligence-mode phases can be either simpler or more complex than the parrot-mode solutions, contradicting a widely held intuition.

Mode-hopping is grokking, but not as we knew it. It's not a one-time phase transition from memorization to generalization, but a recurrent struggle between two computational modes that persists across the entire pretraining trajectory. The model doesn't "arrive" at generalization; it fights for it, over and over again.

### 4.4 A Unified Theory: Signal, Reservoir, and the SNR Gate

If grokking is a phase transition and mode-hopping is a recurrent struggle, what is the underlying mechanism? Elon Litman and Gabe Guo at Stanford proposed a unifying theory in May 2026 <a href="#ref28">[28]</a> that connects these dynamics to a fundamental geometric structure in neural network training.

Their theory decomposes the output space of a neural network, via the empirical Neural Tangent Kernel (eNTK), into two orthogonal subspaces:

- **Signal channel**: The subspace where coherent population signal accumulates through fast linear *drift* across minibatch SGD steps. Training error dissipates rapidly here, and what's learned in this channel is visible at test time.
- **Reservoir**: A vast, high-dimensional subspace orthogonal to the signal channel, where idiosyncratic memorization of noise is trapped in a slow, diffusive *random walk*. Patterns stored here are invisible at test time.

This drift-diffusion decomposition operates in the feature-learning (non-lazy) regime, where the kernel evolves by $\mathcal{O}(1)$ in operator norm, far beyond the stationary-kernel assumptions of classical NTK theory.

From this framework, they derive a population-risk objective from a single training run, with no validation data required. This objective measures the noise leaking into the signal channel and reduces, practically, to an SNR (Signal-to-Noise Ratio) preconditioner that sits on top of Adam. It adds exactly one state vector, with no additional computational cost.

The empirical results are striking:

- **Accelerates grokking by 5×** on modular arithmetic.
- **Suppresses memorization** in Physics-Informed Neural Networks (PINNs) and implicit neural representations.
- **Improves DPO fine-tuning** under noisy preferences while staying 3× closer to the reference policy.

The signal/reservoir framework provides a unified explanation for benign overfitting, double descent, implicit bias, *and* grokking, phenomena that had previously been studied in isolation. Grokking, in this view, is the moment when the signal channel finally wins the drift-diffusion race against the reservoir's random walk.

When read alongside Wen et al.'s mode-hopping <a href="#ref27">[27]</a>, a richer picture emerges: the recurrent struggle between parrot mode and intelligence mode may reflect the ongoing competition between signal accumulation in the channel and noise diffusion in the reservoir. Every pretraining data window reshuffles which circuits get reinforced. The SNR gate offers a practical tool to tilt the balance.

### 4.5 The Generative Frontier: Grokking in Diffusion Models

The memorization-to-generalization transition is not confined to language. Bao Pham, Gabriel Raya, and colleagues at RPI, Tilburg University, and IBM Research showed in May 2025 that the same phase transition governs diffusion models <a href="#ref30">[30]</a>. They cast diffusion models through the lens of Dense Associative Memories (DenseAMs), generalizations of Hopfield networks with superior storage capacity, viewing the diffusion generative process as an attempt at memory retrieval.

Their framework reveals three regimes:

- **Memorization (small data)**: The model creates distinct attractors for each training sample, faithfully reproducing training data point by point.
- **Spurious states (critical intermediate phase)** : As training data exceeds the model's effective memory capacity, new local minima emerge in the energy landscape that are different from any training sample. In classical associative memory theory, these spurious states were considered negative artifacts that hindered accurate retrieval. But in generative modeling, they are the first signatures of creativity: the model begins producing novel samples that blend features from multiple training examples without directly copying any of them.
- **Generalization (large data)** : A continuous manifold of low-energy states forms, enabling smooth interpolation and high-quality novel generation.

This work bridges two previously disconnected literatures and provides an energy-landscape theory of the memorization-to-generalization transition. It suggests that grokking <a href="#ref12">[12]</a>, mode-hopping <a href="#ref27">[27]</a>, and the diffusion generalization transition are all manifestations of the same underlying phenomenon: a phase transition in the model's energy landscape as a function of data volume, governed by the ratio of model capacity to dataset size. The "spurious states" that classical theory dismissed as failures turn out to be the birthplace of generalization itself.

---

## 5. Learning How to Learn: The Meta-Learning Perspective

> *If you want a model that generalizes from few examples, train it to generalize from few examples.*

The meta-learning approach reframes the problem entirely: instead of designing a model that generalizes, design a learning algorithm that generalizes, and let the model learn it.

### 5.1 The Classical Trinity

Three papers from 2016–2017 established the foundations.

Oriol Vinyals and colleagues at DeepMind introduced Matching Networks at NeurIPS 2016 <a href="#ref3">[3]</a>, a framework for one-shot learning that uses an attention mechanism over a small support set of labeled examples. The key innovation: rather than learning a fixed mapping from inputs to outputs, the network learns to *compare* new examples against a handful of labeled ones. This is remarkably close to how humans perform few-shot classification: we don't learn a new category from one example by gradient descent; we compare.

Adam Santoro and colleagues, also at DeepMind, introduced Memory-Augmented Neural Networks (MANNs) at ICML 2016 <a href="#ref4">[4]</a>. Using an external memory module inspired by Neural Turing Machines, MANNs learn to write relevant information to memory and read from it selectively, enabling rapid adaptation across episodes without weight updates.

Chelsea Finn, Pieter Abbeel, and Sergey Levine at Berkeley proposed MAML (Model-Agnostic Meta-Learning) at ICML 2017 <a href="#ref5">[5]</a>. MAML's idea is elegant: optimize for a model initialization that can be fine-tuned to any task in the distribution with just a few gradient steps. The inner loop adapts; the outer loop learns to be adaptable. Unlike Matching Networks or MANNs, MAML doesn't require a special architecture; it works with any model trained by gradient descent.

### 5.2 Meta-Learning Meets Cognitive Science

The meta-learning program achieved its most dramatic validation in a 2023 *Nature* paper by Brenden Lake and Marco Baroni <a href="#ref14">[14]</a>. They trained a standard sequence-to-sequence transformer via meta-learning on synthetic tasks requiring compositional generalization, then tested it against human participants. The result: the model matched or exceeded human performance on held-out compositional tasks. This was a landmark: the first demonstration that a neural network, trained with the right learning objective, could achieve human-level systematic generalization.

Lake's earlier BBS manifesto <a href="#ref1">[1]</a> had laid out exactly this bet: that the path to human-like learning runs through compositionality, causality, and learning-to-learn. The Nature result was proof that the bet was paying off.

### 5.3 Beyond Hand-Designed Meta-Learning: SOAR and Curriculum-Driven Compositional Generalization

Meta-learning's original promise was that models could learn their own learning algorithms. But the meta-training tasks in MAML, Matching Networks, and even Lake and Baroni's Nature experiment were all *hand-designed*. A deeper question emerged in 2026: can a model learn to *generate its own curriculum*, discovering the right sequence of training tasks that leads to generalization, without human engineering?

**SOAR: Teaching models to teach themselves.** Shobhita Sundaram and colleagues at MIT and Meta FAIR introduced SOAR (Self-Optimization via Asymmetric RL) at ICML 2026 <a href="#ref40">[40]</a>, a teacher-student meta-RL framework where a teacher model generates synthetic math problems, a student model trains on them with RL, and the teacher is rewarded solely based on the student's real improvement on a held-out hard dataset, a dataset the teacher never sees. The framework operates at the edge of learnability: tested on the hardest subsets of MATH and HARP where the pretrained Llama-3.2-3B-Instruct achieved 0/128 initial success, SOAR delivered 4× pass@1 and 2× pass@32 on MATH, and 2× pass@1 and 1.5× pass@32 on HARP.

Three findings stand out. First, bi-level meta-RL unlocks learning under sparse binary rewards: the teacher discovers useful stepping-stone problems even for tasks it cannot solve itself. Second, grounded rewards outperform intrinsic rewards: prior LLM self-play methods that used intrinsic curiosity or diversity bonuses suffered from instability and diversity collapse, whereas SOAR's simple signal, whether the student improved on real held-out problems, proved far more robust. Third, question structure matters more than answer correctness: the teacher's value lies in generating well-posed, appropriately calibrated problems, not necessarily in providing correct answers. A hard question with a wrong answer can still be an excellent stepping stone.

**Curriculum-driven compositional generalization.** Where SOAR addresses *what* to learn, Nived Rajaraman and colleagues at Microsoft Research addressed *how sequentially* to learn it <a href="#ref41">[41]</a>. Their paper, *Learning to Reason with Curriculum II: Compositional Generalization*, studies the canonical problem of learning to simulate semiautomata, capturing state tracking, regular language recognition, and modular arithmetic in a unified framework. The key result: an autocurriculum that recursively decomposes long sequences into shorter sub-problems reduces the required supervision from $\Omega(T)$ tokens (direct simulation) to just $2^{O(\sqrt{\log T})}$ tokens, a subpolynomial dependence on sequence length. In the RL with verifiable rewards (RLVR) setting, curriculum reduces the requirement on a pretrained reference model from needing coverage at full sequence length $T$ to coverage at a much shorter block length $B \ll T$, an exponentially weaker condition.

These two papers, published within weeks of each other in mid-2026, converge on the same insight: meta-learning is not just about learning to adapt; it's about learning to teach. The right curriculum can make the difference between a task that requires exponential data and one that requires merely subpolynomial data. The bottleneck, they suggest, is not in the learning algorithm's capacity to generalize, but in our ability to sequence the learning experience.

---

## 6. OOD Generalization: Invariance, Diversity, and Phase Transitions

> *To generalize out-of-distribution, a model must learn what doesn't change.*

**Invariant Risk Minimization.** Martin Arjovsky and colleagues proposed IRM in 2019 <a href="#ref7">[7]</a>, formalizing an idea from causal inference: a predictor that is simultaneously optimal across multiple training environments must depend only on the *invariant* features, the ones whose relationship to the label is stable. IRM spawned an entire subfield of OOD generalization methods, though its practical gains have been more modest than its conceptual clarity would suggest.

**From shortcut to induction head.** The most important recent advance in understanding OOD generalization comes from Ryotaro Kawata and colleagues at NeurIPS 2025 <a href="#ref15">[15]</a>. They proved that a single-layer transformer undergoes a phase transition in what it learns, governed entirely by the diversity of the pretraining data:

- **Low diversity** → the model learns a *positional shortcut*: it attends to tokens based on their absolute position rather than their semantic relationships. OOD generalization fails catastrophically.
- **High diversity** → the model forms an *induction head*: it learns to attend based on content, not position. OOD generalization emerges.

They further derived the optimal pretraining distribution to minimize computational cost while enabling OOD generalization. This paper explains *why* data diversity matters, not as a vague intuition about "more data is better," but as a precise condition for a phase transition in learned mechanisms.

**Task vector geometry.** Hao Yan, Haolin Yang, and Yiqiao Zhong deepened this picture in May 2026 <a href="#ref22">[22]</a> by showing that transformers use two geometrically distinct modes for task inference:

- **In-distribution**: Bayesian retrieval via convex combinations of learned task vectors.
- **OOD**: Extrapolative learning in a subspace nearly orthogonal to the task-vector subspace.

Both modes coexist in a single model. The training distribution shapes the geometry of the task-vector space, and this geometry, in turn, determines whether OOD generalization is possible. It is a mathematical characterization of something practitioners have long suspected: OOD generalization requires the model to learn representations that are *qualitatively different* from those needed for in-distribution performance.

---


---

**📖 Series:** [← Part 1](/blog/2026/generalization-imperative-part-1/)  |  **Part 2**  |  [Part 3 →](/blog/2026/generalization-imperative-part-3/)  |  [Part 4](/blog/2026/generalization-imperative-part-4/)
---

## References

<a id="ref1"></a>[1] Lake, B. M., Ullman, T. D., Tenenbaum, J. B., & Gershman, S. J. (2017). Building Machines That Learn and Think Like People. *Behavioral and Brain Sciences*, 40, e253. arXiv:1604.00289.

<a id="ref3"></a>[3] Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016). Matching Networks for One Shot Learning. *NeurIPS 2016*. arXiv:1606.04080.

<a id="ref4"></a>[4] Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016). Meta-Learning with Memory-Augmented Neural Networks. *ICML 2016*. arXiv:1605.06065.

<a id="ref5"></a>[5] Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML 2017*. arXiv:1703.03400.

<a id="ref7"></a>[7] Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. arXiv:1907.02893.

<a id="ref12"></a>[12] Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

<a id="ref14"></a>[14] Lake, B. M. & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. *Nature*, 623, 115–121. arXiv:2310.01673.

<a id="ref15"></a>[15] Kawata, R., Song, Y., Bietti, A., Nishikawa, N., Suzuki, T., Vaiter, S., & Wu, D. (2025). From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers. *NeurIPS 2025*. arXiv:2512.18634.

<a id="ref20"></a>[20] Truong, X. K. et al. (2026). Spectral Entropy Collapse as a Phase Transition in Delayed Generalisation: An Interventional and Predictive Framework for Grokking. arXiv:2604.13123.

<a id="ref21"></a>[21] Hidajat, K., Stoll, S., & An, J. (2026). Grokking as Structural Inference: Transformers Need Bayesian Lottery Tickets. arXiv:2605.15787.

<a id="ref22"></a>[22] Yan, H., Yang, H., & Zhong, Y. (2026). Task Vector Geometry Underlies Dual Modes of Task Inference in Transformers. arXiv:2605.03780.

<a id="ref27"></a>[27] Wen, J., Wu, Z., Song, D., & Chen, L. (2026). Generalization Dynamics of LM Pre-training. Blog post, May 2026. https://jiaxin-wen.github.io/blog/generalization-dynamics

<a id="ref28"></a>[28] Litman, E. & Guo, G. (2026). A Theory of Generalization in Deep Learning. arXiv:2605.01172.

<a id="ref30"></a>[30] Pham, B., Raya, G., Negri, M., Zaki, M. J., Ambrogioni, L., & Krotov, D. (2025). Memorization to Generalization: Emergence of Diffusion Models from Associative Memory. arXiv:2505.21777.

<a id="ref40"></a>[40] Sundaram, S., Quan, J., Kwiatkowski, A., Ahuja, K., Ollivier, Y., & Kempe, J. (2026). Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability. *ICML 2026*. arXiv:2601.18778.

<a id="ref41"></a>[41] Rajaraman, N., Huang, A., Dudik, M., Schapire, R., Foster, D., & Krishnamurthy, A. (2026). Learning to Reason with Curriculum II: Compositional Generalization. arXiv:2606.27721.


*This is Part 2 of a 4-part series surveying 43 papers on generalization in deep learning. Full references across all parts are available in [Part 4](/blog/2026/generalization-imperative-part-4/).*