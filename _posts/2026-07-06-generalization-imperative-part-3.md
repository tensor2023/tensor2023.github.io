---
layout: post
title: "The Generalization Imperative (3/4): Causality, World Models, and Embodied AI"
date: 2026-07-06 09:00:00
description: Part 3 of 4 — Sections 7–9. Causal representation learning for invariance, world models and physical law discovery, and embodied generalization tested in the real world.
tags: generalization deep-learning machine-learning causality world-models embodied-ai
categories: blog
toc: true
series: generalization-imperative
series_part: 3
---

**📖 Series:** [Part 1](/blog/2026/generalization-imperative-part-1/)  |  [← Part 2](/blog/2026/generalization-imperative-part-2/)  |  **Part 3**  |  [Part 4 →](/blog/2026/generalization-imperative-part-4/)

---

## 7. Causality and Generalization: Learning What Causes What

> *Correlation tells you what happened. Causation tells you what will happen when you act.*

If OOD generalization requires learning features whose relationship to the label is invariant across environments, the natural next question is: *how do we identify those features?* The answer, from a growing body of work, is causality.

The connection between causality and generalization has been recognized since at least IRM <a href="#ref7">[7]</a> formalized the idea that optimal predictors across multiple environments must depend only on causal features. But the causal program in machine learning has since expanded far beyond invariant risk minimization, developing into a rich ecosystem of methods that aim to move models from learning correlations to learning causal mechanisms.

**Causal representation learning.** The central challenge, articulated by Bernhard Schölkopf and colleagues <a href="#ref31">[31]</a>, is that standard representation learning fits correlations in the training distribution, and correlations change when the distribution shifts. A model trained to recognize camels may learn to rely on desert backgrounds; shown a camel on a beach, it fails. Causal representation learning aims to discover the underlying causal variables and their relationships, features that remain stable because they reflect the data-generating process itself, not the accidental correlations of a particular dataset. Key techniques include learning invariant representations across environments, discovering causal graphs from observational data, and using interventional data to identify causal structures.

**Teaching transformers causal reasoning.** A critical 2025 line of work asks whether transformers can be taught to reason causally, rather than merely statistically. At ICML 2025, researchers demonstrated that transformers can internalize causal reasoning through axiomatic training, which injects causal axioms (e.g., the independence of cause and mechanism, the causal Markov condition) directly into the training objective. At EMNLP 2025, a separate study showed that conditional statements in code can elicit and enhance LLMs' latent causal reasoning abilities. The emerging picture: causal reasoning is not an emergent property of scale; it must be explicitly taught, either through architecture, training objective, or data design.

**Causal reinforcement learning.** When an agent acts in an environment, it intervenes on the world, and intervention is the defining operation of causal inference. A 2025 survey systematizing causal RL <a href="#ref33">[33]</a> organized the field into five categories: causal representation learning for state abstraction, counterfactual policy optimization, offline causal RL, causal transfer learning, and causal explainability. The key insight: agents that model the causal structure of their environment can transfer policies across settings that differ superficially but share causal dynamics, a form of generalization inaccessible to agents that learn only correlations.

**From theory to practice.** The practical toolkit is growing. Causal de-biasing removes spurious correlations from training data. Counterfactual data augmentation generates examples that deliberately break spurious associations: camels on beaches, in snow, in forests. Causal graph discovery algorithms, from classic PC and GES to modern deep learning variants, extract causal structure directly from observational data, enabling models to distinguish features that merely co-occur from features that genuinely matter. The Chinese-language monograph *Causal Inference and Machine Learning* <a href="#ref32">[32]</a> devotes an entire chapter to the relationship between causal representation learning and generalization ability, arguing that the problem of generalization under distribution shift is fundamentally a causal one.

The causal program represents the most principled answer to the central question of OOD generalization: *what should a model learn such that its predictions remain valid when the world changes?* The answer is to learn the causal structure, which is simple to state, enormously difficult to implement, and increasingly supported by both theory and practice.

---

## 8. World Models: Learning by Imagination

> *If you can simulate the world, you don't need as much data from it.*

David Ha and Jürgen Schmidhuber's *World Models* at NeurIPS 2018 <a href="#ref6">[6]</a> took a radically different approach to data-efficient generalization. Instead of training an agent entirely on real environment interactions, they proposed a three-part architecture:

1. A VAE compresses visual observations into a compact latent representation.
2. An RNN learns the dynamics of this latent space, the "world model."
3. A small controller learns to act by "dreaming" inside the world model.

The result: agents that could learn complex tasks with very few real environment interactions, by doing most of their learning inside the learned simulator. This is generalization through generation: the model learns a causal model of its environment and uses it to imagine scenarios it hasn't actually seen.

World Models connects naturally to both the architecture hypothesis (the VAE + RNN architecture is inherently recurrent) and the cognitive science perspective (humans don't learn everything from direct experience; we simulate, imagine, and plan). It's a reminder that generalization isn't just about extracting invariants from data; it's also about building models rich enough to generate new data.

### 8.2 From Kepler to Newton: Inductive Biases Shape Learned Physics

A fundamental question lurks beneath all world-model research: even if a model can predict future states accurately, does it actually *understand* the underlying laws, or is it just curve-fitting? Ziming Liu, Sophia Sanborn, Surya Ganguli, and Andreas Tolias at Stanford tackled this question directly in 2026 <a href="#ref42">[42]</a> with a result as elegant as its title: *From Kepler to Newton*.

They trained transformers to predict planetary motion from observational data: positions over time. With standard transformer training using long context windows, the model achieved high predictive accuracy. But when they probed its internal representations, they found it had learned a Keplerian world model: it internally encoded ellipse parameters (semi-major axis, LRL vector) and extrapolated trajectories geometrically, essentially curve-fitting the orbits.

Then they applied three minimal inductive biases that changed everything:

1. **Spatial smoothness**: Use continuous regression instead of discrete tokenization for coordinate inputs, preserving the geometric structure of space.
2. **Spatial stability**: Inject Gaussian noise into input histories during training, preventing catastrophic error accumulation during autoregressive rollouts.
3. **Temporal locality**: Restrict the attention window to just 2 steps, forcing the model to compute local forces rather than globally fitting elliptical curves.

With a context length of 2, the transformer could no longer fit Keplerian orbits. Instead, it discovered Newton's law of gravitation: it internally computed $F = ma$ with near-perfect linearity ($R^2 \approx 0.999$). The context length acts as a bifurcation parameter: long context produces a curve-fitter; short context forces the model to become a physicist.

This result has profound implications for generalization. The model's predictions under both regimes are accurate, but only the Newtonian model generalizes correctly to *counterfactual* scenarios: different masses, different star systems, different numbers of bodies. Kepler knows what will happen if the orbit continues as before. Newton knows what will happen if you change the underlying conditions. The difference is the difference between interpolation and true generalization.

Kepler-to-Newton connects deeply to themes across this survey: it is a phase transition in learned mechanisms <a href="#ref15">[15]</a>, governed by an architectural hyperparameter (context length) rather than data volume; it demonstrates that the right inductive biases can transform a memorizer into a reasoner, echoing the architecture hypothesis (Section 2); and it provides a crisp empirical testbed for the causal program's central claim (Section 7): models must learn mechanisms, not just associations, to generalize under intervention.

---

## 9. Embodied Generalization: Learning to Act in the Physical World

> *A robot that learned to clean one kitchen should be able to clean any kitchen. That, not benchmark scores, is the true test of generalization.*

All the generalization research discussed so far has been tested on language, vision, and algorithmic tasks, abstract domains where the cost of failure is a wrong token or a misclassified image. Embodied AI, robots that act in the physical world, raises the stakes: failure means dropped objects, collisions, or worse. And it raises the bar for generalization: every living room, kitchen, and doorstep is different in ways no dataset can fully cover.

In April 2025, Physical Intelligence (π) released π0.5 <a href="#ref34">[34]</a>, a Vision-Language-Action (VLA) model that achieved something remarkable: controlling a mobile manipulator to perform long-horizon household tasks such as cleaning kitchens, tidying bedrooms, and doing laundry, in brand-new homes never seen during training. The robot walked into unfamiliar environments, understood what it saw, reasoned about what needed to be done, and executed the physical actions to do it, for 10–15 minutes at a stretch.

**The data pyramid.** π0.5's key architectural insight is that generalization in the physical world requires training on far more than robot data. Its pretraining mix is a pyramid where target-robot data sits as a tiny capstone on a massive foundation of diverse experience:

| Data Source              | Description                                                       | Proportion |
| ------------------------ | ----------------------------------------------------------------- | ---------- |
| Web Data (WD)            | Image captioning, VQA, object detection from the internet         | ~50%       |
| Cross-Embodiment (CE)    | Lab data from many different robot types (OXE dataset + in-house) | ~10%       |
| Multi-Environment (ME)   | Static non-mobile robots in diverse indoor settings               | ~5%        |
| High-Level Subtasks (HL) | Human-annotated semantic task breakdowns                          | ~5%        |
| Verbal Instructions (VI) | Human coaches walking robots through steps in natural language    | ~5%        |
| Mobile Manipulator (MM)  | ~400 hours of data from ~100 different homes                      | 2.4%       |

The striking number: 97.6% of the pretraining data does not come from the target robot. Cross-embodiment data teaches physical skills; web data teaches visual and semantic understanding; human instructions teach task decomposition. The actual target-robot data, 400 hours across 100 homes, is a rounding error in the training mix, yet it is sufficient to ground the model's abstract knowledge in physical action.

**Think, then act.** π0.5 uses a hierarchical "think then act" architecture built on a PaliGemma VLM backbone (~3B parameters) plus a ~300M parameter Action Expert. The model first predicts a text-based subtask (e.g., "pick up the plate") using chain-of-thought reasoning over visual input; this is the "think" phase. Then the Action Expert generates continuous 50-step action chunks at 50Hz using flow matching; this is the "act" phase. Decoupling high-level reasoning from low-level control means the model can reason about *what* to do using all its web-learned semantic knowledge, then separately translate that intent into precise motor commands.

**What matters for generalization.** Ablation studies revealed a clear hierarchy. Removing cross-embodiment data caused the largest performance drops, revealing that physical skills learned from other robots transferred to the target robot. Removing web data hurt recognition of out-of-distribution objects. Most remarkably: with only ~100 training homes, the model's performance nearly matched an oracle model trained directly on the test environment. The diversity of training environments, not their absolute number, determined generalization quality.

This result resonates with Kawata et al.'s phase transition theory <a href="#ref15">[15]</a>: data diversity induces a transition from shortcut to induction head. In embodied AI, the "shortcut" is learning a specific kitchen layout; the "induction head" is understanding *kitchen-ness*, the abstract relational structure of counters, sinks, cabinets, and appliances that generalizes across physical spaces. π0.5 also represents a convergence of themes from this survey: it is an architecture designed for generalization (Section 2), it uses meta-learning principles to adapt to each new home (Section 5), it relies on data diversity to induce generalizable circuits (Section 6), and it builds a kind of world model, since the high-level reasoning module simulates task plans before executing them (Section 8). Embodied generalization, it turns out, is not a separate problem; it is the synthesis of every generalization approach, tested against the hardest benchmark: physical reality.

---


---

**📖 Series:** [Part 1](/blog/2026/generalization-imperative-part-1/)  |  [← Part 2](/blog/2026/generalization-imperative-part-2/)  |  **Part 3**  |  [Part 4 →](/blog/2026/generalization-imperative-part-4/)
---

## References

<a id="ref6"></a>[6] Ha, D. & Schmidhuber, J. (2018). World Models. *NeurIPS 2018*. arXiv:1803.10122.

<a id="ref7"></a>[7] Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. arXiv:1907.02893.

<a id="ref15"></a>[15] Kawata, R., Song, Y., Bietti, A., Nishikawa, N., Suzuki, T., Vaiter, S., & Wu, D. (2025). From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers. *NeurIPS 2025*. arXiv:2512.18634.

<a id="ref31"></a>[31] Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*, 109(5), 612–634.

<a id="ref32"></a>[32] 郭若城, 程璐, 刘昊, 刘欢. (2023). 《因果推断与机器学习》. 电子工业出版社. (Chapter 3: 因果表征学习与泛化能力)

<a id="ref33"></a>[33] 因果强化学习统一框架综述 (2025). A Unified Framework for Causal Reinforcement Learning: Survey, Taxonomy, Algorithms, and Applications.

<a id="ref34"></a>[34] Physical Intelligence (π). (2025). π0.5: A Vision-Language-Action Model with Open-World Generalization. Blog post, April 2025. https://www.pi.website/blog/pi05. arXiv:2504.16054.

<a id="ref42"></a>[42] Liu, Z., Sanborn, S., Ganguli, S., & Tolias, A. (2026). From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers. *ICML 2026*. arXiv:2602.06923.


*This is Part 3 of a 4-part series surveying 43 papers on generalization in deep learning. Full references across all parts are available in [Part 4](/blog/2026/generalization-imperative-part-4/).*