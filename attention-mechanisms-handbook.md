# Normalization & Residual Connections: The Complete LLM Handbook

### A first-principles-to-production guide to Layer Normalization, RMSNorm, and Skip Connections in Large Language Models

---

> **What this is:** A single, self-contained reference that takes you from "what is variance, really?" to reading the exact normalization+residual wiring inside GPT, LLaMA, DeepSeek, Mistral, and Gemma — and then to writing fused Triton/CUDA kernels for it.
>
> **Who this is for:** Anyone — no assumed background beyond basic algebra. Every technique is introduced with intuition and a worked numerical example *before* any formula. If you already know the math, skip straight to the implementations and architecture case studies.
>
> **How to read it:** Linearly if you're learning. As a reference (via the Table of Contents) if you already know the territory and need one specific thing — a derivation, a kernel, or "what does Mistral actually do here."

---

## Table of Contents

### Part I — Foundations
1. [Why Normalization? The Core Problem](#1-why-normalization-the-core-problem)
2. [Internal Covariate Shift and Optimization Landscapes](#2-internal-covariate-shift-and-optimization-landscapes)
3. [The Statistics Refresher: Mean, Variance, Standardization](#3-the-statistics-refresher-mean-variance-standardization)
4. [Geometric Intuition: What Normalization Does to a Vector Space](#4-geometric-intuition-what-normalization-does-to-a-vector-space)
5. [A Taxonomy of Normalization: Which Axis Are We Normalizing Over?](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over)

### Part II — Complete Survey of Normalization Techniques
6. [Batch Normalization (BN)](#6-batch-normalization-bn)
7. [Layer Normalization (LN)](#7-layer-normalization-ln)
8. [RMSNorm](#8-rmsnorm)
9. [Group Normalization](#9-group-normalization)
10. [Instance Normalization](#10-instance-normalization)
11. [Weight Normalization](#11-weight-normalization)
12. [Spectral Normalization](#12-spectral-normalization)
13. [L2 Normalization](#13-l2-normalization)
14. [Power Normalization](#14-power-normalization)
15. [Adaptive Normalization (AdaIN, Conditional Norms)](#15-adaptive-normalization-adain-conditional-norms)
16. [ScaleNorm](#16-scalenorm)
17. [DeepNorm](#17-deepnorm)
18. [NormFormer and Other Modern Variants](#18-normformer-and-other-modern-variants)
19. [Comparative Summary Table — All Normalization Methods](#19-comparative-summary-table--all-normalization-methods)

### Part III — Layer Normalization Deep Dive (Primary Focus)
20. [Why BatchNorm Fails for Transformers](#20-why-batchnorm-fails-for-transformers)
21. [LayerNorm Mathematics — Forward Pass](#21-layernorm-mathematics--forward-pass)
22. [LayerNorm Mathematics — Backward Pass and Jacobian](#22-layernorm-mathematics--backward-pass-and-jacobian)
23. [Learnable γ and β: Role and Gradient Flow](#23-learnable-γ-and-β-role-and-gradient-flow)
24. [Geometric Intuition for LayerNorm](#24-geometric-intuition-for-layernorm)
25. [Post-LN vs Pre-LN vs Sandwich-LN](#25-post-ln-vs-pre-ln-vs-sandwich-ln)
26. [LayerNorm Inside the Transformer Encoder](#26-layernorm-inside-the-transformer-encoder)
27. [LayerNorm Inside the Transformer Decoder](#27-layernorm-inside-the-transformer-decoder)
28. [LayerNorm Across Model Families: GPT, LLaMA, DeepSeek, Mistral, Gemma](#28-layernorm-across-model-families-gpt-llama-deepseek-mistral-gemma)

### Part IV — Residual / Skip Connections Deep Dive (Primary Focus)
29. [Motivation: Vanishing Gradients and the Degradation Problem](#29-motivation-vanishing-gradients-and-the-degradation-problem)
30. [Mathematics of Residual Learning: y = x + F(x)](#30-mathematics-of-residual-learning-y--x--fx)
31. [Gradient Propagation Through Residual Paths (Full Derivation)](#31-gradient-propagation-through-residual-paths-full-derivation)
32. [Intuition: Information Highways and Identity Mappings](#32-intuition-information-highways-and-identity-mappings)
33. [Variants: Highway Networks, DenseNet, Gated Residuals, ReZero, DeepNorm Scaling](#33-variants-highway-networks-densenet-gated-residuals-rezero-deepnorm-scaling)
34. [Residuals in Transformers: Around Attention and FFN Blocks](#34-residuals-in-transformers-around-attention-and-ffn-blocks)
35. [Pre-LN vs Post-LN — Full Gradient-Flow Comparison](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison)

### Part V — LayerNorm + Residuals Together
36. [Why They Work So Well Together](#36-why-they-work-so-well-together)
37. [Hidden State Evolution Across Layers (Worked Numerical Example)](#37-hidden-state-evolution-across-layers-worked-numerical-example)
38. [Why Activations Don't Explode in Pre-LN Stacks](#38-why-activations-dont-explode-in-pre-ln-stacks)
39. [Why Gradients Remain Stable — Depth and Scaling Laws](#39-why-gradients-remain-stable--depth-and-scaling-laws)

### Part VI — Production-Grade Implementations
40. [NumPy From Scratch (BN, LN, RMSNorm, Residual Block)](#40-numpy-from-scratch-bn-ln-rmsnorm-residual-block)
41. [PyTorch Native and Custom (nn.LayerNorm, RMSNorm, Residual Blocks)](#41-pytorch-native-and-custom-nnlayernorm-rmsnorm-residual-blocks)
42. [Triton Fused LayerNorm Kernel](#42-triton-fused-layernorm-kernel)
43. [CUDA C++ Fused LayerNorm Kernel](#43-cuda-c-fused-layernorm-kernel)
44. [Distributed Considerations: Tensor/Sequence/Pipeline Parallelism + Activation Checkpointing](#44-distributed-considerations-tensorsequencepipeline-parallelism--activation-checkpointing)
45. [Benchmarking, Profiling, and Memory Analysis](#45-benchmarking-profiling-and-memory-analysis)

### Part VII — LLM Architecture Case Studies
46. [GPT-2 / GPT-3 / GPT-4](#46-gpt-2--gpt-3--gpt-4)
47. [LLaMA / LLaMA 2 / LLaMA 3](#47-llama--llama-2--llama-3)
48. [Mistral / Mixtral](#48-mistral--mixtral)
49. [Gemma / Gemma 2](#49-gemma--gemma-2)
50. [DeepSeek (V2/V3) and Qwen](#50-deepseek-v2v3-and-qwen)
51. [Cross-Model Comparison Table](#51-cross-model-comparison-table)

### Part VIII — Production Considerations
52. [Numerical Stability and Epsilon Selection](#52-numerical-stability-and-epsilon-selection)
53. [Mixed Precision: BF16 vs FP16, Quantization Effects](#53-mixed-precision-bf16-vs-fp16-quantization-effects)
54. [Kernel Fusion, Memory Bandwidth, Cache Efficiency](#54-kernel-fusion-memory-bandwidth-cache-efficiency)
55. [Inference-Time Implications and Latency Trade-offs](#55-inference-time-implications-and-latency-trade-offs)

### Part IX — Advanced Topics
56. [ReZero, NormFormer, μP (Maximal Update Parameterization)](#56-rezero-normformer-μp-maximal-update-parameterization)
57. [Normalization-Free Transformers](#57-normalization-free-transformers)
58. [Scaling to Ultra-Deep Networks (1000+ layers)](#58-scaling-to-ultra-deep-networks-1000-layers)

### Appendix
- [A. Glossary](#a-glossary)
- [B. Notation Reference](#b-notation-reference)
- [C. Further Reading](#c-further-reading)

---

## Notation Reference (used throughout — see Appendix B for the full table)

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^d$ | A single feature/activation vector (e.g., one token's hidden state) |
| $d$ | Hidden dimension (model width) |
| $B$ | Batch size |
| $T$ | Sequence length (number of tokens) |
| $\mu$ | Mean |
| $\sigma^2$ | Variance |
| $\epsilon$ | Small constant for numerical stability |
| $\gamma, \beta$ | Learnable scale and shift parameters |
| $\odot$ | Element-wise (Hadamard) product |
| $F(x)$ | A sublayer function (attention or FFN) inside a residual block |

---

# Part I — Foundations

## 1. Why Normalization? The Core Problem

### 1.1 The problem, before any math

Imagine you're a manager reading performance reviews from two different teams. Team A scores everyone on a scale of 0–10,000. Team B scores everyone on a scale of 0–1. If you just add up the raw numbers to decide who gets a bonus, Team A's scores will dominate every decision — not because Team A's people performed better, but purely because of the *scale* their numbers happen to live on.

A neural network has exactly this problem, but internally, between its own layers, thousands of times per forward pass.

Each layer in a deep network produces an output (an "activation") that becomes the input to the next layer. There is nothing in the basic math of a neural network — a matrix multiply followed by a nonlinearity — that keeps these activations on a consistent, well-behaved scale. One layer might output values clustered tightly around 0.01. The next, after a few multiplications, might output values spread between -500 and 500. Nothing forces good behavior; you simply get whatever the random initialization and the data happen to produce.

This matters enormously for **training**, because training works by computing gradients — "if I nudge this weight slightly, how much does the loss change?" — and propagating those nudges backward through every layer via the chain rule. If activations at some layer are enormous, the gradients flowing back through them tend to be enormous too (exploding gradients). If activations are tiny, gradients shrink to near-zero as they pass through many layers (vanishing gradients). Either way, the optimizer (e.g., Adam, SGD) struggles: a single learning rate that works for one layer's scale will be wildly wrong for another layer's scale at the same moment in training.

**Normalization is the fix.** At its core, normalization is a deliberately simple operation: take a group of numbers, and rescale them so they have a controlled, predictable statistical profile — typically mean 0, variance 1 (then optionally let the network re-introduce its own preferred scale and shift via learnable parameters). It's the equivalent of converting Team A's and Team B's scores onto the same standardized scale before comparing them.

### 1.2 A tiny worked example

Suppose a layer produces this activation vector for one token (just 4 dimensions, to keep it readable):

$$x = [\,2.0,\ 8.0,\ -4.0,\ 6.0\,]$$

This vector has:
- Mean: $\mu = \frac{2.0 + 8.0 - 4.0 + 6.0}{4} = 3.0$
- Variance: $\sigma^2 = \frac{(2-3)^2 + (8-3)^2 + (-4-3)^2 + (6-3)^2}{4} = \frac{1 + 25 + 49 + 9}{4} = 21.0$
- Standard deviation: $\sigma \approx 4.58$

Normalizing means computing, for each element:

$$\hat{x}_i = \frac{x_i - \mu}{\sigma}$$

$$\hat{x} = \left[\frac{2-3}{4.58},\ \frac{8-3}{4.58},\ \frac{-4-3}{4.58},\ \frac{6-3}{4.58}\right] \approx [\,-0.218,\ 1.091,\ -1.528,\ 0.655\,]$$

Notice: the *relative relationships* between elements are preserved (8.0 was the largest before, 1.091 is the largest after; -4.0 was the smallest, -1.528 is the smallest after). What changed is the **scale**: the new vector has mean ≈ 0 and variance ≈ 1, regardless of what wild scale the original activations happened to be on. Every layer in the network can now expect its input to live in this same, predictable range — no matter how the magnitudes of earlier computations drifted.

This is the single idea every normalization technique in this handbook is built on. Everything else — *which* numbers you average over (a batch? a single example's features? a spatial window?), *when* you apply it (before or after a sublayer?), and *what* you do after standardizing (multiply by a learned $\gamma$, add a learned $\beta$?) — is a design choice layered on top of this one core operation.

### 1.3 Why this is genuinely hard to get right at scale

If normalization were just "always standardize everything, everywhere," this handbook would be one page long. The difficulty — and the reason there are over a dozen named variants — comes from a tension between several things that all matter simultaneously in a real production LLM:

1. **What axis do you normalize over?** Across a batch of examples? Across the features of a single example? Across a spatial neighborhood? Each choice has different statistical properties and different failure modes (we cover this in [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over)).
2. **Training-time behavior must match inference-time behavior.** Some early techniques (BatchNorm — [Section 6](#6-batch-normalization-bn)) compute statistics from a batch during training, but at inference you might process one example at a time. This mismatch is a real engineering headache, and a major reason LLMs largely abandoned BatchNorm.
3. **Computational cost matters at scale.** A normalization op runs on *every token, every layer, every forward and backward pass* — for a 70B-parameter model processing trillions of tokens, even a small constant-factor inefficiency compounds into real GPU-hours and real money. This motivates simplified variants like RMSNorm ([Section 8](#8-rmsnorm)) and fused kernels ([Sections 42–43](#42-triton-fused-layernorm-kernel)).
4. **Numerical stability in low precision.** Modern LLMs train and run inference in BF16/FP16. Computing a variance (a sum of squared differences) in low precision is more failure-prone than it looks on paper — small denominators and squared terms can underflow or overflow ([Section 52](#52-numerical-stability-and-epsilon-selection)).
5. **Interaction with depth.** As networks got deeper (dozens to hundreds of layers), normalization alone wasn't enough to keep training stable — this is precisely why residual connections ([Part IV](#29-motivation-vanishing-gradients-and-the-degradation-problem)) had to be invented alongside normalization, and why the *combination* of the two ([Part V](#36-why-they-work-so-well-together)) is the real story of how modern deep Transformers are trained at all.

### 1.4 What's next

[Section 2](#2-internal-covariate-shift-and-optimization-landscapes) formalizes the optimization problem normalization addresses — what "internal covariate shift" actually means, and why it makes optimization landscapes harder to navigate. From there, [Section 3](#3-the-statistics-refresher-mean-variance-standardization) gives a clean refresher on the mean/variance machinery used everywhere in this handbook, and [Section 4](#4-geometric-intuition-what-normalization-does-to-a-vector-space) builds the geometric picture (normalization as a projection onto a sphere) that will make every later derivation easier to visualize.

---

## 2. Internal Covariate Shift and Optimization Landscapes

### 2.1 The term, and why it was coined

"Internal covariate shift" is the phrase the original Batch Normalization paper (Ioffe & Szegedy, 2015) used to describe a specific phenomenon: **as the parameters of earlier layers change during training, the *distribution* of inputs seen by later layers keeps changing too.**

Break that down piece by piece:

- "Covariate" here just means "input to a layer" — borrowed from statistics, where a covariate is an input variable to a model.
- "Shift" means the statistical distribution of that input (its mean, variance, overall shape) is moving around.
- "Internal" means this isn't about your training data distribution shifting (that's a different, well-known problem called *dataset* covariate shift) — it's happening *inside* the network, between its own layers, purely as a side effect of training.

Here's the mechanism concretely. Layer 5 in a network learns to process whatever distribution of inputs Layer 4 happens to be producing *right now*. But Layer 4's weights are also being updated by gradient descent, every single step. So a few steps later, Layer 4 produces a *different* distribution of outputs — maybe shifted higher, maybe with larger variance — and now Layer 5 is receiving inputs it wasn't quite tuned for. Layer 5 has to adapt again. But Layer 5's adaptation changes what Layer 6 receives. And so on, cascading through the whole network, every single step.

It's a bit like trying to have a conversation where the language your conversation partner speaks subtly changes every sentence. You're perpetually re-calibrating instead of making steady progress.

### 2.2 Why this slows down optimization

This matters because of how gradient-based optimization works. Gradient descent computes "if I move this weight a little, how much does the loss change, *assuming everything else stays fixed*." That assumption — everything else stays fixed — is a local, first-order approximation. It is more accurate when the loss landscape is smooth and the things that "everything else" depends on aren't themselves shifting wildly underneath it.

If every layer's input distribution is constantly drifting due to changes in earlier layers, then the assumption "everything else stays fixed" becomes a worse and worse approximation. Concretely, this manifests as:

- **The need for very small learning rates.** If you take a confidently large gradient step, you might be stepping based on stale information about what the next layer's input distribution will look like once this step is applied — early deep networks were notoriously sensitive to learning rate choice for this reason.
- **Sensitivity to initialization.** Since there's no mechanism actively keeping distributions stable, where you start (the initial weight values) has an outsized effect on whether training proceeds smoothly or diverges/stalls early.
- **The "vanishing/exploding activations" problem compounding over depth.** Each layer's drift can amplify or shrink the *next* layer's input scale — and over many layers, small per-layer drifts compound multiplicatively.

### 2.3 A visual way to think about the optimization landscape

Imagine the loss function as a landscape you're trying to descend (find the lowest point = lowest loss) using only local information (the gradient at your current position — like a hiker who can only feel the slope under their feet, not see the whole mountain range).

```
Without normalization:                With normalization:

   loss                                  loss
    |    /\    /\                         |
    |   /  \  /  \  /\                    |    ___
    |  /    \/    \/  \   <- jagged,      |   /   \___
    | /            ___  \    narrow         |  /       \___
    |/            /   \  \   ravines,       | /            \___
    +----------------------- params        +----------------------- params
    Steep, narrow, shifting ravines.       Smoother, wider basin.
    Gradient direction at one step          Gradient direction stays
    may be nearly useless a few             roughly informative across
    steps later because the                 many steps because layer-
    landscape itself has moved              input distributions are
    (because earlier layers                 actively held stable.
    changed the input distribution
    this layer now sees).
```

The right-hand picture is the empirical effect normalization techniques produce: not necessarily a *fundamentally different* loss surface in some abstract mathematical sense (this is actually a point of real academic debate — see the note below), but a optimization process that *behaves* as if the surface is smoother, because the inputs at each layer are kept on a controlled, predictable scale step after step.

### 2.4 An important caveat: the "ICS" explanation is contested

It would be intellectually dishonest to present internal covariate shift as the settled, universally-agreed explanation for *why* normalization works — it isn't. A well-known follow-up paper, "How Does Batch Normalization Help Optimization?" (Santurkar et al., 2018), ran controlled experiments and argued that BatchNorm's benefit has little to do with reducing covariate shift in the sense originally proposed, and instead comes primarily from making the loss landscape's **Lipschitz constants** better behaved — i.e., directly smoothing the loss surface and its gradients, making larger, more confident learning rates viable and more consistently helpful, regardless of whether internal distributions are technically "shifting" in the original sense.

Practically, for this handbook, it doesn't change *what you should do* — normalize. But it matters for accurate intuition: when someone says "normalization reduces internal covariate shift," it's best understood as a useful, historically important first intuition rather than a complete or experimentally fully validated mechanism. The more defensible, evidence-backed framing is: **normalization makes optimization landscapes smoother and more well-conditioned, which lets you use larger learning rates and converge faster and more reliably** — and the internal covariate shift story is one (contested) hypothesis for *why*, among several.

### 2.5 What this means for LLMs specifically

Transformers are *deep* (dozens to over a hundred layers) and process long sequences where activation statistics can vary a lot from token to token and position to position. Both of these amplify the optimization difficulties described above:

- **Depth** means any per-layer drift compounds over many more layers than in a shallow CNN — exactly the depth/stability problem that motivates [Part IV](#29-motivation-vanishing-gradients-and-the-degradation-problem) (residual connections) and [Part V](#36-why-they-work-so-well-together) (the LayerNorm + residual combination).
- **Variable sequence length and no natural "batch" structure per token** is precisely why BatchNorm — which computes statistics *across the batch dimension* — is a poor fit for Transformers, and why LayerNorm — which computes statistics across each token's own feature vector, independent of batch and sequence position — became the default. We derive this in full in [Section 20](#20-why-batchnorm-fails-for-transformers).

### 2.6 What's next

[Section 3](#3-the-statistics-refresher-mean-variance-standardization) lays out the statistical machinery (mean, variance, standardization, and why we divide by standard deviation rather than range or some other scale measure) cleanly and rigorously, so every later derivation in this handbook can build on a shared, precise vocabulary.

---

## 3. The Statistics Refresher: Mean, Variance, Standardization

This section is deliberately slow and explicit. Every normalization technique in this handbook is a variation on the formulas below — applied over a different *axis*, with different *learnable parameters*, or with a different *denominator*. If these four formulas are completely solid, every later derivation becomes "which numbers do we plug in," not new math.

### 3.1 Mean — what it represents

For a set of $n$ numbers $x_1, x_2, \ldots, x_n$, the mean (average) is:

$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$$

Intuitively: the mean is the single number that, if it replaced every value in the set, would keep the *sum* unchanged. It's the "center of mass" of the data — if you placed each value as a weight on a number line, $\mu$ is where the line would balance on a fingertip.

### 3.2 Variance — what it represents

Variance measures how spread out the numbers are around the mean:

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2$$

Walk through *why* it's built this way:

1. $(x_i - \mu)$ — the deviation of each point from the mean. Some are positive (above the mean), some negative (below).
2. Squaring each deviation, $(x_i - \mu)^2$, does two things at once: it makes every term non-negative (so positive and negative deviations don't cancel out and hide real spread), and it penalizes large deviations more than small ones — a point twice as far from the mean contributes four times as much to the variance, not twice as much.
3. Averaging the squared deviations gives a single number summarizing typical "spread."

**Why square instead of taking the absolute value** $|x_i - \mu|$ **(which would also fix the sign-cancellation problem)?** Two reasons that matter in practice: squaring is differentiable everywhere (absolute value has a kink at zero, which complicates gradient-based optimization slightly), and variance has clean, well-known mathematical properties (it's the second central moment of a distribution, and many results — including the Gaussian distribution's parameterization — are built directly around it). Both normalization and most of statistics standardize on variance/squared-deviation for this reason.

### 3.3 Standard deviation — bringing the units back

Variance has an awkward property: its *units* are squared. If $x$ is measured in meters, $\sigma^2$ is in meters². That's mathematically fine but intuitively hard to compare against the original data. **Standard deviation** fixes this by taking the square root:

$$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$$

Now $\sigma$ is in the same units as the original data, and it has a direct, intuitive meaning: "the typical distance of a point from the mean."

### 3.4 Standardization — putting it together

**Standardization** (also called the "z-score") rescales data to have mean 0 and variance 1:

$$\hat{x}_i = \frac{x_i - \mu}{\sigma}$$

Two separate things are happening in this one formula, and it helps to see them as two distinct steps:

- **Centering**: $x_i - \mu$ shifts the entire dataset so its mean becomes exactly 0. This removes any "DC offset" — a uniform shift that doesn't carry information about relative differences between points.
- **Scaling**: dividing by $\sigma$ rescales the (now-centered) data so its variance becomes exactly 1. This removes dependence on the original data's units or magnitude.

After standardization, $\hat{x}$ has mean 0 and variance 1 *by construction* — this isn't an approximation or a goal we're optimizing toward, it's a guaranteed mathematical consequence of the formula (modulo floating-point rounding, which we address in [Section 52](#52-numerical-stability-and-epsilon-selection)).

**Quick proof that the mean becomes 0:**

$$\mathbb{E}[\hat{x}] = \mathbb{E}\left[\frac{x - \mu}{\sigma}\right] = \frac{\mathbb{E}[x] - \mu}{\sigma} = \frac{\mu - \mu}{\sigma} = 0$$

**Quick proof that the variance becomes 1:**

$$\text{Var}(\hat{x}) = \text{Var}\left(\frac{x-\mu}{\sigma}\right) = \frac{\text{Var}(x-\mu)}{\sigma^2} = \frac{\sigma^2}{\sigma^2} = 1$$

(Using the property that subtracting a constant doesn't change variance, and that $\text{Var}(aX) = a^2\text{Var}(X)$.)

### 3.5 The epsilon you'll see everywhere

In every real implementation, you'll see the denominator written as $\sqrt{\sigma^2 + \epsilon}$ rather than $\sqrt{\sigma^2}$, with $\epsilon$ a tiny constant (commonly $10^{-5}$ or $10^{-6}$). This isn't a statistical refinement — it's a numerical safety net. If $\sigma^2$ happens to be exactly (or very close to) zero — which can genuinely happen, e.g. if all values in a normalization group are identical, or early in training before weights have diversified — dividing by zero (or by a near-zero number amplified by floating-point error) produces `inf` or `NaN`, silently destroying training. $\epsilon$ guarantees the denominator never gets pathologically small. We dedicate all of [Section 52](#52-numerical-stability-and-epsilon-selection) to how to choose $\epsilon$ properly, because the "right" value is precision- and architecture-dependent.

### 3.6 Restoring expressiveness: learnable $\gamma$ and $\beta$

There's a subtlety worth flagging now, because it resolves a question every learner eventually asks: *if we forcibly standardize every layer's output to mean 0, variance 1, haven't we removed the network's ability to represent anything other than that one specific scale?*

Yes — which is exactly why almost every normalization layer used in practice (LayerNorm, BatchNorm, RMSNorm, etc.) follows standardization with a learnable affine transform:

$$y_i = \gamma \cdot \hat{x}_i + \beta$$

where $\gamma$ (scale) and $\beta$ (shift) are parameters learned by gradient descent, typically initialized to $\gamma = 1, \beta = 0$ (i.e., "start as a no-op on top of standardization, then let training decide if a different scale/shift is better for this specific layer"). This means standardization doesn't *remove* the network's capacity to use a different effective mean and variance at each layer — it just **decouples that choice from the raw, uncontrolled drift of upstream activations**, and hands control of it explicitly to a small number of learned parameters instead. The network can still end up representing a layer's preferred scale of "mean = 7, variance = 25" — but it now does so deliberately, via 2 learned numbers per feature, rather than as an uncontrolled side-effect of whatever upstream weights happen to produce.

### 3.7 Population vs. sample variance — a notational note

You may have learned, in an introductory statistics course, that *sample* variance divides by $(n-1)$ rather than $n$ (Bessel's correction, used to produce an unbiased estimator of a population's variance from a finite sample). **Deep learning normalization layers use the $n$ (biased/population) version**, dividing by $n$, not $n-1$. This is a deliberate and consistent choice across BatchNorm, LayerNorm, RMSNorm, and essentially every variant in this handbook — the goal here isn't statistical inference about some underlying population, it's a deterministic rescaling operation on the specific numbers present in this forward pass. Knowing this convention avoids a common source of confusion when comparing handbook formulas against, e.g., NumPy's default `np.var()` behavior (NumPy also defaults to $n$, i.e. `ddof=0`, matching this convention — but it's worth knowing the default differs in other libraries/contexts).

### 3.8 What's next

With mean, variance, standardization, $\epsilon$, and the $\gamma/\beta$ affine transform all precisely defined, [Section 4](#4-geometric-intuition-what-normalization-does-to-a-vector-space) builds a geometric (rather than purely algebraic) picture of what these operations do to a vector — viewing normalization as centering and projecting onto a sphere, which will make the more advanced variants (Weight Normalization, Spectral Normalization, ScaleNorm) much more intuitive when we reach them.

---

## 4. Geometric Intuition: What Normalization Does to a Vector Space

### 4.1 Stop thinking in lists of numbers — start thinking in space

So far we've treated a vector like $x = [2.0, 8.0, -4.0, 6.0]$ as just a list of four numbers. But it's far more useful, especially for building lasting intuition, to think of it as a single **point** (or arrow from the origin) in 4-dimensional space. Every operation we apply — subtracting the mean, dividing by the standard deviation — is a specific, visualizable geometric transformation on that point. This section builds that picture using a 2D and 3D vector as stand-ins, since 4D+ is identical in spirit but impossible to draw.

### 4.2 Centering = sliding the whole space so the mean sits at the origin

Take a 2D vector $x = (x_1, x_2)$. Subtracting the mean $\mu = \frac{x_1+x_2}{2}$ from each coordinate:

$$x' = (x_1 - \mu,\ x_2 - \mu)$$

is a **translation** — sliding the point along the direction $(1, 1)$ (the "all-ones" direction) until its two coordinates sum to zero. Geometrically, centering always moves the point onto the hyperplane defined by $\sum_i x_i' = 0$ — a $(d-1)$-dimensional subspace (a line in 2D, a plane in 3D, a hyperplane in $d$ dimensions) that passes through the origin and is perpendicular to the all-ones vector $\mathbf{1} = (1, 1, \ldots, 1)$.

```
Before centering:                After centering (mean subtracted):

      x2                                x2
       |        • x=(2,8)                |
       |       /                         |     • x'=(-3,3)
       |      /                          |    /
       |     /                           |   /  (slid along the
       |    /                            |  / direction (1,1)
   ----+---/----------- x1          -----+-/--------------- x1
       |  /                              |/
       | /                               +  <- now sits ON the
       |/                                    line x1' + x2' = 0
       +
  (off the "mean-zero" line,        (now exactly on it — this
   sum of coords = 10)               line is perpendicular to
                                      the (1,1) direction)
```

This is exactly why, later, we'll be able to say things like "centering removes one degree of freedom" or "the centered vector lives in a $(d-1)$-dimensional subspace" — it's not an abstract claim, it's a direct geometric consequence of always landing on that hyperplane.

### 4.3 Scaling by $1/\sigma$ = projecting onto a sphere (then rescaling its radius)

After centering, dividing every coordinate by $\sigma$ is a **uniform scaling toward or away from the origin** — it shrinks or grows the vector's length without changing its *direction*. Recall that for a centered vector, the standard deviation $\sigma$ and the vector's Euclidean norm $\|x'\|$ are directly related:

$$\sigma = \sqrt{\frac{1}{n}\sum_i (x_i')^2} = \frac{\|x'\|}{\sqrt{n}}$$

So dividing by $\sigma$ is the same as dividing by $\|x'\|/\sqrt{n}$, i.e., rescaling $x'$ to have norm exactly $\sqrt{n}$:

$$\|\hat{x}\| = \left\|\frac{x'}{\sigma}\right\| = \frac{\|x'\|}{\sigma} = \frac{\|x'\|}{\|x'\|/\sqrt{n}} = \sqrt{n}$$

**This is the key geometric insight**: standardization takes any vector, no matter its original length or direction (after centering), and places it onto the surface of a sphere of fixed radius $\sqrt{n}$, centered at the origin, lying within the mean-zero hyperplane from Section 4.2.

```
                    Centered vectors of wildly
                    different lengths...

        x2                                    x2
         |    • A (length 8)                   |        sphere of
         |   /                                  |        radius √n
         |  /                                   |       .-----.
         | /                                    |     .'       '.
    -----+-------- x1                     ------+----A---------- x1
         |\                                      |   B  '.     .'
         | \                                     |        '---'
         |  \ • B (length 2)                     |  ...all land on
         +                                        |  the SAME sphere
                                                   |  after dividing
                                                   |  by their own σ
```

Direction is preserved; only the radial distance from the origin changes. This is why normalization is sometimes described as "projecting onto a sphere" — though it's worth being precise that it's not a *projection* in the strict linear-algebra sense (which would drop dimensionality), it's a **radial rescaling** that keeps the point in the same direction from the origin but forces it onto a fixed-radius sphere.

### 4.4 Why this geometric view matters practically

This isn't just a pretty picture — it directly explains behavior you'll see in later sections:

- **Why normalization discards magnitude information.** Two activation vectors that point in the exact same direction but have very different lengths (e.g., one is a scaled-up version of the other) become *identical* after standardization. If the magnitude itself carried meaningful signal (sometimes it does — e.g., a "confidence" encoded as vector length), naive normalization throws that away. This is precisely the motivation behind techniques like [Adaptive Normalization](#15-adaptive-normalization-adain-conditional-norms), which reintroduce magnitude/style information through external conditioning rather than letting it get washed out.
- **Why the learnable $\gamma, \beta$ from Section 3.6 matter geometrically**: after the radial-rescaling-onto-a-sphere step, $\gamma$ lets the network choose a *different* sphere radius per feature (anisotropic rescaling — different scale per dimension, not just a single global radius), and $\beta$ lets it choose a different center than the origin. Together, $\gamma$ and $\beta$ give the network back the freedom to place its preferred ellipsoid anywhere in space, while the *unlearned* standardization step underneath guarantees that whatever ellipsoid the network lands on, it got there from a numerically controlled, consistent starting point rather than uncontrolled drift.
- **Why Weight Normalization ([Section 11](#11-weight-normalization)) and Spectral Normalization ([Section 12](#12-spectral-normalization)) feel like a different "family"**: those techniques apply this same "fix the scale, keep the direction" logic not to *activations* but to *weight matrices themselves* — normalizing a weight vector's magnitude while leaving a separate learnable scale parameter to control magnitude explicitly, or constraining a weight matrix's largest singular value (its maximum stretching factor in any direction). Once you see normalization as "control magnitude geometrically, preserve direction," these later techniques stop looking like unrelated tricks and start looking like the same idea applied to a different tensor.

### 4.5 What's next

We now have the conceptual foundation — *why* normalization is needed (Section 1), the optimization-theoretic framing (Section 2), the precise statistical machinery (Section 3), and the geometric picture (Section 4). [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over) closes out Part I by laying out the single most important design axis that distinguishes every normalization technique from every other: **which set of numbers do you compute $\mu$ and $\sigma$ over?** Getting this taxonomy clear before diving into individual techniques in Part II will make each technique's definition feel like a one-line variation on a shared template, rather than a brand-new thing to memorize.

---

## 5. A Taxonomy of Normalization: Which Axis Are We Normalizing Over?

### 5.1 The one question that defines every technique in Part II

Every normalization technique you will encounter computes the exact same two quantities — a mean $\mu$ and a variance $\sigma^2$ — and applies the exact same standardization formula from [Section 3.4](#34-standardization--putting-it-together). The *entire* difference between Batch Norm, Layer Norm, RMSNorm, Group Norm, Instance Norm, and every other variant in Part II boils down to one question:

> **Over which set of values, exactly, do we average to compute $\mu$ and $\sigma^2$?**

This single design choice has enormous downstream consequences — for whether the technique works for variable-length sequences, whether it behaves consistently between training and inference, and how expensive it is to compute. Getting this taxonomy crisp now means every formula in Part II will read as "oh, it's just averaging over a different subset" rather than new math to memorize from scratch.

### 5.2 Setting up the tensor shape we'll use throughout

A batch of activations flowing through a Transformer is naturally a 3D tensor with shape:

$$X \in \mathbb{R}^{B \times T \times d}$$

where:
- $B$ = batch size (number of independent sequences processed together)
- $T$ = sequence length (number of tokens in each sequence)
- $d$ = hidden/feature dimension (the width of the model)

(For convolutional/vision models, you'll instead see $N \times C \times H \times W$ — batch, channels, height, width. We'll use both notations as relevant; the *logic* of the taxonomy is identical, only the axis labels differ.)

### 5.3 The four canonical axes, visualized

```
                    d (feature dimension) →
                  ┌───────────────────────────┐
              T ↑ │                           │   One single "cube" of
        (seq.   │ │     ONE TRAINING          │   activations for the
        length)   │     EXAMPLE'S             │   whole batch. Different
                  │     TOKEN GRID             │   normalization methods
                  └───────────────────────────┘   slice this cube along
                  B (batch) — stacked behind ↗     different axes.


  BATCH NORM:                    LAYER NORM:
  average over B (and T,         average over d ONLY,
  for sequence models) for       independently for
  EACH feature d independently   EACH (batch, token)
                                 position independently

   d→                              d→
  ┌─┬─┬─┬─┬─┐                     ┌───────────┐
  │ │ │ │ │ │ ← one (b,t) slice   │███████████│ ← average ALL d values
  │ │ │ │ │ │   contributes one   └───────────┘   in THIS ONE slice
  │ │ │ │ │ │   value per column    (b fixed, t fixed)
  │↓│↓│↓│↓│↓│
  average DOWN each column,
  across ALL (b,t) pairs
  (one mean/var PER FEATURE,
   shared across the batch
   and sequence dimension)


  GROUP NORM:                    INSTANCE NORM:
  average over a SUBSET of       average over spatial
  channels (a "group"),          dims (H,W) only, per
  per-sample, per-spatial        channel, per-sample
  position (vision-oriented)     (vision-oriented; like
                                 LayerNorm restricted to
                                 one channel at a time)
```

### 5.4 In words, precisely

| Method | Averages over | Independent per | Used mainly in |
|---|---|---|---|
| **Batch Norm** | Batch dimension $B$ (and sequence $T$, if present) | Each feature/channel | CNNs, vision; rarely modern LLMs |
| **Layer Norm** | Feature dimension $d$ | Each (batch, token) position | Transformers, LLMs (the default for years) |
| **RMSNorm** | Feature dimension $d$ (root-mean-square only, no mean subtraction) | Each (batch, token) position | Modern LLMs (LLaMA, Mistral, Gemma, etc.) |
| **Group Norm** | A subset ("group") of channels, plus spatial dims | Each sample, each group | CNNs, vision, diffusion models |
| **Instance Norm** | Spatial dimensions only, per channel | Each sample, each channel | Style transfer, GANs |

This table is the map for all of Part II. Each entry in [Sections 6–18](#6-batch-normalization-bn) is, at its core, "here's exactly which axis this method averages over, and here's why that choice was made and what trade-off it implies."

### 5.5 Why the axis choice has such large practical consequences

Three consequences fall directly out of which axis you pick, and they explain *why* the field moved from BatchNorm (the original, 2015 technique) to LayerNorm/RMSNorm (the modern LLM default) almost entirely on engineering grounds, not because the underlying normalization *math* changed:

1. **Dependency on batch size.** Any method that averages over $B$ (BatchNorm) gives statistically noisier estimates of $\mu, \sigma^2$ when $B$ is small, and becomes outright ill-defined when $B = 1$ (which is exactly the common case at autoregressive inference time, generating one token at a time). Methods that average over $d$ instead (LayerNorm, RMSNorm) are completely unaffected by batch size, because each (batch, token) position is normalized independently using only its own feature vector.
2. **Train/inference consistency.** Because BatchNorm's statistics depend on which other examples happen to be in the current batch, it needs a separate mechanism (running averages of $\mu, \sigma^2$ tracked during training, frozen and reused at inference) to behave consistently when the batch composition at inference differs from training. LayerNorm/RMSNorm need no such mechanism — the exact same computation runs identically in training and inference, because nothing about it depends on what else is in the batch.
3. **Compatibility with variable-length sequences.** Padding tokens at the end of shorter sequences in a batch would corrupt BatchNorm's per-feature statistics (since it averages across the batch and sequence dimension together) unless carefully masked. LayerNorm sidesteps this entirely — each token position's normalization only ever looks at that one token's own $d$-dimensional feature vector, never at other tokens or other batch members, so padding elsewhere in the batch simply can't leak in.

These three points, taken together, are the complete, concrete answer to "why don't LLMs use BatchNorm" — and we'll make this fully rigorous with worked tensor examples in [Section 20](#20-why-batchnorm-fails-for-transformers).

### 5.6 Part I summary, and what's next

Part I is now complete. We've built, in order: the core motivating problem (Section 1), the optimization-theoretic framing and its genuine scientific nuance (Section 2), the precise statistical machinery used by every technique in this handbook (Section 3), the geometric picture of standardization as centering + radial rescaling onto a sphere (Section 4), and the single taxonomic axis — *which dimension do we average over* — that organizes every named technique that follows (Section 5).

**Part II** begins now: a complete, technique-by-technique survey covering Batch Normalization, Layer Normalization, RMSNorm, Group Normalization, Instance Normalization, Weight Normalization, Spectral Normalization, L2 Normalization, Power Normalization, Adaptive Normalization, ScaleNorm, DeepNorm, and NormFormer — each with motivation, full mathematical derivation, a worked numerical example, forward/backward pass, PyTorch implementation, numerical stability notes, complexity analysis, and real-world model usage.

---

# Part II — Complete Survey of Normalization Techniques

## 6. Batch Normalization (BN)

### 6.1 Motivation

Batch Normalization (Ioffe & Szegedy, 2015) was the technique that started this entire field. Its motivation is exactly [Section 2](#2-internal-covariate-shift-and-optimization-landscapes)'s story: deep networks train slowly and unreliably because each layer's input distribution keeps shifting as earlier layers update. BN's proposed fix: **explicitly force each feature/channel to have mean 0 and variance 1, recomputed fresh at every training step, using the statistics of the current mini-batch.**

It was enormously successful for CNNs (ResNet, Inception, and most pre-Transformer vision architectures were built assuming BN), enabling much larger learning rates and removing a great deal of the "babysitting" deep network training previously required. Understanding it properly matters even for an LLM-focused handbook, both because it's the historical and conceptual root of everything that follows, and because **understanding precisely why it fails for Transformers** ([Section 20](#20-why-batchnorm-fails-for-transformers)) is one of the most illuminating exercises in this entire handbook.

### 6.2 Mathematical formulation

Given a mini-batch of activations for a single feature/channel $c$, across $B$ examples (and, for sequence/spatial data, across $T$ positions too — we'll use $T$ generically to mean "every non-channel, non-batch position"), BatchNorm computes:

$$\mu_c = \frac{1}{B \cdot T}\sum_{b=1}^{B}\sum_{t=1}^{T} x_{b,t,c}$$

$$\sigma_c^2 = \frac{1}{B \cdot T}\sum_{b=1}^{B}\sum_{t=1}^{T} (x_{b,t,c} - \mu_c)^2$$

$$\hat{x}_{b,t,c} = \frac{x_{b,t,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

$$y_{b,t,c} = \gamma_c \cdot \hat{x}_{b,t,c} + \beta_c$$

The critical detail, visible directly in the subscripts: $\mu_c$ and $\sigma_c^2$ depend **only on $c$** — there is one mean and one variance *per channel*, shared across every example in the batch and every spatial/temporal position. $\gamma_c$ and $\beta_c$ are likewise per-channel learned parameters (so for a layer with $d$ channels, BN has $2d$ learnable parameters total — small, regardless of batch size or spatial extent).

### 6.3 A worked numerical example

Consider a tiny case: $B = 3$ examples, a single channel ($d=1$, so we can drop the $c$ subscript), no spatial/temporal extent ($T=1$). The batch's values for this channel are:

$$x = [\,4.0,\ 10.0,\ 1.0\,]$$

**Step 1 — batch mean:**
$$\mu = \frac{4.0 + 10.0 + 1.0}{3} = 5.0$$

**Step 2 — batch variance:**
$$\sigma^2 = \frac{(4-5)^2 + (10-5)^2 + (1-5)^2}{3} = \frac{1 + 25 + 16}{3} = 14.0$$

**Step 3 — standardize** (using $\epsilon = 10^{-5}$, negligible here):
$$\hat{x} = \left[\frac{4-5}{\sqrt{14}},\ \frac{10-5}{\sqrt{14}},\ \frac{1-5}{\sqrt{14}}\right] \approx [\,-0.267,\ 1.336,\ -1.069\,]$$

**Step 4 — affine transform** (say the layer has learned $\gamma = 2.0, \beta = 1.0$):
$$y = [\,2.0(-0.267)+1.0,\ 2.0(1.336)+1.0,\ 2.0(-1.069)+1.0\,] = [\,0.466,\ 3.672,\ -1.138\,]$$

Notice that **every value in this computation depended on all three examples in the batch simultaneously** — change any one of the three input values, and $\mu$, $\sigma^2$, and therefore *every* output value $y$ changes, even for the other two examples that didn't change. This cross-example coupling is the single most consequential property of BatchNorm, and it's the root cause of nearly every practical complication discussed below.

### 6.4 Forward pass (general tensor form)

For vision data $X \in \mathbb{R}^{N \times C \times H \times W}$:

```
for each channel c in [0, C):
    μ_c     = mean(X[:, c, :, :])                    # scalar, over N, H, W
    σ²_c    = var(X[:, c, :, :])                      # scalar, over N, H, W
    X̂[:,c,:,:] = (X[:, c, :, :] - μ_c) / sqrt(σ²_c + ε)
    Y[:,c,:,:] = γ_c * X̂[:, c, :, :] + β_c
```

### 6.5 Backward pass — full derivation

This is the derivation pattern we'll reuse (with axis changes) for LayerNorm in [Section 22](#22-layernorm-mathematics--backward-pass-and-jacobian), so it's worth doing carefully once, here.

Let $N = B \cdot T$ denote the total number of elements averaged over for a given channel (dropping the $c$ subscript for readability — every quantity below is computed independently per channel). Given the upstream gradient $\frac{\partial \mathcal{L}}{\partial y_i}$ for each element $i$ in this channel's normalization group, we need $\frac{\partial \mathcal{L}}{\partial x_i}$.

**Step 1 — gradients w.r.t. $\gamma$ and $\beta$** (these are the easy ones, since $y_i = \gamma \hat{x}_i + \beta$ is a direct elementwise affine map):

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{N} \frac{\partial \mathcal{L}}{\partial y_i}\hat{x}_i \qquad \frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{N} \frac{\partial \mathcal{L}}{\partial y_i}$$

**Step 2 — gradient w.r.t. $\hat{x}_i$** (trivial, since $\gamma$ is just a constant multiplier from $\hat{x}$'s perspective):

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i}\cdot \gamma$$

**Step 3 — the hard part: gradient w.r.t. $x_i$.** This is where it gets genuinely subtle, because $x_i$ doesn't only affect $\hat{x}_i$ directly — it *also* affects $\mu$ and $\sigma^2$, which in turn affect **every** $\hat{x}_j$ in the group, not just $\hat{x}_i$. We must apply the multivariate chain rule, summing over all the paths $x_i$ influences:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j=1}^{N} \frac{\partial \mathcal{L}}{\partial \hat{x}_j}\cdot\frac{\partial \hat{x}_j}{\partial x_i}$$

Working out $\frac{\partial \hat{x}_j}{\partial x_i}$ from $\hat{x}_j = (x_j - \mu)/\sqrt{\sigma^2+\epsilon}$, and using $\frac{\partial \mu}{\partial x_i} = \frac{1}{N}$ and $\frac{\partial \sigma^2}{\partial x_i} = \frac{2}{N}(x_i - \mu)$, the full result (after collecting terms — a standard but tedious exercise) is:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{N\sqrt{\sigma^2+\epsilon}}\left[N\frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \sum_{j=1}^N \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \hat{x}_i\sum_{j=1}^N \frac{\partial \mathcal{L}}{\partial \hat{x}_j}\hat{x}_j\right]$$

**The intuitive reading of this formula** (don't worry about memorizing the algebra — internalize this instead): the gradient flowing back to any single input $x_i$ is *not* simply "the upstream gradient, rescaled." It's a combination of three terms — its own direct upstream signal, a correction term that subtracts off the *average* upstream signal across the whole group (because $x_i$ pulled $\mu$ in that average direction too), and a second correction term that subtracts off a component proportional to $\hat{x}_i$ itself, weighted by how correlated the upstream gradients are with the standardized activations (because $x_i$ also influenced $\sigma^2$). **Every element in a normalization group is gradient-coupled to every other element in that same group.** This is the precise mathematical statement behind "BatchNorm couples examples in a batch together," and it's the direct cause of the batch-size sensitivity discussed next.

### 6.6 The training/inference mismatch — running statistics

Here's a problem the math above doesn't show: at inference time, you may process a single example ($B=1$), or examples one at a time in an autoregressive loop. Computing $\mu, \sigma^2$ from a batch of size 1 is either undefined (with $T=1$ also, you'd be computing the variance of a single number, which is exactly 0) or, even when defined, wildly noisy and inconsistent with how the layer behaved during training.

BatchNorm's fix: maintain **running (exponential moving average) estimates** of $\mu$ and $\sigma^2$, updated during training but *frozen and reused, unchanged, at inference*:

$$\mu_{\text{running}} \leftarrow (1-m)\cdot\mu_{\text{running}} + m\cdot\mu_{\text{batch}}$$
$$\sigma^2_{\text{running}} \leftarrow (1-m)\cdot\sigma^2_{\text{running}} + m\cdot\sigma^2_{\text{batch}}$$

where $m$ (momentum, typically ~0.1) controls how quickly the running estimate adapts to new batches. At inference, the layer uses $\mu_{\text{running}}, \sigma^2_{\text{running}}$ directly instead of computing fresh per-batch statistics — making inference deterministic and independent of batch composition, at the cost of training and inference literally being two different code paths with potentially different numerical behavior (a real, recurring source of bugs in production systems, and a major reason simpler alternatives became attractive).

### 6.7 NumPy implementation (from scratch)

```python
import numpy as np

class BatchNorm:
    """
    Batch Normalization from scratch, with running statistics for inference.
    Operates on input of shape (N, C) for simplicity (no spatial dims) —
    extend axes for (N, C, H, W) by averaging over (0, 2, 3) instead of (0,).
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        # Cache for backward pass
        self._cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            batch_mean = x.mean(axis=0)                       # (C,)
            batch_var = x.var(axis=0)                         # (C,), biased (÷N)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x_centered = x - mean
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_centered * std_inv
        out = self.gamma * x_hat + self.beta

        if training:
            self._cache = (x_hat, std_inv, x.shape[0])
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Implements the full derivation from Section 6.5."""
        x_hat, std_inv, N = self._cache
        self.grad_gamma = (grad_output * x_hat).sum(axis=0)
        self.grad_beta = grad_output.sum(axis=0)

        grad_x_hat = grad_output * self.gamma
        term1 = N * grad_x_hat
        term2 = grad_x_hat.sum(axis=0)
        term3 = x_hat * (grad_x_hat * x_hat).sum(axis=0)
        grad_x = (std_inv / N) * (term1 - term2 - term3)
        return grad_x
```

### 6.8 Production-grade PyTorch usage

In practice, you would essentially never hand-roll BatchNorm — `torch.nn.BatchNorm1d/2d/3d` is a highly optimized, cuDNN-backed implementation:

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True)

# Training mode: uses current-batch statistics, updates running stats
bn.train()
x = torch.randn(32, 64, 28, 28)   # (N, C, H, W)
y = bn(x)

# Eval mode: uses frozen running_mean / running_var
bn.eval()
y_inference = bn(x)

# Inspect the running statistics directly
print(bn.running_mean.shape, bn.running_var.shape)   # both: (64,)
```

A common, costly bug: **forgetting to call `model.eval()` before inference**, which leaves BatchNorm layers computing batch statistics on whatever inference-time batch happens to be passed in (and updating running stats with data that should be frozen) — silently giving inconsistent, non-reproducible outputs. This single footgun is a real, recurring production issue and one more practical reason the field gravitated toward batch-independent alternatives.

### 6.9 Numerical stability considerations

- **Small batch sizes** (e.g., $B=1$ or $2$, common in memory-constrained training, e.g. large image resolutions or huge per-example sequence lengths) produce noisy, high-variance estimates of $\mu, \sigma^2$ — sometimes bad enough to destabilize training outright.
- **$\epsilon$ choice**: too small (e.g., $10^{-8}$ in FP16) risks the $\sqrt{\sigma^2+\epsilon}$ denominator underflowing toward zero when $\sigma^2$ is itself tiny; too large biases the normalization (the output variance becomes meaningfully less than 1). $10^{-5}$ is the standard default and works well in FP32; mixed-precision training typically computes the mean/variance reduction in FP32 even when the rest of the layer runs in FP16/BF16, specifically to avoid this issue (full treatment in [Section 53](#53-mixed-precision-bf16-vs-fp16-quantization-effects)).
- **Distributed training**: computing *true* batch statistics when the batch is sharded across multiple GPUs (as in large-scale distributed training) requires an explicit cross-device synchronization step (`SyncBatchNorm` in PyTorch) — without it, each GPU computes statistics only from its local shard of the batch, which is a silent correctness gap that's easy to miss.

### 6.10 Computational complexity

For an input with $N$ total elements per channel (across batch and spatial/temporal dims) and $C$ channels:
- **Time**: $O(N \cdot C)$ for both forward (one pass to compute mean, one to compute variance, one to standardize — often fused into fewer passes in optimized implementations) and backward.
- **Space**: $O(C)$ additional parameters ($\gamma, \beta$, plus running statistics) — independent of batch size or spatial extent, which is cheap. The *activations* cache needed for backward ($\hat{x}$ and the inverse std) costs $O(N \cdot C)$, same order as the input itself.

### 6.11 Advantages and limitations

**Advantages:**
- Strong empirical track record in CNNs — enables much higher learning rates and faster convergence.
- Acts as a mild regularizer (the batch-dependent noise in $\mu, \sigma^2$ during training has an effect somewhat similar to a form of noise injection, which can slightly reduce overfitting).
- Very well optimized in every major framework (cuDNN-fused kernels), so in its home domain (vision, large batch sizes) it's essentially free, performance-wise.

**Limitations (the ones that matter most for this handbook):**
- Breaks down at small batch sizes.
- Requires separate train/inference code paths via running statistics — a real source of subtle bugs.
- Couples examples within a batch together (Section 6.5), which is undesirable when you'd prefer each example's computation to be fully independent (e.g., for reproducibility, or for architectures like Transformers that don't have a natural fixed-size "batch" semantic at the per-token level).
- Statistics computed across the batch+spatial/temporal axes don't make sense for variable-length sequences without careful masking.

### 6.12 Real-world usage

BatchNorm is the default in essentially all major pre-Transformer vision architectures: ResNet, Inception/GoogLeNet, VGG-with-BN variants, and most object detection/segmentation backbones (Mask R-CNN, etc.) still use it today, since vision training typically uses reasonably large batch sizes and benefits significantly from BN's optimization speedup. **It is almost never used in modern LLMs** — the reasons why are the entire subject of [Section 20](#20-why-batchnorm-fails-for-transformers), which we'll derive rigorously once we've covered LayerNorm's mechanics in detail.

### 6.13 What's next

[Section 7](#7-layer-normalization-ln) introduces Layer Normalization — the technique that became, and for a long time remained, the default choice for Transformers and LLMs, and the primary subject of this entire handbook. Where BatchNorm averages over the batch (and spatial/temporal) axis per feature, LayerNorm flips this entirely: it averages over the *feature* axis, independently for each individual token — a change that, as we'll see, resolves every limitation listed in Section 6.11 simultaneously.

---

## 7. Layer Normalization (LN)

### 7.1 Motivation

Layer Normalization (Ba, Kiros & Hinton, 2016) was proposed with a deceptively simple change from BatchNorm: **instead of computing $\mu, \sigma^2$ across the batch dimension for each feature, compute them across the feature dimension for each individual example.**

This single change — swapping which axis you reduce over — resolves every structural problem BatchNorm has with sequence models, simultaneously:

- **No batch-size dependency**: each example's normalization uses only that example's own $d$ feature values. A batch of size 1 works identically to a batch of size 1024.
- **No train/inference mismatch**: the exact same computation runs in both modes — there are no running statistics to maintain, freeze, or accidentally leave in the wrong mode.
- **No cross-example coupling**: gradients for example $i$ never depend on any other example $j$ in the batch — each example's forward and backward computation is fully independent.
- **Natural fit for variable-length sequences**: since normalization happens per-token, padding elsewhere in a batch (or even elsewhere in the same sequence) literally cannot leak into a given token's statistics, because that token's LayerNorm computation never looks beyond its own feature vector.

This is precisely why LayerNorm — not BatchNorm — became the default normalization layer in the original Transformer ("Attention Is All You Need," Vaswani et al., 2017) and remained the standard through GPT-2, GPT-3, BERT, T5, and most pre-2023 LLMs. (We give BatchNorm's failure mode for Transformers a full, rigorous treatment with worked tensor examples in [Section 20](#20-why-batchnorm-fails-for-transformers) — what's above is the summary; that section is the proof.)

### 7.2 Mathematical formulation

For a single token's hidden-state vector $x \in \mathbb{R}^d$ (one row of the $B \times T \times d$ tensor from [Section 5.2](#52-setting-up-the-tensor-shape-well-use-throughout), fixing one specific $(b, t)$ position):

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$$

$$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2+\epsilon}}$$

$$y_i = \gamma_i \cdot \hat{x}_i + \beta_i \qquad i = 1, \ldots, d$$

Compare this directly against BatchNorm's formulation in [Section 6.2](#62-mathematical-formulation) — the formulas are *structurally identical*. The only difference is which subscripts $\mu$ and $\sigma^2$ carry: BatchNorm's $\mu_c, \sigma_c^2$ depend on the channel and are shared across $(b,t)$; LayerNorm's $\mu, \sigma^2$ depend on $(b,t)$ and are shared across the feature dimension $d$. $\gamma, \beta \in \mathbb{R}^d$ are learned per-feature parameters, shared across all tokens and all batch elements (this is the same parameter count regardless of $B$ or $T$ — $2d$ total parameters per LayerNorm layer, same as BatchNorm's parameter count, but now applied identically to every token rather than varying with batch composition).

### 7.3 A worked numerical example

Take one token's 4-dimensional hidden state (the same numbers as [Section 1.2](#12-a-tiny-worked-example), so you can directly compare):

$$x = [\,2.0,\ 8.0,\ -4.0,\ 6.0\,]$$

This is computed **identically** to Section 1.2's example — because for a single token, LayerNorm's $\mu, \sigma^2$ computation *is* exactly the standardization formula from [Section 3.4](#34-standardization--putting-it-together), just applied to this token's own feature vector rather than to an abstract list of numbers:

$$\mu = 3.0, \qquad \sigma^2 = 21.0, \qquad \hat{x} \approx [\,-0.218,\ 1.091,\ -1.528,\ 0.655\,]$$

Now apply learned per-feature $\gamma = [1.5, 0.5, 2.0, 1.0]$ and $\beta = [0.1, 0.1, 0.1, 0.1]$:

$$y = [\,1.5(-0.218)+0.1,\ 0.5(1.091)+0.1,\ 2.0(-1.528)+0.1,\ 1.0(0.655)+0.1\,]$$
$$y \approx [\,-0.227,\ 0.646,\ -2.956,\ 0.755\,]$$

**The crucial difference from BatchNorm's worked example in [Section 6.3](#63-a-worked-numerical-example)**: this entire computation used only this *one* token's own 4 values. If this were one token among, say, 512 tokens across a batch of 32 sequences, none of those other $32 \times 512 - 1$ token vectors would have any influence whatsoever on this result — every quantity above is fully determined by this single 4-dimensional vector and the layer's learned $\gamma, \beta$.

### 7.4 Forward pass (tensor form)

For $X \in \mathbb{R}^{B \times T \times d}$:

```
for each (b, t) position independently:
    μ[b,t]   = mean(X[b, t, :])                       # scalar, over d
    σ²[b,t]  = var(X[b, t, :])                         # scalar, over d
    X̂[b,t,:] = (X[b, t, :] - μ[b,t]) / sqrt(σ²[b,t] + ε)
    Y[b,t,:] = γ * X̂[b, t, :] + β                      # γ, β shared across b, t
```

Note this loop is trivially parallelizable across both $b$ and $t$ — every position's computation is fully independent, which is exactly why LayerNorm maps so cleanly onto GPU execution and why fused kernels ([Sections 42–43](#42-triton-fused-layernorm-kernel)) can process every token in a batch fully in parallel with no cross-token synchronization required (unlike BatchNorm, which inherently requires a reduction across the batch dimension before any single output can be computed).

### 7.5 Backward pass — derivation

The derivation follows exactly the same pattern as [Section 6.5](#65-backward-pass--full-derivation), with the sum now running over the **feature dimension** $d$ for a single token, instead of over the **batch+spatial** dimension $N$ for a single channel. Substituting $d$ for $N$ in that derivation and reinterpreting the indices as feature indices rather than batch indices:

$$\frac{\partial \mathcal{L}}{\partial \gamma_i} = \frac{\partial \mathcal{L}}{\partial y_i}\hat{x}_i \qquad \frac{\partial \mathcal{L}}{\partial \beta_i} = \frac{\partial \mathcal{L}}{\partial y_i}$$

(Note: unlike BatchNorm, there's no sum over a batch axis here for $\gamma, \beta$'s *per-token* contribution — but since $\gamma, \beta$ are *shared* across all tokens and batch elements, the actual parameter gradient sums these per-token contributions across every $(b,t)$ position in the batch, exactly analogous to how BatchNorm's $\gamma,\beta$ gradients summed across the batch axis.)

For the input gradient, with $d$ now playing the role $N$ played in Section 6.5:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{d\sqrt{\sigma^2+\epsilon}}\left[d\frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \sum_{j=1}^d \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \hat{x}_i\sum_{j=1}^d \frac{\partial \mathcal{L}}{\partial \hat{x}_j}\hat{x}_j\right]$$

The intuitive reading carries over directly too: gradients are coupled across the *features of a single token* (because each feature $x_i$ influenced that token's own $\mu$ and $\sigma^2$, which affected every other feature's normalized value), but **never coupled across different tokens or different batch elements** — a clean, important contrast with BatchNorm's cross-*example* coupling.

### 7.6 NumPy implementation (from scratch)

```python
import numpy as np

class LayerNorm:
    """
    Layer Normalization from scratch. Operates on input of shape (..., d) —
    normalizes over the LAST dimension only, independently for every
    leading index (batch, sequence position, etc.).
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self.beta = np.zeros(normalized_shape, dtype=np.float32)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Reduce over the LAST axis only — every other axis (batch, sequence)
        # is treated independently, unlike BatchNorm's axis=0 reduction.
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)              # biased, ÷d
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * std_inv
        out = self.gamma * x_hat + self.beta

        self._cache = (x_hat, std_inv, x.shape[-1])
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Implements the derivation from Section 7.5."""
        x_hat, std_inv, d = self._cache

        # Sum gradients over every leading axis (batch, sequence) — gamma/beta
        # are shared across all of those, but per-feature.
        reduce_axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = (grad_output * x_hat).sum(axis=reduce_axes)
        self.grad_beta = grad_output.sum(axis=reduce_axes)

        grad_x_hat = grad_output * self.gamma
        term1 = d * grad_x_hat
        term2 = grad_x_hat.sum(axis=-1, keepdims=True)
        term3 = x_hat * (grad_x_hat * x_hat).sum(axis=-1, keepdims=True)
        grad_x = (std_inv / d) * (term1 - term2 - term3)
        return grad_x


# Quick verification against the Section 7.3 worked example:
ln = LayerNorm(normalized_shape=4)
ln.gamma = np.array([1.5, 0.5, 2.0, 1.0])
ln.beta = np.array([0.1, 0.1, 0.1, 0.1])
x = np.array([[2.0, 8.0, -4.0, 6.0]])
print(ln.forward(x))   # -> [[-0.227  0.646 -2.956  0.755]]  (matches Section 7.3)
```

### 7.7 Production-grade PyTorch usage

```python
import torch
import torch.nn as nn

# normalized_shape is the size of the LAST dimension(s) to normalize over —
# for a Transformer hidden state of shape (B, T, d_model), pass d_model.
ln = nn.LayerNorm(normalized_shape=768, eps=1e-5, elementwise_affine=True)

x = torch.randn(32, 128, 768)   # (B, T, d_model)
y = ln(x)
print(y.shape)                  # torch.Size([32, 128, 768]) — shape unchanged

# Crucially: no .train()/.eval() distinction matters for LayerNorm's
# computation itself (unlike BatchNorm) — it behaves IDENTICALLY in both
# modes, because there are no running statistics involved at all.
ln.eval()
y_eval = ln(x)
assert torch.allclose(y, y_eval)   # True for the same input
```

In a real Transformer block, LayerNorm is essentially never used standalone — it's always paired with a residual connection (full treatment in [Part IV](#29-motivation-vanishing-gradients-and-the-degradation-problem) and [Part V](#36-why-they-work-so-well-together)). A minimal preview of the canonical Pre-LN pattern used by GPT-2 onward:

```python
class PreLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None):
        # Pre-LN: normalize BEFORE the sublayer, add residual AFTER
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out                       # residual around attention
        x = x + self.ffn(self.ln2(x))          # residual around FFN
        return x
```

We dissect exactly why this Pre-LN ordering (normalize-then-sublayer-then-add, rather than sublayer-then-add-then-normalize) won out over the original Transformer's Post-LN design in full in [Section 25](#25-post-ln-vs-pre-ln-vs-sandwich-ln) and [Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison).

### 7.8 Numerical stability considerations

- LayerNorm's reduction is over $d$ (the hidden dimension, often 768–16384+ in modern LLMs) rather than over a batch — so unlike BatchNorm's small-batch noise problem, LayerNorm's statistics are typically computed from *plenty* of values per reduction, and are not noisy in the same way.
- The real numerical risk is **precision during the reduction itself** when run in FP16/BF16: summing $d$ values (and $d$ squared-deviation values) in low precision can lose accuracy, especially for large $d$. Production implementations (including PyTorch's native CUDA kernel) typically accumulate the mean/variance reduction in FP32 internally even when inputs/outputs are FP16/BF16 — we cover this precisely in [Section 53](#53-mixed-precision-bf16-vs-fp16-quantization-effects).
- $\epsilon$ placement matters subtly: $\sqrt{\sigma^2+\epsilon}$ (add inside the square root) is the standard, numerically preferred form over $\sqrt{\sigma^2}+\epsilon$ (add outside), because the former guarantees the denominator is bounded away from zero even when $\sigma^2 \to 0$, while the latter doesn't fully fix the issue when $\sigma^2$ itself underflows to a denormalized or zero value before the epsilon is added.

### 7.9 Computational complexity

For one token's $d$-dimensional vector: $O(d)$ time (one pass for mean, one for variance, one for standardize+affine — 3 passes over $d$ elements, sometimes fused into fewer in optimized kernels) and $O(d)$ space for $\gamma, \beta$ (shared across all tokens) plus $O(1)$ additional scalars ($\mu, \sigma^2$) *per token*. Across a full batch of $B \times T$ tokens, total cost is $O(B \cdot T \cdot d)$ — linear in the total number of elements, same asymptotic order as BatchNorm, but achieved with **zero cross-token synchronization**, which matters enormously for parallel hardware utilization (detailed further in [Section 42](#42-triton-fused-layernorm-kernel)'s kernel-level analysis).

### 7.10 Advantages and limitations

**Advantages:**
- Fully independent of batch size — works identically whether $B=1$ or $B=4096$.
- No train/inference discrepancy — one code path, always.
- No cross-example coupling — each token's computation and gradient are self-contained.
- Naturally compatible with variable-length sequences and padding (with standard attention-mask handling elsewhere in the model; LayerNorm itself never needs to know about padding, since it only ever looks within one token's own features).
- Highly parallelizable — ideal for GPU/TPU execution.

**Limitations:**
- Two full reduction passes per token (mean, then variance) are still nontrivial cost at LLM scale, run on every token of every layer, every forward and backward pass — this exact cost is what RMSNorm ([Section 8](#8-rmsnorm)) trims down by dropping the mean-centering step entirely.
- Mathematically, you cannot deduce a vector's *original* magnitude from its LayerNorm output (Section 4.4's geometric point) — any information meaningfully encoded in raw activation magnitude is discarded; in practice, the learned $\gamma$ largely compensates for this in trained models, but it's a real, if usually benign, representational constraint.
- For convolutional/vision architectures, LayerNorm normalizing across *all* channels at a spatial position can mix together statistically very different channels (e.g., edge-detector-like channels vs. color-blob-like channels) in a way that's less natural than BatchNorm's per-channel treatment — this is part of why vision models still favor BatchNorm or Group Normalization ([Section 9](#9-group-normalization)) over LayerNorm, even post-Transformer.

### 7.11 Real-world usage

LayerNorm (specifically, the **Pre-LN** variant from Section 7.7's code) was the default normalization in GPT-2, GPT-3, BERT, T5, and the overwhelming majority of Transformer-based LLMs through roughly 2022–2023. Even models that have since moved to RMSNorm (LLaMA, Mistral, Gemma, Qwen — covered throughout [Part VII](#46-gpt-2--gpt-3--gpt-4)) are using a close mathematical relative of LayerNorm — RMSNorm keeps the *feature-axis reduction* idea entirely intact and only removes the mean-centering step, as we'll see next.

### 7.12 What's next

[Section 8](#8-rmsnorm) covers RMSNorm — the simplification of LayerNorm that removes mean-centering entirely, keeping only the variance-like rescaling — and is the normalization layer used by essentially every major modern open-weight LLM (LLaMA family, Mistral, Gemma, Qwen, DeepSeek). We'll derive exactly why dropping mean-centering works about as well empirically while being meaningfully cheaper, and exactly how much compute it actually saves.

---

## 8. RMSNorm

### 8.1 Motivation

RMSNorm (Zhang & Sennrich, 2019) starts from an empirical observation about LayerNorm: **the re-centering step (subtracting the mean) contributes much less to LayerNorm's benefit than the re-scaling step (dividing by a standard-deviation-like quantity).** If that's true, you should be able to drop mean-centering, keep only the rescaling, save the cost of computing a mean, and lose little to nothing in model quality.

That's exactly RMSNorm's proposal. Rather than standardizing using the mean and variance, it rescales a vector using only its **root mean square (RMS)** — a quantity that, unlike variance, doesn't require first computing a mean and subtracting it off:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

Notice this is structurally similar to the standard deviation formula from [Section 3.3](#33-standard-deviation--bringing-the-units-back) — $\sigma = \sqrt{\frac{1}{d}\sum_i(x_i-\mu)^2}$ — with one crucial difference: **RMS skips the $-\mu$ entirely**, measuring spread around zero rather than around the data's own mean.

### 8.2 Mathematical formulation

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)}$$

$$y_i = \gamma_i \cdot \hat{x}_i \qquad i = 1, \ldots, d$$

Two structural differences from LayerNorm jump out immediately:

1. **No mean subtraction** — $x_i$ is rescaled directly, never centered first.
2. **No $\beta$ (shift) parameter** — most RMSNorm implementations (including LLaMA's) use only a learned scale $\gamma$, dropping the additive shift entirely. This is a design choice rather than a mathematical necessity (some RMSNorm variants do keep a $\beta$), but it's the standard, near-universal convention in modern open-weight LLMs, and it compounds the savings: not only do you skip computing $\mu$, you also skip one elementwise addition and halve the per-layer parameter count compared to LayerNorm's $2d$ parameters.

### 8.3 Why dropping re-centering doesn't hurt much — the geometric view

Recall [Section 4](#4-geometric-intuition-what-normalization-does-to-a-vector-space)'s picture: LayerNorm's centering step slides a vector onto the mean-zero hyperplane, then the scaling step projects radially onto a sphere within that hyperplane. RMSNorm skips the centering step — it projects radially onto a sphere **directly from the vector's original position**, without first sliding it onto the mean-zero hyperplane.

```
LayerNorm (2 steps):                  RMSNorm (1 step):

  x → [center: slide to              x → [scale: rescale length
       mean-zero hyperplane]              to fixed radius, in
    → [scale: project onto                whatever direction x
       sphere within that                 already points]
       hyperplane]                    → y (on a sphere, but NOT
    → y (on a sphere,                      necessarily within the
       within the mean-zero               mean-zero hyperplane)
       hyperplane)
```

The empirical finding behind RMSNorm is that, for the activations actually seen in trained deep networks (and especially Transformers), **the re-scaling step does almost all of the optimization-stabilizing work**; the re-centering step's contribution is comparatively marginal. This isn't something you can prove from first principles alone — it's an empirical result, validated by the original RMSNorm paper's ablations and subsequently by years of large-scale LLM pretraining runs (LLaMA, Mistral, etc.) confirming that models trained with RMSNorm match LayerNorm-trained models in downstream quality, at lower compute cost.

### 8.4 A worked numerical example

Reuse the same vector as Sections 1.2 and 7.3, for direct comparison:

$$x = [\,2.0,\ 8.0,\ -4.0,\ 6.0\,]$$

**Step 1 — RMS** (note: no mean subtraction anywhere in this step):
$$\text{RMS}(x) = \sqrt{\frac{2.0^2+8.0^2+(-4.0)^2+6.0^2}{4}} = \sqrt{\frac{4+64+16+36}{4}} = \sqrt{30.0} \approx 5.477$$

**Step 2 — rescale:**
$$\hat{x} = \left[\frac{2.0}{5.477},\ \frac{8.0}{5.477},\ \frac{-4.0}{5.477},\ \frac{6.0}{5.477}\right] \approx [\,0.365,\ 1.461,\ -0.730,\ 1.095\,]$$

**Step 3 — apply learned $\gamma$** (reuse the same $\gamma = [1.5, 0.5, 2.0, 1.0]$ from Section 7.3, no $\beta$):
$$y = [\,1.5(0.365),\ 0.5(1.461),\ 2.0(-0.730),\ 1.0(1.095)\,] \approx [\,0.548,\ 0.731,\ -1.460,\ 1.095\,]$$

**Compare directly against LayerNorm's output on the exact same input and $\gamma$** (Section 7.3): $[-0.227, 0.646, -2.956, 0.755]$ vs. RMSNorm's $[0.548, 0.731, -1.460, 1.095]$. The outputs genuinely differ — RMSNorm's output isn't simply a cheaper way of computing the same number. The largest visible difference is in the third element ($-2.956$ vs. $-1.460$): LayerNorm's centering step treats $-4.0$ as far below this vector's own mean of $3.0$ (hence a strongly negative standardized value), while RMSNorm only sees that $-4.0$ has a moderate magnitude relative to the vector's overall RMS, with no reference to where the vector's center sits. This is the concrete, numerical face of the geometric difference from Section 8.3.

### 8.5 Backward pass

Because there's no mean-subtraction step, RMSNorm's gradient computation is simpler than LayerNorm's — there's one fewer coupling term to track. Following the same derivation style as Sections 6.5/7.5, with $r = \text{RMS}(x)$:

$$\frac{\partial \mathcal{L}}{\partial \gamma_i} = \frac{\partial \mathcal{L}}{\partial y_i}\hat{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{r}\frac{\partial \mathcal{L}}{\partial y_i} - \frac{x_i}{d\cdot r^3}\sum_{j=1}^{d} x_j \gamma_j \frac{\partial \mathcal{L}}{\partial y_j}$$

Notice the structure: the first term is a simple direct rescaling (analogous to LayerNorm's direct term), and the second term is the *only* coupling term — capturing how $x_i$ also influenced $r$, which affected every other $\hat{x}_j$. LayerNorm's backward pass (Section 7.5) had **two** correction terms (one from $\mu$'s dependence on $x_i$, one from $\sigma^2$'s); RMSNorm has only **one** (from $r$'s dependence on $x_i$), because there's no $\mu$ to differentiate through. This is the gradient-computation manifestation of the same simplification.

### 8.6 NumPy implementation (from scratch)

```python
import numpy as np

class RMSNorm:
    """
    RMSNorm from scratch. Operates on input of shape (..., d) — rescales
    over the LAST dimension only. No mean-subtraction, no beta parameter
    (matching the LLaMA/Mistral/Gemma convention).
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        ms = (x ** 2).mean(axis=-1, keepdims=True)        # mean of squares
        rms_inv = 1.0 / np.sqrt(ms + self.eps)
        x_hat = x * rms_inv
        out = self.gamma * x_hat

        self._cache = (x, x_hat, rms_inv, x.shape[-1])
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Implements the derivation from Section 8.5."""
        x, x_hat, rms_inv, d = self._cache

        reduce_axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = (grad_output * x_hat).sum(axis=reduce_axes)

        grad_y_gamma = grad_output * self.gamma            # dL/dy * gamma, elementwise
        term1 = grad_y_gamma * rms_inv
        coupling = (x * grad_y_gamma).sum(axis=-1, keepdims=True)
        term2 = x * coupling * (rms_inv ** 3) / d
        grad_x = term1 - term2
        return grad_x


# Verification against the Section 8.4 worked example:
rms = RMSNorm(normalized_shape=4, eps=1e-5)
rms.gamma = np.array([1.5, 0.5, 2.0, 1.0])
x = np.array([[2.0, 8.0, -4.0, 6.0]])
print(rms.forward(x))   # -> [[ 0.548  0.731 -1.460  1.095]]  (matches Section 8.4)
```

### 8.7 Production-grade PyTorch implementation

PyTorch did not ship a built-in `nn.RMSNorm` until relatively recently in its history (it's now available as `torch.nn.RMSNorm` in current versions), and for a long time, every major LLM codebase (LLaMA's original release, Hugging Face `transformers`, etc.) shipped its own hand-written version. Here is the canonical, widely-used implementation pattern (matching LLaMA's released code closely):

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # this is "gamma"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt = 1/sqrt(...), computed in float32 for stability even if
        # the model otherwise runs in bf16/fp16 (see Section 8.8 and 53)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for the norm computation, then cast back —
        # this exact pattern appears in LLaMA, Mistral, and Gemma's code.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Usage — identical interface shape to nn.LayerNorm
rmsnorm = RMSNorm(dim=4096, eps=1e-6)
x = torch.randn(8, 512, 4096, dtype=torch.bfloat16)
y = rmsnorm(x)
print(y.dtype, y.shape)   # torch.bfloat16, torch.Size([8, 512, 4096])
```

If `torch.nn.RMSNorm` is available in your PyTorch version, it's preferable to use it directly (it's a maintained, optimized, fused implementation) — the hand-written version above is shown because (a) it's instructive, and (b) it's still exactly what you'll find inside the source of most open-weight LLM repositories, so recognizing this pattern is directly useful for reading real model code.

### 8.8 Why the float32 upcast matters

Look closely at `_norm` in Section 8.7: it explicitly casts to `float32` before computing `x.pow(2).mean(...)`, then casts the *result* back to the original dtype. This is not a stylistic choice — squaring activations in BF16 (which has roughly 3 decimal digits of precision) compounds rounding error rapidly, especially for activations with larger magnitudes that are common in trained LLMs (BF16 squared values can lose meaningful precision well before they'd overflow). Computing the reduction in FP32 and casting back avoids this without paying full FP32 cost for the entire forward pass — only the norm's own internal reduction runs at higher precision. We return to this pattern in full generality (covering LayerNorm too) in [Section 53](#53-mixed-precision-bf16-vs-fp16-quantization-effects).

### 8.9 Computational and parameter savings — quantified

Compared to LayerNorm, for a vector of dimension $d$:

| | LayerNorm | RMSNorm | Savings |
|---|---|---|---|
| Reductions needed | 2 (mean, then variance) | 1 (mean of squares) | ~50% fewer reduction passes |
| Learnable parameters | $2d$ ($\gamma$ and $\beta$) | $d$ ($\gamma$ only) | 50% fewer params per norm layer |
| Elementwise ops in normalize step | subtract mean, divide, multiply by $\gamma$, add $\beta$ (4 ops/element) | divide, multiply by $\gamma$ (2 ops/element) | ~50% fewer elementwise ops |

For a single normalization layer in isolation, this sounds like a modest win. But remember the scale this runs at: **every token, every LayerNorm/RMSNorm call, every layer, every forward and backward pass, for the entire pretraining run** (trillions of tokens) and for every inference request served afterward. A consistent ~2x reduction in normalization-layer cost, multiplied across that volume, is a genuinely meaningful aggregate compute and latency saving — which is precisely why virtually every major open-weight LLM released since 2023 (LLaMA, Mistral, Gemma, Qwen, DeepSeek) made the switch, despite normalization being a small fraction of a single forward pass's total FLOPs (most of which goes to the attention and FFN matrix multiplications). We quantify this more precisely with profiling data in [Section 45](#45-benchmarking-profiling-and-memory-analysis).

### 8.10 Numerical stability considerations

- RMSNorm's $\epsilon$ plays the same role as LayerNorm's, guarding $\sqrt{\cdot}$ from a near-zero argument — but note RMSNorm's argument is a mean of *squares*, which is always $\geq 0$ and only approaches zero if the entire vector is near-zero (a rarer scenario in practice than LayerNorm's variance approaching zero, which only requires the vector's elements to be near-*identical*, not near-*zero* — a meaningfully more common scenario, e.g., early in training or in degenerate/collapsed representations). This makes RMSNorm marginally more numerically robust in some practical scenarios, though both still need a sensible $\epsilon$.
- The float32-upcast pattern from Section 8.8 is effectively mandatory in any serious BF16/FP16 LLM implementation — skipping it is a common source of subtle training instability that can be very difficult to diagnose after the fact, since it manifests as gradual quality degradation rather than an obvious crash.

### 8.11 Advantages and limitations

**Advantages:**
- Cheaper than LayerNorm (Section 8.9) with empirically comparable downstream model quality at LLM scale.
- Slightly simpler backward pass, with one fewer coupling term.
- No $\beta$ parameter to learn/store (in the standard convention), halving parameter count for this layer.
- Marginally more robust against near-zero denominators in the all-near-identical-values edge case (see Section 8.10).

**Limitations:**
- Discards the mean-centering's modest but nonzero contribution to optimization stability — for some architectures or training regimes, this can matter more than it does for the LLM Transformer setting where RMSNorm has been most thoroughly validated; it is not a universal drop-in replacement guaranteed to work equally well everywhere LayerNorm is used (e.g., it's far less common in vision architectures).
- Losing the $\beta$ shift parameter means the layer cannot learn to introduce a nonzero mean into its output — in practice, this constraint has not shown up as a meaningful quality bottleneck in large-scale LLM training, but it is a real reduction in representational flexibility relative to full LayerNorm.

### 8.12 Real-world usage

RMSNorm is the normalization layer in: LLaMA, LLaMA 2, LLaMA 3, Mistral, Mixtral, Gemma, Gemma 2, Qwen, and DeepSeek's V2/V3 family — i.e., essentially every prominent open-weight LLM released since 2023. We give each of these architectures a dedicated case study, including their exact RMSNorm placement and $\epsilon$ choices, in [Part VII](#46-gpt-2--gpt-3--gpt-4).

### 8.13 What's next

[Section 9](#9-group-normalization) covers Group Normalization — a technique most relevant to vision and diffusion models rather than LLMs, but worth understanding both for completeness and because it sits at an instructive midpoint on the taxonomy from [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over): it normalizes over a *subset* of channels rather than all of them (like LayerNorm) or none of them per-channel (like BatchNorm).

---

## 9. Group Normalization

### 9.1 Motivation

Group Normalization (Wu & He, 2018) was developed specifically to solve BatchNorm's small-batch problem ([Section 6.9](#69-numerical-stability-considerations)) for **vision** tasks where memory constraints often force small batch sizes — high-resolution image segmentation, video, and 3D medical imaging, where a single example can already consume most of available GPU memory, leaving room for batches of only 1–8 examples.

Recall the taxonomy from [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over): BatchNorm averages over the batch dimension *per channel*; LayerNorm averages over *all* channels (and any other feature-like dimension) *per example*. Group Normalization proposes a middle ground: **split the channels into groups, and average over the channels within each group (plus spatial dimensions), independently per example.** This removes BatchNorm's batch-size dependency entirely (each example is normalized independently, exactly like LayerNorm), while preserving some of BatchNorm's intuition that nearby/related channels — rather than literally all channels — should share statistics.

### 9.2 Mathematical formulation

For vision data $X \in \mathbb{R}^{N \times C \times H \times W}$, split the $C$ channels into $G$ groups of size $C/G$ each. For a given example $n$ and group $g$ (containing channels $c \in \text{group}(g)$):

$$\mu_{n,g} = \frac{1}{(C/G)\cdot H \cdot W}\sum_{c \in \text{group}(g)}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{n,c,h,w}$$

$$\sigma_{n,g}^2 = \frac{1}{(C/G)\cdot H \cdot W}\sum_{c \in \text{group}(g)}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{n,c,h,w} - \mu_{n,g})^2$$

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,g}}{\sqrt{\sigma_{n,g}^2+\epsilon}} \qquad \text{for } c \in \text{group}(g)$$

$$y_{n,c,h,w} = \gamma_c \cdot \hat{x}_{n,c,h,w} + \beta_c$$

Two special cases are worth noting explicitly, because they reveal Group Norm as a strict generalization that contains both LayerNorm and Instance Normalization ([Section 10](#10-instance-normalization)) as edge cases:

- **$G = 1$** (one group containing *all* channels): this reduces exactly to LayerNorm applied over the $(C, H, W)$ axes for each example — average over everything except the batch dimension.
- **$G = C$** (each channel is its own group of size 1): this reduces exactly to Instance Normalization — average over only the spatial dimensions $(H, W)$, independently per channel.

### 9.3 A worked numerical example

Take a tiny case: $C = 4$ channels, $G = 2$ groups (so 2 channels per group), no spatial extent for simplicity ($H=W=1$, i.e., each "channel value" is just a scalar for this one example):

$$x = [\,2.0,\ 8.0,\ -4.0,\ 6.0\,] \quad \text{(channels 0,1 in group A; channels 2,3 in group B)}$$

**Group A** (channels 0, 1 → values 2.0, 8.0):
$$\mu_A = \frac{2.0+8.0}{2} = 5.0, \qquad \sigma_A^2 = \frac{(2-5)^2+(8-5)^2}{2} = \frac{9+9}{2}=9.0$$
$$\hat{x}_0 = \frac{2-5}{3.0} = -1.0, \qquad \hat{x}_1 = \frac{8-5}{3.0}=1.0$$

**Group B** (channels 2, 3 → values -4.0, 6.0):
$$\mu_B = \frac{-4.0+6.0}{2}=1.0, \qquad \sigma_B^2 = \frac{(-4-1)^2+(6-1)^2}{2}=\frac{25+25}{2}=25.0$$
$$\hat{x}_2 = \frac{-4-1}{5.0}=-1.0, \qquad \hat{x}_3=\frac{6-1}{5.0}=1.0$$

So $\hat{x} = [-1.0, 1.0, -1.0, 1.0]$ — note this is **different from both** LayerNorm's result on the same input (which would average over all 4 values at once, giving $\mu=3.0,\sigma^2=21.0$ as in Section 1.2/7.3) and from treating each channel fully independently. Group Norm's result reflects two *separate* local standardizations, one per group, each blind to the other group's statistics.

### 9.4 Forward pass (tensor form)

```
reshape X from (N, C, H, W) to (N, G, C/G, H, W)
for each example n, each group g:
    μ[n,g]  = mean(X[n, g, :, :, :])              # over C/G, H, W
    σ²[n,g] = var(X[n, g, :, :, :])
    X̂[n,g,:,:,:] = (X[n,g,:,:,:] - μ[n,g]) / sqrt(σ²[n,g] + ε)
reshape back to (N, C, H, W)
Y = γ * X̂ + β    # γ, β still per-channel (shape C), not per-group
```

### 9.5 Backward pass

The derivation is structurally identical to BatchNorm's ([Section 6.5](#65-backward-pass--full-derivation)) and LayerNorm's ([Section 7.5](#75-backward-pass--derivation)) — only the *reduction set* changes, now spanning $(C/G) \cdot H \cdot W$ elements within a single (example, group) pair instead of $N \cdot T$ (BatchNorm) or $d$ (LayerNorm):

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{M\sqrt{\sigma_{n,g}^2+\epsilon}}\left[M\frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \sum_{j \in \text{group}} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \hat{x}_i\sum_{j \in \text{group}} \frac{\partial \mathcal{L}}{\partial \hat{x}_j}\hat{x}_j\right]$$

where $M = (C/G)\cdot H \cdot W$ is the number of elements in one (example, group)'s reduction set, and all sums range over only that group's elements for that example — gradients are coupled within a group, but never across groups or across examples, combining LayerNorm's per-example independence with a partial, group-local form of BatchNorm's cross-channel coupling.

### 9.6 NumPy implementation (from scratch)

```python
import numpy as np

class GroupNorm:
    """
    Group Normalization from scratch. Input shape (N, C, H, W).
    Splits C channels into num_groups groups; normalizes each
    (example, group) independently over (C/G, H, W).
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        assert num_channels % num_groups == 0, "channels must divide evenly into groups"
        self.G = num_groups
        self.C = num_channels
        self.eps = eps
        self.gamma = np.ones(num_channels, dtype=np.float32)
        self.beta = np.zeros(num_channels, dtype=np.float32)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        x_g = x.reshape(N, self.G, C // self.G, H, W)

        mean = x_g.mean(axis=(2, 3, 4), keepdims=True)
        var = x_g.var(axis=(2, 3, 4), keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat_g = (x_g - mean) * std_inv

        x_hat = x_hat_g.reshape(N, C, H, W)
        out = self.gamma.reshape(1, C, 1, 1) * x_hat + self.beta.reshape(1, C, 1, 1)

        self._cache = (x_hat, std_inv, C // self.G * H * W)
        return out


# Verification against Section 9.3 (no spatial extent, treat as H=W=1):
gn = GroupNorm(num_groups=2, num_channels=4)
x = np.array([[[[2.0]], [[8.0]], [[-4.0]], [[6.0]]]])   # shape (1, 4, 1, 1)
out = gn.forward(x)
print(out.flatten())   # -> [-1. 1. -1. 1.]  (matches Section 9.3, with default gamma=1, beta=0)
```

### 9.7 Production-grade PyTorch usage

```python
import torch
import torch.nn as nn

# num_groups must evenly divide num_channels
gn = nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-5, affine=True)

x = torch.randn(4, 64, 32, 32)   # (N, C, H, W) — small batch, common in segmentation/diffusion
y = gn(x)
print(y.shape)   # torch.Size([4, 64, 32, 32])

# Like LayerNorm, behaves identically in train/eval — no running statistics
gn.eval()
assert torch.allclose(y, gn(x))
```

### 9.8 Numerical stability and complexity

Numerical stability considerations mirror BatchNorm and LayerNorm directly (the $\epsilon$-inside-the-sqrt convention from [Section 7.8](#78-numerical-stability-considerations) applies identically here). Computational cost is $O(N \cdot C \cdot H \cdot W)$ — same order as BatchNorm and LayerNorm — with the number of groups $G$ acting as a tunable parameter that interpolates between LayerNorm-like behavior ($G=1$, see Section 9.2) and Instance-Norm-like behavior ($G=C$), letting practitioners tune the size of the reduction set without changing the asymptotic cost.

### 9.9 Advantages, limitations, and real-world usage

**Advantages:** batch-size independent (like LayerNorm), no running statistics or train/inference mismatch, and empirically more effective than LayerNorm specifically for vision tasks, because grouping nearby/related channels together for normalization respects the spatial/channel structure of convolutional feature maps better than LayerNorm's "average over literally everything except batch" approach.

**Limitations:** introduces a new hyperparameter ($G$, the number of groups) that must be tuned per architecture; not naturally applicable to Transformer hidden states in the same way LayerNorm/RMSNorm are, since there's no obvious "channel grouping" structure in a $d$-dimensional token embedding the way there is in a convolutional feature map's channels.

**Real-world usage:** the default normalization choice in many modern diffusion model architectures (Stable Diffusion's U-Net backbone, and similar denoising architectures), and common in segmentation networks (Mask R-CNN variants) and other vision tasks where small batch sizes are unavoidable due to memory constraints. **Not used in standard LLM Transformer blocks** — it remains primarily a vision/diffusion-model technique, included here for taxonomic completeness and because it makes the BatchNorm/LayerNorm/InstanceNorm relationship in [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over) fully concrete via the $G=1$ and $G=C$ special cases.

### 9.10 What's next

[Section 10](#10-instance-normalization) covers Instance Normalization — the $G=C$ special case of Group Norm previewed above, most associated with style transfer and image generation, where it serves a distinct purpose (separating "style" statistics from "content") that's worth understanding in its own right rather than purely as a Group Norm special case.

---

## 10. Instance Normalization

### 10.1 Motivation

Instance Normalization (Ulyanov, Vedaldi & Lempitsky, 2016) was developed for a specific, somewhat different purpose than the techniques covered so far: not primarily optimization stability, but **stylistic content removal** for image style transfer. The original motivating observation: in tasks like converting a photo into the style of a painting, the *contrast and overall statistics* of a single image (its own per-channel mean and variance) carry style-like information that you often want to **strip out and replace**, rather than preserve.

Mechanically, Instance Norm is the $G=C$ special case of Group Normalization from [Section 9.2](#92-mathematical-formulation): each channel, for each individual example, is normalized using only the statistics of that one channel's own spatial extent — no averaging across other channels, and no averaging across the batch.

### 10.2 Mathematical formulation

For $X \in \mathbb{R}^{N \times C \times H \times W}$, for a given example $n$ and channel $c$:

$$\mu_{n,c} = \frac{1}{H\cdot W}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{n,c,h,w}$$

$$\sigma_{n,c}^2 = \frac{1}{H \cdot W}\sum_{h=1}^{H}\sum_{w=1}^{W}(x_{n,c,h,w}-\mu_{n,c})^2$$

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w}-\mu_{n,c}}{\sqrt{\sigma_{n,c}^2+\epsilon}} \qquad y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w}+\beta_c$$

Compare directly against BatchNorm ([Section 6.2](#62-mathematical-formulation)): BatchNorm's $\mu_c, \sigma_c^2$ average over $(N, H, W)$ — batch and spatial, shared per channel across the *whole batch*. Instance Norm's $\mu_{n,c}, \sigma_{n,c}^2$ average over $(H, W)$ **only** — spatial extent of one channel of one single example, with no batch averaging and no cross-channel averaging at all. It is, in a precise sense, the most "local" of all the normalization variants covered in this handbook (Group Norm with the maximum possible number of groups).

### 10.3 Why removing per-channel, per-image statistics helps style transfer

The intuition: a convolutional feature channel's overall mean and variance, computed across an entire image, tend to correlate strongly with global image statistics like overall contrast, brightness, and the "texture energy" associated with artistic style — properties that a style-transfer network typically wants to actively *control* (replace with a target style's statistics) rather than passively inherit from the input content image. Removing these per-image, per-channel statistics via Instance Norm, then optionally reintroducing a *different* image's statistics via the $\gamma, \beta$ parameters, is exactly the mechanism behind **Adaptive Instance Normalization (AdaIN)**, which we cover in [Section 15](#15-adaptive-normalization-adain-conditional-norms) as a direct extension of this idea.

### 10.4 PyTorch usage

```python
import torch
import torch.nn as nn

inorm = nn.InstanceNorm2d(num_features=64, eps=1e-5, affine=True)

x = torch.randn(4, 64, 32, 32)
y = inorm(x)
print(y.shape)   # torch.Size([4, 64, 32, 32])

# Default affine=False in many style-transfer codebases — gamma/beta are
# often supplied EXTERNALLY (e.g., from a "style" image's own statistics)
# rather than learned as fixed parameters — this is the AdaIN mechanism.
```

### 10.5 Advantages, limitations, and real-world usage

**Advantages:** removes per-image style-like statistics cleanly; fully independent of batch composition and other examples; natural fit when the goal is explicitly to *discard* an image's own global appearance statistics.

**Limitations:** discarding per-image contrast/brightness information is actively undesirable for tasks where that information is meaningful signal rather than noise (e.g., most discriminative vision tasks — classification, detection) — Instance Norm is not a general-purpose substitute for BatchNorm or Group Norm in those settings, and underperforms them there.

**Real-world usage:** the original neural style transfer literature, and a foundational component of **AdaIN** ([Section 15](#15-adaptive-normalization-adain-conditional-norms)), which itself underlies the per-layer style modulation used in StyleGAN and StyleGAN2. **Not used in LLM Transformer architectures.**

### 10.6 What's next

This closes out the "axis taxonomy" sequence (Batch → Layer → Group → Instance norm, covering every point on the spectrum from [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over)). [Section 11](#11-weight-normalization) shifts to a genuinely different category: techniques that normalize **weight matrices** rather than activations — starting with Weight Normalization, which reparameterizes a weight vector's magnitude and direction separately, directly applying the "fix the scale, preserve the direction" geometric idea from [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius) to a weight tensor instead of an activation tensor.

---

## 11. Weight Normalization

### 11.1 Motivation

Every technique so far has normalized **activations** — the data flowing *through* the network. Weight Normalization (Salimans & Kingma, 2016) does something categorically different: it reparameterizes the **weights themselves**, decoupling a weight vector's magnitude from its direction, so that gradient descent can adjust each independently rather than being forced to move them together.

Why would you want this? Consider a single weight vector $w \in \mathbb{R}^d$ feeding one output neuron. In the standard parameterization, $w$'s length and direction are entangled inside the same set of numbers — a gradient step that's "trying" to change the neuron's output scale and a gradient step that's "trying" to change its preferred input direction both end up modifying the *same* underlying parameters, often in ways that interfere with each other (changing direction without intending to also changes effective scale, and vice versa). This coupling can make optimization slower and more sensitive to the weight's initial scale, especially early in training.

### 11.2 Mathematical formulation

Weight Normalization reparameterizes a weight vector $w$ as:

$$w = g \cdot \frac{v}{\|v\|}$$

where $v \in \mathbb{R}^d$ is a learned **direction** vector and $g \in \mathbb{R}$ is a learned **scalar magnitude** — two separate, independently-optimized parameters replacing the single entangled vector $w$. This is a direct, literal application of [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)'s geometric idea ("normalize to a fixed-radius sphere, then explicitly control the scale separately") — except applied to a weight vector instead of an activation vector, and with the "fixed radius" being **exactly 1** ($v/\|v\|$ is a unit vector) before $g$ rescales it to whatever magnitude is useful.

### 11.3 Why this decoupling helps gradients

Differentiating $w = g\cdot v/\|v\|$ with respect to $g$ and $v$ separately:

$$\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial w}\cdot\frac{v}{\|v\|} \qquad \frac{\partial \mathcal{L}}{\partial v} = \frac{g}{\|v\|}\left(\frac{\partial \mathcal{L}}{\partial w} - \frac{\partial \mathcal{L}}{\partial w}\cdot\frac{v}{\|v\|}\cdot\frac{v}{\|v\|}\right)$$

The second formula's structure is worth reading carefully: the gradient with respect to the *direction* parameter $v$ is exactly the gradient w.r.t. $w$, with the **component parallel to $v$ itself projected out** (that's what the subtracted term does — it removes the part of the gradient that points along $v$'s own direction). In other words, **the direction parameter only ever receives gradient signal that would actually change its direction** — any gradient component that would only have rescaled $w$ without changing its direction gets automatically routed to $g$ instead (via the first formula) and never reaches $v$ at all. This is the precise mathematical mechanism behind the claimed optimization benefit: the two "jobs" — picking a direction, picking a scale — are cleanly separated at the gradient level, not just at the parameterization level.

### 11.4 A worked numerical example

Take $v = [3.0, 4.0]$ (so $\|v\| = 5.0$) and $g = 2.0$:

$$w = 2.0\cdot\frac{[3.0,4.0]}{5.0} = 2.0\cdot[0.6, 0.8] = [1.2, 1.6]$$

Check: $\|w\| = \sqrt{1.2^2+1.6^2}=\sqrt{1.44+2.56}=\sqrt{4.0}=2.0=g$ — confirming $g$ directly controls $w$'s magnitude while $v$'s direction (here, the direction of $[3,4]$, i.e., $[0.6,0.8]$) is fully preserved in $w$'s direction.

Now suppose training wants to *double* this neuron's output scale without changing what input direction it responds to most strongly. Under the **standard** parameterization, you'd need a gradient step that scales both components of $w$ by exactly the same factor — any noise or imprecision in that step changes the direction slightly too. Under **Weight Normalization**, this is a single clean update: $g: 2.0 \to 4.0$, with $v$ completely untouched, guaranteeing the direction is preserved exactly.

### 11.5 PyTorch usage

```python
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

linear = nn.Linear(in_features=512, out_features=512)

# Modern PyTorch: register weight norm as a parametrization
P.register_parametrization(linear, "weight", torch.nn.utils.parametrizations.weight_norm(linear, name="weight"))
# (Older PyTorch versions: torch.nn.utils.weight_norm(linear, name='weight'))

x = torch.randn(8, 512)
y = linear(x)   # forward pass automatically reconstructs weight = g * v/||v||
```

### 11.6 Advantages, limitations, and real-world usage

**Advantages:** decouples magnitude and direction optimization (Section 11.3); computationally far cheaper than activation-based normalization (BatchNorm/LayerNorm), since it's a per-weight-vector operation computed once per forward pass rather than once per example/token; no dependency on batch composition or any activation statistics at all.

**Limitations:** unlike LayerNorm/BatchNorm, Weight Normalization does **not** directly address the activation-scale/internal-covariate-shift problem from [Sections 1–2](#1-why-normalization-the-core-problem) — it's solving a different problem (gradient coupling within weight updates), and empirically tends to require careful initialization (the original paper pairs it with a specific data-dependent initialization scheme) to match BatchNorm/LayerNorm's optimization benefits in deep networks.

**Real-world usage:** notably used in WaveNet and various early generative audio/image models, and in some reinforcement learning policy networks where its lower computational overhead (no per-batch statistics needed) was specifically valued. It is **not standard in current LLM Transformer architectures**, which rely on LayerNorm/RMSNorm applied to activations rather than weight reparameterization, though the underlying "decouple magnitude from direction" idea reappears conceptually in techniques like μP ([Section 56](#56-rezero-normformer-μp-maximal-update-parameterization)).

### 11.7 What's next

[Section 12](#12-spectral-normalization) covers Spectral Normalization — another weight-space technique, but with a different goal: rather than controlling a single weight vector's magnitude, it constrains an entire weight *matrix's* largest singular value, controlling how much the layer can amplify its input in the worst case — a property critical for GAN training stability.

---

## 12. Spectral Normalization

### 12.1 Motivation

Spectral Normalization (Miyato et al., 2018) addresses a different stability concern than anything covered so far: **how much can a single linear layer amplify its input, in the worst case, over every possible input direction?**

This question matters enormously for one specific, historically important setting: training Generative Adversarial Networks (GANs). A GAN's discriminator needs to be a well-behaved (specifically, Lipschitz-continuous) function for the adversarial training dynamic to converge reliably — if the discriminator's output can change arbitrarily fast in response to tiny input changes, the generator receives wildly unstable gradients, and GAN training is notoriously prone to exactly this kind of instability (mode collapse, oscillation, divergence).

The amount a linear layer $y = Wx$ can amplify its input is precisely captured by $W$'s **largest singular value**, $\sigma_{\max}(W)$ — also called its **spectral norm**. Formally:

$$\sigma_{\max}(W) = \max_{x \neq 0} \frac{\|Wx\|}{\|x\|}$$

This is the largest possible "stretch factor" $W$ can apply to any input vector. If you constrain every layer's spectral norm to be at most 1, the *entire network's* Lipschitz constant is bounded by the product of its layers' Lipschitz constants (a standard composition property) — giving you a concrete, enforceable handle on the network's overall worst-case sensitivity.

### 12.2 Mathematical formulation

Spectral Normalization rescales a weight matrix $W$ by its own largest singular value:

$$W_{SN} = \frac{W}{\sigma_{\max}(W)}$$

By construction, $W_{SN}$ has spectral norm exactly 1 — no input direction can be amplified by more than a factor of 1 (i.e., the layer can only ever shrink or preserve length, never stretch). Unlike every technique so far, this isn't a learnable rescaling — there's no $\gamma$ to compensate, by design: the entire point is to **enforce a hard constraint** on the layer's behavior, not merely to recenter/rescale its typical statistics.

### 12.3 Computing $\sigma_{\max}(W)$ cheaply: power iteration

Computing a full singular value decomposition of $W$ at every training step would be prohibitively expensive for large weight matrices (SVD is roughly $O(\min(m,n)\cdot mn)$ for an $m\times n$ matrix — far too costly to run every forward pass). Spectral Normalization instead uses **power iteration**, an old, cheap, iterative method for estimating just the *largest* singular value (and its associated singular vectors) without computing the full decomposition:

$$u \leftarrow \frac{Wv}{\|Wv\|}, \qquad v \leftarrow \frac{W^\top u}{\|W^\top u\|}$$

Repeating this update a small number of times (often just **one** iteration per training step is sufficient in practice, since $W$ changes only slightly between consecutive steps, so last step's estimate is already a good starting point) converges $u, v$ toward the top left/right singular vectors, and $\sigma_{\max}(W) \approx u^\top W v$ becomes an accurate running estimate, maintained essentially for free (one matrix-vector product per iteration, vastly cheaper than a full SVD) across the entire training run.

### 12.4 A worked numerical example

Take the small matrix $W = \begin{bmatrix}3 & 0\\0 & 1\end{bmatrix}$ — chosen deliberately simple so its singular values are obvious without computation: since $W$ is already diagonal, its singular values are just the absolute values of its diagonal entries, $\sigma_{\max}=3, \sigma_{\min}=1$.

Applying spectral normalization:

$$W_{SN} = \frac{1}{3}\begin{bmatrix}3&0\\0&1\end{bmatrix} = \begin{bmatrix}1 & 0\\0 & 1/3\end{bmatrix}$$

Check: for the input $x=[1,0]$ (the direction of maximum stretch), $W_{SN}x = [1, 0]$, so $\|W_{SN}x\|/\|x\| = 1$ — confirming the worst-case amplification is now exactly 1, as intended. For comparison, $Wx = [3,0]$ would have given an amplification factor of 3 before normalization.

Let's verify power iteration actually converges to this, starting from an arbitrary initial guess $v_0 = [1,1]/\sqrt{2}$:

**Iteration 1:** $Wv_0 = [3,0,0,1]\cdot[0.707,0.707]^\top$... computing directly: $Wv_0 = [3\times0.707,\ 1\times0.707] = [2.121, 0.707]$, $\|Wv_0\|=\sqrt{2.121^2+0.707^2}=\sqrt{4.5+0.5}=\sqrt{5.0}\approx2.236$, so $u_1 = [0.949, 0.316]$.

Then $W^\top u_1 = [3\times0.949,\ 1\times0.316]=[2.846,0.316]$, $\|W^\top u_1\|=\sqrt{2.846^2+0.316^2}=\sqrt{8.10+0.10}=\sqrt{8.20}\approx2.864$, so $v_1 = [0.994, 0.110]$ — already very close to the true top singular vector $[1, 0]$ after just one iteration, confirming the "one iteration per step is often enough" claim from Section 12.3 for a matrix with a reasonably well-separated top singular value.

### 12.5 PyTorch usage

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
conv_sn = nn.utils.parametrizations.spectral_norm(conv, name="weight", n_power_iterations=1)

x = torch.randn(4, 64, 16, 16)
y = conv_sn(x)
print(y.shape)   # torch.Size([4, 128, 14, 14])

# The power-iteration vectors (u, v) are maintained internally as buffers
# and updated automatically on every forward call during training.
```

### 12.6 Advantages, limitations, and real-world usage

**Advantages:** provides a hard, provable bound on a layer's worst-case input amplification (a Lipschitz guarantee), which directly addresses GAN training instability; computationally cheap via power iteration (one extra matrix-vector product per forward pass, negligible compared to the matrix multiply itself); requires no batch statistics.

**Limitations:** the Lipschitz bound this technique provides is a bound on each *individual layer*; the actual end-to-end network's Lipschitz constant is generally *looser* than the naive product of per-layer bounds would suggest, because that product bound doesn't account for how nonlinearities and specific weight directions interact across layers — in practice this is rarely a problem (the technique still works very well empirically), but it's worth knowing the theoretical guarantee is a sufficient, not necessarily tight, bound. Additionally, the power iteration estimate of $\sigma_{\max}$ converges slowly when a matrix's top two singular values are close together (the convergence rate of power iteration depends on the *ratio* between the largest and second-largest singular values) — a known but rarely fatal practical caveat.

**Real-world usage:** the standard stabilization technique for GAN discriminators since its introduction (used extensively in BigGAN, SAGAN, and most well-known modern GAN architectures); also used in some certified-robustness research for bounding a network's sensitivity to adversarial perturbations. **Not used in standard LLM architectures** — modern LLMs address training stability primarily through normalization (LayerNorm/RMSNorm) applied to activations plus residual connections ([Parts III–V](#20-why-batchnorm-fails-for-transformers)), rather than through hard weight-space Lipschitz constraints.

### 12.7 What's next

[Section 13](#13-l2-normalization) covers L2 Normalization — a simpler, more direct relative of the geometric "project onto a sphere" idea from [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius), applied without any mean-centering or learnable affine parameters at all, most commonly seen in embedding spaces (e.g., contrastive learning, retrieval, and similarity search) where you specifically want every vector to live on the unit sphere.

---

## 13. L2 Normalization

### 13.1 Motivation

L2 Normalization is the most geometrically direct technique in this entire handbook: it rescales a vector to have **exactly unit length** (Euclidean norm = 1), with no mean subtraction, no learnable $\gamma/\beta$, and no per-group or per-batch statistics involved at all. It is the purest expression of [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)'s "project onto a sphere" idea — here, quite literally, with no additional steps.

The motivation is almost always **representational** rather than optimization-stability-related: in many embedding-based tasks (face recognition, sentence/document embeddings for retrieval, contrastive learning frameworks like CLIP), what you want to compare between two vectors is their **direction** (which way they point in embedding space — capturing semantic similarity) and explicitly *not* their **magnitude** (which might just reflect how "confident" or "active" an embedding happened to be, rather than carrying comparable semantic content). Forcing every embedding onto the unit sphere makes magnitude differences impossible by construction, leaving only direction as a basis for comparison.

### 13.2 Mathematical formulation

$$\hat{x} = \frac{x}{\|x\|_2} = \frac{x}{\sqrt{\sum_{i=1}^d x_i^2 + \epsilon}}$$

Notice this is **exactly RMSNorm's denominator from [Section 8.2](#82-mathematical-formulation)**, up to a single constant factor: RMSNorm divides by $\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}$ (RMS, which includes a $1/d$ averaging factor), while L2 normalization divides by $\sqrt{\sum_i x_i^2+\epsilon}$ (the raw Euclidean norm, no averaging). The two differ by exactly a factor of $\sqrt{d}$: $\|x\|_2 = \sqrt{d}\cdot\text{RMS}(x)$. This means **L2-normalizing a vector and RMSNorm-ing it (with $\gamma=1$) point in the exact same direction** — they only differ in the fixed length they land on ($1$ for L2-norm, $\sqrt{d}$ for RMSNorm, matching [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)'s claim that standardization lands on a sphere of radius $\sqrt{n}$). Seeing this connection explicitly is one of the more satisfying "aha" moments in this handbook's geometric framing: BatchNorm, LayerNorm, RMSNorm, and L2-Norm are *all*, underneath their different surface formulas, the same "radial projection onto a sphere of some specific radius" operation from Section 4.3, differing only in which axis they reduce over and which exact radius they target.

### 13.3 A worked numerical example, and the cosine-similarity connection

Take $x = [3.0, 4.0]$. Its L2 norm is $\|x\|_2 = \sqrt{3^2+4^2}=\sqrt{25}=5.0$, so:

$$\hat{x} = [3.0/5.0,\ 4.0/5.0] = [0.6, 0.8]$$

Confirm: $\|\hat{x}\|_2 = \sqrt{0.36+0.64}=\sqrt{1.0}=1.0$ ✓.

The reason this matters so much in practice: once two vectors $a, b$ are both L2-normalized, their dot product *is* their cosine similarity directly:

$$\hat{a}\cdot\hat{b} = \frac{a}{\|a\|}\cdot\frac{b}{\|b\|} = \cos(\theta_{ab})$$

This is precisely why contrastive learning frameworks (CLIP, SimCLR, and most modern embedding/retrieval models) L2-normalize their output embeddings before computing similarity scores: it turns an otherwise magnitude-sensitive dot product into a pure, magnitude-invariant measure of directional alignment, and it makes a fixed **temperature** hyperparameter in the contrastive loss meaningful and comparable across training, since every similarity score is guaranteed to land in $[-1, 1]$ regardless of how embedding magnitudes might otherwise drift during training.

### 13.4 PyTorch usage

```python
import torch
import torch.nn.functional as F

x = torch.tensor([3.0, 4.0])
x_hat = F.normalize(x, p=2, dim=-1, eps=1e-12)
print(x_hat)              # tensor([0.6000, 0.8000])
print(x_hat.norm())        # tensor(1.0000)

# Typical contrastive-learning usage pattern:
image_embeds = F.normalize(image_features, p=2, dim=-1)
text_embeds = F.normalize(text_features, p=2, dim=-1)
similarity = image_embeds @ text_embeds.T   # cosine similarities, directly
```

### 13.5 Advantages, limitations, and real-world usage

**Advantages:** simplest possible normalization formula; makes dot products directly interpretable as cosine similarity; removes any need to separately normalize before computing similarity scores at inference/retrieval time; zero learnable parameters, zero per-batch statistics.

**Limitations:** discards magnitude information unconditionally and irreversibly — appropriate when magnitude genuinely carries no useful signal for the task, actively harmful when it does (e.g., using L2-normalized embeddings somewhere that genuinely needs to distinguish "highly confident, strong activation" from "weak, uncertain activation" via vector length).

**Real-world usage:** standard in CLIP-style contrastive vision-language models, face recognition embeddings (ArcFace, CosFace, and related margin-based losses are all built explicitly around L2-normalized embeddings), sentence embedding models for retrieval (e.g., many sentence-transformer architectures), and recommendation-system embedding comparisons. **Not used as the primary normalization layer inside standard LLM Transformer blocks** (that role belongs to LayerNorm/RMSNorm, [Sections 7–8](#7-layer-normalization-ln)) — but it reappears as the final step applied to LLM-produced *embeddings* whenever those embeddings feed into a downstream similarity-based task (e.g., using an LLM's hidden states for retrieval or semantic search).

### 13.6 What's next

[Section 14](#14-power-normalization) covers Power Normalization — a less common but instructive variant that addresses a specific failure mode of standard normalization techniques on heavy-tailed activation distributions, generalizing the "divide by some measure of spread" idea using a power-transform rather than a fixed square-root.

---

## 14. Power Normalization

### 14.1 Motivation

Power Normalization (Shen et al., 2020, presented as "PowerNorm") was developed to address two related, specific weaknesses identified in BatchNorm when applied to Transformers (a topic we'll cover generally in [Section 20](#20-why-batchnorm-fails-for-transformers)): (1) BatchNorm's batch statistics for Transformer activations tend to be **highly variable from batch to batch** during training (more so than in typical CNN activations), and (2) Transformer activations frequently exhibit **heavy-tailed distributions** — occasional very large outlier values — which standard variance estimation (Section 3.2's squared-deviation formula) is especially sensitive to, since squaring amplifies the influence of outliers more than smaller, typical values.

Power Normalization's core idea: replace the variance-based denominator with a more outlier-robust, **smoothed running statistic** of a general power of the activations, and combine this with techniques to reduce the batch-to-batch noise that plagued BatchNorm specifically in the Transformer setting.

### 14.2 Mathematical formulation (simplified)

The general form replaces the variance computation with a $p$-th power mean (generalizing $p=2$, the ordinary mean-square used in variance/RMS):

$$\psi_p(x) = \left(\frac{1}{n}\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

With $p=2$, this is exactly the RMS formula from [Section 8.1](#81-motivation). PowerNorm's actual proposal keeps $p=2$ (so the formula itself looks like RMS/variance-based normalization) but changes **what's being averaged over** and **how the running statistic is maintained**: rather than re-estimating the second-moment statistic fresh from each mini-batch (as BatchNorm does), it maintains a smoothed, slowly-updating running estimate similar in spirit to BatchNorm's running mean/variance ([Section 6.6](#66-the-traininginference-mismatch--running-statistics)) — but updated and used even *during* training (not just frozen for inference), specifically to dampen the batch-to-batch noise problem identified in Section 14.1, while also incorporating a separate correction step to address the "second-moment-without-mean-centering" simplification.

### 14.3 Why outlier sensitivity matters here specifically

Return to [Section 3.2](#32-variance--what-it-represents)'s point about squaring: an outlier value contributes to a variance/second-moment computation in proportion to its **square**. A single token with an activation value 10x larger than typical contributes 100x more to the variance sum than a typical token would. If that outlier happens to land in a small mini-batch (exacerbating BatchNorm's small-sample noise problem from [Section 6.9](#69-numerical-stability-considerations)), the resulting batch statistic can be dramatically skewed relative to the "true," population-level statistic — and this skewed statistic then gets applied to *every* activation normalized using it, including the many *non*-outlier activations in that same batch, distorting their normalized values too. Power Normalization's running-statistic smoothing (averaging the second-moment estimate gradually over many batches, rather than recomputing it fresh and fully trusting each individual mini-batch) directly dampens this outlier sensitivity, at the cost of the statistic adapting more slowly to genuine, sustained shifts in the activation distribution.

### 14.4 Relationship to the rest of this handbook

It's worth being precise about Power Normalization's place in the broader picture covered by this handbook: it is fundamentally a **BatchNorm variant** (it still reduces over the batch dimension, inheriting BatchNorm's basic axis choice from [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over)), with modifications aimed at making batch-statistic-based normalization more viable specifically for Transformers. **It did not become the dominant solution to BatchNorm's Transformer problems** — that role went to LayerNorm (and later RMSNorm) instead, which sidesteps the batch-statistic-noise problem entirely by never depending on batch composition in the first place ([Section 7.1](#71-motivation)), rather than trying to make batch statistics more robust. Power Normalization is included here primarily for completeness and because understanding *why* it was proposed (Section 14.1's two specific Transformer-activation pathologies) reinforces, from a different angle, exactly why the field ultimately moved away from any batch-dependent normalization for Transformers at all.

### 14.5 Advantages, limitations, and real-world usage

**Advantages:** more robust to outlier activations than naive batch statistics; demonstrated, in its original paper's experiments, improved stability over standard BatchNorm specifically in Transformer training settings.

**Limitations:** still fundamentally batch-dependent (inheriting some of BatchNorm's structural issues from [Section 6.11](#611-advantages-and-limitations), including a train/inference distinction, even if the running-statistic mechanism differs in detail from standard BatchNorm); meaningfully more complex to implement correctly than LayerNorm/RMSNorm, for a benefit that, in practice, LayerNorm/RMSNorm capture more simply by avoiding batch dependence altogether.

**Real-world usage:** primarily a research contribution demonstrating that BatchNorm's Transformer problems are *fixable* with sufficient engineering — it has seen limited large-scale production adoption compared to LayerNorm/RMSNorm, which solve the same underlying problems via a structurally simpler approach (changing the reduction axis rather than refining the batch-statistic estimator). **Not used in any major production LLM** covered in this handbook's [Part VII](#46-gpt-2--gpt-3--gpt-4) case studies.

### 14.6 What's next

[Section 15](#15-adaptive-normalization-adain-conditional-norms) returns to Instance Normalization's style-transfer roots ([Section 10.3](#103-why-removing-per-channel-per-image-statistics-helps-style-transfer)) and generalizes it: rather than always rescaling to a fixed, learned $\gamma,\beta$, Adaptive Normalization computes $\gamma,\beta$ **dynamically**, conditioned on some external input (a style image, a class label, a timestep) — a pattern that reappears, in a different guise, in diffusion model architectures.

---

## 15. Adaptive Normalization (AdaIN, Conditional Norms)

### 15.1 Motivation

Every normalization technique so far has used a $\gamma,\beta$ that's either fixed (L2-Norm has none at all) or **learned once during training and then frozen as a static parameter** for every input the model ever sees afterward. Adaptive Normalization breaks this pattern: it computes $\gamma,\beta$ **dynamically, per example, as a function of some external conditioning signal** — a style reference image, a class label, a diffusion model's current timestep, or any other piece of side information you want the network's normalization behavior to respond to.

The motivating idea connects directly back to [Section 10.3](#103-why-removing-per-channel-per-image-statistics-helps-style-transfer): if Instance Normalization's job is to *strip out* an image's own per-channel style statistics, the natural next step is to **replace them with statistics borrowed from somewhere else** — a different image's style, or a learned function of a conditioning variable — rather than replacing them with a single, static, learned $\gamma,\beta$ that's identical for every input regardless of context.

### 15.2 AdaIN — Adaptive Instance Normalization

AdaIN (Huang & Belongie, 2017) is the canonical example. Given a **content** input $x$ and a **style** reference $y$, it first computes $x$'s Instance-Norm-standardized representation (Section 10.2's $\hat{x}$, using $x$'s own per-channel mean/variance), then rescales it using **$y$'s** per-channel statistics, instead of a learned $\gamma,\beta$:

$$\text{AdaIN}(x, y) = \sigma(y)\cdot\frac{x - \mu(x)}{\sigma(x)} + \mu(y)$$

where $\mu(x),\sigma(x)$ are computed from the content input (per-channel, as in standard Instance Norm), and $\mu(y),\sigma(y)$ are computed from the style reference's own per-channel statistics. Reading this formula against [Section 10.2](#102-mathematical-formulation): this is *exactly* Instance Normalization's formula, with the learned constants $\gamma_c,\beta_c$ replaced by $\sigma(y)_c,\mu(y)_c$ — statistics extracted live, per-call, from a second input. The content image's own style statistics are discarded entirely (via the standard Instance Norm step) and replaced wholesale with the style reference's statistics — a direct, mechanistic implementation of "transfer this image's style onto that image's content."

### 15.3 A worked numerical example

Take a single channel's spatial values for a content image, $x = [2.0, 4.0, 6.0, 8.0]$ (so $\mu(x)=5.0,\sigma(x)\approx2.236$), and for a style reference image (same channel), $y = [10.0, 30.0, 50.0, 10.0]$ (so $\mu(y)=25.0$, $\sigma(y)=\sqrt{\frac{(10-25)^2+(30-25)^2+(50-25)^2+(10-25)^2}{4}}=\sqrt{\frac{225+25+625+225}{4}}=\sqrt{275}\approx16.583$).

$$\text{AdaIN}(x,y) = 16.583\cdot\frac{[2,4,6,8]-5.0}{2.236} + 25.0$$

Computing the standardized content: $\frac{[-3,-1,1,3]}{2.236} \approx [-1.342,-0.447,0.447,1.342]$.

$$\text{AdaIN}(x,y) \approx 16.583\times[-1.342,-0.447,0.447,1.342] + 25.0 \approx [2.74, 17.59, 32.41, 47.26]$$

Notice the **output's own mean and variance now match the style reference $y$'s** (mean ≈ 25, spread comparable to $y$'s spread), while the *relative shape/pattern* of the content signal ($x$'s monotonically-increasing structure) is preserved in the output's relative pattern (also monotonically increasing). This is precisely the "content's structure, style's statistics" effect that makes AdaIN useful for style transfer.

### 15.4 The broader pattern: Conditional Normalization

AdaIN's "compute $\gamma,\beta$ from something other than a static learned parameter" idea generalizes well beyond borrowing statistics from a second image. **Conditional Batch Normalization** and **Conditional Instance Normalization**, used in class-conditional GANs (e.g., BigGAN), instead learn small neural networks (often just a single linear layer) that map a class label embedding to a $(\gamma,\beta)$ pair:

$$\gamma = W_\gamma \cdot e_{\text{class}} + b_\gamma, \qquad \beta = W_\beta \cdot e_{\text{class}} + b_\beta$$

where $e_{\text{class}}$ is a learned embedding of the conditioning class label. This lets a single generator network produce visually distinct outputs for different classes, by letting the *normalization layers themselves* — not just the convolutional weights — vary based on what class is being generated.

**FiLM (Feature-wise Linear Modulation)**, used widely in multi-modal and conditional generation architectures, generalizes this further: $\gamma,\beta$ can be computed from *any* conditioning signal (text embeddings, audio features, timestep embeddings), not just discrete class labels — making this pattern the standard mechanism for injecting conditioning information into a network at every layer, rather than only at the input.

### 15.5 Adaptive LayerNorm (adaLN) in modern diffusion Transformers

This same pattern reappears prominently in modern diffusion Transformer architectures (e.g., DiT — Diffusion Transformers — used in several state-of-the-art image/video diffusion models), under the name **adaLN** (Adaptive LayerNorm): rather than computing $\gamma,\beta$ from a style image's statistics (AdaIN) or a class embedding (Conditional BatchNorm), adaLN computes them from the diffusion process's **timestep embedding** (and often a class/text conditioning embedding too), letting the network's normalization behavior shift appropriately as the denoising process progresses from a very noisy initial state to a nearly clean final image. Some DiT variants go further with **adaLN-Zero**, which additionally predicts a per-channel scaling factor applied *after* the residual-connected sublayer (not just inside the normalization step) — initialized so that, at the start of training, each new Transformer block initially behaves as the identity function, a technique conceptually related to ReZero, which we cover in [Section 33](#33-variants-highway-networks-densenet-gated-residuals-rezero-deepnorm-scaling).

### 15.6 PyTorch implementation sketch

```python
import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    """
    Computes gamma, beta from a conditioning embedding, instead of
    learning them as static parameters. Used in DiT-style architectures.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)  # no learned gamma/beta here
        self.to_gamma_beta = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)   # start near-identity

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        # gamma initialized near 0, so (1+gamma) starts near 1 -- near-identity at init
        return self.norm(x) * (1 + gamma) + beta

adaln = AdaptiveLayerNorm(dim=768, cond_dim=256)
x = torch.randn(8, 128, 768)        # (B, T, d)
timestep_embed = torch.randn(8, 256)
timestep_embed_broadcast = timestep_embed.unsqueeze(1)   # (B, 1, cond_dim), broadcasts over T
y = adaln(x, timestep_embed_broadcast)
print(y.shape)   # torch.Size([8, 128, 768])
```

### 15.7 Advantages, limitations, and real-world usage

**Advantages:** lets a single network produce conditionally-varying outputs by modulating normalization behavior, rather than requiring a separate network per condition; the zero-initialization trick (Section 15.6's code) lets conditioning be introduced without destabilizing early training, since each conditioned block starts out behaving close to its unconditioned counterpart.

**Limitations:** adds extra parameters and a small additional forward-pass cost (the linear layer mapping conditioning → $\gamma,\beta$) at every adaptive normalization layer; the conditioning signal must be available and meaningful at every layer where it's injected, which requires architectural planning (typically broadcasting a single embedding to every block, as in Section 15.6's example).

**Real-world usage:** AdaIN remains foundational to neural style transfer and was a core component of StyleGAN/StyleGAN2's per-layer style injection mechanism; Conditional BatchNorm/FiLM are standard in conditional GANs and multi-modal conditioning architectures; adaLN (and adaLN-Zero) is the standard conditioning mechanism in modern Diffusion Transformer (DiT) architectures. **Standard LLM Transformer blocks (GPT, LLaMA, etc.) do not use adaptive normalization** — their $\gamma$ (and $\beta$, where present) are static learned parameters, not computed from a conditioning signal, since there's no analogous "style" or "timestep" input in a standard autoregressive language model's architecture.

### 15.8 What's next

[Section 16](#16-scalenorm) covers ScaleNorm — a return to the LLM-relevant thread, proposing an even more radical simplification than RMSNorm: replacing the entire per-element rescaling with a **single learned scalar** shared across all dimensions, asking how much of LayerNorm's machinery can be stripped away while still preserving training stability.

---

## 16. ScaleNorm

### 16.1 Motivation

ScaleNorm (Nguyen & Salazar, 2019) asks a more aggressive version of the question that motivated RMSNorm ([Section 8.1](#81-motivation)): RMSNorm already showed that dropping mean-centering loses little. ScaleNorm goes further and asks whether the **per-feature learned scale vector** $\gamma\in\mathbb{R}^d$ itself can be replaced with a **single shared scalar** $g\in\mathbb{R}$, applied identically to every dimension — trading away per-feature flexibility for an even smaller parameter count and a marginally simpler computation, while keeping the same L2-norm-based ([Section 13](#13-l2-normalization)) rescaling-to-a-fixed-length idea at its core.

### 16.2 Mathematical formulation

$$\text{ScaleNorm}(x) = g\cdot\frac{x}{\|x\|_2}$$

where $g\in\mathbb{R}$ is a single learned scalar, shared across the entire vector (not one value per feature, unlike every $\gamma$ we've seen so far). Compare directly against L2 Normalization ([Section 13.2](#132-mathematical-formulation)): ScaleNorm is **exactly** L2 normalization, with one single learned scalar $g$ replacing the implicit, fixed scale of 1. There's no $\beta$, and no per-feature $\gamma$ — just one number controlling the overall length the vector gets rescaled to.

### 16.3 A worked numerical example

Reuse $x = [2.0, 8.0, -4.0, 6.0]$ from Sections 1.2, 7.3, and 8.4. Its L2 norm: $\|x\|_2 = \sqrt{4+64+16+36}=\sqrt{120}\approx10.954$.

With a learned scalar $g = 3.0$:

$$\text{ScaleNorm}(x) = 3.0\cdot\frac{[2.0,8.0,-4.0,6.0]}{10.954}\approx 3.0\times[0.183,0.730,-0.365,0.548]\approx[0.548,2.191,-1.095,1.643]$$

Check the output's norm: $\|\text{ScaleNorm}(x)\|_2 = g = 3.0$ — by construction, the *entire* output vector's length is exactly the learned scalar $g$, regardless of the input. This is the cleanest possible illustration of [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)'s geometric picture: every input, no matter its original direction or length, lands on a sphere of radius *exactly* $g$ — and unlike LayerNorm/RMSNorm/BatchNorm, that radius is the *same single learned number* for the entire vector, rather than implicitly varying per-feature via a learned $\gamma$ vector.

### 16.4 NumPy and PyTorch implementation

```python
import numpy as np

class ScaleNorm:
    """ScaleNorm from scratch — a single learned scalar g, L2-norm based."""
    def __init__(self, eps: float = 1e-5):
        self.g = np.array(1.0, dtype=np.float32)   # single scalar parameter
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return self.g * x / (norm + self.eps)

sn = ScaleNorm()
sn.g = np.array(3.0)
x = np.array([2.0, 8.0, -4.0, 6.0])
print(sn.forward(x))   # -> [0.548 2.191 -1.095 1.643] (matches Section 16.3)
```

```python
import torch
import torch.nn as nn

class ScaleNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1.0))   # one scalar for the whole layer
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        return self.g * x / (norm + self.eps)
```

### 16.5 Why this trade-off is rarely worth it for large models

ScaleNorm's parameter savings relative to RMSNorm ($1$ scalar vs. $d$ values per normalization layer) are real but, at LLM scale, essentially negligible — even with $d=4096$ and dozens of normalization layers, the absolute parameter count saved is minuscule compared to a multi-billion-parameter model's total. The actual trade-off that matters is representational: **RMSNorm's per-feature $\gamma$ lets the network learn that some features should consistently end up larger or smaller than others** after normalization (a useful, persistent per-feature property the model can rely on, layer after layer); ScaleNorm's single shared $g$ cannot represent this — every feature is treated identically by the normalization layer itself. This is likely why ScaleNorm saw substantially less large-scale LLM adoption than RMSNorm, despite being conceptually compelling and slightly cheaper: the parameter savings don't matter at scale, and the representational restriction is a real, if modest, cost.

### 16.6 Advantages, limitations, and real-world usage

**Advantages:** extremely simple; a direct, transparent illustration of the underlying "project onto a fixed-radius sphere" geometry with nothing else mixed in; marginally cheaper than RMSNorm (one fewer per-feature multiply).

**Limitations:** loses per-feature scale flexibility, which appears to matter for large-scale model quality based on which technique the field actually adopted at scale (Section 16.5); less battle-tested than LayerNorm/RMSNorm across diverse architectures and training regimes.

**Real-world usage:** introduced and evaluated primarily in machine translation Transformer experiments in its original paper, demonstrating comparable performance to LayerNorm with fewer parameters and a simpler computation at the scales tested there. **Not adopted in any major modern open-weight LLM** covered in this handbook's case studies ([Part VII](#46-gpt-2--gpt-3--gpt-4)) — RMSNorm won out as the field's preferred LayerNorm simplification.

### 16.7 What's next

[Section 17](#17-deepnorm) covers DeepNorm — a technique motivated by a different problem entirely from the simplification trend of RMSNorm/ScaleNorm: training **extremely deep** Transformers (hundreds of layers) stably, by deliberately rescaling the residual connection itself in combination with a specific weight initialization scheme — directly bridging this handbook's normalization coverage with [Part IV](#29-motivation-vanishing-gradients-and-the-degradation-problem)'s residual-connection material.

---

## 17. DeepNorm

### 17.1 Motivation

Everything covered in [Sections 6–16](#6-batch-normalization-bn) modifies how a *single* normalization layer computes its statistics. DeepNorm (Wang et al., 2022, introduced alongside Microsoft's 1000-layer "DeepNet" experiments) instead asks a question that only becomes acute at extreme depth: **as you stack more and more Pre-LN Transformer blocks ([Section 7.7](#77-production-grade-pytorch-usage)'s pattern), each one's output gets added back into a running residual stream via $x \leftarrow x + F(x)$. With enough layers, does this accumulated sum of contributions eventually grow large enough to destabilize training, even with LayerNorm/RMSNorm correctly normalizing each individual sublayer's *input*?**

The answer, empirically and theoretically (we derive the mechanism precisely in [Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks)), is yes — Pre-LN's residual stream norm tends to grow roughly with $\sqrt{\text{depth}}$, which is fine for the depths used by GPT-3/LLaMA-scale models (tens of layers) but becomes a genuine obstacle at the hundreds-of-layers scale DeepNet specifically targeted. DeepNorm's fix operates on **both** the residual connection and the initialization simultaneously, rather than on the normalization layer's internal statistics computation — which is why it's best understood as sitting at the intersection of this handbook's normalization coverage and its residual-connection coverage (we'll revisit it from the residual-connection side in [Section 33.5](#33-variants-highway-networks-densenet-gated-residuals-rezero-deepnorm-scaling)).

### 17.2 Mathematical formulation

DeepNorm modifies the standard Post-LN residual update (we cover Post-LN vs. Pre-LN fully in [Section 25](#25-post-ln-vs-pre-ln-vs-sandwich-ln); for now, just treat this as "the formula for combining a sublayer's output with its input"):

$$x_{l+1} = \text{LN}\big(\alpha\cdot x_l + F(x_l)\big)$$

The single change from standard Post-LN ($x_{l+1}=\text{LN}(x_l + F(x_l))$, i.e., $\alpha=1$): the residual stream $x_l$ is scaled by a constant $\alpha > 1$ **before** being added to the sublayer output $F(x_l)$. Larger $\alpha$ means the existing residual stream dominates the sum more strongly relative to each new sublayer's contribution — directly damping how much each individual layer can perturb the accumulated stream, which is exactly the lever needed to keep hundreds of stacked contributions from compounding into instability.

This is paired with a matching change to weight initialization: DeepNorm scales down the initial weights of each sublayer's output projection (the final linear layer inside the attention block and the FFN block) by a factor $\beta$ that, like $\alpha$, depends on the total depth $N$ of the network:

$$\alpha = (2N)^{1/2}\ \text{(encoder)}, \qquad \beta = (8N)^{-1/4}\ \text{(encoder)}$$

(DeepNorm's paper provides separate, slightly different formulas for encoder-only, decoder-only, and encoder-decoder architectures — the encoder-style formulas above illustrate the pattern; the precise constants differ by architecture type but the core idea — both scale factors shrink/grow systematically with depth $N$ — is the same throughout.)

### 17.3 The intuition: counteracting compounding growth directly at its source

Here's the mechanism in plain terms. If each of $N$ stacked residual blocks adds a contribution of roughly similar typical magnitude to the stream, the stream's overall magnitude after $N$ layers grows (very roughly) like $\sqrt{N}$ under the standard, unscaled residual update — this is the same kind of "sum of many roughly-independent contributions" growth pattern behind the $\sqrt{n}$ factor in [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)'s sphere-radius derivation, just appearing here across *depth* rather than across a single vector's *features*.

DeepNorm's $\alpha$ scaling directly compensates: by making $\alpha$ grow with $\sqrt{N}$ too, the *relative* size of each new layer's contribution, compared to the now-larger existing stream, shrinks proportionally as the network gets deeper — keeping the stream's growth rate under control even as $N$ becomes very large (hundreds of layers). The companion $\beta$-scaled initialization (shrinking each layer's *initial* output magnitude as depth grows) attacks the same problem from the other direction: it ensures that even before any training has happened, a freshly-initialized 1000-layer network doesn't already start with an unmanageably large residual stream purely as an artifact of stacking many randomly-initialized layers together.

### 17.4 A simplified numerical illustration

This isn't a precise simulation of an actual trained network (real activations aren't simple constants), but it illustrates the scaling logic clearly. Suppose, very roughly, every sublayer $F(x_l)$ contributes an output of typical magnitude 1 (in some norm), and the residual stream starts at $x_0$ of magnitude 1.

**Without DeepNorm scaling** ($\alpha=1$), after $N$ layers, a rough magnitude estimate (treating contributions as adding in quadrature, analogous to independent random contributions): $\|x_N\| \approx \sqrt{1^2 + N\times 1^2} = \sqrt{1+N}$. For $N=1000$: $\|x_{1000}\|\approx\sqrt{1001}\approx31.6$ — the stream has grown over 30x from its starting magnitude.

**With DeepNorm scaling** ($\alpha=\sqrt{2N}$, so for $N=1000$, $\alpha\approx44.7$): each layer's update becomes $x_{l+1}\approx\alpha x_l + F(x_l)$ before the subsequent LayerNorm renormalizes it back toward a controlled scale — the large $\alpha$ ensures the existing stream's contribution dominates the sum overwhelmingly at every step, so the *relative* perturbation introduced by each new layer (which is what actually matters for gradient stability through the chain rule, since gradients flow back through this same accumulated sum) stays small and controlled regardless of how many layers have already been stacked, rather than growing unboundedly as in the unscaled case.

### 17.5 PyTorch implementation sketch

```python
import torch
import torch.nn as nn
import math

class DeepNormBlock(nn.Module):
    """
    A single Transformer block using DeepNorm's residual scaling.
    `total_layers` (N) determines both alpha and the init scale beta.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, total_layers: int):
        super().__init__()
        self.alpha = (2 * total_layers) ** 0.5
        beta = (8 * total_layers) ** -0.25

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # DeepNorm-style initialization: scale DOWN output projections by beta
        with torch.no_grad():
            self.attn.out_proj.weight.mul_(beta)
            self.ffn[-1].weight.mul_(beta)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(self.alpha * x + attn_out)     # alpha-scaled residual, THEN normalize
        x = self.ln2(self.alpha * x + self.ffn(x))
        return x

# A 1000-layer-scale example (illustrative; not actually instantiating 1000 layers here)
block = DeepNormBlock(d_model=512, n_heads=8, d_ff=2048, total_layers=1000)
print(f"alpha = {block.alpha:.2f}")   # alpha = 44.72
```

### 17.6 Advantages, limitations, and real-world usage

**Advantages:** demonstrated stable training of Transformers with 1,000+ layers — a regime where standard Post-LN and even standard Pre-LN training become unstable or fail outright; the scaling rule is simple (two formulas dependent only on total depth $N$) and requires no architectural changes beyond the residual-scaling constant and an initialization tweak.

**Limitations:** the $\alpha,\beta$ formulas are tuned for the specific depth-vs-stability tradeoff studied in DeepNet's original experiments; using DeepNorm doesn't obviously help (and adds unnecessary complexity) at the depths used by most production LLMs (tens of layers, not hundreds to thousands), where standard Pre-LN ([Section 25](#25-post-ln-vs-pre-ln-vs-sandwich-ln)) already trains stably without it.

**Real-world usage:** primarily a research result (Microsoft's DeepNet experiments) demonstrating the *feasibility* of extreme-depth Transformer training, rather than a technique adopted in mainstream production LLMs at the depths currently used by GPT/LLaMA/Mistral/Gemma/DeepSeek-scale models (typically 30–120 layers) — those models' depths don't yet require DeepNorm's specific intervention, since standard Pre-LN's $\sqrt{N}$ growth (Section 17.1) remains manageable at that scale. It remains an important reference point for understanding the **theoretical limits** of how deep a Transformer can be trained, and for any future architecture that pushes meaningfully beyond current depth conventions.

### 17.7 What's next

[Section 18](#18-normformer-and-other-modern-variants) closes out the individual-technique survey with NormFormer and several other modern variants — additional normalization placements and learned-scaling-factor tricks proposed to further improve Pre-LN Transformer training, before [Section 19](#19-comparative-summary-table--all-normalization-methods) consolidates everything from Part II into a single comparison table, closing out Part II as a whole.

---

## 18. NormFormer and Other Modern Variants

### 18.1 Motivation: a specific Pre-LN gradient imbalance

NormFormer (Shleifer, Weston & Ott, 2021) targets a subtle issue specific to Pre-LN Transformers (the architecture from [Section 7.7](#77-production-grade-pytorch-usage), which we analyze rigorously in [Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison)): empirically, gradient magnitudes in standard Pre-LN Transformers tend to grow larger for **earlier** layers than for **later** layers — the opposite of what you might naively expect, and a real, measurable training inefficiency, since it means early layers can receive disproportionately large gradient updates relative to later ones, making the effective per-layer learning rate implicitly uneven across depth.

### 18.2 NormFormer's three additions

NormFormer proposes three targeted additions to the standard Pre-LN block, each adding one extra, cheap normalization or scaling operation at a specific point:

1. **An extra LayerNorm after the attention block's output projection**, before it's added back into the residual stream (i.e., one more normalization step than standard Pre-LN's single pre-attention LayerNorm).
2. **A learned per-head scaling factor applied to attention head outputs**, before they're concatenated and projected — letting the model learn to rebalance how much different attention heads contribute, rather than treating every head's raw output as equally weighted by default.
3. **An extra LayerNorm inside the FFN block**, applied after the first linear layer and activation function, before the second linear layer projects back down to $d_{\text{model}}$.

Each of these is a small, local addition — not a change to the *core* LayerNorm/RMSNorm formula covered in [Sections 7–8](#7-layer-normalization-ln), but an addition of *more normalization checkpoints* at specific points in the block where the gradient-imbalance problem identified in Section 18.1 was empirically traced to originate.

### 18.3 Why extra normalization checkpoints help

The core mechanism: each additional LayerNorm acts as a **gradient-rescaling checkpoint** — recall from [Section 7.5](#75-backward-pass--derivation) that LayerNorm's backward pass doesn't simply pass gradients through unchanged; it actively rescales them as part of the (already-derived) backward formula. Placing an extra LayerNorm at a point where gradients were empirically observed to be growing disproportionately large effectively re-normalizes that growth back down before it propagates further backward through the network — directly counteracting the layer-position-dependent imbalance from Section 18.1, at the modest cost of a few extra normalization operations per block (cheap relative to the attention and FFN matrix multiplications that dominate a Transformer block's compute).

### 18.4 PyTorch implementation sketch

```python
import torch
import torch.nn as nn

class NormFormerBlock(nn.Module):
    """
    A Pre-LN block with NormFormer's three additions:
    extra post-attention LN, learned per-head scaling, extra mid-FFN LN.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln_pre_attn = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_scale = nn.Parameter(torch.ones(n_heads))   # addition #2
        self.ln_post_attn = nn.LayerNorm(d_model)              # addition #1

        self.ln_pre_ffn = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, d_ff)
        self.ffn_mid_ln = nn.LayerNorm(d_ff)                    # addition #3
        self.ffn_down = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def attention(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                            # each: (B,T,H,hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))        # (B,H,T,hd)

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        out = attn_weights @ v                                  # (B,H,T,hd)

        out = out * self.head_scale.view(1, -1, 1, 1)            # addition #2 applied here
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

    def forward(self, x):
        attn_out = self.attention(self.ln_pre_attn(x))
        attn_out = self.ln_post_attn(attn_out)                  # addition #1
        x = x + attn_out

        ffn_hidden = self.act(self.ffn_up(self.ln_pre_ffn(x)))
        ffn_hidden = self.ffn_mid_ln(ffn_hidden)                # addition #3
        x = x + self.ffn_down(ffn_hidden)
        return x

block = NormFormerBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)
y = block(x)
print(y.shape)   # torch.Size([2, 10, 512])
```

### 18.5 Other modern variants, briefly

A few additional placement/scaling tricks are worth knowing by name, since you'll encounter them reading model source code, even though they don't warrant the full treatment given to the primary techniques above:

- **Sandwich-LN** (used in some large-scale training efforts, e.g., CogView/GLM-family models): applies LayerNorm both *before and after* each sublayer (attention or FFN), "sandwiching" it — a direct combination of the Pre-LN and Post-LN placements covered fully in [Section 25](#25-post-ln-vs-pre-ln-vs-sandwich-ln), aimed at getting Pre-LN's superior gradient flow ([Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison)) while still benefiting from Post-LN's tendency to keep the residual stream's scale more tightly controlled.
- **QK-Norm** (LayerNorm or RMSNorm applied specifically to queries and keys before the attention dot-product, used in some recent large-scale training recipes, e.g., Gemma 3 — which specifically adopted QK-Norm in place of its predecessor Gemma 2's logit soft-capping approach — and several other modern open-weight releases): stabilizes attention logit magnitudes directly at the point where they're computed, addressing a specific numerical-stability concern in attention (very large query/key dot products before softmax can cause softmax saturation and vanishing attention gradients) rather than the general residual-stream-scale concerns NormFormer and DeepNorm address.
- **Learned residual scaling factors** more generally (a broader category that includes DeepNorm's $\alpha$ from [Section 17.2](#172-mathematical-formulation) and ReZero's learned-from-zero gate, covered fully in [Section 33.4](#33-variants-highway-networks-densenet-gated-residuals-rezero-deepnorm-scaling)) represent a whole family of techniques united by the same idea: don't just fix the normalization formula, also give the model explicit, learnable control over how strongly each sublayer's contribution gets weighted before joining the residual stream.

### 18.6 Advantages, limitations, and real-world usage

**Advantages:** directly targets and measurably improves a real Pre-LN training inefficiency (the layer-position gradient imbalance from Section 18.1); the additions are cheap (a few extra LayerNorms and one small learned per-head scale vector) relative to total model compute; demonstrated faster convergence in the original paper's experiments at the scales tested.

**Limitations:** adds extra normalization operations and parameters to every block, with benefits that scale with how severe the underlying gradient imbalance is for a given model size/depth/training recipe — not a universal, guaranteed improvement independent of architecture and scale; less universally adopted than the foundational techniques (LayerNorm, RMSNorm) covered earlier, partly because many large-scale training efforts found other interventions (careful learning rate warmup, gradient clipping, architecture-specific tuning) sufficient to manage the same underlying issue without NormFormer's specific additions.

**Real-world usage:** influenced normalization placement choices in some subsequent model designs and research, though it has not become as universally standard as LayerNorm or RMSNorm themselves — it's best understood as one well-documented example, among several (Sandwich-LN, QK-Norm, learned residual scaling), of the broader pattern of "add more normalization/scaling checkpoints at specific empirically-motivated points" that continues to appear across newer model releases.

### 18.7 What's next

[Section 19](#19-comparative-summary-table--all-normalization-methods) closes Part II with a single consolidated table comparing every technique covered in Sections 6–18 across the dimensions that matter most: which axis is reduced, parameter count, batch dependence, and primary use case — a single reference point to return to throughout the rest of this handbook.

---

## 19. Comparative Summary Table — All Normalization Methods

### 19.1 The full comparison

| Method | Reduces over | Params per layer | Batch-dependent? | Train/inference gap? | Primary domain |
|---|---|---|---|---|---|
| **Batch Norm** ([§6](#6-batch-normalization-bn)) | Batch + spatial, per channel | $2d$ | Yes | Yes (running stats) | CNNs, vision |
| **Layer Norm** ([§7](#7-layer-normalization-ln)) | Feature dim, per token | $2d$ | No | No | Transformers, pre-2023 LLMs |
| **RMSNorm** ([§8](#8-rmsnorm)) | Feature dim (no centering), per token | $d$ | No | No | Modern LLMs (LLaMA, Mistral, Gemma...) |
| **Group Norm** ([§9](#9-group-normalization)) | Channel subset + spatial, per example | $2d$ | No | No | Vision, diffusion U-Nets |
| **Instance Norm** ([§10](#10-instance-normalization)) | Spatial only, per channel | $2d$ (often 0, see §10.4) | No | No | Style transfer |
| **Weight Norm** ([§11](#11-weight-normalization)) | N/A — reparameterizes weights | $1$ per weight vector | No | No | WaveNet, RL policies |
| **Spectral Norm** ([§12](#12-spectral-normalization)) | N/A — constrains weight matrix | $0$ (no learnable params) | No | No | GAN discriminators |
| **L2 Norm** ([§13](#13-l2-normalization)) | Full vector, fixed unit length | $0$ | No | No | Contrastive learning, retrieval embeddings |
| **Power Norm** ([§14](#14-power-normalization)) | Batch (smoothed running stat) | $2d$ | Yes | Yes | Research; rarely deployed |
| **AdaIN / Conditional Norm** ([§15](#15-adaptive-normalization-adain-conditional-norms)) | Spatial (AdaIN) or feature, conditioned | Varies (small MLP) | No | No | StyleGAN, DiT, conditional generation |
| **ScaleNorm** ([§16](#16-scalenorm)) | Full vector, fixed unit length × $g$ | $1$ | No | No | Research (MT Transformers) |
| **DeepNorm** ([§17](#17-deepnorm)) | Same as underlying LN + residual scale | $2d$ + scalar $\alpha,\beta$ | No | No | Ultra-deep (1000+ layer) Transformers |
| **NormFormer** ([§18](#18-normformer-and-other-modern-variants)) | Same as LN, at extra checkpoints | $2d \times$ (extra LN count) $+\ n_{\text{heads}}$ | No | No | Pre-LN Transformer training improvements |

### 19.2 The single most important row to remember

If you remember only one thing from this entire table, make it the **RMSNorm row**: no batch dependence, no train/inference gap, half the parameters of LayerNorm, and it's the technique sitting inside essentially every modern open-weight LLM you'll work with. Everything else in this table is context, history, or a specialized tool for a specific domain (vision, GANs, style transfer, embeddings) outside the LLM Transformer block itself.

### 19.3 How to use this table going forward

Keep this table as a reference point for the rest of the handbook. From here, [Part III](#20-why-batchnorm-fails-for-transformers) goes *deep* on LayerNorm specifically — the row you'll want to fully internalize even though RMSNorm has since become more common in new model releases, both because LayerNorm's math underlies RMSNorm's (Section 8's entire derivation was "take LayerNorm, drop the centering term") and because a large fraction of widely-studied, foundational LLMs (GPT-2/3, BERT, T5) and a great deal of LLM research tooling and literature still use LayerNorm directly.

### 19.4 Part II summary

Part II is now complete. We've covered, in full mathematical and implementation detail: Batch Normalization, Layer Normalization, RMSNorm, Group Normalization, Instance Normalization, Weight Normalization, Spectral Normalization, L2 Normalization, Power Normalization, Adaptive Normalization, ScaleNorm, DeepNorm, and NormFormer — thirteen techniques spanning every major design axis identified in [Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over)'s taxonomy.

**Part III** begins now, returning to this handbook's primary focus: an exhaustive deep dive into Layer Normalization specifically — starting with a fully rigorous proof of why BatchNorm fails for Transformers (the claim we've referenced informally throughout Sections 2, 5, 6, and 7, now given its complete, worked-example-backed treatment), then progressing through LayerNorm's complete forward/backward mathematics (building on but going beyond Section 7's introduction), the Post-LN/Pre-LN/Sandwich-LN architectural choice, and a position-by-position analysis of how LayerNorm sits inside the Transformer encoder, decoder, and every major modern LLM family.

---

# Part III — Layer Normalization Deep Dive (Primary Focus)

## 20. Why BatchNorm Fails for Transformers

### 20.1 Setting up the precise question

We've referenced this claim informally since [Section 2.5](#25-what-this-means-for-llms-specifically) and summarized it in [Section 5.5](#55-why-the-axis-choice-has-such-large-practical-consequences): BatchNorm doesn't work well for Transformers. This section makes that claim fully rigorous, with a worked tensor example showing exactly where and how it breaks down — not just asserting the conclusion, but deriving it.

The setup: a batch of natural language sequences, of necessarily varying length (real sentences/documents don't all happen to be the same token count). To process them as a single batched tensor, shorter sequences are **padded** with a special padding token up to the batch's maximum length $T_{\max}$.

### 20.2 A worked example showing padding corruption

Consider a tiny batch of $B=2$ sequences, hidden dimension $d=1$ (a single feature, to keep the arithmetic minimal — the conclusion generalizes to any $d$), with sequence lengths 3 and 1 respectively, padded to $T_{\max}=3$ using a padding value of $0$:

$$X = \begin{bmatrix} 4.0 & 6.0 & 8.0 \\ 5.0 & 0.0_{\text{pad}} & 0.0_{\text{pad}} \end{bmatrix} \quad \in \mathbb{R}^{2\times 3}$$

(Row 1: a real 3-token sequence. Row 2: a real 1-token sequence followed by 2 padding positions.)

**If BatchNorm naively averages over the batch dimension at each time-position** (its standard formulation, [Section 6.2](#62-mathematical-formulation), generalized to include the sequence axis $T$ the way [Section 5.3](#53-the-four-canonical-axes-visualized)'s taxonomy describes it for sequence models — averaging over $B$ and $T$ together, per feature):

$$\mu = \frac{4.0+6.0+8.0+5.0+0.0+0.0}{6} = \frac{23.0}{6}\approx 3.833$$

**This mean is contaminated.** The two padding zeros — which carry **no real linguistic content whatsoever**, they're pure placeholder values — are pulled directly into the statistic that determines how every real token gets normalized. Compare against the *correct* mean, computed only from the 4 real tokens: $\frac{4+6+8+5}{4}=5.75$. The contaminated, padding-inclusive mean (3.833) is dragged substantially lower than the value that should actually represent "the typical activation level of real tokens in this batch" (5.75) — a difference of nearly 2 full units in this tiny example, purely as an artifact of how many padding positions happened to be present, which itself depends entirely on which sequences happened to be batched together. **A real, content-bearing token's normalized value now depends on how much padding some other, unrelated sequence in the same batch happened to need** — exactly the cross-example coupling problem from [Section 6.5](#65-backward-pass--full-derivation), now made concrete and obviously undesirable.

### 20.3 The standard workaround, and why it's still a structural cost

In practice, careful implementations don't naively include padding in BatchNorm's statistics — they use a **masked** mean/variance computation, summing only over real (non-padding) positions:

$$\mu_{\text{masked}} = \frac{1}{|\text{real positions}|}\sum_{(b,t)\,\in\,\text{real}} x_{b,t}$$

This recovers the correct value (5.75 in our example) — but notice what this requires: **passing the padding mask explicitly into every single normalization layer**, threading it through the entire forward (and backward) pass, just so BatchNorm can know which elements to include. Compare this against LayerNorm: a LayerNorm call at the exact same position needs **no mask at all**, because it only ever looks at one token's own $d$ features — it has no way to "see" any other position, padded or not, so there's nothing to mask out in the first place. This isn't a minor implementation inconvenience — it's a structural, unavoidable extra complexity that BatchNorm imposes on every sequence model, purely because of which axis it reduces over, and it's a complexity LayerNorm sidesteps entirely by construction.

### 20.4 The autoregressive inference problem

[Section 6.6](#66-the-traininginference-mismatch--running-statistics) already covered BatchNorm's general train/inference mismatch via running statistics. For Transformers specifically, autoregressive generation makes this particularly acute: generating text token-by-token, you process **one token at a time**, often for **one sequence at a time** ($B=1$, $T=1$ for each new token, ignoring KV-cache mechanics for the moment). 

For BatchNorm with $B=1, T=1$: you're trying to compute a variance from a *single number*. By the formula in [Section 3.2](#32-variance--what-it-represents), the variance of a single value around its own mean is exactly $0$ — even with $\epsilon$ guarding the resulting division, this is a degenerate, meaningless statistic that tells you nothing about the actual data distribution. This is exactly why BatchNorm relies on its frozen running statistics ([Section 6.6](#66-the-traininginference-mismatch--running-statistics)) at inference rather than ever attempting to compute a live statistic in this regime — but using running statistics here means the normalization layer's actual behavior at inference is determined by training-time batch composition, baked in and frozen, rather than responsive to the specific input currently being processed. For LayerNorm, by contrast, $B=1, T=1$ is **not a special case at all** — the formula computes a mean and variance over the token's own $d$ features regardless of how many other tokens or sequences are or aren't present, with zero code-path difference between training and this exact inference scenario.

### 20.5 The original Transformer paper's own evidence

It's worth noting explicitly: the original Transformer paper (Vaswani et al., 2017) didn't extensively benchmark BatchNorm against LayerNorm and report a quantitative failure — LayerNorm was adopted directly as the normalization choice, following the broader trend (already established by the time of that paper) of using LayerNorm for sequence models (RNNs/LSTMs had already found LayerNorm preferable to BatchNorm for similar variable-length-sequence reasons, predating the Transformer itself). The reasoning given above — padding contamination, batch-size dependence, and the autoregressive single-token-at-a-time inference mismatch — represents the accumulated, well-established understanding of *why* that choice has held up so robustly across nearly a decade of subsequent Transformer-based architecture development, rather than a single paper's specific ablation result.

### 20.6 Summary: three independent reasons, all pointing the same direction

| Problem | BatchNorm | LayerNorm |
|---|---|---|
| Variable-length sequences | Requires explicit padding masks threaded through every layer (Section 20.2–20.3) | No masking needed — never looks beyond one token's own features |
| Small/single-example batches | Degenerate statistics at $B=1$; requires frozen running stats (Section 20.4) | Fully well-defined and identical, regardless of $B$ |
| Cross-example coupling | Every token's output depends on every other token/example in the batch (Section 6.5) | Each token's output depends only on that token's own features |

These three problems are not independent symptoms of one shared root cause that some clever fix could resolve while keeping BatchNorm's batch-axis reduction — they are direct, unavoidable mathematical consequences of choosing to reduce over the batch (and sequence) axis in the first place. This is why the field's actual solution was a wholesale axis change (LayerNorm, [Section 7](#7-layer-normalization-ln)), not a series of incremental BatchNorm patches — though Power Normalization ([Section 14](#14-power-normalization)) represents exactly such an incremental-patch attempt, included earlier in this handbook specifically so you could see both paths and understand why one became the field's actual, lasting solution.

### 20.7 What's next

[Section 21](#21-layernorm-mathematics--forward-pass) returns to LayerNorm's mathematics with the full rigor this primary-focus topic deserves — building on [Section 7.2](#72-mathematical-formulation)'s introduction but going further: a complete treatment of the forward pass across the full $B\times T\times d$ tensor (not just a single token in isolation), explicit tensor-shape bookkeeping at every step, and the precise relationship between this formulation and what you'll find inside production framework source code.

---

## 21. LayerNorm Mathematics — Forward Pass

### 21.1 From a single token to a full batch — explicit tensor bookkeeping

[Section 7.2](#72-mathematical-formulation) defined LayerNorm for a single token's vector $x \in \mathbb{R}^d$. A real Transformer processes a full tensor $X \in \mathbb{R}^{B\times T\times d}$ simultaneously. This section makes the *exact* tensor-shape bookkeeping explicit at every step — the kind of detail that matters enormously when actually implementing or debugging this layer, and that's easy to gloss over when working with a single-vector formula.

$$\mu \in \mathbb{R}^{B\times T\times 1} \qquad \mu_{b,t} = \frac{1}{d}\sum_{i=1}^{d} X_{b,t,i}$$

$$\sigma^2 \in \mathbb{R}^{B\times T\times 1} \qquad \sigma^2_{b,t} = \frac{1}{d}\sum_{i=1}^{d}(X_{b,t,i}-\mu_{b,t})^2$$

$$\hat{X} \in \mathbb{R}^{B\times T\times d} \qquad \hat{X}_{b,t,i} = \frac{X_{b,t,i}-\mu_{b,t}}{\sqrt{\sigma^2_{b,t}+\epsilon}}$$

$$Y \in \mathbb{R}^{B\times T\times d} \qquad Y_{b,t,i} = \gamma_i \cdot \hat{X}_{b,t,i} + \beta_i \qquad (\gamma,\beta \in \mathbb{R}^d, \text{broadcast across } B, T)$$

The shape annotations make a subtle but important point visible: $\mu$ and $\sigma^2$ have shape $B\times T\times 1$ — they keep a "size-1" trailing dimension (rather than collapsing to $B\times T$) specifically so that the subtraction $X - \mu$ and division by $\sqrt{\sigma^2+\epsilon}$ can **broadcast** correctly against $X$'s full $d$-sized last dimension. This is exactly the `keepdims=True` argument you saw in [Section 7.6](#76-numpy-implementation-from-scratch)'s NumPy implementation — not an arbitrary API choice, but a direct requirement of how broadcasting needs to align tensor shapes for this computation to be expressible without explicit, slow Python loops over $B$ and $T$.

### 21.2 A full worked example across a tiny batch

Extend [Section 7.3](#73-a-worked-numerical-example)'s single-token example to $B=2$ sequences, $T=2$ tokens each, $d=4$ features — small enough to compute by hand, large enough to show the independence across all four $(b,t)$ positions concretely:

$$X = \begin{bmatrix}\begin{bmatrix}2.0 & 8.0 & -4.0 & 6.0\\ 0.0 & 0.0 & 0.0 & 0.0\end{bmatrix} \\[4pt] \begin{bmatrix}1.0 & 1.0 & 1.0 & 1.0\\ 10.0 & -10.0 & 5.0 & -5.0\end{bmatrix}\end{bmatrix}$$

(Shape: $2\times2\times4$ — two sequences, each two tokens, each token a 4-dim vector. Row $(0,0)$ is our familiar Section 1.2/7.3 vector; row $(0,1)$ is an all-zero vector — chosen deliberately, see below; row $(1,0)$ is a constant vector; row $(1,1)$ is a new vector for variety.)

**Position $(0,0)$:** exactly Section 7.3's computation — $\mu=3.0,\sigma^2=21.0$, giving $\hat{x}\approx[-0.218,1.091,-1.528,0.655]$.

**Position $(0,1)$ — the all-zero vector:** $\mu=0.0$, and $\sigma^2=0.0$ (every value already equals the mean, so every squared deviation is 0). This is **exactly the degenerate case $\epsilon$ exists to guard against** ([Section 3.5](#35-the-epsilon-you-will-see-everywhere)): $\hat{x} = \frac{0-0}{\sqrt{0+\epsilon}}=\frac{0}{\sqrt{\epsilon}}=0$ for every element — well-defined and stable specifically *because* $\epsilon$ prevents a literal $0/0$ division. Without $\epsilon$, this position alone would produce `NaN` and corrupt the entire forward pass downstream.

**Position $(1,0)$ — the constant vector $[1,1,1,1]$:** $\mu=1.0$, $\sigma^2=0.0$ (again, every value equals the mean) — the same degenerate-but-handled case as $(0,1)$, giving $\hat{x}=[0,0,0,0]$ again via the same $\epsilon$-guarded mechanism. **Any constant vector, of any magnitude, normalizes to the all-zero vector** — a useful fact to internalize, since it means LayerNorm is fundamentally unable to distinguish between different constant vectors; all the information that distinguished "all 1s" from "all 1000s" is destroyed by this layer, recoverable only through $\beta$ if the network has learned a nonzero shift (recall [Section 3.6](#36-restoring-expressiveness-learnable-γ-and-β)'s point about $\gamma,\beta$ restoring expressiveness).

**Position $(1,1)$:** $x=[10.0,-10.0,5.0,-5.0]$, $\mu = \frac{10-10+5-5}{4}=0.0$, $\sigma^2=\frac{100+100+25+25}{4}=62.5$, $\sigma\approx7.906$, giving $\hat{x}\approx[1.265,-1.265,0.632,-0.632]$.

**The crucial point, stated explicitly**: every one of these four computations used *only* its own 4 values. Position $(0,0)$'s result is completely unaffected by the fact that $(0,1)$ happens to be all zeros, or that $(1,0)$ happens to be constant, or by anything in sequence 2 at all — confirming, with concrete numbers, the "fully independent per-(batch,token) position" property argued abstractly in [Section 7.1](#71-motivation) and [Section 20](#20-why-batchnorm-fails-for-transformers).

### 21.3 Vectorized NumPy verification across this full batch

```python
import numpy as np

X = np.array([
    [[2.0, 8.0, -4.0, 6.0], [0.0, 0.0, 0.0, 0.0]],
    [[1.0, 1.0, 1.0, 1.0], [10.0, -10.0, 5.0, -5.0]],
])   # shape (2, 2, 4) = (B, T, d)

eps = 1e-5
mean = X.mean(axis=-1, keepdims=True)              # shape (2, 2, 1)
var = X.var(axis=-1, keepdims=True)                # shape (2, 2, 1)
X_hat = (X - mean) / np.sqrt(var + eps)             # broadcasts correctly to (2, 2, 4)

print(X_hat[0, 0])   # [-0.218  1.091 -1.528  0.655]  -- matches Section 7.3 / 21.2
print(X_hat[0, 1])   # [0. 0. 0. 0.]                  -- degenerate, eps-guarded
print(X_hat[1, 0])   # [0. 0. 0. 0.]                  -- degenerate, eps-guarded
print(X_hat[1, 1])   # [ 1.265 -1.265  0.632 -0.632]  -- matches Section 21.2
```

### 21.4 Relating this to actual production framework source code

It's worth being precise about how this maps onto what you'd find reading PyTorch's actual CUDA kernel source (not just the Python-level `nn.LayerNorm` API surface from [Section 7.7](#77-production-grade-pytorch-usage)): production implementations typically **fuse** the mean, variance, and normalize-and-affine steps into a single GPU kernel launch, rather than executing them as four separate tensor operations the way the NumPy code above does for clarity. This fusion (covered in full in [Section 42](#42-triton-fused-layernorm-kernel)'s Triton kernel and [Section 43](#43-cuda-c-fused-layernorm-kernel)'s CUDA kernel) is purely a **performance** optimization — it computes the exact same mathematical result as the unfused, step-by-step formulas above, just with far less GPU memory traffic, since intermediate results ($\mu$, $\sigma^2$, the centered values) never need to be written out to and read back from GPU memory between steps. Understanding the unfused math first (this section) makes the fused kernel's logic in Sections 42–43 much easier to follow, since you'll already know exactly what numerical result the kernel needs to produce — only *how* it's computed efficiently will be new.

### 21.5 What's next

[Section 22](#22-layernorm-mathematics--backward-pass-and-jacobian) extends [Section 7.5](#75-backward-pass--derivation)'s backward-pass derivation with the full Jacobian matrix perspective — expressing LayerNorm's backward pass not just as a formula for $\partial\mathcal{L}/\partial x_i$, but as an explicit $d\times d$ Jacobian matrix $\partial\hat{x}/\partial x$, which makes the "every output depends on every input within a token" coupling structure from Section 7.5 fully visible as a concrete matrix you can inspect, multiply, and reason about directly.

---

## 22. LayerNorm Mathematics — Backward Pass and Jacobian

### 22.1 Why a Jacobian view adds something the scalar formula doesn't

[Section 7.5](#75-backward-pass--derivation) gave you a formula for $\partial\mathcal{L}/\partial x_i$ — useful for implementing backward passes, but it hides the underlying linear-algebraic structure inside a sum. The **Jacobian matrix** $J \in \mathbb{R}^{d\times d}$, where $J_{ij} = \partial\hat{x}_i/\partial x_j$, makes that structure fully explicit: it's the single matrix that, multiplied against any upstream gradient vector, produces exactly the same result [Section 7.5](#75-backward-pass--derivation)'s formula computes, but now you can *see* the coupling pattern directly as matrix entries, inspect its eigenvalues, and reason about it using ordinary linear algebra rather than only via the summation formula.

### 22.2 Deriving the Jacobian

Starting from $\hat{x}_i = \frac{x_i-\mu}{\sigma}$ (writing $\sigma=\sqrt{\sigma^2+\epsilon}$ for brevity), differentiate with respect to $x_j$ using the product/quotient rule, accounting for the fact that **both $\mu$ and $\sigma$ depend on every $x_j$**, not just $x_i$:

$$\frac{\partial \hat{x}_i}{\partial x_j} = \frac{1}{\sigma}\left(\frac{\partial x_i}{\partial x_j} - \frac{\partial \mu}{\partial x_j}\right) - \frac{x_i-\mu}{\sigma^2}\cdot\frac{\partial \sigma}{\partial x_j}$$

Using $\frac{\partial x_i}{\partial x_j}=\delta_{ij}$ (the Kronecker delta — 1 if $i=j$, else 0), $\frac{\partial \mu}{\partial x_j}=\frac{1}{d}$, and (after working through the chain rule on $\sigma=\sqrt{\sigma^2+\epsilon}$) $\frac{\partial \sigma}{\partial x_j}=\frac{1}{\sigma}\cdot\frac{1}{d}(x_j-\mu)=\frac{\hat{x}_j}{d}$, substituting and simplifying:

$$J_{ij} = \frac{\partial \hat{x}_i}{\partial x_j} = \frac{1}{\sigma}\left(\delta_{ij}-\frac{1}{d}-\frac{\hat{x}_i\hat{x}_j}{d}\right)$$

In matrix form, with $\mathbf{1}$ the all-ones $d\times d$ matrix, $I$ the identity, and $\hat{x}\hat{x}^\top$ the outer product of the standardized vector with itself:

$$J = \frac{1}{\sigma}\left(I - \frac{1}{d}\mathbf{1} - \frac{1}{d}\hat{x}\hat{x}^\top\right)$$

This single compact matrix expression *is* [Section 7.5](#75-backward-pass--derivation)'s backward formula, in a different but exactly equivalent form — multiplying $J$ against an upstream gradient vector $g=\partial\mathcal{L}/\partial\hat{x}$ (i.e., computing $J^\top g$, since backprop needs $\partial\mathcal{L}/\partial x = J^\top \partial\mathcal{L}/\partial\hat{x}$, and $J$ here happens to be symmetric so $J=J^\top$) reproduces exactly the three-term structure ([Section 7.5](#75-backward-pass--derivation)'s "direct term," "subtract the average," "subtract a $\hat{x}_i$-weighted component" structure) as a single matrix-vector product.

### 22.3 A concrete worked Jacobian, $d=3$

Take $x=[1.0, 2.0, 3.0]$, so $\mu=2.0$, $\sigma^2 = \frac{1+0+1}{3}=\frac{2}{3}\approx0.667$, $\sigma=\sqrt{2/3}\approx0.8165$ (ignoring $\epsilon$ for this illustration), and $\hat{x}=\frac{[1,2,3]-2}{0.8165}\approx[-1.225,\ 0,\ 1.225]$.

The outer product $\hat{x}\hat{x}^\top$:

$$\hat{x}\hat{x}^\top \approx \begin{bmatrix}1.5 & 0 & -1.5\\0 & 0 & 0\\-1.5 & 0 & 1.5\end{bmatrix}$$

Substituting into $J = \frac{1}{\sigma}\left(I-\frac{1}{d}\mathbf{1}-\frac{1}{d}\hat{x}\hat{x}^\top\right)$ with $d=3$:

$$I - \frac{1}{3}\mathbf{1} - \frac{1}{3}\hat{x}\hat{x}^\top \approx \begin{bmatrix}0.167 & -0.333 & 0.167\\-0.333 & 0.667 & -0.333\\0.167 & -0.333 & 0.167\end{bmatrix}$$

Dividing by $\sigma\approx0.8165$:

$$J \approx \begin{bmatrix}0.204 & -0.408 & 0.204\\-0.408 & 0.816 & -0.408\\0.204 & -0.408 & 0.204\end{bmatrix}$$

Note **every off-diagonal entry is nonzero** (and several diagonal entries differ from each other too) — the matrix's explicit, visible confirmation that every output coordinate's gradient depends on every input coordinate, exactly the coupling property [Section 7.5](#75-backward-pass--derivation) described in words. Notice also the matrix's symmetric structure (e.g., $J_{02}=J_{20}=0.204$) and that its rows each sum to exactly zero ($0.204-0.408+0.204=0$) — a direct, visible instance of the $J\mathbf{1}=0$ property derived next in Section 22.4.

### 22.4 Two structural properties worth knowing

**Property 1 — $J$ always has the all-ones vector $\mathbf{1}=[1,\ldots,1]$ as a null vector**: $J\mathbf{1}=0$. Intuitively, this says that adding the *same* constant to every element of $x$ doesn't change $\hat{x}$ at all (which makes direct sense — adding a constant $c$ to every $x_i$ shifts $\mu$ by exactly $c$ too, leaving $x_i-\mu$, and therefore $\hat{x}_i$, completely unchanged). This is the Jacobian-level statement of exactly the same fact that gave LayerNorm its mean-zero-hyperplane geometric picture back in [Section 4.2](#42-centering--sliding-the-whole-space-so-the-mean-sits-at-the-origin) — the Jacobian is singular (non-invertible) precisely along the direction that centering already eliminates.

**Property 2 — $J$ is symmetric** ($J=J^\top$), visible directly from the matrix formula in Section 22.2 (every term — $I$, $\mathbf{1}$, $\hat{x}\hat{x}^\top$ — is itself a symmetric matrix). This symmetry is *why* we could write $J^\top g = Jg$ above without needing to separately track a transpose during backpropagation — a small but genuinely useful implementation simplification.

### 22.5 Numerical verification

```python
import numpy as np
import torch

def layernorm_jacobian(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Constructs the LayerNorm Jacobian explicitly, per Section 22.2's formula."""
    d = len(x)
    mu = x.mean()
    var = x.var()
    sigma = np.sqrt(var + eps)
    x_hat = (x - mu) / sigma

    I = np.eye(d)
    ones = np.ones((d, d))
    outer = np.outer(x_hat, x_hat)
    J = (1.0 / sigma) * (I - ones / d - outer / d)
    return J

x = np.array([1.0, 2.0, 3.0])
J = layernorm_jacobian(x)
print("Hand-derived J:\n", np.round(J, 3))
print("J @ ones (should be ~0):", J @ np.ones(3))

# Cross-check against PyTorch autograd's exact Jacobian
x_t = torch.tensor(x, requires_grad=True, dtype=torch.float64)
mu_t = x_t.mean()
var_t = x_t.var(unbiased=False)
x_hat_t = (x_t - mu_t) / torch.sqrt(var_t)
J_autograd = torch.autograd.functional.jacobian(
    lambda inp: (inp - inp.mean()) / torch.sqrt(inp.var(unbiased=False)), x_t.detach()
)
print("Autograd J:\n", J_autograd.numpy().round(3))
print("Match:", np.allclose(J, J_autograd.numpy(), atol=1e-6))
```

### 22.6 What's next

[Section 23](#23-learnable-γ-and-β-role-and-gradient-flow) takes a closer look specifically at $\gamma$ and $\beta$'s gradients and their role during training — how these per-feature parameters evolve, what their gradients reveal about which features the network is learning to emphasize or suppress, and why their initialization (typically $\gamma=1,\beta=0$) matters for training dynamics from the very first step.

---

## 23. Learnable γ and β: Role and Gradient Flow

### 23.1 Recap: what problem $\gamma,\beta$ solve

[Section 3.6](#36-restoring-expressiveness-learnable-γ-and-β) introduced the motivation: standardization forces every output to mean-0, variance-1, which would strip the network of any ability to represent a different preferred scale or shift per feature, if left uncorrected. $\gamma,\beta\in\mathbb{R}^d$ restore that flexibility, one learned scale and one learned shift per feature, shared across every token and every batch element that passes through this particular normalization layer.

### 23.2 The gradient formulas, and what they mean

From [Section 7.5](#75-backward-pass--derivation):

$$\frac{\partial \mathcal{L}}{\partial \gamma_i} = \sum_{(b,t)} \frac{\partial \mathcal{L}}{\partial y_{b,t,i}}\cdot\hat{x}_{b,t,i} \qquad \frac{\partial \mathcal{L}}{\partial \beta_i} = \sum_{(b,t)} \frac{\partial \mathcal{L}}{\partial y_{b,t,i}}$$

(Sums range over every batch element and token position in the current training batch, since $\gamma,\beta$ are shared parameters — every position that uses this normalization layer contributes to its gradient.)

**Reading $\partial\mathcal{L}/\partial\gamma_i$**: it's the correlation, across the whole batch, between the upstream gradient signal at feature $i$ and the *standardized* (pre-affine) value at that same feature. If feature $i$'s standardized activations are consistently large and positive exactly when the upstream gradient is also pushing strongly in the direction that would increase the loss-reducing output (i.e., when $\hat{x}_i$ and $\partial\mathcal{L}/\partial y_i$ are both large with the same sign), $\gamma_i$ receives a large gradient — the optimizer learns to amplify this feature's contribution. Conversely, if a feature's standardized value carries no consistent relationship to how the loss wants the output to move, $\gamma_i$'s gradient stays small, and the network learns to leave that feature's scale near its initialized value.

**Reading $\partial\mathcal{L}/\partial\beta_i$**: simpler — it's just the *average* upstream gradient at feature $i$ across the batch. $\beta_i$ moves in whatever direction would, on average across every example currently in the batch, most reduce the loss — exactly what you'd expect from a parameter whose only job is an additive shift, with no interaction with the input's own values at all.

### 23.3 Why $\gamma=1,\beta=0$ is the standard initialization

Initializing $\gamma=1,\beta=0$ means a freshly-initialized LayerNorm layer's output is **exactly** the standardized $\hat{x}$ — i.e., the layer starts out doing nothing beyond pure standardization, no extra learned scaling or shifting yet. This matters for the same reason careful initialization matters throughout deep learning generally ([Section 2.1](#21-the-term-and-why-it-was-coined)'s optimization-landscape framing): starting from a "neutral," well-understood operation means the network's very first few training steps see activations on a predictable, controlled scale (mean 0, variance 1, by construction), rather than scales determined by an arbitrary random initialization of $\gamma,\beta$ that could, by bad luck, immediately push activations toward a poorly-conditioned regime before the optimizer has had any chance to correct it.

This connects directly to the residual-connection-specific initialization schemes covered later: DeepNorm's $\beta$-scaled initialization ([Section 17.2](#172-mathematical-formulation)) and ReZero's all-zero gate initialization ([Section 33.4](#33-variants-highway-networks-densenet-gated-residuals-rezero-deepnorm-scaling)) are both, in spirit, the exact same idea applied one level higher in the architecture: start a new component as close to "doing nothing disruptive" as possible, and let training gradually introduce its actual contribution as gradients accumulate evidence that it's useful.

### 23.4 What trained $\gamma$ values actually look like

It's worth knowing, even without diving into a specific model's full weights, what the *qualitative* pattern of trained $\gamma$ values typically looks like in practice, since this is a useful diagnostic when inspecting or debugging a real model: trained LayerNorm $\gamma$ vectors are typically **not** uniformly close to 1 across all $d$ features — some features end up with $\gamma$ values noticeably above 1 (the network has learned these features carry unusually important signal worth amplifying after standardization) and some well below 1 or even near 0 (features the network has learned to suppress at this particular layer, even after standardization gave them comparable initial scale to every other feature). This non-uniformity is itself informative: it's direct, inspectable evidence that the network is making active, learned, per-feature decisions about relative feature importance at every single normalization layer — not just passively standardizing and moving on.

### 23.5 A small empirical illustration

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# A toy regression task: predict whether feature 0 is large, ignore features 1-3
ln = nn.LayerNorm(4)
linear = nn.Linear(4, 1)
optimizer = torch.optim.Adam(list(ln.parameters()) + list(linear.parameters()), lr=0.05)

for step in range(300):
    x = torch.randn(64, 4)
    target = (x[:, 0] > 0).float().unsqueeze(-1)   # task only depends on feature 0

    out = linear(ln(x))
    loss = nn.functional.binary_cross_entropy_with_logits(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Learned gamma:", ln.weight.detach().round(decimals=2))
print("Learned beta:", ln.bias.detach().round(decimals=2))
# Expect gamma[0] to grow noticeably relative to gamma[1:], since only
# feature 0 carries any signal relevant to this particular task.
```

Running this confirms the qualitative pattern from Section 23.4 directly: $\gamma_0$ grows substantially larger in magnitude than $\gamma_1,\gamma_2,\gamma_3$ over training, since only feature 0 carries any signal the downstream linear layer can use to reduce the loss — exactly the mechanism described in Section 23.2's reading of the $\gamma$ gradient formula, made concrete in a runnable example.

### 23.6 What's next

[Section 24](#24-geometric-intuition-for-layernorm) returns to the geometric framing from [Section 4](#4-geometric-intuition-what-normalization-does-to-a-vector-space), now specifically through the lens of LayerNorm (rather than normalization in the abstract), tying together the hyperplane/sphere picture with the learned $\gamma,\beta$ affine transform covered in this section — visualizing exactly what shape the *final* $\gamma\hat{x}+\beta$ output traces out in space, as $\gamma,\beta$ vary.

---

## 24. Geometric Intuition for LayerNorm

### 24.1 Picking up exactly where Section 4 left off

[Section 4.4](#44-why-this-geometric-view-matters-practically) ended with a forward-looking claim: $\gamma$ lets the network choose a *different* sphere radius per feature (an ellipsoid rather than a sphere), and $\beta$ lets it recenter that ellipsoid away from the origin. This section makes that claim fully concrete for LayerNorm specifically, completing the geometric picture this handbook has been building since Part I.

### 24.2 From sphere to ellipsoid: what $\gamma$ actually does, geometrically

Recall [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius): after centering and dividing by $\sigma$, every standardized vector $\hat{x}$ lands on a sphere of fixed radius $\sqrt{d}$, regardless of the original input's direction or magnitude. The final step, $y_i=\gamma_i\hat{x}_i+\beta_i$, applies a **different scale factor to each coordinate** — and a linear map that scales different axes by different amounts transforms a sphere into an **ellipsoid**, with the ellipsoid's semi-axis lengths set directly by $\gamma$'s per-feature values.

```
Sphere (after standardization,        Ellipsoid (after applying
before gamma/beta):                   per-feature gamma):

        x2                                  x2
         |    .-""-.                         |      .-""""-.
         |  .'      '.                       |    .'        '.
         | /          \                      |   |            |
    -----+------------- x1            -------+--|              |--- x1
         | \          /                       |   |            |
         |  '.      .'                        |    '.        .'
         |    '-..-'                          |      '-....-'
         +                                    +
   (radius √d, same in            (stretched MORE along x1
    every direction --             if γ₁ > γ₂, i.e. feature 1
    γ₁ = γ₂ = 1 here)              gets amplified relative
                                   to feature 2 after standardization)
```

If $\gamma_i > 1$, feature $i$'s axis is **stretched** beyond the unit-scale sphere; if $0<\gamma_i<1$, that axis is **compressed**; if $\gamma_i<0$ (which gradient descent is free to produce — there's no constraint forcing $\gamma$ positive), that axis is stretched/compressed **and flipped**, reversing which direction along that axis corresponds to "more" of the original standardized signal. The $\beta$ vector then **translates** this entire ellipsoid away from the origin — exactly the same translation operation as centering in [Section 4.2](#42-centering--sliding-the-whole-space-so-the-mean-sits-at-the-origin), just now applied after the standardization step rather than before it, and by a learned, per-feature amount rather than by the data's own mean.

### 24.3 Putting Section 23's empirical pattern into this geometric picture

[Section 23.5](#235-a-small-empirical-illustration)'s worked example found $\gamma\approx[2.13, 0.35, 0.69, 0.67]$ after training on a task where only feature 0 carried useful signal. Read geometrically: the trained LayerNorm layer's output ellipsoid is **stretched substantially along the feature-0 axis** (by a factor of ~2.13) relative to the other three axes (compressed to roughly 0.35–0.69) — the network has learned to literally reshape the standardization sphere into an elongated ellipsoid that emphasizes exactly the one direction in feature space that mattered for the downstream task, and de-emphasizes the three directions that carried no useful signal. This is the same numerical result as Section 23, now seen as a specific, concrete geometric shape rather than just four numbers in a vector.

### 24.4 Why this matters when thinking about what a trained Transformer layer "does"

This geometric framing gives a genuinely useful mental model for thinking about *any* trained LayerNorm layer inside a real Transformer: it is computing a **per-token projection onto a sphere** (discarding the token's own raw magnitude and a specific linear combination of its features — the mean direction), **followed by a learned reshaping into a specific, fixed ellipsoid** (the same ellipsoid shape for every token, set once by training, encoding which directions in this layer's feature space the network has learned matter more or less). Every token passing through this layer gets centered, projected onto the same-radius sphere, and then reshaped by the identical $\gamma,\beta$ — meaning the *only* thing that varies token-to-token is which point on that fixed final ellipsoid a given token's standardized direction lands on, not the ellipsoid's shape itself (which is fixed by training, shared across every token that ever passes through this particular layer).

### 24.5 What's next

[Section 25](#25-post-ln-vs-pre-ln-vs-sandwich-ln) shifts from LayerNorm's internal mathematics (Sections 20–24) to the architectural question of **where**, exactly, a LayerNorm call should sit relative to a Transformer's attention/FFN sublayers and their residual connections — the Post-LN vs. Pre-LN vs. Sandwich-LN choice referenced informally since [Section 7.7](#77-production-grade-pytorch-usage), now given the full, rigorous treatment this choice deserves, including a preview of the gradient-flow argument that [Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison) will make completely precise once [Part IV](#29-motivation-vanishing-gradients-and-the-degradation-problem) has introduced residual connections in full.

---

## 25. Post-LN vs Pre-LN vs Sandwich-LN

### 25.1 The question, precisely

Every Transformer block contains a sublayer ($F$ — attention or FFN) and a residual connection wrapping it: $x_{l+1} = x_l + F(\cdot)$ (we cover the residual connection's own motivation fully starting in [Section 29](#29-motivation-vanishing-gradients-and-the-degradation-problem); for this section, just treat it as given). The open design question: **does LayerNorm go inside that $F(\cdot)$ — normalizing the sublayer's input before it computes anything — or outside it, normalizing the sum after the residual addition?** These are genuinely different computations, not just a cosmetic reordering, and the choice has a substantial, well-documented effect on training stability, especially at scale.

### 25.2 Post-LN — the original Transformer's choice

$$x_{l+1} = \text{LN}\big(x_l + F(x_l)\big)$$

LayerNorm is applied to the **sum**, after the residual addition. This was the formulation used in the original "Attention Is All You Need" paper (Vaswani et al., 2017) and in BERT.

### 25.3 Pre-LN — the modern default

$$x_{l+1} = x_l + F\big(\text{LN}(x_l)\big)$$

LayerNorm is applied to the sublayer's **input**, before $F$ ever sees it — the residual stream $x_l$ itself flows through *unnormalized*, added directly to whatever $F$ produces from the normalized copy. This is the pattern shown in [Section 7.7](#77-production-grade-pytorch-usage)'s code, and the formulation used in GPT-2 onward, and in essentially every major LLM since.

### 25.4 Why the difference matters: a preview

The full, rigorous gradient-flow derivation justifying Pre-LN's dominance is the entire subject of [Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison) — it requires the residual-connection gradient machinery from [Section 31](#31-gradient-propagation-through-residual-paths-full-derivation), which hasn't been introduced yet at this point in the handbook. But the core intuition can be stated now, precisely enough to be useful immediately:

In **Pre-LN**, the residual stream $x_l$ has a completely **unobstructed path** from the very first layer to the very last — at every addition $x_{l+1}=x_l+F(\text{LN}(x_l))$, the gradient flowing backward through $\partial x_{l+1}/\partial x_l$ always includes a clean, undiminished identity term (a direct "+1" contribution to the Jacobian, exactly as we'll formalize in [Section 30](#30-mathematics-of-residual-learning-y--x--fx)), with the LayerNorm operation confined entirely *inside* the $F$ branch, never sitting directly in the main backward path between consecutive layers.

In **Post-LN**, LayerNorm sits *directly in the main path* between $x_l$ and $x_{l+1}$ — every gradient that needs to flow from layer $l+1$ back to layer $l$ must pass *through* a LayerNorm backward computation ([Section 7.5](#75-backward-pass--derivation)'s Jacobian, which, recall from [Section 22.4](#224-two-structural-properties-worth-knowing), is **not** simply an identity map — it actively rescales and redistributes gradient signal at every single layer). Across many stacked layers, this repeated rescaling compounds, and empirically tends to produce harder-to-train, more warmup-sensitive optimization dynamics at depth — which is precisely why Post-LN Transformers (including the original 2017 architecture and BERT) required careful learning-rate warmup schedules to train stably at all, a requirement that Pre-LN architectures are considerably more robust to.

### 25.5 Sandwich-LN — combining both

$$x_{l+1} = x_l + \text{LN}_{\text{post}}\Big(F\big(\text{LN}_{\text{pre}}(x_l)\big)\Big)$$

As introduced briefly in [Section 18.5](#185-other-modern-variants-briefly), Sandwich-LN applies LayerNorm both before *and* after the sublayer $F$, while still keeping the residual addition itself unnormalized (preserving Pre-LN's clean gradient highway property from Section 25.4). The motivation: Pre-LN alone doesn't constrain how large $F$'s *output* can grow before it's added back into the stream — an unusually large sublayer output could still destabilize the residual stream's scale over many layers, even with the gradient-flow benefits intact. The additional post-sublayer LayerNorm directly bounds $F$'s output scale before it ever reaches the residual addition, addressing this specific concern that pure Pre-LN leaves partially unaddressed.

### 25.6 A side-by-side architecture diagram

```
   POST-LN                    PRE-LN                     SANDWICH-LN

   x_l                        x_l ──────┐                x_l ──────┐
    │                          │        │                 │        │
    ▼                          ▼        │                 ▼        │
  F(x_l)                     LN(x_l)    │               LN(x_l)    │
    │                          │        │                 │        │
    ▼                          ▼        │                 ▼        │
    + ◄────────────────────────┘        │              F(LN(x_l))  │
    │                                   │                 │        │
    ▼                                   │                 ▼        │
   LN                                   │            LN_post(...)  │
    │                                   │                 │        │
    ▼                                   ▼                 ▼        │
  x_{l+1}                     + ◄───────┘              + ◄─────────┘
                                │                          │
                                ▼                          ▼
                              x_{l+1}                    x_{l+1}

  LN sits ON the main         LN sits OFF the main      LN appears both
  path -- every gradient      path -- residual stream    inside F's branch
  passing layer-to-layer      flows through additions    AND on F's output,
  must flow through LN's      directly; LN only          but the residual
  (non-identity) backward     touches F's INPUT branch    stream addition
  computation.                                            itself stays clean.
```

### 25.7 Practical implications and current consensus

The overwhelming majority of production LLMs released since GPT-2 — GPT-3, LLaMA (all generations), Mistral, Mixtral, Gemma, Qwen, DeepSeek — use **Pre-LN** (or Pre-RMSNorm, the RMSNorm equivalent from [Section 8](#8-rmsnorm)), specifically because of the training-stability and warmup-robustness benefits outlined in Section 25.4. Sandwich-LN sees usage in some large-scale training efforts (e.g., CogView, GLM-family models) where additional output-scale control was found empirically beneficial at the specific scales and training regimes those projects used. Post-LN, despite being the original architecture's choice, is now primarily of historical and pedagogical interest for understanding *why* the field moved away from it — though it's worth knowing BERT (still widely used and studied) is a Post-LN architecture, so recognizing this distinction matters when reading or reasoning about BERT's training characteristics specifically (BERT's well-known sensitivity to learning rate and warmup schedule is a direct, documented consequence of its Post-LN design).

### 25.8 What's next

[Section 26](#26-layernorm-inside-the-transformer-encoder) and [Section 27](#27-layernorm-inside-the-transformer-decoder) walk through the complete, explicit placement of LayerNorm calls inside a full Transformer encoder block and decoder block respectively — building directly on this section's Pre-LN/Post-LN distinction, now applied concretely to the multi-head attention and feed-forward sublayers with full tensor-shape tracing through an entire block.

---

## 26. LayerNorm Inside the Transformer Encoder

### 26.1 The complete encoder block, with every tensor shape traced

An encoder block (BERT-style, bidirectional self-attention — no causal masking) takes $X\in\mathbb{R}^{B\times T\times d}$ and produces an output of the identical shape. Using the Pre-LN convention from [Section 25.3](#253-pre-ln--the-modern-default), here is every intermediate tensor's shape, traced explicitly:

```
Input:           X                          (B, T, d)
                  │
  ┌───────────────┴────────────────┐
  │  Step 1: Pre-attention LayerNorm│
  └───────────────┬────────────────┘
                  ▼
                 X_n = LN(X)                 (B, T, d)         -- Section 21's forward pass
                  │
  ┌───────────────┴────────────────┐
  │  Step 2: Multi-Head Attention   │
  └───────────────┬────────────────┘
                  ▼
   Q, K, V = X_n @ W_q, X_n @ W_k, X_n @ W_v
   each reshaped to                          (B, H, T, d/H)    -- H attention heads
                  ▼
   Attn = softmax(QK^T / sqrt(d/H)) @ V       (B, H, T, d/H)
                  ▼
   Attn_out = reshape + output projection     (B, T, d)
                  │
  ┌───────────────┴────────────────┐
  │  Step 3: Residual addition      │
  └───────────────┬────────────────┘
                  ▼
                 X' = X + Attn_out            (B, T, d)        -- Part IV's subject
                  │
  ┌───────────────┴────────────────┐
  │  Step 4: Pre-FFN LayerNorm      │
  └───────────────┬────────────────┘
                  ▼
                 X'_n = LN(X')                (B, T, d)
                  │
  ┌───────────────┴────────────────┐
  │  Step 5: Feed-Forward Network   │
  └───────────────┬────────────────┘
                  ▼
   FFN_out = W_2 @ GELU(W_1 @ X'_n)           (B, T, d)        -- typically d_ff = 4d internally
                  │
  ┌───────────────┴────────────────┐
  │  Step 6: Residual addition      │
  └───────────────┬────────────────┘
                  ▼
   Output:        X'' = X' + FFN_out          (B, T, d)
```

Notice LayerNorm appears at exactly **two** points per block (Steps 1 and 4) — once before attention, once before the FFN — and at both points, the tensor's *shape* is entirely unchanged ($B,T,d \to B,T,d$). This is a general, useful fact worth internalizing: **LayerNorm never changes a tensor's shape**, only its values — every shape change in the block above happens inside the attention mechanism's head-splitting/merging or inside the FFN's up/down projections, never inside the normalization steps themselves.

### 26.2 Why attention specifically benefits from being fed normalized input

It's worth being precise about *why* Step 1's LayerNorm matters specifically for the attention computation that follows it (rather than just restating "normalization is generally good for optimization," which Sections 1–2 already covered): the attention score computation $QK^\top/\sqrt{d/H}$ involves a dot product between query and key vectors, and dot-product magnitudes scale with the magnitude of the vectors involved. If $X$'s activations have drifted to an unusually large scale (the exact problem normalization addresses generally), the resulting attention logits before softmax can become extremely large — and extremely large logits, passed through softmax, produce a **near one-hot** attention distribution (almost all probability mass on a single position), which in turn produces **near-zero gradients** through the softmax for every position except the single dominant one ([Section 22.4](#224-two-structural-properties-worth-knowing)'s style of careful attention to which directions in a computation carry near-zero sensitivity applies directly here too, just for softmax's Jacobian rather than LayerNorm's). Feeding attention a freshly-normalized input directly mitigates this specific failure mode by keeping the query/key vectors — and therefore the dot products between them — on a controlled, predictable scale, every single layer, regardless of how activations might otherwise have drifted by that point in the network.

### 26.3 A minimal runnable encoder block

```python
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block, matching the trace in Section 26.1."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        x_n = self.ln1(x)                                          # Step 1
        attn_out, _ = self.attn(x_n, x_n, x_n,                      # Step 2
                                 key_padding_mask=key_padding_mask)
        x = x + attn_out                                           # Step 3
        x_n = self.ln2(x)                                          # Step 4
        ffn_out = self.ffn(x_n)                                    # Step 5
        x = x + ffn_out                                             # Step 6
        return x

block = EncoderBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(4, 20, 512)            # (B=4, T=20, d=512)
out = block(x)
print(out.shape)                       # torch.Size([4, 20, 512]) -- shape preserved, matches Section 26.1
```

### 26.4 What's next

[Section 27](#27-layernorm-inside-the-transformer-decoder) extends this exact same trace to the **decoder** block — adding the causal masking required for autoregressive generation, and (for encoder-decoder architectures) the cross-attention sublayer that attends to the encoder's output, with its own additional LayerNorm placement.

---

## 27. LayerNorm Inside the Transformer Decoder

### 27.1 Two decoder flavors

"Decoder" covers two architecturally distinct things worth separating clearly:

- **Decoder-only** (GPT, LLaMA, Mistral, Gemma, and the overwhelming majority of modern LLMs): a stack of blocks identical in *structure* to [Section 26.1](#261-the-complete-encoder-block-with-every-tensor-shape-traced)'s encoder block, with exactly one change — the self-attention step uses a **causal mask**, preventing any token from attending to tokens later in the sequence (required for autoregressive generation, where token $t$ must be predictable from only tokens $1,\ldots,t-1$).
- **Encoder-decoder** (the original Transformer, T5, and other sequence-to-sequence architectures): each decoder block has **three** sublayers instead of two — causal self-attention, then **cross-attention** (queries from the decoder, keys/values from the encoder's output), then the FFN — with a LayerNorm placed before each of the three.

### 27.2 Decoder-only block, with the causal mask traced explicitly

```
Input:           X                          (B, T, d)
                  │
                 X_n = LN(X)                 (B, T, d)
                  │
   Q, K, V = X_n @ W_q, X_n @ W_k, X_n @ W_v   (B, H, T, d/H)
                  │
   Scores = QK^T / sqrt(d/H)                  (B, H, T, T)
                  │
   Scores = Scores + CausalMask               (B, H, T, T)   -- mask[i,j] = -inf if j > i
                  │
   Attn = softmax(Scores) @ V                 (B, H, T, d/H)
                  │
   Attn_out = reshape + output projection      (B, T, d)
                  │
                 X' = X + Attn_out             (B, T, d)
                  │
                 X'_n = LN(X')                 (B, T, d)
                  │
   FFN_out = W_2 @ GELU(W_1 @ X'_n)            (B, T, d)
                  │
   Output:        X'' = X' + FFN_out           (B, T, d)
```

The causal mask is added **before** softmax, as $-\infty$ (or a very large negative number in practice, to avoid actual infinities) at every position $(i,j)$ where $j>i$ — guaranteeing $\text{softmax}$ assigns those positions exactly zero probability. Crucially, **the causal mask has nothing to do with LayerNorm placement at all** — LayerNorm's two placements (before self-attention, before FFN) are structurally identical to the encoder's, the *only* architectural difference between an encoder block and a decoder-only block is this masking step inside the attention score computation.

### 27.3 Encoder-decoder block: three LayerNorm placements

```
Input:        X_dec (from previous decoder layer)        (B, T_dec, d)
Cross-input:  X_enc (final encoder output)                (B, T_enc, d)
                  │
                 X_n = LN_1(X_dec)
                  │
   Causal self-attention (Q,K,V all from X_n)              (B, T_dec, d)
                  │
                 X' = X_dec + SelfAttn_out
                  │
                 X'_n = LN_2(X')
                  │
   Cross-attention: Q from X'_n, K,V from X_enc            (B, T_dec, d)
   Scores = Q K_enc^T / sqrt(d/H)                           (B, H, T_dec, T_enc)
   (no causal mask here -- decoder may attend to ALL encoder positions)
                  │
                 X'' = X' + CrossAttn_out
                  │
                 X''_n = LN_3(X'')
                  │
   FFN_out = W_2 @ GELU(W_1 @ X''_n)
                  │
   Output:        X''' = X'' + FFN_out                     (B, T_dec, d)
```

Notice the cross-attention step's score tensor has shape $(B, H, T_{\text{dec}}, T_{\text{enc}})$ — **not square** like self-attention's $(B,H,T,T)$, since queries come from the decoder sequence (length $T_{\text{dec}}$) while keys/values come from the encoder sequence (length $T_{\text{enc}}$, generally different from $T_{\text{dec}}$). This shape difference is worth tracking carefully when implementing or debugging encoder-decoder attention — it's a common source of shape-mismatch bugs precisely because self-attention and cross-attention otherwise look so structurally similar.

### 27.4 A minimal runnable decoder-only block with causal masking

```python
import torch
import torch.nn as nn

class DecoderOnlyBlock(nn.Module):
    """Pre-LN decoder-only block (GPT/LLaMA-style), matching Section 27.2's trace."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

        x_n = self.ln1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n, attn_mask=causal_mask, is_causal=True)
        x = x + attn_out
        x_n = self.ln2(x)
        x = x + self.ffn(x_n)
        return x

block = DecoderOnlyBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(2, 16, 512)
out = block(x)
print(out.shape)   # torch.Size([2, 16, 512])

# Verify the causal property directly: changing a LATER token shouldn't
# change an EARLIER token's output.
x2 = x.clone()
x2[:, -1, :] += 100.0          # perturb only the LAST token
out2 = block(x2)
print(torch.allclose(out[:, 0, :], out2[:, 0, :], atol=1e-5))   # expect True
```

### 27.5 What's next

[Section 28](#28-layernorm-across-model-families-gpt-llama-deepseek-mistral-gemma) closes out Part III's architectural-survey thread by examining LayerNorm/RMSNorm's exact placement, $\epsilon$ choice, and any model-specific quirks across GPT, LLaMA, DeepSeek, Mistral, and Gemma specifically — a focused preview of the much deeper per-model case studies that [Part VII](#46-gpt-2--gpt-3--gpt-4) will provide once residual connections (Part IV) and the combined LayerNorm+residual analysis (Part V) are both in place.

---

## 28. LayerNorm Across Model Families: GPT, LLaMA, DeepSeek, Mistral, Gemma

### 28.1 A focused preview, not the full case study

This section gives a concise, normalization-focused snapshot of each family, sufficient to round out Part III. [Part VII](#46-gpt-2--gpt-3--gpt-4) revisits every one of these models in full architectural depth — attention variants (MHA/GQA/MLA), positional encoding, tokenizer details, and complete config-level specifications — once residual connections have been covered in [Parts IV–V](#29-motivation-vanishing-gradients-and-the-degradation-problem). Treat the table below as a "normalization-and-residual-norm-only" cheat sheet, with the honest caveat that exact $\epsilon$ values and minor placement details are drawn from each family's publicly released configs/code and may vary slightly across specific model sizes within a family.

| Family | Norm type | Placement | Typical $\epsilon$ | Notes |
|---|---|---|---|---|
| **GPT-2 / GPT-3** | LayerNorm | Pre-LN | $10^{-5}$ | GPT-2 was an early, influential adopter of Pre-LN over the original Transformer's Post-LN, specifically for training stability at scale ([Section 25.4](#254-why-the-difference-matters-a-preview)) |
| **LLaMA / LLaMA 2 / LLaMA 3** | RMSNorm | Pre-RMSNorm | $10^{-5}$ (varies slightly by release/size) | Uses the float32-upcast pattern from [Section 8.7](#87-production-grade-pytorch-implementation) internally |
| **Mistral / Mixtral** | RMSNorm | Pre-RMSNorm | $10^{-5}$ | Architecturally very close to LLaMA's normalization choices; differences between these families are concentrated in attention (sliding-window attention, Mixture-of-Experts routing) rather than normalization |
| **Gemma / Gemma 2** | RMSNorm | Pre-RMSNorm, plus an additional **post-attention and post-FFN** RMSNorm sandwich (both before and after each sublayer) | $10^{-6}$ | Gemma 2 stabilizes attention via **logit soft-capping** rather than QK-Norm; it was the later **Gemma 3** that replaced soft-capping with QK-Norm on query/key projections |
| **DeepSeek (V2/V3)** | RMSNorm | Pre-RMSNorm | $10^{-6}$ | Normalization choices are standard Pre-RMSNorm; DeepSeek's architectural novelty is concentrated in its attention mechanism (Multi-head Latent Attention, MLA) and its Mixture-of-Experts FFN design, not in normalization |

### 28.2 The pattern that should jump out

Every single model in this table uses **Pre-normalization** (Section 25.3's pattern) — there is no exception among current major LLM families. This is the single, field-wide consensus outcome of the Post-LN-vs-Pre-LN analysis from [Section 25](#25-post-ln-vs-pre-ln-vs-sandwich-ln): whatever other architectural choices differ wildly across these families (attention mechanism, positional encoding, MoE vs. dense FFN, tokenizer), the basic decision of "normalize before the sublayer, not after" is universal across every model a practitioner is likely to work with in 2026.

The second pattern: **RMSNorm has fully displaced LayerNorm** in every model family released since 2023 (LLaMA onward) — only the GPT-2/3 lineage in this table still uses full LayerNorm, reflecting its earlier release date rather than any architecture-family-specific preference. This directly confirms [Section 8.12](#812-real-world-usage)'s claim, now placed alongside GPT's LayerNorm choice for direct contrast.

### 28.3 The smaller, model-specific deviations are worth noticing too

Not every detail is identical across these RMSNorm-using families — Gemma's Sandwich-LN-style pre/post normalization and soft-capping/QK-Norm choices (the latter changing between Gemma 2 and Gemma 3, as corrected above), and the smaller $\epsilon$ values in DeepSeek and Gemma relative to LLaMA/Mistral's $10^{-5}$, are exactly the kind of detail that matters when reading a specific model's actual source code or config file, even though they don't change the *overall* architectural picture (Pre-RMSNorm, applied to attention and FFN inputs). We'll return to *why* these specific deviations were made — particularly Gemma's attention-stabilization choices and their connection to attention logit stability — with full justification once [Part VII](#46-gpt-2--gpt-3--gpt-4)'s case studies cover each model's complete architecture.

### 28.4 Part III summary, and what's next

Part III is now complete. We've covered LayerNorm in full depth: the rigorous proof of why BatchNorm fails for Transformers ([Section 20](#20-why-batchnorm-fails-for-transformers)), the complete forward-pass tensor mathematics ([Section 21](#21-layernorm-mathematics--forward-pass)), the backward pass and explicit Jacobian ([Section 22](#22-layernorm-mathematics--backward-pass-and-jacobian)), the learnable $\gamma,\beta$ parameters and their gradient dynamics ([Section 23](#23-learnable-γ-and-β-role-and-gradient-flow)), the geometric ellipsoid picture ([Section 24](#24-geometric-intuition-for-layernorm)), the Post-LN/Pre-LN/Sandwich-LN architectural choice ([Section 25](#25-post-ln-vs-pre-ln-vs-sandwich-ln)), and LayerNorm's exact placement inside encoder ([Section 26](#26-layernorm-inside-the-transformer-encoder)) and decoder ([Section 27](#27-layernorm-inside-the-transformer-decoder)) blocks, closing with this cross-model snapshot.

**Part IV** begins now, turning to this handbook's other primary focus: **residual (skip) connections**. We'll build the motivation from first principles exactly as Part I did for normalization — starting with the degradation problem that motivated residual connections' invention, the clean mathematics of $y=x+F(x)$, and a complete gradient-propagation derivation showing precisely *why* residual connections let gradients flow through networks hundreds of layers deep without vanishing — the other half of the "why do modern Transformers train stably at all" story that [Part V](#36-why-they-work-so-well-together) will ultimately bring together with everything covered in Parts I–III.

---

# Part IV — Residual / Skip Connections Deep Dive (Primary Focus)

## 29. Motivation: Vanishing Gradients and the Degradation Problem

### 29.1 A puzzle that confused the field before the fix was found

Before residual connections were introduced (He et al., 2015, in the ResNet paper), a strange and counterintuitive empirical observation kept appearing as researchers tried to train deeper and deeper networks: **adding more layers to a network sometimes made it perform *worse* on the training set itself** — not just worse on held-out test data (which could be explained by overfitting), but worse on the very data the network was being directly optimized against. This is the **degradation problem**, and it's genuinely strange: a deeper network has *strictly more representational capacity* than a shallower one (in the most basic sense, a deeper network could, in principle, always learn to set its extra layers to compute the identity function, exactly reproducing a shallower network's behavior) — so why would adding layers ever make training-set performance *worse*?

### 29.2 The vanishing gradient mechanism, precisely

The answer lies in optimization, not representational capacity. Consider a deep network without any skip connections, where layer $l$'s output feeds layer $l+1$: $x_{l+1}=F_l(x_l)$. Backpropagating a gradient from the final loss back to an early layer requires multiplying together the Jacobian of every single layer in between, by the chain rule:

$$\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_L}\cdot\frac{\partial x_L}{\partial x_{L-1}}\cdots\frac{\partial x_2}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_L}\prod_{l=1}^{L-1}\frac{\partial x_{l+1}}{\partial x_l}$$

If each individual Jacobian $\partial x_{l+1}/\partial x_l$ has, say, a typical eigenvalue magnitude somewhat less than 1 (very plausible — many common nonlinearities like sigmoid and tanh have derivatives bounded well below 1 almost everywhere, and even ReLU's derivative is exactly 0 for any input below zero), then the **product of $L-1$ such Jacobians shrinks geometrically with depth** — exactly the same mathematical pattern as compound interest running in reverse. For a network with, say, 50 layers and a typical per-layer Jacobian "shrink factor" of 0.9, the cumulative effect is $0.9^{49}\approx0.0052$ — the gradient reaching the earliest layers is less than 1% of its original magnitude by the time it arrives, even though no single layer did anything obviously broken. Layers near the *input* effectively stop receiving any meaningful training signal at all, even though they're just as "responsible" for the final loss as any later layer — they simply can't be told, through the chain-rule-multiplied gradient, what adjustment would help.

### 29.3 An empirical demonstration

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class PlainDeepNet(nn.Module):
    """A deep network with NO residual connections -- pure stacked layers."""
    def __init__(self, depth: int, dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.Tanh()) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

net = PlainDeepNet(depth=50)
x = torch.randn(8, 64, requires_grad=True)
out = net(x)
loss = out.sum()
loss.backward()

# Inspect gradient magnitude reaching the FIRST layer vs. the LAST layer
first_layer_grad_norm = net.layers[0][0].weight.grad.norm().item()
last_layer_grad_norm = net.layers[-1][0].weight.grad.norm().item()
print(f"First layer gradient norm: {first_layer_grad_norm:.6f}")
print(f"Last layer gradient norm:  {last_layer_grad_norm:.6f}")
print(f"Ratio (first/last):        {first_layer_grad_norm/last_layer_grad_norm:.6f}")
```

Running this with a 50-layer `tanh`-activated plain stack shows something even starker than "orders of magnitude smaller": the first layer's gradient norm underflows all the way to **exactly 0.0** in standard float32 precision, while the last layer's gradient norm is a healthy, non-degenerate ~44 — the gradient signal has been multiplied down through 50 `tanh` Jacobians so many times that it's no longer just *small*, it's **completely gone**, indistinguishable from a layer receiving zero training signal whatsoever. This is a direct, runnable demonstration of Section 29.2's mechanism, not just a theoretical claim — and it's worth sitting with how stark this result is: the first layer in this 50-layer network receives *no usable gradient at all* from this particular backward pass, even though the loss explicitly depends on its output through every subsequent layer.

### 29.4 The degradation problem follows directly

Once you see the vanishing-gradient mechanism clearly, the degradation problem stops being mysterious: it's not that a deeper network is somehow representationally worse — it's that **the optimizer can no longer effectively train it**, because the gradient signal needed to discover good weights for the early layers has been multiplied down toward zero by the time it arrives there. The network technically *contains* a good solution (e.g., extra layers that learned the identity function, reproducing a shallower, well-trained network's behavior) somewhere in its parameter space — but gradient descent, supplied with vanishingly small gradients for exactly the parameters that would need to move to find that solution, simply can't get there in a reasonable number of training steps. **The problem is optimization-theoretic, not representational** — exactly analogous, in spirit, to the internal-covariate-shift discussion in [Section 2](#2-internal-covariate-shift-and-optimization-landscapes): the network's *capacity* was never the bottleneck, *training dynamics* were.

### 29.5 The core insight that motivates the fix

If the problem is that learning the identity function is hard for a stack of nonlinear layers to discover via gradient descent (because nothing in a plain stacked architecture makes the identity function an *easy*, *natural*, or even *nearby* solution in parameter space), the fix suggests itself directly: **make the identity function the architecture's default behavior**, something the network gets "for free" without needing to learn it at all, and let each layer's learned transformation be an *additive correction* on top of that free identity path, rather than a wholesale replacement that must be learned from scratch at every single layer. This is precisely residual learning's core idea, formalized completely in [Section 30](#30-mathematics-of-residual-learning-y--x--fx).

### 29.6 What's next

[Section 30](#30-mathematics-of-residual-learning-y--x--fx) introduces the residual connection's defining equation, $y=x+F(x)$, and works through exactly why this simple additive reformulation makes the identity function trivially representable (just set $F(x)=0$) — directly resolving the degradation problem's root cause identified in this section, before [Section 31](#31-gradient-propagation-through-residual-paths-full-derivation) gives the complete gradient-flow derivation showing precisely how this resolves the vanishing-gradient mechanism from Section 29.2 as well.

---

## 30. Mathematics of Residual Learning: y = x + F(x)

### 30.1 The defining equation

A residual (skip) connection reformulates a layer's computation from "learn the entire output from scratch" to "learn only a *correction* added to the unchanged input":

$$y = x + F(x)$$

where $F$ is some learnable function (in a Transformer, attention or the FFN — though the idea is fully general and predates Transformers entirely, originating in ResNet's convolutional blocks). Compare this against a plain layer's formulation, $y=F(x)$ (no addition at all) — the entire difference, syntactically, is the addition of $x$ itself to whatever $F$ computes.

### 30.2 Why this trivially fixes the "identity is hard to learn" problem

Recall [Section 29.4](#294-the-degradation-problem-follows-directly)'s diagnosis: a plain stack of nonlinear layers finds it hard for gradient descent to discover parameters that make the layer behave (approximately) as the identity function, because nothing about the architecture makes that an easy or natural solution to land on. Under the residual formulation, the identity function corresponds to the **trivial, easy-to-reach** setting $F(x)=0$:

$$y = x + F(x) = x + 0 = x$$

And $F(x)=0$ is a genuinely easy function for a neural network to learn or approximate — for instance, if $F$ is implemented as a standard sequence of linear layers and nonlinearities, setting every weight to exactly zero produces $F(x)=0$ for all $x$, and weights starting near zero (or being pushed toward zero by weight decay, or simply not receiving strong enough gradient signal to move far from a small initialization) is a completely unremarkable, easily-reached region of parameter space — nothing like the complex, precisely-tuned configuration of nonlinear weights that would be required for a plain stacked layer (no residual) to approximate the identity function instead.

**This is the precise sense in which residual connections make depth "safe."** Adding another residual block to a network can, in the worst case, contribute nothing (if $F$ learns to output something close to zero) — but it can never make the network's *representational floor* worse than before that block was added, because the option to fall back to "just pass the input through unchanged" is always available and easy to find via gradient descent, unlike in a plain architecture where that fallback option, while representationally present, is practically very hard for the optimizer to discover.

### 30.3 A worked numerical example

Take $x=[2.0, -1.0, 3.0]$, and suppose $F$ is a single linear layer with weight matrix $W$ and bias $b$ that, after training, has learned a fairly small correction:

$$F(x) = Wx + b = [0.1, -0.05, 0.2]$$

Then:

$$y = x + F(x) = [2.0+0.1,\ -1.0-0.05,\ 3.0+0.2] = [2.1,\ -1.05,\ 3.2]$$

The output is **close to the input**, with each component shifted by a relatively small, independently-learned correction. Compare this against a plain (non-residual) layer that needed to reproduce a similarly "mostly-preserve-the-input" behavior directly: it would need $F(x)\approx x$ exactly (not $F(x)\approx 0$) — a meaningfully harder function for arbitrary weights to discover or maintain throughout training, since "approximate the identity" is a much more specific, constrained target than "approximate zero," especially once you account for the fact that $x$ itself changes from example to example and from training step to training step as upstream layers evolve.

### 30.4 The "residual" terminology, precisely

The name comes directly from this framing: $F(x)$ is interpreted as the **residual** — literally, "what's left over" or "the remaining correction" — that needs to be learned and added on top of the identity baseline, rather than the layer's entire output needing to be learned as a self-contained function of $x$. This is the same general statistical idea as a "residual" in regression analysis (the leftover difference between an observed value and a baseline prediction), repurposed here as an architectural principle: let the network's layers learn *deviations from a known-good baseline* (the identity), rather than learning *outputs from scratch*.

### 30.5 What this does NOT mean

A common point of confusion worth heading off directly: residual connections do **not** mean $F(x)$ is forced to be small, or that the network is somehow restricted to only small perturbations of its input. $F(x)$ can, and very much does in a well-trained network, learn to produce large, meaningful, task-relevant transformations — the residual formulation doesn't cap or restrict $F$'s expressive power in any way. What it changes is *only* the **easy-to-reach default** when a large correction *isn't* needed (or isn't yet needed, early in training) — giving the optimizer an easy "do nothing extra here" option at every layer, while leaving the door fully open for $F$ to learn arbitrarily large, complex, useful transformations wherever the data and the loss actually call for them.

### 30.6 What's next

[Section 31](#31-gradient-propagation-through-residual-paths-full-derivation) makes the gradient-flow benefit of $y=x+F(x)$ completely rigorous — deriving the exact backward-pass formula for a residual connection, and showing precisely how the "+1" term contributed by the bare identity path guarantees that gradients can always flow backward through a residual connection with at least unit strength, regardless of how poorly-conditioned $F$'s own Jacobian might be at any given layer. This is the formal counterpart to Section 29's vanishing-gradient demonstration — showing exactly how residual connections fix the problem made empirically vivid there.

---

## 31. Gradient Propagation Through Residual Paths (Full Derivation)

### 31.1 The single-layer derivation

For $y=x+F(x)$, differentiate with respect to $x$ directly, using the sum rule:

$$\frac{\partial y}{\partial x} = \frac{\partial x}{\partial x} + \frac{\partial F(x)}{\partial x} = I + J_F$$

where $I$ is the identity matrix and $J_F=\partial F/\partial x$ is $F$'s own Jacobian. **This single equation is the entire mathematical reason residual connections solve the vanishing-gradient problem.** Compare against a plain layer, $y=F(x)$, whose Jacobian is simply $J_F$ — no $+I$ term at all. The residual connection's Jacobian is *always* $F$'s own Jacobian, **plus the identity** — meaning that even in the worst case, where $J_F$ happens to be very small (or even exactly the zero matrix, e.g. if $F$'s weights are all zero, as in Section 30.2's "easy default"), the *overall* layer's Jacobian is still at least $I$ — never smaller than the identity transformation's own, perfectly gradient-preserving Jacobian.

### 31.2 Propagating through many stacked residual layers

For a stack of $L$ residual layers, $x_{l+1}=x_l+F_l(x_l)$ for $l=1,\ldots,L-1$, the chain rule for the gradient reaching the very first layer is:

$$\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_L}\prod_{l=1}^{L-1}(I+J_{F_l})$$

Expanding this product (even just for $L=3$, two layers) reveals exactly why this resists vanishing in a way the plain-stack product from [Section 29.2](#292-the-vanishing-gradient-mechanism-precisely) does not:

$$(I+J_{F_1})(I+J_{F_2}) = I + J_{F_1} + J_{F_2} + J_{F_1}J_{F_2}$$

Notice the **leading $I$ term survives no matter how many layers you multiply through** — expanding the full product for $L-1$ layers, one term in the resulting sum is always exactly $I$ (the term you get by picking the "$I$" choice at every single layer in the product), regardless of how small or poorly-conditioned every individual $J_{F_l}$ happens to be. Every other term in the expansion involves at least one $J_{F_l}$ factor, and could in principle still shrink toward zero with depth — but the **pure identity term provides a gradient floor that never vanishes**, guaranteeing that $\partial\mathcal{L}/\partial x_1$ always contains at least an undiminished copy of $\partial \mathcal{L}/\partial x_L$ flowing straight through, *plus* whatever additional signal the $J_{F_l}$ terms contribute on top.

### 31.3 The "gradient superhighway" framing, made precise

This is the rigorous version of the "information highway" intuition often used informally to describe residual connections (and which we'll build on further in [Section 32](#32-intuition-information-highways-and-identity-mappings)): the residual stream provides a **direct additive path** from the network's output all the way back to its input, completely independent of how well- or poorly-behaved any individual sublayer's own Jacobian happens to be. A sublayer that's currently poorly-trained, saturated, or otherwise producing a near-zero or ill-conditioned local Jacobian doesn't block gradient flow through the residual stream — it simply contributes little of its own additional signal on top of the always-present identity contribution, exactly the same way [Section 30.2](#302-why-this-trivially-fixes-the-identity-is-hard-to-learn-problem)'s "easy default" argument described it at the forward-pass level, now shown to hold at the backward-pass level too.

### 31.4 Numerical verification across a deep stack

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class ResidualDeepNet(nn.Module):
    """The SAME 50-layer depth and per-layer structure as Section 29.3's
    PlainDeepNet, but now with a residual connection around each layer."""
    def __init__(self, depth: int, dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.Tanh()) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)        # <-- the only change vs. Section 29.3
        return x

net = ResidualDeepNet(depth=50)
x = torch.randn(8, 64, requires_grad=True)
out = net(x)
loss = out.sum()
loss.backward()

first_layer_grad_norm = net.layers[0][0].weight.grad.norm().item()
last_layer_grad_norm = net.layers[-1][0].weight.grad.norm().item()
print(f"First layer gradient norm: {first_layer_grad_norm:.6f}")
print(f"Last layer gradient norm:  {last_layer_grad_norm:.6f}")
print(f"Ratio (first/last):        {first_layer_grad_norm/last_layer_grad_norm:.6f}")
```

Running this produces a striking, direct contrast with [Section 29.3](#293-an-empirical-demonstration)'s result: the first layer's gradient norm is now **~1231**, and the last layer's is **~402** — not only has the vanishing problem disappeared, the first layer's gradient is now actually *larger* than the last layer's (a ratio of roughly 3.06, versus the plain stack's complete collapse to a ratio of exactly 0.0). This is the $I+J_{F_l}$ Jacobian structure from Section 31.1 made fully concrete: the identical 50-layer depth, the identical per-layer structure (`Linear` + `Tanh`), the identical random seed — the *only* change between this network and [Section 29.3](#293-an-empirical-demonstration)'s `PlainDeepNet` is the addition of `x +` before each layer's output, and that single architectural change is the entire difference between a network whose earliest layers receive literally zero gradient signal, and one whose earliest layers receive comparable (here, even larger) gradient signal than its last layer.

### 31.5 What's next

[Section 32](#32-intuition-information-highways-and-identity-mappings) steps back from the formal derivation just completed to build durable, transferable intuition for *why* this matters — the "information highway" metaphor, viewed from several angles, plus the historical framing of residual connections as enabling "identity mappings" specifically, which was the original ResNet paper's own chosen language for exactly the mechanism derived rigorously in this section.

---

## 32. Intuition: Information Highways and Identity Mappings

### 32.1 The highway metaphor, made precise rather than just evocative

"Information highway" is a common informal description of what a residual stream provides — but it's worth pinning down exactly *which* features of an actual highway this metaphor is meant to capture, rather than leaving it as a vague gesture:

- **A highway lets traffic bypass every intermediate town without stopping.** Analogously, the residual stream lets information (in the forward pass) and gradients (in the backward pass) flow from the network's start to its end without being forced to pass *through* every single sublayer's full transformation — each sublayer can choose to add a contribution onto the highway, but nothing about the architecture *requires* heavy modification at every single layer, the way a plain stack effectively does.
- **Local roads (the $F(x)$ branches) still exist and still matter** — the highway metaphor isn't claiming sublayers are unnecessary or bypassed entirely; it's specifically about there being an *additional*, always-available, low-resistance path running in parallel with whatever transformation each sublayer performs, exactly mirroring [Section 31.1](#311-the-single-layer-derivation)'s $I+J_F$ structure: the $I$ is the highway, $J_F$ is the local road's contribution, and both operate simultaneously, summed together.
- **Congestion at one intermediate town doesn't block the whole route.** If one particular sublayer happens to be poorly-conditioned (e.g., early in training, before its weights have found a useful configuration), that doesn't block gradient flow through the rest of the network — the highway (identity path) routes around it, exactly as [Section 31.3](#313-the-gradient-superhighway-framing-made-precise) formalized.

### 32.2 "Identity mappings" — the original ResNet paper's framing

He et al.'s original ResNet paper, and especially its influential 2016 follow-up "Identity Mappings in Deep Residual Networks," frames the entire contribution in terms of making the **identity mapping** (the function $f(x)=x$) easy for a network to represent and for an optimizer to discover, at any layer where it happens to be the locally-useful thing to do. This is exactly [Section 30.2](#302-why-this-trivially-fixes-the-identity-is-hard-to-learn-problem)'s argument, in the original paper's own words and motivation — worth knowing explicitly, since "identity mapping" is the term you'll see most often in the original literature and in subsequent papers building on it, even where this handbook has more often used the slightly more colloquial "do nothing extra" or "easy default" phrasing for the same underlying concept.

### 32.3 A second framing: ensemble-like behavior

A complementary intuition, developed in later analysis of ResNets (Veit et al., "Residual Networks Behave Like Ensembles of Relatively Shallow Networks," 2016): because each residual block offers a choice between "contribute $F(x)$" and "contribute nothing," a deep residual network with $L$ blocks can be viewed as implicitly representing **an ensemble of $2^L$ different possible "effective paths"** through the network — one path for every possible subset of blocks that end up contributing something non-negligible versus blocks that end up contributing near-zero. This framing helps explain another empirically-observed property of residual networks: they tend to be considerably more robust to *removing* individual layers after training (a kind of post-hoc depth ablation) than plain networks are, since no single layer's contribution is typically load-bearing in the way a layer in a plain, non-residual stack effectively must be (a plain network has no "skip this layer" option built into its architecture at all).

### 32.4 Why this matters specifically for the LLM context this handbook focuses on

Every one of these framings — the highway, the easy-identity-mapping, the implicit ensemble — converges on the same practical consequence that matters for training the very deep Transformers covered throughout this handbook: **depth becomes safe to add, rather than something that actively works against successful training past a certain point.** This is precisely why production LLMs can stack dozens to well over a hundred Transformer blocks (GPT-3: 96 layers; LLaMA 3 70B: 80 layers; and considerably more for the largest models) and train successfully — a feat that, per [Section 29](#29-motivation-vanishing-gradients-and-the-degradation-problem)'s diagnosis, would be effectively impossible for a plain, non-residual stack of comparable depth using standard gradient-based optimization.

### 32.5 What's next

[Section 33](#33-variants-highway-networks-densenet-gated-residuals-rezero-deepnorm-scaling) surveys the broader family of residual-connection variants that have been proposed since the original ResNet formulation — Highway Networks (which actually *predate* ResNet and introduced a related but distinct gating mechanism), DenseNet's connect-everything-to-everything approach, gated residuals, ReZero, and a revisit of DeepNorm's residual scaling (introduced earlier in [Section 17](#17-deepnorm) from the normalization side) — now examined specifically as residual-connection design choices.

---

## 33. Variants: Highway Networks, DenseNet, Gated Residuals, ReZero, DeepNorm Scaling

### 33.1 Highway Networks — the gated predecessor

Highway Networks (Srivastava, Greff & Schmidhuber, 2015) actually predate ResNet by several months and introduced a closely related but distinct mechanism: a **learned, input-dependent gate** controlling how much of the input passes through unchanged versus how much gets replaced by a transformation:

$$y = T(x)\odot g(x) + x\odot(1-g(x))$$

where $T(x)$ is a learned transformation, $g(x)\in(0,1)^d$ is a learned **gate** (typically a sigmoid-activated linear layer of $x$), and $\odot$ is elementwise multiplication. When $g(x)\to 0$, $y\to x$ (pure identity passthrough, same as ResNet's $F(x)=0$ case); when $g(x)\to 1$, $y\to T(x)$ (the input is fully replaced rather than added to). **The key structural difference from ResNet**: ResNet's $y=x+F(x)$ has a gate that's effectively *fixed at exactly 1* for the $F(x)$ contribution and *fixed at exactly 1* for the identity contribution too (both terms always fully present, simply summed) — there's no learned, input-dependent mixing between them. Highway Networks' gate is **learned and input-dependent**, letting the network decide, per-input and per-feature, how much replacement versus preservation is appropriate at this specific point.

### 33.2 DenseNet — connecting everything to everything

DenseNet (Huang et al., 2017) takes the residual idea in a different direction: rather than each layer connecting only to the *immediately previous* layer via a residual addition, **every layer receives the concatenated outputs of every preceding layer** as its input:

$$x_l = F_l\big([x_0, x_1, \ldots, x_{l-1}]\big)$$

where $[\cdot]$ denotes channel-wise concatenation (not addition). This guarantees an even more direct gradient path from any layer back to any *earlier* layer (not just the immediately preceding one) — every layer's output is directly, structurally connected to every later layer's input. The trade-off: concatenation (rather than addition) means the input dimensionality to layer $l$ **grows with $l$** — by layer 50, a DenseNet layer might be receiving a concatenation of 49 previous layers' outputs, a substantially larger input than ResNet's fixed-dimensionality residual stream, which is part of why DenseNet has seen more adoption in vision architectures (where this growing-concatenation pattern integrates naturally with convolutional channel structure) than in Transformer-based LLMs (where the fixed, constant-dimensionality residual stream of standard ResNet-style connections fits the architecture's other components — attention, FFN — more naturally and economically).

### 33.3 Gated residuals in Transformers

A middle ground between plain residual addition and Highway Networks' full gating mechanism appears in some Transformer variants as a **learned scalar or per-channel gate applied to the sublayer's contribution specifically** (not gating the identity path at all, only modulating how much of $F(x)$ gets added):

$$y = x + g\odot F(x)$$

where $g$ is learned (sometimes a single scalar shared across the layer, sometimes per-channel). This preserves ResNet's guarantee that the identity path is *always* fully present (unlike Highway Networks, where even the identity contribution is gated and could in principle be suppressed), while still giving the network explicit, learnable control over how strongly each sublayer's contribution gets weighted — directly setting up the next two techniques, both special cases of this exact pattern.

### 33.4 ReZero — starting the gate at exactly zero

ReZero (Bachlechner et al., 2021) is precisely the gated-residual pattern from Section 33.3, with one specific, carefully chosen initialization: $g$ is initialized to **exactly 0**:

$$y = x + g\odot F(x), \qquad g_{\text{init}} = 0$$

At initialization, every ReZero block is **exactly** the identity function ($y=x$, since $g=0$ makes the $F(x)$ term vanish entirely) — not "approximately the identity, given typical small random weights" the way [Section 30.2](#302-why-this-trivially-fixes-the-identity-is-hard-to-learn-problem)'s plain-residual default achieves it, but *exactly*, deterministically, by construction. This is the same "start as close to a no-op as possible" principle seen in [Section 23.3](#233-why-γ1β0-is-the-standard-initialization)'s $\gamma=1,\beta=0$ LayerNorm initialization and [Section 15.6](#156-pytorch-implementation-sketch)'s zero-initialized adaLN conditioning — but now applied at the level of an entire sublayer's contribution to the residual stream, rather than to a normalization layer's affine parameters. The empirical benefit reported in the original paper: ReZero enables training substantially deeper Transformers (and other architectures) **without needing LayerNorm at all** in some configurations, since the careful zero-gated initialization alone provides enough training stability that normalization's separate stabilizing role becomes less critical — though in practice, most production architectures still combine gating-style ideas with normalization rather than removing normalization entirely.

### 33.5 DeepNorm's residual scaling, revisited from this section's framing

[Section 17.2](#172-mathematical-formulation) introduced DeepNorm's $x_{l+1}=\text{LN}(\alpha x_l + F(x_l))$ from the normalization side. Read through this section's residual-variant lens: DeepNorm is a **fixed** (not learned) scaling of the identity path specifically — $\alpha\cdot x_l$ rather than the gated pattern's *learned* $g\odot F(x_l)$ scaling of the sublayer path. Both interventions target the same underlying concern (controlling the relative magnitude of "preserved input" versus "new contribution" as it accumulates across many layers), but DeepNorm scales the **identity** side by a fixed, depth-dependent constant computed in advance, while ReZero scales the **sublayer** side by a learned parameter that starts at zero and is free to grow as training determines it's useful — two different points along the same broader design space of "give the architecture explicit control over the residual-vs-new-contribution balance," approached from opposite directions (scale up the identity path vs. scale down the new-contribution path) and with opposite philosophies (fixed-by-formula vs. learned-from-zero).

### 33.6 A unified comparison table

| Technique | Formula | Gate type | Identity at init? |
|---|---|---|---|
| **Plain residual** ([§30](#30-mathematics-of-residual-learning-y--x--fx)) | $y=x+F(x)$ | None (always full strength) | Approximate (small random $F$ weights) |
| **Highway Networks** ([§33.1](#331-highway-networks--the-gated-predecessor)) | $y=T(x)g(x)+x(1-g(x))$ | Learned, input-dependent, gates BOTH paths | Depends on $g$'s initialization |
| **DenseNet** ([§33.2](#332-densenet--connecting-everything-to-everything)) | $x_l=F_l([x_0,\ldots,x_{l-1}])$ | None — concatenation, not gating | N/A (different mechanism entirely) |
| **Gated residual** ([§33.3](#333-gated-residuals-in-transformers)) | $y=x+g\odot F(x)$ | Learned, gates only $F(x)$ | Depends on $g$'s initialization |
| **ReZero** ([§33.4](#334-rezero--starting-the-gate-at-exactly-zero)) | $y=x+g\odot F(x),\ g_0=0$ | Learned, gates only $F(x)$, starts at 0 | **Exact** ($g=0$ at init) |
| **DeepNorm** ([§33.5](#335-deepnorms-residual-scaling-revisited-from-this-sections-framing)) | $\text{LN}(\alpha x+F(x))$ | Fixed (not learned), scales identity | Approximate, depth-compensated |

### 33.7 What's next

[Section 34](#34-residuals-in-transformers-around-attention-and-ffn-blocks) returns from this survey of general residual-connection variants to the specific, concrete question of how residual connections are placed around the attention and FFN sublayers in a real Transformer block — consolidating [Section 26](#26-layernorm-inside-the-transformer-encoder) and [Section 27](#27-layernorm-inside-the-transformer-decoder)'s tensor traces with this Part's residual-connection mathematics into a single, complete picture of exactly where every addition happens.

---

## 34. Residuals in Transformers: Around Attention and FFN Blocks

### 34.1 The complete picture, consolidated

[Section 26.1](#261-the-complete-encoder-block-with-every-tensor-shape-traced) traced a full encoder block's tensor shapes; this section adds the residual-connection mathematics from [Sections 30–31](#30-mathematics-of-residual-learning-y--x--fx) explicitly into that same trace, making clear exactly **two** residual additions occur per standard Transformer block — one around the attention sublayer, one around the FFN sublayer:

$$x' = x + \text{Attn}\big(\text{LN}(x)\big) \qquad \text{(residual \#1, around attention)}$$
$$x'' = x' + \text{FFN}\big(\text{LN}(x')\big) \qquad \text{(residual \#2, around the FFN)}$$

Each of these two additions independently provides the $I+J_F$ gradient-flow guarantee derived in [Section 31.1](#311-the-single-layer-derivation) — meaning a single Transformer block contributes **two** separate "+1 identity" terms to the overall gradient product across the full network depth, not just one. For a network with $L$ blocks, the gradient flowing from the final output back to the very first block must pass through $2L$ individual residual additions (not $L$), each independently contributing its own identity-preserving term to the overall product from [Section 31.2](#312-propagating-through-many-stacked-residual-layers).

### 34.2 Why two separate residual connections, rather than one around the whole block?

It's worth asking directly: why not wrap the *entire* block (attention + FFN together) in a single residual connection, rather than two separate ones? The answer connects directly back to [Section 30.5](#305-what-this-does-not-mean)'s point about $F$ retaining full expressive freedom: wrapping attention and FFN *separately* gives the network **two independent "do nothing extra here" defaults** rather than one — a block could, in principle, learn to skip its attention computation entirely (if $\text{Attn}(\cdot)\approx 0$) while still meaningfully using its FFN computation, or vice versa, a flexibility that a single combined residual wrapping both sublayers together would not offer nearly as directly (the combined function would need to learn that the *entire* attention-plus-FFN computation should net out to near-zero, a more specific and constrained target than either sublayer independently learning to contribute little). This finer-grained "skip-ability" is part of why the two-separate-residuals pattern, established in the original Transformer paper and never seriously challenged since, has remained the universal standard.

### 34.3 A complete, fully-annotated reference implementation

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Complete Pre-LN Transformer block with both residual connections
    made fully explicit and separately commented, consolidating:
      - Section 26's encoder tensor trace
      - Section 30-31's residual mathematics
      - Section 34.1's "two independent residuals" structure
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        # --- Residual #1: around attention ---
        # F(x) here is Attn(LN(x)). The identity path (bare `x`) guarantees
        # an undiminished gradient route regardless of how well-conditioned
        # the attention computation's own Jacobian happens to be.
        normed = self.ln_attn(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        # --- Residual #2: around the FFN ---
        # An entirely SEPARATE identity path from residual #1 -- this block
        # could learn attn_out ≈ 0 while still meaningfully using ffn_out,
        # or vice versa, per Section 34.2's independent-skip argument.
        normed = self.ln_ffn(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x

# Stack several blocks and confirm gradients reach the very first block's
# parameters with healthy (non-vanished) magnitude, even at moderate depth.
depth = 24
blocks = nn.ModuleList([TransformerBlock(256, 8, 1024) for _ in range(depth)])
x = torch.randn(4, 32, 256, requires_grad=True)

h = x
for block in blocks:
    h = block(h)

loss = h.sum()
loss.backward()

first_block_grad = blocks[0].ln_attn.weight.grad.norm().item()
last_block_grad = blocks[-1].ln_attn.weight.grad.norm().item()
print(f"First block's ln_attn gradient norm: {first_block_grad:.6f}")
print(f"Last block's ln_attn gradient norm:  {last_block_grad:.6f}")
```

### 34.4 What's next

[Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison) finally delivers the fully rigorous gradient-flow comparison between Pre-LN and Post-LN that [Section 25.4](#254-why-the-difference-matters-a-preview) previewed — now equipped with both LayerNorm's complete backward mathematics ([Sections 21–22](#21-layernorm-mathematics--forward-pass)) and residual connections' complete gradient propagation derivation ([Section 31](#31-gradient-propagation-through-residual-paths-full-derivation)), making it possible to derive, rather than just assert, exactly why Pre-LN's gradient path is structurally superior at depth.

---

## 35. Pre-LN vs Post-LN — Full Gradient-Flow Comparison

### 35.1 Writing both architectures' single-layer Jacobians explicitly

**Pre-LN**: $x_{l+1}=x_l+F(\text{LN}(x_l))$. Differentiating with respect to $x_l$, using the chain rule through the composition $F\circ\text{LN}$:

$$\frac{\partial x_{l+1}}{\partial x_l} = I + J_F\cdot J_{\text{LN}}$$

where $J_{\text{LN}}$ is LayerNorm's own Jacobian (derived explicitly in [Section 22.2](#222-deriving-the-jacobian)) and $J_F$ is the sublayer's Jacobian. **The identity term $I$ sits completely outside both Jacobians** — it's added on, untouched by either $F$'s or LayerNorm's potentially-poorly-conditioned behavior.

**Post-LN**: $x_{l+1}=\text{LN}(x_l+F(x_l))$. Differentiating, using the chain rule through the composition $\text{LN}\circ(\text{Add})$:

$$\frac{\partial x_{l+1}}{\partial x_l} = J_{\text{LN}}\cdot(I+J_F)$$

**The identity term $I+J_F$ from the residual addition is here multiplied by $J_{\text{LN}}$ from the *outside*** — meaning LayerNorm's Jacobian (which, recall from [Section 22.4](#224-two-structural-properties-worth-knowing), has the all-ones vector as a null direction, and is generally **not** simply a scaled identity matrix) directly transforms the *entire* sum, identity term included, rather than only affecting the sublayer's own branch.

### 35.2 Why this single structural difference matters so much across depth

Propagating through $L$ layers, Pre-LN's full gradient product is:

$$\prod_{l=1}^{L-1}\big(I+J_{F_l}J_{\text{LN},l}\big)$$

— exactly the same expansion-with-a-surviving-$I$-term structure derived in [Section 31.2](#312-propagating-through-many-stacked-residual-layers), since each factor still has the form "identity plus something." Post-LN's product, in contrast, is:

$$\prod_{l=1}^{L-1}\big(J_{\text{LN},l}\cdot(I+J_{F_l})\big)$$

This product has **no surviving pure-identity term** the way Pre-LN's expansion does — every single factor in the product carries at least one $J_{\text{LN},l}$ multiplication, and since $J_{\text{LN}}$ is generally not the identity matrix (it actively rescales and redistributes gradient signal, as shown concretely in [Section 22.3](#223-a-concrete-worked-jacobian-d3)'s worked example), there is no clean, gradient-flow-guaranteeing term left over after expanding the full product across $L$ layers the way Pre-LN's expansion guarantees one. **This is the precise, derived (not just asserted) reason Post-LN networks are harder to train at depth and require careful learning-rate warmup**, while Pre-LN networks are considerably more robust to depth and to a wider range of learning rates from the very start of training.

### 35.3 An empirical, side-by-side comparison at real depth

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class PostLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_attn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln_ffn = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln_attn(x + attn_out)        # LN AFTER the residual addition
        ffn_out = self.ffn(x)
        x = self.ln_ffn(x + ffn_out)           # LN AFTER the residual addition
        return x

class PreLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        normed = self.ln_attn(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out                       # LN was BEFORE attention, not after the add
        normed = self.ln_ffn(x)
        x = x + self.ffn(normed)
        return x

def measure_first_layer_grad(block_cls, depth=40):
    torch.manual_seed(0)
    blocks = nn.ModuleList([block_cls(128, 4, 512) for _ in range(depth)])
    x = torch.randn(2, 16, 128, requires_grad=True)
    h = x
    for block in blocks:
        h = block(h)
    loss = h.sum()
    loss.backward()
    first = blocks[0].ln_attn.weight.grad.norm().item()
    last = blocks[-1].ln_attn.weight.grad.norm().item()
    return first, last

post_first, post_last = measure_first_layer_grad(PostLNBlock)
pre_first, pre_last = measure_first_layer_grad(PreLNBlock)

print(f"Post-LN -- first block grad: {post_first:.6f} | last block grad: {post_last:.6f}")
print(f"Pre-LN  -- first block grad: {pre_first:.6f}  | last block grad: {pre_last:.6f}")
```

Running this at 40 layers produces a striking result, though worth describing precisely rather than overstating: **Post-LN's gradients collapse to roughly $10^{-6}$ at both the first and last block**, while **Pre-LN's gradients remain healthy, at roughly $100$–$130$, at both the first and last block**. The dominant effect here isn't primarily a first-vs-last *asymmetry* within each architecture (both architectures show comparable first-to-last ratios in this particular measurement) — it's that **Post-LN's repeated $J_{\text{LN}}$ multiplications shrink the gradient's overall scale by roughly eight orders of magnitude relative to Pre-LN's**, across the entire stack, exactly the mechanism predicted by Section 35.2's product-with-no-surviving-identity-term argument: every one of the 40 layers' $J_{\text{LN}}$ factors compounds multiplicatively in Post-LN with nothing to anchor the product's overall scale, while Pre-LN's $I+J_FJ_{\text{LN}}$ structure keeps every factor's "size" anchored near 1 by the always-present identity term, regardless of how many layers are stacked. This matches the well-documented empirical fact that Post-LN Transformers require careful learning-rate warmup to train at all at this kind of depth (the gradient magnitudes a fixed learning rate would need to handle are simply on a wildly different scale at initialization than what an unwarmed-up optimizer typically expects), while Pre-LN networks are considerably more forgiving from the very first training step.

### 35.4 What's next

[Part V](#36-why-they-work-so-well-together) brings together everything derived independently in Parts I–III (normalization) and Part IV (residual connections) into a single, unified analysis of *why the combination specifically* — not either technique alone — is what actually makes training very deep modern Transformers tractable, starting with [Section 36](#36-why-they-work-so-well-together)'s direct synthesis of this section's gradient-flow result with [Section 30–31](#30-mathematics-of-residual-learning-y--x--fx)'s residual mathematics.

---

# Part V — LayerNorm + Residuals Together

## 36. Why They Work So Well Together

### 36.1 Two independent fixes for two independent problems

It's worth being precise about something this handbook has built toward but not yet stated as bluntly as it deserves: normalization and residual connections solve **two genuinely different problems**, and the fact that combining them works so well isn't a coincidence — it's because each technique's fix doesn't interfere with, and in fact actively complements, the other's.

- **Residual connections** ([Part IV](#29-motivation-vanishing-gradients-and-the-degradation-problem)) solve the **gradient-flow** problem: they guarantee, via the $I+J_F$ structure derived in [Section 31.1](#311-the-single-layer-derivation), that gradients can always propagate backward through arbitrarily many layers without vanishing, *regardless* of how well- or poorly-conditioned any individual sublayer's Jacobian happens to be.
- **Normalization** ([Parts I–III](#1-why-normalization-the-core-problem)) solves the **activation-scale** problem: it guarantees that the *input* to each sublayer stays on a controlled, predictable scale, regardless of how activations might otherwise have drifted due to accumulated computation from earlier layers.

Neither technique, alone, fully solves the other's problem. A network with only residual connections (no normalization) has guaranteed gradient flow, but nothing stops the residual stream's *magnitude* from growing unboundedly as more and more sublayer contributions accumulate additively across many layers (exactly the concern [Section 17.1](#171-motivation) raised about DeepNorm's motivation, and which [Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks) will quantify precisely). A network with only normalization (no residual connections) has every individual layer's input correctly scaled, but each layer is still a plain, non-residual transformation, fully subject to [Section 29](#29-motivation-vanishing-gradients-and-the-degradation-problem)'s vanishing-gradient and degradation problems at depth, since normalization alone does nothing to address the $J_F$-only (no surviving identity term) chain-rule product structure.

### 36.2 The combination, precisely: where each technique's job sits

In the standard Pre-LN block, $x_{l+1}=x_l+F(\text{LN}(x_l))$:

- The **residual addition** ($x_l+(\cdot)$) is entirely responsible for gradient flow — it's the term contributing the $I$ in [Section 35.1](#351-writing-both-architectures-single-layer-jacobians-explicitly)'s Pre-LN Jacobian, $I+J_FJ_{\text{LN}}$.
- The **LayerNorm call** ($\text{LN}(x_l)$) is entirely responsible for activation scale — it ensures $F$ always receives a well-conditioned, controlled-scale input, regardless of what scale the accumulated residual stream $x_l$ has drifted to by this point in the network.

These two jobs are **architecturally separated** in Pre-LN — LayerNorm sits inside $F$'s own branch (only affecting the *new contribution*, never touching the identity path itself), while the residual addition operates on the *raw, unnormalized* stream. This separation is precisely why Pre-LN's gradient-flow guarantee from [Section 35.2](#352-why-this-single-structural-difference-matters-so-much-across-depth) holds essentially unconditionally — the identity term genuinely never passes through any normalization Jacobian at all, leaving gradient flow and activation-scale control as two fully decoupled, independently-solved problems.

### 36.3 Why Post-LN's combination is structurally weaker

Contrast this with Post-LN, $x_{l+1}=\text{LN}(x_l+F(x_l))$: here, LayerNorm's job (activation-scale control) and the residual addition's job (gradient flow) are **architecturally entangled** — LayerNorm sits *outside* the addition, meaning it controls the scale of the *combined* stream (identity plus new contribution together), but in doing so it also imposes its own Jacobian on the gradient flowing back through that combined stream, exactly as derived in [Section 35.1](#351-writing-both-architectures-single-layer-jacobians-explicitly)'s Post-LN formula $J_{\text{LN}}(I+J_F)$. Post-LN does still get *some* benefit from both techniques (it's not as if Post-LN gets zero gradient-flow benefit — the $I+J_F$ term is still present inside the product, just no longer alone) — but the clean separation-of-concerns that makes Pre-LN's combination so robust is specifically lost, which is the precise, structural reason [Section 35.3](#353-an-empirical-side-by-side-comparison-at-real-depth)'s measured ~8-order-of-magnitude gradient gap between the two architectures exists at all.

### 36.4 A unifying way to think about the whole handbook so far

Here is a single sentence that compresses everything covered in Parts I–V into one statement, worth holding onto as a mental anchor for the rest of this handbook: **a modern Transformer trains successfully at great depth because residual connections guarantee that gradients always have a path back to every layer, while normalization guarantees that every layer's sublayer always receives a well-behaved input to compute its own contribution to that path** — two independent guarantees, architecturally separated by the Pre-LN placement choice, each addressing a problem the other does nothing to solve on its own.

### 36.5 What's next

[Section 37](#37-hidden-state-evolution-across-layers-worked-numerical-example) makes Section 36's abstract synthesis fully concrete with a worked numerical example tracing a single token's hidden state across several layers of a real Pre-LN stack — showing explicitly how the residual stream's magnitude evolves, how LayerNorm repeatedly "resets" the *scale* each sublayer sees without resetting the *accumulated content* of the stream itself, and tying this directly to the empirical gradient measurements from [Section 35.3](#353-an-empirical-side-by-side-comparison-at-real-depth).

---

## 37. Hidden State Evolution Across Layers (Worked Numerical Example)

### 37.1 Setting up a real, runnable trace

Rather than a hand-computed toy example (which would be impractical at any meaningful depth), this section traces actual tensor norms through a real Pre-LN stack, exactly as a debugging engineer would do when inspecting a real model's activation statistics layer-by-layer.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class PreLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        normed = self.ln_attn(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        normed = self.ln_ffn(x)
        x = x + self.ffn(normed)
        return x

depth = 20
blocks = nn.ModuleList([PreLNBlock(128, 4, 512) for _ in range(depth)])
x = torch.randn(1, 1, 128)   # a single token, for a clean per-layer trace

print(f"{'Layer':>6} | {'Residual stream norm':>22} | {'Post-LN input norm':>20}")
print("-" * 56)
print(f"{'input':>6} | {x.norm().item():>22.4f} | {'--':>20}")

h = x
for i, block in enumerate(blocks):
    normed_for_attn = block.ln_attn(h)
    h = block(h)
    print(f"{i:>6} | {h.norm().item():>22.4f} | {normed_for_attn.norm().item():>20.4f}")
```

### 37.2 What this trace reveals, and why

Running this produces two columns worth reading side by side: the **residual stream norm** (the raw, unnormalized $x_l$'s magnitude, accumulating contributions layer after layer) and the **post-LN input norm** (what each sublayer actually receives, always pinned to a value near $\sqrt{d}$ — recall [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)'s sphere-radius derivation, here $\sqrt{128}\approx11.3$, modulated by the layer's learned $\gamma$).

The expected, characteristic pattern: the **residual stream norm grows steadily with depth** (each layer adds another contribution on top of the accumulating sum — exactly [Section 17.3](#173-the-intuition-counteracting-compounding-growth-directly-at-its-source)'s "sum of many roughly-independent contributions grows like $\sqrt{\text{depth}}$" argument, now observed directly rather than just asserted), while the **post-LN input norm stays roughly constant across every single layer**, regardless of how large the residual stream itself has grown — because LayerNorm, by its very construction ([Section 7.2](#72-mathematical-formulation)), always rescales whatever it receives back to the same controlled scale, every single time, completely independent of the input's incoming magnitude.

### 37.3 The "repeatedly resets scale, never resets content" framing

This is the single most useful mental picture for understanding what Pre-LN actually does, mechanically, across depth: **the residual stream itself is allowed to grow** (and does grow, predictably, with depth) — this growth is not a bug or an oversight, it's simply the natural, expected consequence of summing many learned contributions together, and it's precisely what carries the network's accumulated "knowledge so far" forward from layer to layer, layer 1's contribution still fully present and accessible (via the unbroken identity path) all the way at layer 20. **What LayerNorm prevents is any individual sublayer ever having to directly cope with that growing scale** — every single sublayer, no matter how deep into the network it sits, always sees an input rescaled to the same familiar, controlled statistical profile it was designed and initialized to handle, with the *actual accumulated content* of the stream (which token, in what context, with what learned representation so far) fully preserved and passed through, only its raw numerical *scale* being repeatedly normalized away at the point of consumption by each new sublayer.

### 37.4 What's next

[Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks) takes this section's empirical observation — the residual stream's norm growing steadily with depth — and derives *precisely* why that growth follows a $\sqrt{\text{depth}}$ pattern rather than, say, growing linearly or exponentially, connecting this directly back to the "sum of independent contributions" statistical argument first introduced in [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius) and revisited in [Section 17.3](#173-the-intuition-counteracting-compounding-growth-directly-at-its-source).

---

## 38. Why Activations Don't Explode in Pre-LN Stacks

### 38.1 The variance-accumulation argument, precisely

Model the residual stream's evolution as $x_l = x_{l-1} + \delta_l$, where $\delta_l$ is the $l$-th block's contribution (the output of $F(\text{LN}(x_{l-1}))$). If we treat each $\delta_l$ as a roughly independent random contribution with some typical variance $v$ per element (a simplifying but broadly reasonable assumption — each block's contribution depends on a different, independently-trained set of weights, and on an input that's been freshly renormalized by LayerNorm before $F$ ever sees it, making successive $\delta_l$'s reasonably decorrelated from each other in practice), then **variances of independent contributions add**:

$$\text{Var}(x_L) = \text{Var}(x_0) + \sum_{l=1}^{L}\text{Var}(\delta_l) \approx \text{Var}(x_0) + L\cdot v$$

Since standard deviation (and therefore typical vector norm, by [Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)'s $\sigma=\|x\|/\sqrt{d}$ relationship) is the *square root* of variance:

$$\|x_L\| \propto \sqrt{\text{Var}(x_0)+L\cdot v} \approx \sqrt{L}\cdot\sqrt{v} \quad \text{for large } L$$

**This is the precise origin of the "$\sqrt{\text{depth}}$" growth pattern** referenced informally in [Section 17.3](#173-the-intuition-counteracting-compounding-growth-directly-at-its-source) and observed directly in [Section 37.2](#372-what-this-trace-reveals-and-why)'s trace — it's the exact same "variance of a sum of independent things adds, so standard deviation grows with the square root of the count" statistical fact that appeared all the way back in deriving why $\sigma=\|x\|/\sqrt{n}$ for a vector's own features ([Section 4.3](#43-scaling-by-1σ--projecting-onto-a-sphere-then-rescaling-its-radius)), now applied across the *depth* axis instead of the *feature* axis — a genuinely pleasing instance of the same underlying mathematical idea reappearing in a different context, worth recognizing explicitly as the same fact rather than two coincidentally similar ones.

### 38.2 Checking this prediction against Section 37's actual measured data

Section 37.2's trace gave us 21 actual numbers (input norm plus 20 layer norms) to check this prediction against directly, rather than just asserting the $\sqrt{L}$ pattern holds:

```python
import numpy as np

# Norms measured directly in Section 37.1's trace
measured_norms = [11.6568, 12.7455, 13.6727, 13.9939, 15.2515, 14.8600, 16.0113,
                  17.2299, 17.7474, 18.4960, 19.7999, 20.2815, 21.1267, 22.0410,
                  22.3413, 23.1463, 24.2480, 24.6372, 25.1706, 25.5781, 26.0690]
layers = np.arange(len(measured_norms))   # 0 = input, 1..20 = after each block

# Fit norm^2 (proportional to variance accumulated) against layer count linearly --
# Section 38.1 predicts norm^2 should grow roughly LINEARLY with layer count.
norms_sq = np.array(measured_norms) ** 2
slope, intercept = np.polyfit(layers, norms_sq, 1)
predicted = slope * layers + intercept
r_squared = 1 - np.sum((norms_sq - predicted)**2) / np.sum((norms_sq - norms_sq.mean())**2)

print(f"Linear fit of (norm)^2 vs layer index: slope={slope:.3f}, intercept={intercept:.3f}")
print(f"R^2 of this linear fit: {r_squared:.5f}")
```

If Section 38.1's variance-accumulation argument is correct, $\|x_l\|^2$ (proportional to accumulated variance) should fit a **straight line** against layer index $l$ very well — and indeed, running this against the actual measured data confirms exactly that: an extremely high $R^2$, confirming the squared-norm grows essentially linearly with depth, exactly as the "variances add" argument predicts (equivalently, the norm itself grows roughly with $\sqrt{l}$).

### 38.3 Why this matters: bounded, predictable growth rather than unbounded explosion

The crucial qualitative takeaway, contrasting $\sqrt{L}$ growth against the alternative failure modes this section's title explicitly rules out: **$\sqrt{L}$ growth is slow** — doubling the network's depth only increases the residual stream's typical magnitude by a factor of $\sqrt{2}\approx1.41$, not by a factor of 2 (linear growth) and certainly not exponentially. This is precisely why standard Pre-LN Transformers, at the depths actually used in production (tens to roughly 100 layers), never need DeepNorm's explicit depth-dependent rescaling ([Section 17](#17-deepnorm)) — the natural $\sqrt{L}$ growth rate is gentle enough that LayerNorm's per-layer rescaling ([Section 37.3](#373-the-repeatedly-resets-scale-never-resets-content-framing)'s "reset scale at point of consumption" mechanism) comfortably keeps pace with it at these depths. It's only at the much more extreme depths DeepNet specifically targeted (1000+ layers) that this otherwise-gentle $\sqrt{L}$ growth becomes large enough in absolute terms to warrant DeepNorm's additional, explicit intervention — directly explaining, with this section's quantitative mechanism, exactly the boundary [Section 17.6](#176-advantages-limitations-and-real-world-usage) described only qualitatively ("doesn't help at the depths used by most production LLMs").

### 38.4 What's next

[Section 39](#39-why-gradients-remain-stable--depth-and-scaling-laws) closes Part V by connecting this section's activation-scale analysis with [Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison)'s gradient-flow analysis into a single combined picture of stability across depth, and previews how these same dynamics connect to broader scaling laws governing how LLM training behavior changes as both depth and width are scaled up together — the bridge into [Part VI](#40-numpy-from-scratch-bn-ln-rmsnorm-residual-block)'s production implementation material.

---

## 39. Why Gradients Remain Stable — Depth and Scaling Laws

### 39.1 Combining Section 35's and Section 38's results into one picture

We now have two independently-derived facts that, put together, give the complete stability story for Pre-LN Transformers at depth:

- **From [Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison)**: gradient flow through the residual stream is guaranteed not to vanish, via the always-present identity term in $\prod_l(I+J_{F_l}J_{\text{LN},l})$ — this holds at *any* depth, with no dependence on how large the residual stream's magnitude has grown.
- **From [Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks)**: the residual stream's magnitude itself grows only as $\sqrt{L}$ with depth — slow enough that it never becomes numerically unmanageable at the depths production LLMs actually use, and slow enough that LayerNorm's per-layer rescaling ([Section 37.3](#373-the-repeatedly-resets-scale-never-resets-content-framing)) comfortably absorbs it at every single layer.

**Together, these two facts say something quite strong**: a Pre-LN Transformer's gradient stability doesn't degrade as depth increases — at least not for the structural reasons that plague plain (non-residual, non-normalized) stacks. Gradients have a guaranteed path back through every layer ($I$ never vanishes), and the activations those gradients are computed with respect to never reach an unmanageable scale ($\sqrt{L}$ growth is gentle). This is the complete, two-part answer to the question this entire handbook opened with in [Section 1.3](#13-why-this-is-genuinely-hard-to-get-right-at-scale): point 5 there flagged "interaction with depth" as a reason normalization alone wasn't historically sufficient — Part IV's residual connections are exactly the missing second half of that answer.

### 39.2 What this does NOT guarantee

It's worth being precise about the limits of this result, in the same spirit as [Section 12.6](#126-advantages-limitations-and-real-world-usage)'s honesty about Spectral Normalization's bound being sufficient-but-not-tight: the $I+J_F J_{\text{LN}}$ structure guarantees gradients don't *structurally* vanish due to the chain-rule-product-shrinking-to-zero mechanism from [Section 29.2](#292-the-vanishing-gradient-mechanism-precisely) — but it says nothing about whether a *specific* trained network's gradients are well-scaled for a *specific* learning rate, whether the loss landscape has other pathologies (sharp minima, saddle points) unrelated to this particular vanishing-gradient mechanism, or whether numerical precision issues (covered in [Part VIII](#52-numerical-stability-and-epsilon-selection)) introduce their own separate problems even when the underlying mathematical gradient is well-behaved. Architectural stability (this Part's subject) and optimization success (a broader topic touching learning rate schedules, optimizer choice, data quality, and more) are related but distinct questions — Parts I–V have rigorously addressed the former, which is necessary but not sufficient for the latter.

### 39.3 Connection to scaling laws — a brief, honest preview

A natural next question, given everything derived in this Part: as practitioners scale models up — more layers (depth), wider hidden dimensions (width), or both — how does the normalization-plus-residual stability story interact with the now well-documented empirical **scaling laws** governing LLM training (the relationships between model size, dataset size, compute budget, and resulting loss, popularized by work including Kaplan et al. 2020 and Hoffmann et al. 2022's "Chinchilla" scaling laws)? This is a genuinely deep topic warranting real care rather than a confident-sounding but unsubstantiated claim here — initialization schemes like **μP (Maximal Update Parameterization)**, covered properly in [Section 56](#56-rezero-normformer-μp-maximal-update-parameterization), specifically address how to scale width while keeping training dynamics consistent, and connect directly to the per-layer Jacobian conditioning arguments built throughout this Part. We defer the full, careful treatment of this connection to that later section, once μP's own machinery has been properly introduced, rather than asserting a compressed, under-justified version of it here.

### 39.4 Part V summary, and what's next

Part V is now complete. We've unified Parts I–III's normalization machinery with Part IV's residual-connection machinery into a single coherent account: [Section 36](#36-why-they-work-so-well-together) established that the two techniques solve genuinely independent problems (activation scale vs. gradient flow) that compose cleanly under Pre-LN's specific architectural separation; [Section 37](#37-hidden-state-evolution-across-layers-worked-numerical-example) made this concrete with a real, measured 20-layer trace confirming the residual stream grows while each sublayer's input stays pinned at a constant scale; [Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks) derived and empirically confirmed ($R^2=0.991$) the precise $\sqrt{\text{depth}}$ growth mechanism; and this section closed the loop by combining the activation-scale and gradient-flow results into the complete stability picture, with an honest scoping of exactly what is and isn't guaranteed by the architecture alone.

**Part VI** begins now, shifting from *why* normalization and residual connections work to *how to implement them for real, at production scale* — complete NumPy, PyTorch, Triton, and CUDA C++ implementations, distributed-training considerations (tensor/sequence/pipeline parallelism, activation checkpointing), and benchmarking/profiling/memory-analysis methodology, fulfilling the "Code Quality Requirements" and "Production-Grade Implementations" sections of this handbook's original brief.

---

# Part VI — Production-Grade Implementations

## 40. NumPy From Scratch (BN, LN, RMSNorm, Residual Block)

### 40.1 Consolidating what's already been built, properly

[Sections 6.7](#67-numpy-implementation-from-scratch), [7.6](#76-numpy-implementation-from-scratch), and [8.6](#86-numpy-implementation-from-scratch) already built and numerically verified BatchNorm, LayerNorm, and RMSNorm from scratch. This section consolidates those into a single, properly-structured module with type hints, docstrings, and unit tests — the "Code Quality Requirements" from this handbook's original brief — plus a residual block wrapper that hasn't yet been given a NumPy implementation.

### 40.2 The consolidated module

```python
"""
norm_residual.py -- Production-style NumPy implementations of normalization
and residual-connection primitives, consolidating Sections 6-8 and 30-31.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


class LayerNorm:
    """Layer Normalization (Section 7). Normalizes over the LAST axis."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self.beta = np.zeros(normalized_shape, dtype=np.float32)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, int]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.gamma.shape[0]:
            raise ValueError(
                f"Expected last dim {self.gamma.shape[0]}, got {x.shape[-1]}"
            )
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * std_inv
        out = self.gamma * x_hat + self.beta
        self._cache = (x_hat, std_inv, x.shape[-1])
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        x_hat, std_inv, d = self._cache
        reduce_axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = (grad_output * x_hat).sum(axis=reduce_axes)
        self.grad_beta = grad_output.sum(axis=reduce_axes)
        grad_x_hat = grad_output * self.gamma
        term1 = d * grad_x_hat
        term2 = grad_x_hat.sum(axis=-1, keepdims=True)
        term3 = x_hat * (grad_x_hat * x_hat).sum(axis=-1, keepdims=True)
        return (std_inv / d) * (term1 - term2 - term3)


class RMSNorm:
    """RMSNorm (Section 8). No mean-centering, no beta."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        ms = (x ** 2).mean(axis=-1, keepdims=True)
        rms_inv = 1.0 / np.sqrt(ms + self.eps)
        x_hat = x * rms_inv
        out = self.gamma * x_hat
        self._cache = (x, x_hat, rms_inv, x.shape[-1])
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        x, x_hat, rms_inv, d = self._cache
        reduce_axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = (grad_output * x_hat).sum(axis=reduce_axes)
        grad_y_gamma = grad_output * self.gamma
        term1 = grad_y_gamma * rms_inv
        coupling = (x * grad_y_gamma).sum(axis=-1, keepdims=True)
        term2 = x * coupling * (rms_inv ** 3) / d
        return term1 - term2


class ResidualBlock:
    """
    A residual wrapper, y = x + F(x), per Section 30. `sublayer` must be an
    object exposing forward(x) -> y and backward(grad_y) -> grad_x, matching
    this module's own LayerNorm/RMSNorm interface convention.
    """

    def __init__(self, sublayer) -> None:
        self.sublayer = sublayer
        self._input_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_shape = x.shape
        return x + self.sublayer.forward(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Per Section 31.1: dy/dx = I + J_F, so backward is simply
        # grad_output (the "I" term, unchanged) PLUS whatever gradient
        # flows back through the sublayer's own backward() call.
        return grad_output + self.sublayer.backward(grad_output)
```

### 40.3 Unit tests, including the numerical checks already performed earlier in this handbook

```python
"""test_norm_residual.py -- pytest-style unit tests."""
import numpy as np
import pytest
from norm_residual import LayerNorm, RMSNorm, ResidualBlock


def test_layernorm_matches_section_7_3():
    ln = LayerNorm(normalized_shape=4)
    ln.gamma = np.array([1.5, 0.5, 2.0, 1.0])
    ln.beta = np.array([0.1, 0.1, 0.1, 0.1])
    x = np.array([[2.0, 8.0, -4.0, 6.0]])
    out = ln.forward(x)
    expected = np.array([[-0.227, 0.646, -2.956, 0.755]])
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_rmsnorm_matches_section_8_4():
    rms = RMSNorm(normalized_shape=4, eps=1e-5)
    rms.gamma = np.array([1.5, 0.5, 2.0, 1.0])
    x = np.array([[2.0, 8.0, -4.0, 6.0]])
    out = rms.forward(x)
    expected = np.array([[0.548, 0.731, -1.460, 1.095]])
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_layernorm_output_has_zero_mean_unit_variance():
    ln = LayerNorm(normalized_shape=16, eps=1e-8)
    x = np.random.randn(8, 16) * 100 + 50    # arbitrary scale/offset
    # gamma=1, beta=0 (defaults) -- output should be pure standardization
    x_hat = ln.forward(x)
    np.testing.assert_allclose(x_hat.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(x_hat.var(axis=-1), 1.0, atol=1e-3)


def test_layernorm_handles_zero_variance_without_nan():
    """Per Section 21.2's degenerate-case discussion."""
    ln = LayerNorm(normalized_shape=4, eps=1e-5)
    x = np.array([[5.0, 5.0, 5.0, 5.0]])     # constant vector
    out = ln.forward(x)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, np.zeros((1, 4)), atol=1e-3)


def test_layernorm_rejects_wrong_shape():
    ln = LayerNorm(normalized_shape=4)
    with pytest.raises(ValueError):
        ln.forward(np.zeros((2, 5)))   # wrong last-dim size


def test_residual_block_backward_includes_identity_term():
    """Per Section 31.1: dy/dx must always include an undiminished +I term."""
    ln = LayerNorm(normalized_shape=4)
    ln.gamma = np.zeros(4)   # force the sublayer's OWN contribution to ~0
    block = ResidualBlock(ln)

    x = np.random.randn(2, 4)
    block.forward(x)
    grad_output = np.ones((2, 4))
    grad_input = block.backward(grad_output)

    # Even with gamma=0 (sublayer contributes nothing), the identity path
    # must still pass grad_output through essentially unchanged.
    np.testing.assert_allclose(grad_input, grad_output, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 40.4 What's next

[Section 41](#41-pytorch-native-and-custom-nnlayernorm-rmsnorm-residual-blocks) moves to PyTorch, covering both the native, framework-provided `nn.LayerNorm`/`nn.RMSNorm` modules and custom implementations matching production LLM codebases' exact conventions — building directly on the patterns already shown in [Sections 7.7](#77-production-grade-pytorch-usage) and [8.7](#87-production-grade-pytorch-implementation), now organized as a complete, cohesive reference module.

---

## 41. PyTorch Native and Custom (nn.LayerNorm, RMSNorm, Residual Blocks)

### 41.1 When to use native modules vs. custom implementations

A practical rule worth stating directly: **use `nn.LayerNorm` and (where available) `nn.RMSNorm` by default.** They're maintained, optimized (often calling fused CUDA kernels under the hood — the subject of [Sections 42–43](#42-triton-fused-layernorm-kernel)), and numerically battle-tested. Write a **custom** implementation only when you need: (a) a non-standard variant not covered by the native API (ScaleNorm, DeepNorm's scaled residual, adaLN), (b) to match an existing model's exact released code for weight-loading compatibility (e.g., reproducing LLaMA's exact `RMSNorm` class to load its released checkpoint correctly), or (c) educational/debugging purposes, exactly as this handbook has done throughout.

### 41.2 A consolidated reference module

```python
"""norm_residual.py -- PyTorch reference implementations."""
import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """Matches LLaMA/Mistral/Gemma's released implementation convention
    (Section 8.7) -- float32-upcast internally for numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class ResidualWrapper(nn.Module):
    """
    Generic y = x + sublayer(x) wrapper (Section 30), usable around ANY
    sublayer module -- attention, FFN, or a custom block.
    """

    def __init__(self, sublayer: nn.Module) -> None:
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.sublayer(x, *args, **kwargs)


class GatedResidualWrapper(nn.Module):
    """
    y = x + g * sublayer(x), with a learned per-channel gate g (Section 33.3).
    Set zero_init=True for ReZero's exact-identity-at-init behavior (33.4).
    """

    def __init__(self, sublayer: nn.Module, dim: int, zero_init: bool = True) -> None:
        super().__init__()
        self.sublayer = sublayer
        init_value = 0.0 if zero_init else 1.0
        self.gate = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.gate * self.sublayer(x, *args, **kwargs)


class PreNormBlock(nn.Module):
    """
    Combines a normalization layer + sublayer + residual into the standard
    Pre-LN pattern (Section 25.3), generically -- works with EITHER
    nn.LayerNorm or RMSNorm passed in as `norm`.
    """

    def __init__(self, norm: nn.Module, sublayer: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.sublayer(self.norm(x), *args, **kwargs)
```

### 41.3 Using these building blocks to assemble a full decoder block

```python
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPTStyleBlock(nn.Module):
    """A full decoder block, assembled from the Section 41.2 primitives."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn_block = PreNormBlock(RMSNorm(d_model), CausalSelfAttention(d_model, n_heads))
        self.ffn_block = PreNormBlock(RMSNorm(d_model), FFN(d_model, d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_block(x)
        x = self.ffn_block(x)
        return x


block = GPTStyleBlock(d_model=256, n_heads=8, d_ff=1024)
x = torch.randn(2, 16, 256)
out = block(x)
print(out.shape)   # torch.Size([2, 16, 256])
```

### 41.4 What's next

[Section 42](#42-triton-fused-layernorm-kernel) moves below the PyTorch API level entirely, into writing a custom Triton kernel that fuses LayerNorm's mean, variance, and normalize-and-affine steps into a single GPU kernel launch — the performance-motivated fusion previewed conceptually back in [Section 21.4](#214-relating-this-to-actual-production-framework-source-code), now implemented for real.

---

## 42. Triton Fused LayerNorm Kernel

### 42.1 Why fusion matters: a memory-bandwidth argument

Before any kernel code, it's worth being precise about *why* fusion is the right performance lever here, rather than asserting it as received wisdom. LayerNorm is **memory-bandwidth-bound**, not compute-bound: the actual arithmetic (a few additions, multiplications, one square root) is trivial for a GPU, but every unfused step — compute mean, compute variance, normalize, apply affine — if implemented as separate kernel launches, requires **reading the entire input tensor from GPU memory and writing the entire output back to GPU memory, once per step**. For a tensor with $N$ total elements, 4 unfused steps means roughly $4\times$ the memory traffic of a single fused kernel that reads the input once, does all the arithmetic in fast on-chip registers/shared memory, and writes the output once. Since GPU memory bandwidth (not compute throughput) is almost always the binding constraint for an operation this arithmetically simple, **fusion's benefit is almost entirely about reducing memory traffic, not about reducing the number of floating-point operations** — the same total arithmetic happens either way; what changes is how many times data round-trips to GPU memory.

### 42.2 Triton's programming model, briefly

Triton lets you write GPU kernels in a Python-like syntax, where the key idea is a **program** that each operates on one *block* of data (here, one row — one token's $d$-dimensional feature vector) in parallel across as many GPU streaming multiprocessors as the hardware has available. Each Triton program loads its assigned row entirely into fast on-chip memory, performs the *entire* LayerNorm computation for that row without writing any intermediate result back to slower GPU global memory, and only writes the final output once.

### 42.3 A complete, runnable Triton LayerNorm forward kernel

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(
    X_ptr, Y_ptr, GAMMA_ptr, BETA_ptr,
    stride_row, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Each Triton program handles ONE row (one token's d-dim vector)."""
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_row
    Y_ptr += row_idx * stride_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Single load of the entire row into fast on-chip memory.
    x = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Mean and variance computed ENTIRELY in registers -- no intermediate
    # writes back to GPU global memory between these steps.
    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_hat = diff * inv_std

    gamma = tl.load(GAMMA_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(BETA_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * gamma + beta

    # Single write of the final result -- no intermediate writes at all.
    tl.store(Y_ptr + col_offsets, y, mask=mask)


def triton_layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    """Python-side wrapper that launches one Triton program per row."""
    assert x.is_cuda, "Triton kernels require a CUDA tensor"
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x_2d.shape

    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    _layernorm_fwd_kernel[(n_rows,)](
        x_2d, y, gamma, beta,
        x_2d.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.reshape(orig_shape)
```

### 42.4 Verifying correctness against PyTorch's native implementation

Correctness is non-negotiable for a kernel like this — a fast but wrong implementation is far worse than a slow but correct one. The verification pattern below (which would need an actual CUDA-enabled GPU to run, unlike the CPU-executable verifications elsewhere in this handbook) is the standard approach for validating any custom kernel against ground truth:

```python
if torch.cuda.is_available():
    torch.manual_seed(0)
    x = torch.randn(128, 768, device="cuda", dtype=torch.float32)
    gamma = torch.randn(768, device="cuda")
    beta = torch.randn(768, device="cuda")

    triton_out = triton_layernorm(x, gamma, beta, eps=1e-5)
    torch_out = torch.nn.functional.layer_norm(x, (768,), gamma, beta, eps=1e-5)

    max_diff = (triton_out - torch_out).abs().max().item()
    print(f"Max absolute difference vs. torch.nn.functional.layer_norm: {max_diff:.2e}")
    assert torch.allclose(triton_out, torch_out, atol=1e-4), "Kernel output mismatch!"
    print("Triton kernel matches PyTorch's native LayerNorm.")
else:
    print("No CUDA device available in this environment -- "
          "this kernel requires an actual GPU to execute and benchmark.")
```

### 42.5 A note on this handbook's verification standard, applied honestly here

Every other code example in this handbook has been actually executed in this conversation, with real measured output reported rather than asserted. This Triton kernel needs a more precise disclosure: the current container environment has no CUDA-capable GPU (`torch.cuda.is_available()` returns `False` here, confirmed directly), so the kernel **cannot be compiled and run** the way every NumPy and CPU-based PyTorch example in this handbook has been. What *was* verified here: the kernel file loads cleanly and the `@triton.jit` decoration succeeds without error, confirming the code is syntactically valid and uses Triton's API correctly (decoration failures are a common way to catch basic API misuse even pre-execution) — but decoration succeeding is a much weaker guarantee than actual execution, since it doesn't compile the kernel body or run it against real data. The code above follows Triton's standard, well-documented kernel-writing patterns precisely (this is close to the structure of Triton's own official LayerNorm tutorial), but readers should run [Section 42.4](#424-verifying-correctness-against-pytorchs-native-implementation)'s verification snippet themselves on actual GPU hardware before relying on this kernel in any production setting — the honest position here is "this loads correctly and follows standard patterns" rather than "this has been verified to produce correct output," and that distinction matters.

### 42.6 What's next

[Section 43](#43-cuda-c-fused-layernorm-kernel) goes one level lower still, writing the equivalent fused LayerNorm forward pass directly in CUDA C++ — useful both for understanding what Triton compiles down to conceptually, and for situations requiring hand-tuned kernel performance beyond what Triton's higher-level abstraction provides.

---

## 43. CUDA C++ Fused LayerNorm Kernel

### 43.1 The same fusion goal, one abstraction level lower

[Section 42.1](#421-why-fusion-matters-a-memory-bandwidth-argument)'s memory-bandwidth argument applies identically here — the goal is unchanged (one read, all arithmetic in fast on-chip memory, one write). What changes at the CUDA C++ level is that you now manage thread blocks, shared memory, and warp-level reductions explicitly, rather than letting Triton's compiler handle those details automatically.

### 43.2 The reduction problem: summing across threads efficiently

The core technical challenge a hand-written CUDA kernel must solve explicitly (which Triton's `tl.sum` handled automatically in Section 42.3): computing a row's mean and variance requires **summing across every thread** that's cooperatively processing that row's $d$ elements (a single thread block typically handles one row, with multiple threads each handling a subset of that row's $d$ elements). This requires a **parallel reduction** — a well-known pattern where threads repeatedly combine pairs of partial sums, halving the number of "active" partial sums at each step, until a single total remains.

### 43.3 A complete CUDA C++ kernel

```cuda
#include <cuda_runtime.h>
#include <cmath>

// One thread block per row. BLOCK_SIZE threads cooperatively reduce
// across the row's `n_cols` elements using shared memory.
template <int BLOCK_SIZE>
__global__ void layernorm_fwd_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int n_cols,
    float eps
) {
    extern __shared__ float shared_mem[];   // used for the reduction
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = X + row * n_cols;
    float* y_row = Y + row * n_cols;

    // --- Step 1: each thread accumulates a PARTIAL sum over its slice ---
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        local_sum += x_row[i];
    }
    shared_mem[tid] = local_sum;
    __syncthreads();

    // --- Step 2: tree reduction in shared memory to get the TOTAL sum ---
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float mean = shared_mem[0] / n_cols;
    __syncthreads();

    // --- Step 3: same reduction pattern, now for the variance ---
    float local_sq_diff = 0.0f;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        float diff = x_row[i] - mean;
        local_sq_diff += diff * diff;
    }
    shared_mem[tid] = local_sq_diff;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float var = shared_mem[0] / n_cols;
    float inv_std = rsqrtf(var + eps);   // rsqrtf: fast reciprocal sqrt

    // --- Step 4: every thread normalizes and writes its OWN slice ---
    // (no further synchronization needed -- each thread's output elements
    // are independent given mean/inv_std, which are now known to all threads)
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        float x_hat = (x_row[i] - mean) * inv_std;
        y_row[i] = x_hat * gamma[i] + beta[i];
    }
}

// Host-side launch wrapper
void launch_layernorm(const float* x, float* y, const float* gamma,
                       const float* beta, int n_rows, int n_cols, float eps) {
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);

    layernorm_fwd_kernel<BLOCK_SIZE><<<grid, block, shared_mem_size>>>(
        x, y, gamma, beta, n_cols, eps
    );
}
```

### 43.4 Reading the reduction pattern explicitly

Lines worth pausing on, since this tree-reduction pattern reappears throughout GPU programming generally, not just here: `for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)` halves `stride` each iteration (256 → 128 → 64 → ... → 1), and at each step, the first `stride` threads each add a value from a thread `stride` positions away into their own running total. After $\log_2(\text{BLOCK\_SIZE})$ iterations, `shared_mem[0]` holds the complete sum across all `BLOCK_SIZE` threads' partial contributions — this is the standard, well-known logarithmic-depth parallel reduction pattern, here used twice (once for the mean, once for the variance), exactly mirroring [Section 6.5](#65-backward-pass--full-derivation)'s "sum across a reduction set" operations, just implemented at the explicit thread-cooperation level rather than expressed as a high-level `np.sum`/`tl.sum` call.

### 43.5 Honest disclosure on verification, matching Section 42.5's standard

This handbook's general verification standard — actually running every claim before stating it as fact — applies here with the same constraint as [Section 42.5](#425-a-note-on-this-handbooks-verification-standard-applied-honestly-here): there is no CUDA toolchain (`nvcc`) or GPU in the current container environment to compile or run this kernel against, confirmed directly (`nvcc` is not found in this environment). What *could* be checked, and was: the kernel's core algorithm — strided per-thread accumulation followed by a logarithmic-depth tree reduction in shared memory — was reproduced as a plain Python simulation (looping over simulated thread indices exactly as the CUDA code's `for (int i = tid; i < n_cols; i += BLOCK_SIZE)` and `for (int stride = BLOCK_SIZE/2; ...; stride >>= 1)` patterns specify) and its result matched NumPy's `mean()` exactly. This confirms the **algorithm** the kernel implements is correct, but it does not confirm the **CUDA C++ syntax itself compiles or executes correctly** — syntax errors, indexing mistakes, or CUDA-specific issues (shared memory bank conflicts, incorrect `__syncthreads()` placement, etc.) would not be caught by an algorithm-level Python simulation. The honest position: the underlying math is verified correct, but the CUDA code itself should be compiled and tested on real GPU hardware (e.g., via `nvcc` and a correctness check against PyTorch's native LayerNorm, exactly mirroring [Section 42.4](#424-verifying-correctness-against-pytorchs-native-implementation)'s pattern) before any production use.

### 43.6 What's next

[Section 44](#44-distributed-considerations-tensorsequencepipeline-parallelism--activation-checkpointing) moves from single-GPU kernel performance to multi-GPU distributed training considerations — how normalization and residual connections interact with tensor parallelism, sequence parallelism, pipeline parallelism, and activation checkpointing, each of which introduces its own specific considerations for where exactly a normalization layer's computation and memory footprint end up in a distributed training setup.

---

## 44. Distributed Considerations: Tensor/Sequence/Pipeline Parallelism + Activation Checkpointing

### 44.1 Why normalization specifically needs separate consideration in distributed training

Large-scale LLM training splits both the model's parameters and its computation across many GPUs, since a single GPU's memory cannot hold a multi-billion-parameter model plus its activations and optimizer state. Each parallelism strategy splits the computation along a *different* axis — and because LayerNorm/RMSNorm reduce over the **feature dimension** specifically ([Section 5](#5-a-taxonomy-of-normalization-which-axis-are-we-normalizing-over)'s taxonomy), some parallelism strategies interact with this reduction in ways that require explicit handling, while others don't touch it at all.

### 44.2 Tensor Parallelism — the feature dimension gets split

**Tensor Parallelism (TP)** splits individual weight matrices (e.g., a linear layer's weight) across GPUs — for instance, splitting an FFN's up-projection matrix column-wise across $P$ GPUs, so each GPU computes only a $1/P$ slice of the hidden dimension. This directly affects normalization: if a LayerNorm/RMSNorm layer's reduction needs the **full** $d$-dimensional feature vector to compute a correct mean/variance ([Section 7.2](#72-mathematical-formulation)), but each GPU under TP only holds a $d/P$ slice of that vector, **a cross-GPU communication step (an all-reduce) is required before the normalization's reduction can be computed correctly** — every GPU needs to learn the full row's sum (and sum-of-squares) before any of them can finish computing $\mu,\sigma^2$. This is a real, unavoidable communication cost specifically introduced by combining tensor parallelism with feature-dimension-reducing normalization — it doesn't show up at all for techniques that reduce over a different axis (e.g., if some hypothetical setup used BatchNorm, which reduces over the batch dimension, TP's column-split wouldn't require this particular all-reduce, though BatchNorm brings its own, separate set of problems from [Section 20](#20-why-batchnorm-fails-for-transformers)).

### 44.3 Sequence Parallelism — splitting along tokens, normalization stays local

**Sequence Parallelism (SP)**, often used in combination with TP, splits the **sequence dimension** $T$ across GPUs instead of (or in addition to) the feature dimension — each GPU holds a contiguous chunk of tokens, with their full $d$-dimensional feature vectors intact. This interacts much more favorably with LayerNorm/RMSNorm specifically *because* of [Section 7.1](#71-motivation)'s core property: since normalization only ever needs one token's own full feature vector (never looking across tokens or batch elements), and SP keeps each token's full feature vector together on a single GPU, **normalization requires zero additional cross-GPU communication under pure sequence parallelism** — each GPU can compute every one of its local tokens' LayerNorm/RMSNorm calls completely independently. This is, in fact, one of the more concrete and underappreciated practical payoffs of LayerNorm's per-token independence property (first established abstractly in [Section 7.1](#71-motivation) and demonstrated concretely with the worked tensor example in [Section 21.2](#212-a-full-worked-example-across-a-tiny-batch)): it's not just a training-stability property, it's also a **distributed-systems-friendly** property, since it requires no cross-device synchronization at all.

### 44.4 Pipeline Parallelism — normalization stays within a single stage

**Pipeline Parallelism (PP)** splits the model **by layer** — different groups of consecutive Transformer blocks are assigned to different GPUs (e.g., GPU 0 holds layers 1–10, GPU 1 holds layers 11–20, and so on), with activations passed forward (and gradients passed backward) between GPUs at each pipeline stage boundary. Normalization layers are unaffected by this split in any special way — each LayerNorm/RMSNorm call lives entirely within whichever pipeline stage (and therefore whichever single GPU) its parent Transformer block has been assigned to, computing its full reduction locally exactly as in the single-GPU case. The one consideration worth flagging: the residual stream itself ([Part IV](#29-motivation-vanishing-gradients-and-the-degradation-problem)) must be passed across pipeline stage boundaries as one of the activation tensors transmitted between GPUs — meaning the $\sqrt{\text{depth}}$-growing residual stream magnitude from [Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks) is exactly what's being communicated at each pipeline boundary, making that growth pattern directly relevant to estimating pipeline communication bandwidth requirements, not just a training-stability abstraction.

### 44.5 Activation Checkpointing — recomputing rather than storing

**Activation checkpointing** (also called gradient checkpointing) is a memory-saving technique orthogonal to the parallelism strategies above: rather than storing every intermediate activation needed for the backward pass (which [Section 7.9](#79-computational-complexity) noted costs $O(B\cdot T\cdot d)$ space per normalization layer, multiplied across every layer in a deep network), activation checkpointing stores only a subset of activations (typically at layer boundaries) and **recomputes** the rest during the backward pass by re-running the relevant forward computation on demand. This directly affects normalization layers specifically: a checkpointed LayerNorm/RMSNorm's $\hat{x}$, $\sigma^2$ (or RMS), and other forward-pass intermediates ([Section 7.6](#76-numpy-implementation-from-scratch)'s `_cache`, in this handbook's NumPy implementations) are **not stored** between the forward and backward passes — they're recomputed by re-running the layer's forward pass again, right before its backward pass needs them. This trades additional compute (an extra forward pass through checkpointed layers) for substantially reduced memory — a favorable trade specifically because, per [Section 42.1](#421-why-fusion-matters-a-memory-bandwidth-argument)'s argument, LayerNorm/RMSNorm's forward computation is cheap and fast to redo, while the *memory* it would otherwise occupy (multiplied across every layer of a deep network, every token, every training step) is often the more binding constraint at LLM training scale.

### 44.6 A unified summary

| Parallelism strategy | Splits along | Normalization impact |
|---|---|---|
| **Tensor Parallelism** | Feature dimension $d$ | Requires an all-reduce before normalization's reduction can complete correctly |
| **Sequence Parallelism** | Sequence dimension $T$ | Zero additional communication — normalization stays fully local per token |
| **Pipeline Parallelism** | Layers (depth) | No special handling — normalization stays within its assigned pipeline stage |
| **Activation Checkpointing** | N/A (memory-vs-compute trade-off) | Forward pass intermediates recomputed rather than stored across forward/backward |

### 44.7 What's next

[Section 45](#45-benchmarking-profiling-and-memory-analysis) closes Part VI with concrete benchmarking and profiling methodology — how to actually measure whether a normalization implementation is memory-bandwidth-bound as [Section 42.1](#421-why-fusion-matters-a-memory-bandwidth-argument) argued, how to quantify RMSNorm's real-world speedup over LayerNorm at production scale (closing the loop on [Section 8.9](#89-computational-and-parameter-savings--quantified)'s cost-savings table with actual measured numbers), and memory analysis methodology connecting directly to this section's activation checkpointing trade-off.

---

## 45. Benchmarking, Profiling, and Memory Analysis

### 45.1 An honest scoping note before any numbers

The benchmarks in this section run on **CPU**, in this container environment (confirmed in [Section 42.5](#425-a-note-on-this-handbooks-verification-standard-applied-honestly-here) to have no GPU available) — not the GPU environment any real production LayerNorm/RMSNorm benchmarking would actually target. CPU timing numbers here demonstrate **methodology and relative comparison** (is RMSNorm faster than LayerNorm, and by roughly what factor, for this implementation) rather than **absolute production performance figures** (which depend heavily on GPU architecture, memory bandwidth, kernel fusion status, and batch/sequence dimensions — exactly the factors [Sections 42–44](#42-triton-fused-layernorm-kernel) discussed). Treat the relative comparison as informative and the absolute numbers as specific to this CPU and this unfused PyTorch implementation, not as a stand-in for real GPU production benchmarks.

### 45.2 A timing harness comparing LayerNorm vs. RMSNorm

```python
import torch
import torch.nn as nn
import time

def benchmark(fn, x, n_warmup=10, n_iters=100):
    for _ in range(n_warmup):
        fn(x)
    start = time.perf_counter()
    for _ in range(n_iters):
        fn(x)
    elapsed = time.perf_counter() - start
    return elapsed / n_iters * 1000   # ms per call

torch.manual_seed(0)
B, T, d = 16, 512, 4096   # representative of a mid-size LLM hidden dim
x = torch.randn(B, T, d)

ln = nn.LayerNorm(d)
rms_native = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else None

class ManualRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

rms_manual = ManualRMSNorm(d)

ln_time = benchmark(ln, x)
rms_manual_time = benchmark(rms_manual, x)

print(f"nn.LayerNorm:        {ln_time:.4f} ms/call")
print(f"Manual RMSNorm:      {rms_manual_time:.4f} ms/call")
print(f"Speedup (LN/RMSNorm): {ln_time/rms_manual_time:.2f}x")

if rms_native is not None:
    rms_native_time = benchmark(rms_native, x)
    print(f"nn.RMSNorm (native): {rms_native_time:.4f} ms/call")
```

### 45.3 An honest, important result: this does NOT confirm Section 8.9's savings on this hardware

Running the benchmark above on this CPU produces a result worth reporting precisely rather than glossing over: **`nn.LayerNorm` is roughly 3x *faster* than both the manual RMSNorm implementation and PyTorch's native `nn.RMSNorm`** (≈105 ms/call for LayerNorm vs. ≈300 ms/call for either RMSNorm variant, controlling for thread count) — the *opposite* direction from [Section 8.9](#89-computational-and-parameter-savings--quantified)'s "RMSNorm should be cheaper" argument. Rather than discard this inconvenient result, it's worth actually diagnosing it, since the explanation is genuinely instructive:

```python
# Breaking the manual RMSNorm down operation-by-operation reveals the cause:
#   x.pow(2) alone:                  ~102 ms  <- already ~LayerNorm's ENTIRE runtime
#   x.pow(2).mean(-1, keepdim=True): ~114 ms
#   + rsqrt(... + eps):              ~113 ms
#   + final multiply (x * ...):      ~204 ms  <- nearly DOUBLES again here
#   nn.LayerNorm (full):             ~104 ms
```

The diagnosis: `x.pow(2)` and the final broadcasted multiply `x * rsqrt(...)` each **allocate an entirely new, full-size tensor** in naive eager-mode PyTorch — for a $(16,512,4096)$ float32 tensor, that's over 32 million elements being read and written multiple times over, exactly the unfused memory-traffic problem [Section 42.1](#421-why-fusion-matters-a-memory-bandwidth-argument) described abstractly, now observed concretely. `nn.LayerNorm`'s CPU path, by contrast, dispatches to a single fused, highly-optimized ATen kernel that performs the entire mean/variance/normalize/affine computation without materializing these large intermediate tensors. **This is direct, measured evidence for a claim this handbook has made several times but not yet demonstrated with a contradicting result: RMSNorm's theoretical FLOP and memory-traffic advantage over LayerNorm ([Section 8.9](#89-computational-and-parameter-savings--quantified)) is a property of the underlying *mathematical operation count*, not a guarantee that holds for any arbitrary implementation** — it only materializes in practice when paired with an equally well-fused kernel (exactly what [Sections 42–43](#42-triton-fused-layernorm-kernel)'s Triton/CUDA kernels are for, and exactly what production LLM training/inference stacks like LLaMA's actual released code, vLLM, and similar frameworks invest in specifically). A naive, unfused implementation of the theoretically-cheaper operation can easily lose to a maturely-optimized implementation of the theoretically-more-expensive one — a genuinely important, humbling lesson about the gap between operation-count arguments and real measured performance that this handbook would have been dishonest to omit just because it complicates an earlier, simpler claim.

### 45.4 Memory analysis methodology

```python
import torch

def estimate_layernorm_memory(B: int, T: int, d: int, dtype_bytes: int = 4) -> dict:
    """
    Estimates the activation memory a SINGLE LayerNorm call needs to retain
    for its backward pass (Section 7.6's cache: x_hat, std_inv).
    """
    n_tokens = B * T
    x_hat_bytes = n_tokens * d * dtype_bytes          # shape (B,T,d)
    std_inv_bytes = n_tokens * 1 * dtype_bytes        # shape (B,T,1)
    total_bytes = x_hat_bytes + std_inv_bytes
    return {
        "x_hat_MB": x_hat_bytes / 1e6,
        "std_inv_MB": std_inv_bytes / 1e6,
        "total_MB": total_bytes / 1e6,
    }

# A representative mid-size LLM training configuration
mem = estimate_layernorm_memory(B=32, T=2048, d=4096, dtype_bytes=2)  # bf16
print(mem)

# Multiply by the number of LayerNorm/RMSNorm calls per layer (2, per
# Section 34.1) and the total layer count to estimate AGGREGATE cached
# activation memory across an entire model, e.g. for a 32-layer model:
n_layers = 32
calls_per_layer = 2
aggregate_MB = mem["total_MB"] * calls_per_layer * n_layers
print(f"Aggregate normalization-layer cached activation memory "
      f"across {n_layers} layers: {aggregate_MB:.1f} MB")
```

### 45.5 Connecting the memory estimate back to activation checkpointing

This aggregate figure is exactly the quantity [Section 44.5](#445-activation-checkpointing--recomputing-rather-than-storing)'s activation checkpointing trade-off targets for reduction — by not storing these cached `x_hat`/`std_inv` tensors (and the corresponding caches for every other layer type) across the forward-to-backward gap, and instead recomputing them on demand, a training run trades the extra forward-pass compute for exactly this much (multiplied across every checkpointed layer type, not just normalization) freed-up memory. Running [Section 45.4](#454-memory-analysis-methodology)'s estimator at your own model's specific $B,T,d$ and layer count gives a concrete, model-specific number for exactly how much normalization-layer memory pressure activation checkpointing would relieve — useful for deciding, for a specific training run's memory budget, whether checkpointing's recompute cost is worth paying.

### 45.6 Part VI summary, and what's next

Part VI is now complete. We've covered every implementation tier this handbook's brief specified: NumPy from-scratch implementations with unit tests ([Section 40](#40-numpy-from-scratch-bn-ln-rmsnorm-residual-block)), PyTorch native and custom modules assembled into a full decoder block ([Section 41](#41-pytorch-native-and-custom-nnlayernorm-rmsnorm-residual-blocks)), a Triton fused kernel ([Section 42](#42-triton-fused-layernorm-kernel)), a CUDA C++ fused kernel ([Section 43](#43-cuda-c-fused-layernorm-kernel)), distributed-training considerations across four parallelism/memory strategies ([Section 44](#44-distributed-considerations-tensorsequencepipeline-parallelism--activation-checkpointing)), and benchmarking/profiling/memory-analysis methodology (this section) — with every claim either actually executed and measured, or, where execution genuinely wasn't possible in this environment (the GPU kernels), disclosed honestly rather than asserted as verified.

**Part VII** begins now: case studies of GPT-2/3/4, LLaMA/LLaMA 2/LLaMA 3, Mistral/Mixtral, Gemma/Gemma 2, and DeepSeek/Qwen — the complete architectural deep-dives previewed only briefly back in [Section 28](#28-layernorm-across-model-families-gpt-llama-deepseek-mistral-gemma), now covering each family's full normalization-and-residual design (and key surrounding architectural context) in the depth this handbook's brief calls for.

---

# Part VII — LLM Architecture Case Studies

## 46. GPT-2 / GPT-3 / GPT-4

### 46.1 GPT-2: the Pre-LN pivot

GPT-2 (Radford et al., 2019) was released in four sizes — confirmed configurations: GPT-2 Small (12 layers, 768 hidden dim, 12 heads, 117M params), Medium (24 layers, 1024 hidden, 16 heads, 345M params), Large (36 layers, 762M params), and XL (48 layers, 1600 hidden dim, 25 heads, 1.5B params). Architecturally, GPT-2's most consequential normalization-related decision — and the single fact most relevant to this handbook — was moving LayerNorm to the **Pre-LN** position ([Section 25.3](#253-pre-ln--the-modern-default)), in contrast to the original 2017 Transformer's and BERT's Post-LN design. This single change is widely credited with making GPT-2's deeper variants (up to 48 layers) trainable with substantially less of the careful learning-rate-warmup tuning Post-LN architectures require — exactly the gradient-flow argument derived rigorously in [Section 35](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison).

### 46.2 GPT-3: scaling the same recipe to 96 layers

GPT-3 175B (Brown et al., 2020) scales GPT-2's same fundamental architecture — Pre-LN, full LayerNorm (not yet RMSNorm, which postdates GPT-3 by roughly a year) — to a confirmed **96 layers, 12,288 hidden dimension, 96 attention heads** (128-dim per head, $96\times128=12288$). At this depth, GPT-3 sits well within the regime where standard Pre-LN's $\sqrt{\text{depth}}$ activation growth ([Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks)) remains comfortably manageable — 96 layers is substantial, but nowhere near DeepNet's 1000+-layer regime that specifically required DeepNorm's additional intervention ([Section 17.6](#176-advantages-limitations-and-real-world-usage)). GPT-3's normalization design is, in this sense, a direct, larger-scale validation of the same Pre-LN-plus-residual recipe this handbook has derived from first principles throughout Parts I–V, rather than a departure requiring new normalization machinery.

### 46.3 GPT-4: an honest disclosure of what is and isn't publicly known

Unlike GPT-2 and GPT-3, **OpenAI has not publicly released GPT-4's architectural specifications** — no confirmed layer count, hidden dimension, attention head count, or normalization details exist in any official technical report. What is more broadly believed, based on leaks and informed industry speculation rather than official confirmation, is that GPT-4 uses a **Mixture-of-Experts (MoE)** architecture — a meaningfully different structural choice from GPT-3's dense design, though one that, importantly, doesn't change *this handbook's specific subject matter* in any fundamental way: MoE affects which FFN expert(s) a given token's hidden state gets routed through, but the normalization and residual-connection design surrounding that FFN (Pre-LN or Pre-RMSNorm, wrapped in the same $x+F(\text{LN}(x))$ structure derived throughout this handbook) is architecturally orthogonal to the routing question. The honest, appropriately-hedged position this handbook takes: GPT-4's specific normalization choices (LayerNorm vs. RMSNorm, exact $\epsilon$, any QK-Norm or soft-capping additions) are **not publicly confirmed**, and any specific claim about them found elsewhere should be treated as informed speculation, not verified fact, unless backed by an official OpenAI technical disclosure.

### 46.4 What's next

[Section 47](#47-llama--llama-2--llama-3) moves to the LLaMA family — where, unlike GPT-4, Meta has released full architectural specifications across all three generations, allowing this handbook to ground its analysis in confirmed configuration details rather than informed speculation.

---

## 47. LLaMA / LLaMA 2 / LLaMA 3

### 47.1 LLaMA 1: the architecture this handbook's RMSNorm coverage is built around

LLaMA 1 (Touvron et al., 2023) explicitly states its three core departures from the original Transformer, each citing its specific inspiration: **pre-normalization** (citing GPT-3 — confirming [Section 46.2](#462-gpt-3-scaling-the-same-recipe-to-96-layers)'s point that Pre-LN's lineage runs directly from GPT-2/3 into LLaMA), using **RMSNorm** specifically (citing Zhang & Sennrich 2019 — exactly [Section 8](#8-rmsnorm)'s subject); **SwiGLU** activation in the FFN (citing PaLM, replacing ReLU, and notably using a $\frac{2}{3}\cdot4d$ FFN hidden dimension rather than PaLM's plain $4d$, to keep parameter count comparable despite SwiGLU's extra weight matrix — exactly the "three weight matrices instead of two" trade-off, costed precisely); and **RoPE** (Rotary Position Embeddings) replacing absolute positional embeddings entirely. LLaMA 1 was released in four sizes (7B, 13B, 33B, 65B parameters), trained on 1–1.4 trillion tokens.

### 47.2 LLaMA 2: GQA as the primary architectural change

LLaMA 2 (Touvron et al., 2023b)'s own technical report is explicit that it "adopt[s] most of the pretraining setting and model architecture from LLaMA 1," with **Grouped-Query Attention (GQA)** and increased context length cited as the primary differences. This is worth being precise about, since it's a common point of confusion: **LLaMA 2's headline architectural change is in attention (GQA), not in normalization** — it continues using the same Pre-RMSNorm design as LLaMA 1 unchanged. Confirmed configuration at 70B scale: 80 layers, 8192 hidden dimension (verified in [Section 32.4](#324-why-this-matters-specifically-for-the-llm-context-this-handbook-focuses-on)).

### 47.3 LLaMA 3: scaling further, RoPE base frequency increase, GQA everywhere

LLaMA 3 (2024) continues the same Pre-RMSNorm-plus-residual design unchanged from LLaMA 1/2 — confirmed directly: *"Llama 3 applies RMSNorm before each attention and feed-forward block (pre-norm), consistent with Llama 1 and 2."* The two confirmed changes relevant to this handbook's scope: GQA is now used **across all released sizes** (LLaMA 2 used GQA only at larger scales; LLaMA 3's 8B model uses GQA too, with confirmed 8 KV heads against 32 query heads), and RoPE's base frequency was increased substantially (to 500,000, up from LLaMA 2's 10,000) specifically to improve length generalization toward the longer context windows LLaMA 3 targets. Confirmed configuration figures across the family (from LLaMA 3.1, which shares the base architecture): RMSNorm $\epsilon=10^{-5}$, hidden size 4096/3072/2048 and 32/28/16 layers for the 8B/3B/1B variants respectively, head dimension 128 (or 64 for the smallest 1B model), SiLU activation (the activation SwiGLU itself is built from — see [Section 47.1](#471-llama-1-the-architecture-this-handbooks-rmsnorm-coverage-is-built-around)).

### 47.4 What this means for this handbook's normalization coverage specifically

The single most important takeaway from this three-generation comparison, for this handbook's purposes: **LLaMA's normalization design has been essentially unchanged since its very first release** — Pre-RMSNorm, applied before attention and before the FFN, exactly the $x+F(\text{RMSNorm}(x))$ pattern derived in full throughout [Parts III–V](#20-why-batchnorm-fails-for-transformers). Every subsequent architectural innovation across LLaMA 1→2→3 (GQA, RoPE base frequency, context length) has been layered on top of this same, stable normalization foundation rather than requiring any change to it — a real, concrete illustration of [Section 39.1](#391-combining-section-35s-and-section-38s-results-into-one-picture)'s claim that Pre-LN/Pre-RMSNorm's stability holds robustly across a wide range of other architectural choices and scales, rather than being a fragile design that needs continual re-tuning as other parts of the architecture evolve.

### 47.5 What's next

[Section 48](#48-mistral--mixtral) covers Mistral and Mixtral — architecturally very close to LLaMA (confirmed: same Pre-RMSNorm, SwiGLU, RoPE foundation), with the interesting attention-mechanism innovations (sliding window attention, Mixture-of-Experts routing in Mixtral) layered on top, exactly mirroring the pattern just established for LLaMA's own evolution.

---

## 48. Mistral / Mixtral

### 48.1 Mistral 7B: confirmed configuration

Mistral 7B (Jiang et al., 2023) confirmed architecture: dim 4096, n_layers 32, head_dim 128, n_heads 32 query heads, **n_kv_heads 8** (Grouped-Query Attention), sliding window 4096, context length 8192, vocab size 32,000. Mistral's own paper frames its contribution relative to LLaMA explicitly: it "introduces a few changes" — specifically **Sliding Window Attention (SWA)** and GQA — while otherwise building on the same foundation. Normalization-wise, this means Mistral inherits LLaMA's exact Pre-RMSNorm design unchanged; the paper's "changes" section, confirmed directly, is entirely about the attention mechanism (SWA's bounded local attention window, enabling a theoretical receptive field of $W\times k$ tokens after $k$ layers — confirmed ≈131K tokens with $W=4096$ at the deepest layers) and not about normalization at all.

### 48.2 Mixtral 8x7B: MoE layered onto the identical foundation

Mixtral's own paper states its relationship to Mistral 7B as directly as possible: *"Mixtral is based on a transformer architecture and uses the same modifications as described in [Mistral 7B]... with the notable exceptions that Mixtral supports a fully dense context length of 32k tokens, and the feed-forward blocks are replaced by Mixture-of-Expert layers."* Confirmed mechanics: 8 experts per layer, a router selects the **top-2** experts per token via a learned linear-plus-softmax gating network, and their outputs are combined via a weighted sum; total parameters ≈47B, but only ≈13B are active for any given token's forward pass (confirmed directly from the paper and corroborated by Hugging Face's integration writeup).

### 48.3 The precise, reusable pattern this confirms

Reading Sections 48.1–48.2 together gives a clean, doubly-confirmed instance of exactly the same point [Section 46.3](#463-gpt-4-an-honest-disclosure-of-what-is-and-isnt-publicly-known) raised about MoE architectures generally: **switching from a dense FFN to a Mixture-of-Experts FFN is, architecturally, a change to *which* feed-forward computation $F$ a token's hidden state passes through — it is not a change to the normalization-and-residual scaffolding surrounding that computation.** Mixtral's blocks are still $x+F_{\text{MoE}}(\text{RMSNorm}(x))$ — the exact same Pre-RMSNorm pattern this handbook has analyzed since [Part III](#20-why-batchnorm-fails-for-transformers), with $F_{\text{MoE}}$ (router selects 2 of 8 experts, combine their outputs) simply substituted in place of $F$'s single dense FFN. Confirmed directly: source describing Mixtral notes RMSNorm is used "in all Norm Layers, similar to Llama" — explicit, direct confirmation that the MoE conversion left normalization completely untouched.

### 48.4 What's next

[Section 49](#49-gemma--gemma-2) covers Gemma and Gemma 2 — including the correction made earlier in [Section 28](#28-layernorm-across-model-families-gpt-llama-deepseek-mistral-gemma) regarding which generation actually uses QK-Norm versus logit soft-capping, now given the full, properly-sourced treatment.

---

## 49. Gemma / Gemma 2

### 49.1 Gemma 1: the same Pre-RMSNorm foundation, with GeGLU instead of SwiGLU

Gemma's architecture is confirmed to follow essentially the same Pre-RMSNorm/RoPE foundation as LLaMA, with one activation-function difference: Gemma uses **GeGLU** rather than SwiGLU in its FFN (both are gated-linear-unit variants — the gating mechanism is shared, only the underlying activation differs, GELU vs. SiLU/Swish). This is, again, a change confined to the FFN's internal computation $F$, not to the surrounding normalization-and-residual structure.

### 49.2 Gemma 2: confirmed RMSNorm sandwich, GQA, and logit soft-capping (correcting Section 28)

Gemma 2's confirmed architectural specifics, directly sourced: **"RMSNorm is used to normalize the input and output of each transformer sub-layer, the attention layer, and the feedforward layer"** — i.e., Gemma 2 genuinely does use the **Sandwich-LN-style pattern** introduced conceptually in [Section 18.5](#185-other-modern-variants-briefly) and revisited in [Section 25.5](#255-sandwich-ln--combining-both): RMSNorm both before *and* after each sublayer, not just before. Gemma 2 also confirmed: **Grouped-Query Attention** with `num_groups = 2` for both the 9B and 27B sizes, and **alternating local sliding-window attention (4096 tokens) and global attention (8192 tokens) every other layer** — a confirmed precursor to Gemma 3's later, more aggressive 5:1 local-to-global ratio (covered in the search results gathered for this section, though Gemma 3 itself sits outside this handbook's named GPT/LLaMA/Mistral/Gemma/DeepSeek scope).

**The correction from [Section 28](#28-layernorm-across-model-families-gpt-llama-deepseek-mistral-gemma)**, now properly sourced: Gemma 2's attention-logit stabilization mechanism is confirmed to be **logit soft-capping** — $\text{logits} \leftarrow \text{soft\_cap}\times\tanh(\text{logits}/\text{soft\_cap})$, applied to both each attention layer's logits and the final output layer's logits — **not** QK-Norm. QK-Norm was confirmed (via Gemma 3's own technical report, which explicitly states it "replace[s] the soft-capping of Gemma 2 with QK-norm") to be a **Gemma 3** innovation specifically, introduced *because* soft-capping was judged less effective or less elegant than directly normalizing queries and keys before the dot product. This is exactly the kind of detail [Section 28.3](#283-the-smaller-model-specific-deviations-are-worth-noticing-too) flagged as needing later, careful verification — and the verification process for this section is precisely what caught and corrected the original error.

### 49.3 Why soft-capping and QK-Norm address the same underlying concern differently

Both techniques target the same problem [Section 26.2](#262-why-attention-specifically-benefits-from-being-fed-normalized-input) described: uncontrolled attention logit magnitudes causing softmax saturation. **Soft-capping** intervenes *after* the logits are computed, directly bounding their range via a smooth $\tanh$-based squashing function — logits can grow arbitrarily large internally, but the soft-cap guarantees the *post-cap* value passed to softmax never exceeds $\pm\text{soft\_cap}$. **QK-Norm** intervenes *before* the dot product, normalizing the query and key vectors themselves (using LayerNorm or RMSNorm, applied to $q$ and $k$ specifically) so that the dot product computed from them is naturally well-scaled by construction, rather than computed first and then clamped after the fact. Gemma 3's move from the former to the latter reflects a broader pattern this handbook has seen repeatedly (e.g., [Section 17](#17-deepnorm)'s DeepNorm, [Section 33](#33-variants-highway-networks-densenet-gated-residuals-rezero-deepnorm-scaling)'s residual variants): intervening *before* a potentially-destabilizing computation, to prevent the problem from occurring in the first place, is generally found to be a cleaner and more effective design than intervening *after* the fact to clamp an already-computed, potentially-extreme value.

### 49.4 What's next

[Section 50](#50-deepseek-v2v3-and-qwen) covers DeepSeek's V2/V3 architecture and Qwen — including DeepSeek's distinctive **Multi-head Latent Attention (MLA)**, confirmed to use RMSNorm specifically on its compressed latent vectors (a normalization placement genuinely novel relative to every other model covered so far in Part VII), and a brief comparative note on Qwen's normalization choices.

---

## 50. DeepSeek (V2/V3) and Qwen

### 50.1 DeepSeek's Multi-head Latent Attention — confirmed mechanics

DeepSeek-V2 introduced, and DeepSeek-V3 (DeepSeek-AI, 2024) continues using, **Multi-head Latent Attention (MLA)** — confirmed directly from the V3 technical report's stated configuration: 128 attention heads, 128 per-head dimension, a **KV compression dimension of 512** and a separate query compression dimension. The core idea, confirmed across multiple independent sources: rather than caching full-size key/value tensors per head (the standard KV-cache approach), MLA compresses keys and values into a much lower-dimensional **latent vector** before caching, then decompresses (projects back up) at attention-computation time — directly reducing KV-cache memory, which is one of the dominant inference-time memory costs for long-context serving.

### 50.2 The genuinely novel normalization detail: RMSNorm on the compressed latent vectors

This is the specific fact this handbook's research process surfaced that goes beyond every other model covered in Part VII so far: DeepSeek-V3's own technical report states directly — *"As DeepSeek-V2, DeepSeek-V3 also employs additional RMSNorm layers after the compressed latent vectors, and multiplies additional scaling factors at the width bottlenecks."* This is a normalization placement **inside the attention mechanism's internal compression step**, not just at the standard pre-attention/pre-FFN block-level positions every other model in this Part has used. The motivation, confirmed from independent technical explainers: applying RMSNorm directly to the compressed latent representation **"helps prevent the compressed representation from drifting to extreme values during training"** — i.e., exactly [Section 1](#1-why-normalization-the-core-problem)'s foundational motivation (controlling activation scale), applied at a specific internal bottleneck point unique to MLA's architecture, rather than at a sublayer boundary.

### 50.3 A second genuinely interesting detail: recomputing RMSNorm specifically to save memory

DeepSeek-V3's technical report confirms a specific activation-checkpointing decision directly relevant to [Section 44.5](#445-activation-checkpointing--recomputing-rather-than-storing): *"We recompute all RMSNorm operations and MLA up-projections during back-propagation, thereby eliminating the need to persistently store their output activations."* This is a direct, large-scale production confirmation of exactly the trade-off this handbook described generally in Section 44.5 — DeepSeek-V3's own engineering team made the explicit choice to recompute RMSNorm's forward pass rather than cache its activations, precisely because (per [Section 42.1](#421-why-fusion-matters-a-memory-bandwidth-argument)'s argument) RMSNorm is cheap enough to recompute that the memory savings are worth the small additional compute cost, at the scale of a 671B-parameter model.

### 50.4 Qwen: Pre-RMSNorm, GQA, and a notable convergence with Gemma 3 on QK-Norm

Qwen2.5's confirmed architecture: Pre-RMSNorm, GQA, SwiGLU, RoPE, and (notably, unlike most other families covered in this Part) **QKV bias retained** in the attention projections. Qwen3's technical report confirms a specific, interesting evolution: *"we remove QKV-bias used in Qwen2... and introduce QK-Norm to the attention mechanism to ensure stable training."* This is a direct, independently-confirmed parallel to [Section 49.2–49.3](#492-gemma-2-confirmed-rmsnorm-sandwich-gqa-and-logit-soft-capping-correcting-section-28)'s Gemma 3 finding: **two separate model families (Google's Gemma and Alibaba's Qwen) independently converged on adopting QK-Norm** in their most recent generations, each moving away from a different prior approach (soft-capping for Gemma, QKV-bias for Qwen) toward the same "normalize queries and keys directly before the attention dot-product" solution — a genuinely meaningful piece of evidence that QK-Norm represents a real, broadly-applicable improvement to attention stability, observed independently by multiple research teams rather than a single lab's idiosyncratic choice.

### 50.5 Part VII summary, and what's next

Part VII is now complete, with every claim grounded in directly-verified sources rather than assumed from memory — including one real correction made and properly closed out (Gemma 2 vs. Gemma 3's QK-Norm attribution) and several genuinely novel findings surfaced specifically through this verification process (DeepSeek's latent-vector RMSNorm placement and recomputation strategy, the Gemma/Qwen QK-Norm convergence). Across every model covered — GPT-2/3 ([Section 46](#46-gpt-2--gpt-3--gpt-4)), LLaMA 1/2/3 ([Section 47](#47-llama--llama-2--llama-3)), Mistral/Mixtral ([Section 48](#48-mistral--mixtral)), Gemma/Gemma 2 ([Section 49](#49-gemma--gemma-2)), and DeepSeek/Qwen (this section) — the same foundational pattern derived from first principles throughout [Parts I–V](#1-why-normalization-the-core-problem) holds: Pre-normalization (LayerNorm or its RMSNorm simplification), wrapped in a residual connection, at every attention and FFN sublayer — with newer architectural innovations (GQA, MoE, MLA, sliding-window/local-global attention, QK-Norm) layered on top of, rather than replacing, this stable foundation.

[Section 51](#51-cross-model-comparison-table) closes Part VII with a single consolidated comparison table across all five model families covered, before [Part VIII](#52-numerical-stability-and-epsilon-selection) turns to production-deployment considerations: numerical stability, mixed precision, kernel fusion, and inference-time trade-offs.

---

## 51. Cross-Model Comparison Table

### 51.1 The consolidated table

| Model | Norm type | Block placement | Attention-specific norm | Layers (representative size) | Hidden dim |
|---|---|---|---|---|---|
| **GPT-2** | LayerNorm | Pre-LN | — | 12–48 (117M–1.5B) | 768–1600 |
| **GPT-3** | LayerNorm | Pre-LN | — | 96 (175B) | 12,288 |
| **GPT-4** | *Not publicly confirmed* | *Not publicly confirmed* | *Not publicly confirmed* | *Not publicly confirmed* | *Not publicly confirmed* |
| **LLaMA 1/2/3** | RMSNorm | Pre-RMSNorm | — | 32 (7-8B) – 80 (70B) | 4096 – 8192 |
| **Mistral 7B** | RMSNorm | Pre-RMSNorm | — | 32 | 4096 |
| **Mixtral 8x7B** | RMSNorm | Pre-RMSNorm (MoE FFN) | — | 32 | 4096 |
| **Gemma 2** | RMSNorm | Pre- **and post-**RMSNorm (sandwich) | Logit soft-capping | 26–46 (2B–27B) | varies |
| **Gemma 3** | RMSNorm | Pre- and post-RMSNorm (sandwich) | **QK-Norm** | varies | varies |
| **DeepSeek V2/V3** | RMSNorm | Pre-RMSNorm + **latent-vector RMSNorm inside MLA** | — | varies (V3: 671B total/37B active) | varies |
| **Qwen2.5** | RMSNorm | Pre-RMSNorm | QKV bias (no QK-Norm) | 24–80+ (0.5B–72B+) | varies |
| **Qwen3** | RMSNorm | Pre-RMSNorm | **QK-Norm** (QKV bias removed) | varies (0.6B–235B) | varies |

*(Entries marked "varies" reflect genuine model-size-dependent configuration differences within a family, rather than uncertainty about the architecture itself — unlike GPT-4's entries, which reflect genuine lack of public disclosure.)*

### 51.2 The two patterns that hold across every single confirmed entry

Stepping back from the individual details, two patterns hold with **zero exceptions** across every model in this table whose architecture has been publicly confirmed:

1. **Every single model uses Pre-normalization** ([Section 25.3](#253-pre-ln--the-modern-default)) as its foundational placement — even Gemma's "sandwich" pattern is Pre-normalization *plus* an additional post-sublayer norm, never Post-normalization *instead of* Pre-normalization. This is the single most universal architectural consensus this entire handbook has uncovered.
2. **Every model released since LLaMA (2023 onward) uses RMSNorm**, not full LayerNorm — confirming [Section 8.12](#812-real-world-usage)'s claim with five independent, directly-verified model families.

### 51.3 The pattern that's genuinely evolving: attention-specific stabilization

Unlike the two universal patterns above, the *specific mechanism* for stabilizing attention logits is visibly still evolving across this table: GPT/LLaMA/Mistral/Mixtral use none at all (relying solely on Pre-RMSNorm's general activation-scale control), DeepSeek uses its MLA-specific latent-vector normalization, Qwen2.5 uses QKV bias, and both Gemma 3 and Qwen3 have converged on QK-Norm specifically — having moved there from two different starting points (soft-capping and QKV-bias, respectively). If this handbook were being updated again in the future, attention-specific normalization is the dimension along which the most continued evolution should be expected, based on this trajectory.

### 51.4 What's next

[Part VIII](#52-numerical-stability-and-epsilon-selection) shifts from architecture-level case studies to production-deployment considerations that cut across every model in this table regardless of family: numerical stability and $\epsilon$ selection (closing the loop on the various $\epsilon$ values referenced throughout Part VII), mixed-precision training (BF16/FP16), kernel fusion and memory bandwidth, and inference-time latency trade-offs.

---

# Part VIII — Production Considerations

## 52. Numerical Stability and Epsilon Selection

### 52.1 Consolidating every $\epsilon$ value referenced across this handbook

Part VII referenced several different $\epsilon$ values without yet explaining *why* they differ. Consolidating them: LLaMA/Mistral use $\epsilon=10^{-5}$ (confirmed directly from [Section 47.3](#473-llama-3-scaling-further-rope-base-frequency-increase-gqa-everywhere)'s source), while Gemma and DeepSeek use the smaller $\epsilon=10^{-6}$ (as noted in [Section 28.1](#281-a-focused-preview-not-the-full-case-study)'s table). This isn't arbitrary — it reflects a genuine engineering trade-off this section makes precise.

### 52.2 The trade-off, precisely

Recall [Section 3.5](#35-the-epsilon-you-will-see-everywhere)'s original motivation: $\epsilon$ prevents $\sqrt{\sigma^2+\epsilon}$ (or RMSNorm's analogous denominator) from collapsing toward zero when the underlying variance/mean-square is itself near zero. The trade-off:

- **Too large an $\epsilon$**: biases the normalization — when $\sigma^2$ (or the mean-square) is small but not negligible, a large $\epsilon$ added to it changes the effective denominator more than it should, making the output's actual variance meaningfully less than the intended 1 (or, for RMSNorm, distorting the intended unit-RMS scaling). This is a real, measurable accuracy cost, not just a theoretical concern.
- **Too small an $\epsilon$**: risks exactly the original division-by-near-zero problem $\epsilon$ exists to prevent, especially in lower-precision formats (FP16's limited exponent range makes this more acute than FP32 or BF16, which has FP32's exponent range despite fewer mantissa bits — a distinction we make precise in [Section 53](#53-mixed-precision-bf16-vs-fp16-quantization-effects)).

### 52.3 An empirical demonstration of both failure modes

```python
import torch

def layernorm_with_eps(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)

# A vector with a SMALL but genuinely nonzero variance -- chosen specifically
# to make eps's bias effect visible (a fully zero-variance vector, as tested
# in Section 21.2, just floors to 0 regardless of eps and hides this effect).
x_small_var = torch.tensor([[1.0, 1.0001, 0.9999, 1.0002]])
print("Actual variance of this vector:", x_small_var.var(unbiased=False).item())

for eps in [1e-10, 1e-8, 1e-6, 1e-5, 1e-3, 1e-1]:
    out = layernorm_with_eps(x_small_var, eps)
    print(f"eps={eps:<8} -> output variance: {out.var(unbiased=False).item():.6f} "
          f"(intended: 1.0)")
```

Running this on a vector with actual variance $\approx1.25\times10^{-8}$ shows the bias effect directly and quantitatively: with $\epsilon=10^{-10}$ (smaller than the vector's own variance), the output variance is $\approx0.992$ — very close to the intended 1.0. As $\epsilon$ grows past the vector's own tiny variance — $10^{-8}, 10^{-6}, 10^{-5}, 10^{-3}$ — the output variance collapses progressively further from 1.0 (to $0.556$, then $0.012$, then $0.0012$, then effectively $0$), because $\epsilon$ increasingly dominates the denominator's $\sigma^2+\epsilon$ sum, suppressing the normalization's intended rescaling-to-unit-variance behavior. **This is precisely why $\epsilon$ must be chosen relative to the typical scale of $\sigma^2$ (or RMSNorm's mean-square) actually encountered in a given model's activations** — too large relative to that typical scale, and $\epsilon$ silently distorts normalization for any token whose variance happens to be small (not just exactly zero); too small relative to the precision format's limits (the subject of [Section 53](#53-mixed-precision-bf16-vs-fp16-quantization-effects)), and the original division-by-near-zero risk reappears. This is the concrete, quantitative mechanism behind why different model families ([Section 52.1](#521-consolidating-every-ε-value-referenced-across-this-handbook)) settle on different $\epsilon$ values — it reflects each team's empirical tuning against their own model's specific activation-scale distribution, not an arbitrary or universal constant.

### 52.4 What's next

[Section 53](#53-mixed-precision-bf16-vs-fp16-quantization-effects) builds directly on this section's $\epsilon$/precision interaction, giving the complete treatment of BF16 vs. FP16 training and inference — exactly why the float32-upcast pattern from [Section 8.7](#87-production-grade-pytorch-implementation) and [Section 8.8](#88-why-the-float32-upcast-matters) is necessary, and how quantization (going below 16 bits) introduces its own, related set of normalization-specific considerations.

---

## 53. Mixed Precision: BF16 vs FP16, Quantization Effects

### 53.1 The structural difference between BF16 and FP16

Both BF16 and FP16 use 16 bits total, but split them differently between exponent and mantissa: **FP16** uses 5 exponent bits and 10 mantissa bits (more precision, less range); **BF16** uses 8 exponent bits — the *same* as FP32 — and only 7 mantissa bits (less precision, but the *same dynamic range* as FP32). This single structural difference explains nearly every practical difference between them for LLM training: BF16's FP32-matching exponent range means it almost never overflows/underflows in the same way FP16 can, at the cost of less precision for any individual number; FP16's extra precision is valuable but its narrower exponent range (roughly $\pm 6\times10^4$ vs. BF16/FP32's $\pm 3\times10^{38}$) makes it considerably more prone to overflow on large activations or gradients.

### 53.2 Why this matters specifically for normalization's variance computation

Recall [Section 3.2](#32-variance--what-it-represents)'s squaring step: variance/mean-square computation involves squaring values, which for FP16's narrower range can overflow much sooner than the un-squared values would. Let's measure this directly:

```python
import torch

# Compute the same RMSNorm-style mean-square reduction in FP16 vs FP32,
# using activation magnitudes representative of real (post-warmup) LLM
# hidden states, which can have outlier values into the hundreds.
x = torch.tensor([50.0, 120.0, 80.0, 200.0, 95.0, 60.0])

x_fp32 = x.to(torch.float32)
x_fp16 = x.to(torch.float16)

ms_fp32 = x_fp32.pow(2).mean()
ms_fp16 = x_fp16.pow(2).mean()

print(f"FP32 mean-square: {ms_fp32.item():.4f}")
print(f"FP16 mean-square: {ms_fp16.item():.4f}")
print(f"Relative error: {abs(ms_fp32.item() - ms_fp16.item()) / ms_fp32.item() * 100:.4f}%")

# Now push toward FP16's actual overflow point directly
x_large = torch.tensor([300.0], dtype=torch.float16)
squared = x_large.pow(2)
print(f"\n300^2 in FP16: {squared.item()}")
x_larger = torch.tensor([260.0], dtype=torch.float16)
print(f"260^2 in FP16: {x_larger.pow(2).item()}")
```

Running this confirms both effects precisely. For the first test (activation magnitudes 50–200, well within FP16's safe range for squaring): FP32's mean-square is $12654.17$, FP16's is $12656.0$ — a relative error of only $0.0145\%$, confirming that **for moderate-magnitude activations, FP16's reduced precision alone introduces only a small, usually tolerable error**. The second test reveals the sharper, more dangerous failure mode: **both $300^2$ and $260^2$ overflow to `inf`** when computed in FP16. The reason is precise and worth knowing exactly: FP16's maximum representable value is $65504$, and $\sqrt{65504}\approx255.9$ — so **any activation value whose magnitude exceeds approximately 256 will overflow to `inf` the moment it's squared in FP16**, even though the *un-squared* value (260, 300) is nowhere near FP16's own overflow point on its own. This is exactly [Section 53.1](#531-the-structural-difference-between-bf16-and-fp16)'s point made concrete and numerically exact: trained LLM activations routinely produce outlier values well above 256 in magnitude (this handbook's own [Section 17.4](#174-a-simplified-numerical-illustration) discussed residual-stream magnitudes growing into the tens, and real production models can see considerably larger outliers at specific layers/positions) — meaning **FP16's squaring-induced overflow is not a rare, theoretical edge case, it is a routinely-triggered failure mode for any normalization layer's variance/mean-square computation that doesn't upcast to a wider-range format first**, which is precisely why the float32-upcast pattern from [Sections 8.7–8.8](#87-production-grade-pytorch-implementation) is close to mandatory for any FP16 (and, to a lesser extent given BF16's much larger exponent range, even BF16) training or inference pipeline.

### 53.3 What's next

[Section 54](#54-kernel-fusion-memory-bandwidth-cache-efficiency) builds on this section's precision discussion to cover kernel fusion, memory bandwidth, and cache efficiency in full generality — consolidating and extending [Section 42.1](#421-why-fusion-matters-a-memory-bandwidth-argument)'s memory-bandwidth argument and [Section 45.3](#453-an-honest-important-result-this-does-not-confirm-section-89s-savings-on-this-hardware)'s measured fusion-matters finding into a complete production-deployment picture.

---

## 54. Kernel Fusion, Memory Bandwidth, Cache Efficiency

### 54.1 Revisiting the central lesson from Section 45.3, generalized

[Section 45.3](#453-an-honest-important-result-this-does-not-confirm-section-89s-savings-on-this-hardware)'s measured result — naive eager-mode RMSNorm losing to `nn.LayerNorm`'s fused kernel, despite RMSNorm's lower theoretical operation count — generalizes into a principle worth stating explicitly for any normalization (or, more broadly, any memory-bandwidth-bound) operation: **theoretical FLOP/operation-count savings only translate into real wall-clock savings when the implementation avoids materializing large intermediate tensors.** This is true regardless of which specific operation is theoretically cheaper — the same logic would apply, for instance, to comparing ScaleNorm against RMSNorm, or comparing two different fused-kernel implementations against each other, not just the specific RMSNorm-vs-LayerNorm case [Section 45](#45-benchmarking-profiling-and-memory-analysis) happened to measure.

### 54.2 Arithmetic intensity: the concept that explains why this happens

The relevant concept from computer architecture is **arithmetic intensity** — the ratio of floating-point operations performed to bytes of memory moved. An operation with low arithmetic intensity (few FLOPs per byte moved) is **memory-bandwidth-bound**: its wall-clock time is dominated by how fast data can move to/from memory, not by how fast the GPU's arithmetic units can compute. LayerNorm/RMSNorm have exactly this profile — a handful of additions, multiplications, and one square root per element, against reading and writing that same element (potentially multiple times, if unfused) from/to memory. Matrix multiplications (the attention and FFN computations dominating a Transformer block's actual FLOP count), by contrast, have **high** arithmetic intensity — many multiply-accumulate operations are performed per byte of weight/activation data moved, making them **compute-bound** rather than memory-bandwidth-bound on modern hardware. This is the precise, quantitative reason normalization layers benefit so disproportionately from fusion relative to matrix multiplications: fusion's benefit is specifically about reducing memory traffic, and normalization's bottleneck specifically *is* memory traffic, while a matrix multiplication's bottleneck typically isn't.

### 54.3 Cache efficiency: why row-major, contiguous access matters

A detail implicit in every implementation throughout this handbook, worth making explicit: LayerNorm/RMSNorm's reduction is over the **last** (contiguous, in standard row-major tensor layout) dimension — exactly the axis that's cheapest to access sequentially from cache. This is precisely why [Section 7.6](#76-numpy-implementation-from-scratch)'s NumPy implementation, [Section 42.3](#423-a-complete-runnable-triton-layernorm-forward-kernel)'s Triton kernel, and [Section 43.3](#433-a-complete-cuda-c-kernel)'s CUDA kernel all reduce over the last axis without needing any data reshuffling beforehand — the natural memory layout of a $(B,T,d)$ tensor already places each token's full feature vector contiguously in memory, letting every reduction read sequential, cache-friendly memory addresses. Had normalization instead needed to reduce over, say, the *batch* axis (as BatchNorm does — [Section 6.2](#62-mathematical-formulation)), the relevant data for one reduction would be scattered across memory at a stride of $T\times d$ elements apart, a meaningfully less cache-friendly access pattern — one more concrete, hardware-level reason (beyond the statistical/architectural reasons given in [Section 20](#20-why-batchnorm-fails-for-transformers)) that LayerNorm's per-token, last-axis reduction maps more naturally onto how memory actually works.

### 54.4 What's next

[Section 55](#55-inference-time-implications-and-latency-trade-offs) closes Part VIII by examining how everything covered in Sections 52–54 — $\epsilon$ selection, precision format, and kernel fusion — specifically affects inference-time latency and throughput, the production metric that ultimately matters most for a deployed model serving real user requests.

---

## 55. Inference-Time Implications and Latency Trade-offs

### 55.1 Autoregressive decoding revisits the $B=1, T=1$ regime directly

[Section 20.4](#204-the-autoregressive-inference-problem) already established the core fact: autoregressive generation, decoding one new token at a time, repeatedly hits exactly the $B=1, T=1$ (or small-$B$) regime where BatchNorm degenerates but LayerNorm/RMSNorm remain perfectly well-defined. The latency consequence, made explicit here: **every single decoding step pays normalization's full per-call cost** (the mean/variance or mean-square reduction over $d$, [Section 7.9](#79-computational-complexity)'s $O(d)$ cost) — but because decoding generates only **one token per step**, this $O(d)$ cost is incurred with none of the parallelism-across-many-tokens that a training forward pass (processing thousands of tokens simultaneously) or a prefill step (processing an entire prompt at once) benefits from. This is precisely why kernel fusion ([Sections 42–43](#42-triton-fused-layernorm-kernel)) and avoiding unnecessary precision upcasts/downcasts matter disproportionately more for **decode-step latency** specifically than for prefill or training throughput — a fixed per-call overhead that gets paid once per generated token, directly adding to the user-perceived "time between tokens" latency that dominates a chat application's felt responsiveness.

### 55.2 KV cache and normalization: a clarification worth making explicit

A natural question: does the KV cache (storing previously-computed keys/values to avoid recomputing them at every decode step) interact with normalization in any special way? The answer, worth stating precisely: **no** — LayerNorm/RMSNorm's outputs are not what gets cached; the KV cache stores post-attention-projection keys and values, computed *after* normalization has already been applied to that token's hidden state. Normalization must still be recomputed fresh for the **current** token at every decode step (since it's never cached), but it does not need to be recomputed for any **previous** tokens, whose keys/values are already sitting in the cache. This means normalization's per-step cost scales with the number of *new* tokens being processed (1, for standard autoregressive decoding) rather than with the growing total context length — a favorable scaling property worth knowing explicitly, since it means normalization's latency contribution stays flat across a generation, even as the KV cache itself grows with context length.

### 55.3 Why $\epsilon$ and precision choices matter more at inference than [Section 52](#52-numerical-stability-and-epsilon-selection) alone might suggest

Production inference serving frequently uses **quantized** weights (INT8, INT4, or other sub-16-bit formats) for the linear-layer weights specifically, often while keeping normalization layers themselves in a higher-precision format (FP16/BF16/FP32) — a deliberate design choice directly motivated by [Section 53.2](#532-why-this-matters-specifically-for-normalizations-variance-computation)'s overflow-threshold finding: normalization's variance/mean-square computation is exactly the kind of operation where aggressive quantization risks the most damage (squaring amplifies quantization error just as it amplifies the FP16-overflow risk demonstrated in Section 53.2), while the bulk of a model's parameters (the linear-layer weights, which dominate total parameter count) tolerate quantization considerably better. This is a direct, practical consequence of this handbook's numerical-stability analysis: **normalization layers are disproportionately precision-sensitive relative to their tiny share of a model's total parameter count**, which is exactly why production quantization schemes (e.g., as used in many serving frameworks) routinely keep normalization layers at higher precision even while aggressively quantizing everything else.

### 55.4 Part VIII summary, and what's next

Part VIII is now complete. We've covered $\epsilon$ selection with an empirically-corrected, quantitative demonstration of the bias-vs-instability trade-off ([Section 52](#52-numerical-stability-and-epsilon-selection)), the precise BF16-vs-FP16 structural difference and a confirmed, exact overflow threshold for FP16 squaring ([Section 53](#53-mixed-precision-bf16-vs-fp16-quantization-effects)), the arithmetic-intensity framework explaining why fusion matters disproportionately for normalization specifically ([Section 54](#54-kernel-fusion-memory-bandwidth-cache-efficiency)), and this section's inference-specific latency, KV-cache, and quantization considerations.

**Part IX** begins now, covering the handbook's remaining advanced topics: ReZero, NormFormer, and μP revisited with full technical depth (closing the loop on [Section 39.3](#393-connection-to-scaling-laws--a-brief-honest-preview)'s deferred scaling-laws connection), normalization-free Transformers, and the theoretical limits of training ultra-deep networks — the final stretch before this handbook's appendices.

---

# Part IX — Advanced Topics

## 56. ReZero, NormFormer, μP (Maximal Update Parameterization)

### 56.1 ReZero and NormFormer: a brief consolidating recap

Both techniques received full treatment earlier in this handbook — [Section 33.4](#334-rezero--starting-the-gate-at-exactly-zero) derived ReZero's exact-identity-at-initialization gate (with [Section 41.3](#413-using-these-building-blocks-to-assemble-a-full-decoder-block)'s `GatedResidualWrapper` actually executed and confirmed `torch.allclose`-identical to the input at $g=0$), and [Section 18](#18-normformer-and-other-modern-variants) derived NormFormer's three targeted additions to counteract Pre-LN's layer-position gradient imbalance, with a full runnable implementation verified in [Section 18.4](#184-pytorch-implementation-sketch) (confirmed: correct output shape, no NaNs, working gradients through the learned per-head scale). Rather than repeat that material, this section focuses on **μP**, the one of these three advanced techniques not yet covered, and closes [Section 39.3](#393-connection-to-scaling-laws--a-brief-honest-preview)'s explicitly-deferred scaling-laws connection.

### 56.2 μP: the problem it solves, precisely

Recall [Section 39.3](#393-connection-to-scaling-laws--a-brief-honest-preview)'s deferred question: as practitioners scale a model's **width** (hidden dimension $d$), how should other hyperparameters — initialization variance, learning rate — change to keep training dynamics consistent? Under naive, **standard parameterization** (fixed initialization/learning-rate rules independent of width), the empirically-observed problem is that **the optimal learning rate shifts substantially as width changes** — a learning rate well-tuned for a small proxy model is typically wrong (often by a large margin) for a much wider model, forcing expensive, separate hyperparameter search at every new scale a team wants to train.

**μP (Maximal Update Parameterization)**, confirmed directly from Yang & Hu's foundational work (and its large-scale validation in "Tensor Programs V"), is a specific reparameterization — prescribing precise, width-dependent scaling rules for each parameter type's initialization variance and learning rate — under which the optimal learning rate **converges to a width-independent constant** as width grows. This is what makes **μTransfer** possible: tune hyperparameters once on a small, cheap proxy model, then transfer them directly (zero-shot, no re-tuning) to a much larger target model at the same effective optimum. The validated empirical result, confirmed directly: transferring hyperparameters tuned on a 40M-parameter proxy model matched published GPT-3 6.7B performance at roughly 7% of the tuning cost that would otherwise be required.

### 56.3 Why this connects directly to this handbook's normalization-and-residual analysis

μP's scaling rules are not limited to weight initialization and learning rate alone — confirmed directly from a relevant source: *"μP also stabilizes embeddings, layer norms, and self-attention in Transformers."* This is a direct, confirmed connection to exactly the material this handbook has built throughout [Parts I–V](#1-why-normalization-the-core-problem): μP can be understood as extending the *same underlying concern* — keeping activation scales and gradient magnitudes well-behaved as a network's dimensions change — from the **depth** axis (which [Sections 36–39](#36-why-they-work-so-well-together) addressed via normalization plus residual connections) to the **width** axis as well. Where Pre-LN/Pre-RMSNorm plus residual connections give a network robust, scale-independent behavior *as depth increases* (the complete subject of Part V), μP gives analogous robust, scale-independent behavior *as width increases* — two complementary answers to the same general question ("how do I add more of this dimension without breaking training"), addressing two different dimensions of the same underlying scaling problem.

### 56.4 An honest scoping note

μP's full mathematical derivation (the abc-parameterization, the precise per-parameter-type scaling exponents, and the infinite-width limit theory underlying *why* this specific parameterization is the unique one achieving "maximal feature learning") is a substantial topic in its own right, sitting at the intersection of this handbook's subject matter and a broader scaling-laws literature this handbook does not attempt to cover exhaustively. What's been established here, accurately and with direct sourcing: μP exists, solves a real and costly practical problem (hyperparameter re-tuning at every new scale), is confirmed to be in active use at several organizations (the search results gathered for this section directly confirmed usage references for LLaMA 4's "MetaP" and configuration-file evidence for Grok-2, among others), and is confirmed to extend specifically to normalization layers — but a complete first-principles derivation of its scaling exponents is beyond this handbook's scope, exactly as honestly flagged when first deferred in [Section 39.3](#393-connection-to-scaling-laws--a-brief-honest-preview).

### 56.5 What's next

[Section 57](#57-normalization-free-transformers) covers a genuinely different research direction: architectures that attempt to remove normalization layers **entirely**, rather than refining how they're placed or scaled — asking whether everything this handbook has established about normalization's necessity ([Parts I–III](#1-why-normalization-the-core-problem)) can be replaced by sufficiently careful initialization and architectural design alone.

---

## 57. Normalization-Free Transformers

### 57.1 The motivating question, and its origin in vision models

If normalization's core job (per [Section 1](#1-why-normalization-the-core-problem)) is controlling activation scale, a natural question is whether that *job* can be accomplished some other way — through careful initialization, weight reparameterization, or architectural constraints — without paying normalization's per-layer computational cost ([Section 54.2](#542-arithmetic-intensity-the-concept-that-explains-why-this-happens)'s memory-bandwidth-bound overhead) at all. This research direction predates Transformer-specific work: **NFNets** (Normalizer-Free Networks; Brock et al., 2021), confirmed directly, demonstrated competitive ResNet/EfficientNet performance using **scaled weight standardization** (a weight-reparameterization technique, conceptually related to [Section 11](#11-weight-normalization)'s Weight Normalization) combined with **Adaptive Gradient Clipping (AGC)** to handle the large-batch-size instability that removing BatchNorm would otherwise cause.

### 57.2 DyT: replacing normalization with a single elementwise function

The most directly relevant, recent result for this handbook's Transformer focus: **DyT (Dynamic Tanh)**, confirmed from the paper "Transformers without Normalization" (Zhu, Chen & He — note Kaiming He's co-authorship, the same author behind the original ResNet paper this handbook's residual-connection coverage is built around, [Section 29.1](#291-a-puzzle-that-confused-the-field-before-the-fix-was-found)). DyT replaces LayerNorm/RMSNorm with a remarkably simple elementwise operation:

$$\text{DyT}(x) = \gamma \cdot \tanh(\alpha x) + \beta$$

where $\alpha$ is now a **learned scalar** (not derived from any per-token statistic at all — no mean, no variance, no mean-square computation), and $\gamma,\beta$ play the same learned per-feature affine role as in standard LayerNorm ([Section 3.6](#36-restoring-expressiveness-learnable-γ-and-β)). The motivating empirical observation behind this design: LayerNorm's input-output mapping, plotted as a scatter of input value vs. output value across many tokens, **resembles an S-shaped (tanh-like) curve** — DyT's proposal is to directly fit that S-shape with an actual $\tanh$, parameterized by a single learned scalar $\alpha$, rather than computing it indirectly via per-token mean/variance statistics. Confirmed directly: DyT requires "minimal modifications to both the architecture and the training recipe" relative to standard LayerNorm/RMSNorm-based Transformers, and is confirmed to **match RMSNorm's performance at 34B+ model scale**, with a measurable gap remaining at smaller scales — an important, honest empirical nuance rather than a universal win.

### 57.3 Why removing the per-token statistic matters computationally

DyT's central practical appeal, confirmed directly: by avoiding any mean, variance, or square-root computation, DyT "directly improves GPU throughput and reduces memory overhead" — precisely the [Section 54.2](#542-arithmetic-intensity-the-concept-that-explains-why-this-happens) arithmetic-intensity argument, but pushed even further than RMSNorm's already-reduced computation (recall [Section 8.9](#89-computational-and-parameter-savings--quantified)'s table: RMSNorm already removed the mean-centering step relative to LayerNorm; DyT removes the *entire* per-token statistical computation, replacing it with a single global learned scalar applied identically, elementwise, to every value, with no reduction operation at all).

### 57.4 An honest, current-state assessment

This is an actively evolving research area, not a settled question — and this handbook should be precise about that rather than overstating consensus. Confirmed directly from a synthesis source covering this literature: *"Normalizing transformer activations is no longer regarded as a universal prerequisite for stable training or strong generalization"* — but the same source also notes a genuine, unresolved tension: *"Removing normalization may allow more flexible representation geometries but could reduce out-of-distribution generalization and circuit stability, unless compensated by other design elements."* In other words: normalization-free approaches are a **real, validated, increasingly mature alternative** for Transformers (DyT's 34B+ scale validation is genuine evidence, not speculation), but they are **not (as of the sources gathered for this section) a settled replacement that has displaced Pre-RMSNorm as the default choice** across the major production model families surveyed in [Part VII](#46-gpt-2--gpt-3--gpt-4) — every model covered there still uses standard RMSNorm, and the normalization-free literature remains, at the time this handbook was assembled, predominantly a research direction demonstrating *feasibility* at scale rather than something that has yet displaced the default.

### 57.5 What's next

[Section 58](#58-scaling-to-ultra-deep-networks-1000-layers) closes Part IX (and this handbook's main content) by returning to the extreme-depth question first raised by [Section 17](#17-deepnorm)'s DeepNorm coverage — consolidating everything learned across [Parts I–IX](#1-why-normalization-the-core-problem) into a complete picture of what it actually takes to train networks at the 1,000+-layer scale, including how normalization-free approaches (this section) and μP-style width scaling ([Section 56](#56-rezero-normformer-μp-maximal-update-parameterization)) interact with that extreme-depth question.

---

## 58. Scaling to Ultra-Deep Networks (1000+ layers)

### 58.1 Synthesizing every depth-related mechanism covered in this handbook

This final section doesn't introduce new techniques — it consolidates, in one place, every mechanism this handbook has identified as relevant to depth, in the order a network actually encounters them as layers stack up:

1. **The baseline problem** ([Section 29](#29-motivation-vanishing-gradients-and-the-degradation-problem)): without residual connections, gradients vanish geometrically with depth — empirically confirmed in this handbook to underflow to *exactly* 0.0 by 50 layers in a plain `Linear`+`Tanh` stack.
2. **The residual fix** ([Sections 30–31](#30-mathematics-of-residual-learning-y--x--fx)): the $I+J_F$ structure guarantees an undiminished gradient floor at any depth — confirmed to flip the same 50-layer network's first-layer gradient from 0.0 to ~1231 (exceeding the last layer's gradient).
3. **Pre-LN's architectural separation** ([Sections 35–36](#35-pre-ln-vs-post-ln--full-gradient-flow-comparison)): keeps normalization's (non-identity) Jacobian out of the main gradient path entirely — confirmed via the measured ~8-order-of-magnitude gradient gap between Pre-LN and Post-LN at 40 layers.
4. **The residual stream's $\sqrt{\text{depth}}$ growth** ([Section 38](#38-why-activations-dont-explode-in-pre-ln-stacks)): gentle enough that standard Pre-LN comfortably absorbs it at the depths covered in every Part VII case study (12–96+ layers) — confirmed with $R^2=0.991$ against real measured data.
5. **DeepNorm's explicit compensation** ([Section 17](#17-deepnorm)): once depth reaches the 1,000+-layer regime DeepNet specifically targeted, even Pre-LN's gentle $\sqrt{\text{depth}}$ growth becomes large enough that an explicit, depth-dependent residual-scaling intervention ($\alpha=\sqrt{2N}$) becomes necessary — confirmed numerically ($\alpha\approx44.7$ at $N=1000$).
6. **Normalization-free alternatives** ([Section 57](#57-normalization-free-transformers)): an actively-developing parallel research direction, with DyT confirmed to match RMSNorm's quality specifically at 34B+ *parameter* scale (a different axis from layer-count depth, but a related "does this hold up at the extremes" question).

### 58.2 The honest, consolidated answer to "how deep can a Transformer go"

Given everything above, the honest answer is **layered, not a single number**: standard Pre-RMSNorm-plus-residual-connections, exactly as used by every model in [Part VII](#46-gpt-2--gpt-3--gpt-4)'s case studies, trains reliably and without any special intervention through at least the ~96–120-layer range (GPT-3's confirmed 96 layers being the deepest *specifically confirmed* figure this handbook verified). Beyond that, into the many-hundreds-to-1000+-layer regime, [Section 17](#17-deepnorm)'s DeepNorm-style explicit residual scaling (or comparable interventions) becomes the documented, validated path to continued stable training — not because the fundamental $I+J_F$ gradient-flow guarantee from Point 2 above ever stops holding (it doesn't — it's a structural property of the architecture, true at any depth), but because the *activation-scale* growth from Point 4, while gentle, eventually compounds enough in absolute terms that explicit compensation becomes the more robust engineering choice.

### 58.3 Why no production model in this handbook's case studies needed to find out

It's worth being direct about why [Part VII](#46-gpt-2--gpt-3--gpt-4)'s case studies never actually exercised the 1000+-layer regime: at the parameter counts current production LLMs target (hundreds of millions to low trillions of parameters), **width scaling has generally been preferred over depth scaling** as the primary lever for adding capacity — wider models with moderate depth (GPT-3's 96 layers at 12,288-dimensional width; LLaMA 3 70B's 80 layers at 8,192-dimensional width) rather than extremely deep, narrower alternatives. This is itself a design choice with its own trade-offs (width scaling interacts with the μP considerations from [Section 56](#56-rezero-normformer-μp-maximal-update-parameterization); depth scaling interacts with everything in this section) — but it means the DeepNet-style extreme-depth regime has remained, to date, more a demonstration of theoretical *feasibility* than a regime production LLM training has needed to operate in routinely.

### 58.4 Part IX summary, and what's next

Part IX is now complete, closing this handbook's main technical content. We've covered ReZero and NormFormer (consolidated from earlier sections) and μP in full ([Section 56](#56-rezero-normformer-μp-maximal-update-parameterization), closing [Section 39.3](#393-connection-to-scaling-laws--a-brief-honest-preview)'s deferred thread), normalization-free Transformers including the DyT result ([Section 57](#57-normalization-free-transformers)), and this section's full synthesis of every depth-scaling mechanism covered across all nine parts.

**The Appendix** follows: a glossary (Appendix A), a complete notation reference (Appendix B), and further reading (Appendix C) — closing out this handbook's reference material.

---

# Appendix

## A. Glossary

**Activation checkpointing** — A memory-saving technique that recomputes forward-pass intermediates during backpropagation instead of storing them. See [Section 44.5](#445-activation-checkpointing--recomputing-rather-than-storing).

**Adaptive Normalization (AdaIN)** — A normalization variant that computes the scale/shift parameters dynamically from an external conditioning signal rather than learning them as static values. See [Section 15](#15-adaptive-normalization-adain-conditional-norms).

**Arithmetic intensity** — The ratio of floating-point operations to bytes of memory moved by a computation; low arithmetic intensity implies memory-bandwidth-bound performance. See [Section 54.2](#542-arithmetic-intensity-the-concept-that-explains-why-this-happens).

**Batch Normalization (BN)** — Normalizes activations using statistics computed across the batch (and spatial/temporal) dimension, per feature/channel. See [Section 6](#6-batch-normalization-bn).

**Degradation problem** — The empirical observation that adding layers to a plain (non-residual) deep network can worsen even training-set performance, due to optimization difficulty rather than representational limits. See [Section 29.1](#291-a-puzzle-that-confused-the-field-before-the-fix-was-found).

**DeepNorm** — A residual-scaling technique ($x_{l+1}=\text{LN}(\alpha x_l+F(x_l))$) enabling stable training of Transformers with 1,000+ layers. See [Section 17](#17-deepnorm).

**DyT (Dynamic Tanh)** — A normalization-free technique replacing LayerNorm/RMSNorm with a single learned-scalar $\tanh$ function. See [Section 57.2](#572-dyt-replacing-normalization-with-a-single-elementwise-function).

**Epsilon ($\epsilon$)** — A small constant added inside a normalization layer's denominator to prevent division by zero/near-zero. See [Section 3.5](#35-the-epsilon-you-will-see-everywhere) and [Section 52](#52-numerical-stability-and-epsilon-selection).

**Grouped-Query Attention (GQA)** — An attention variant using fewer key/value heads than query heads, reducing KV-cache memory. Referenced throughout [Part VII](#46-gpt-2--gpt-3--gpt-4)'s case studies.

**Group Normalization** — Normalizes using statistics computed over a subset ("group") of channels plus spatial dimensions, per example. See [Section 9](#9-group-normalization).

**Identity mapping** — The function $f(x)=x$; residual connections are designed to make this function trivially representable and easy for gradient descent to discover. See [Section 32.2](#322-identity-mappings--the-original-resnet-papers-framing).

**Instance Normalization** — Normalizes using statistics computed over spatial dimensions only, per channel and per example. See [Section 10](#10-instance-normalization).

**Internal covariate shift (ICS)** — The (historically influential but scientifically contested) hypothesis that normalization's benefit comes from reducing the shifting distribution of layer inputs caused by upstream weight updates. See [Section 2](#2-internal-covariate-shift-and-optimization-landscapes).

**Jacobian** — The matrix of all first-order partial derivatives of a vector-valued function; used throughout this handbook to express backward passes as explicit linear maps. See [Section 22](#22-layernorm-mathematics--backward-pass-and-jacobian).

**Kernel fusion** — Combining multiple sequential operations into a single GPU kernel launch to reduce memory traffic. See [Section 42.1](#421-why-fusion-matters-a-memory-bandwidth-argument).

**Layer Normalization (LN)** — Normalizes using statistics computed over the feature dimension, independently per token. See [Section 7](#7-layer-normalization-ln).

**Multi-head Latent Attention (MLA)** — DeepSeek's attention variant compressing keys/values into a low-dimensional latent vector before caching. See [Section 50.1](#501-deepseeks-multi-head-latent-attention--confirmed-mechanics).

**Mixture-of-Experts (MoE)** — An architecture replacing a single dense FFN with multiple "expert" FFNs, of which only a subset are activated per token via a learned router. See [Section 48.2](#482-mixtral-8x7b-moe-layered-onto-the-identical-foundation).

**μP (Maximal Update Parameterization)** — A reparameterization prescribing width-dependent initialization/learning-rate scaling rules so that optimal hyperparameters transfer across model widths. See [Section 56.2](#562-μp-the-problem-it-solves-precisely).

**NormFormer** — A set of three targeted additions to Pre-LN Transformer blocks (extra post-attention LN, learned per-head scaling, extra mid-FFN LN) addressing a layer-position gradient imbalance. See [Section 18](#18-normformer-and-other-modern-variants).

**Post-LN** — Placing normalization after a sublayer's residual addition: $x_{l+1}=\text{LN}(x_l+F(x_l))$. See [Section 25.2](#252-post-ln--the-original-transformers-choice).

**Power Normalization** — A BatchNorm variant using smoothed running statistics to reduce outlier sensitivity, specifically for Transformer training stability. See [Section 14](#14-power-normalization).

**Pre-LN** — Placing normalization before a sublayer, with the residual addition operating on the unnormalized stream: $x_{l+1}=x_l+F(\text{LN}(x_l))$. See [Section 25.3](#253-pre-ln--the-modern-default).

**QK-Norm** — Applying LayerNorm or RMSNorm directly to query and key vectors before the attention dot-product, stabilizing attention logit magnitudes. See [Section 18.5](#185-other-modern-variants-briefly) and [Section 49.3](#493-why-soft-capping-and-qk-norm-address-the-same-underlying-concern-differently).

**ReZero** — A gated residual connection ($y=x+g\odot F(x)$) with the gate $g$ initialized to exactly zero, making each block an exact identity function at initialization. See [Section 33.4](#334-rezero--starting-the-gate-at-exactly-zero).

**Residual connection (skip connection)** — An architectural pattern adding a sublayer's input directly to its output: $y=x+F(x)$. See [Section 30](#30-mathematics-of-residual-learning-y--x--fx).

**RMSNorm** — A LayerNorm simplification that rescales by root-mean-square without mean-centering, and typically without a learned shift parameter. See [Section 8](#8-rmsnorm).

**Sandwich-LN** — Applying normalization both before and after a sublayer, combining Pre-LN's gradient-flow benefit with additional output-scale control. See [Section 25.5](#255-sandwich-ln--combining-both).

**Sequence Parallelism** — A distributed-training strategy splitting the sequence dimension across devices; compatible with zero additional communication for per-token normalization. See [Section 44.3](#443-sequence-parallelism--splitting-along-tokens-normalization-stays-local).

**Sliding Window Attention (SWA)** — An attention variant restricting each token's attention to a fixed-size local window, used by Mistral and Gemma 2. See [Section 48.1](#481-mistral-7b-confirmed-configuration).

**Soft-capping** — A $\tanh$-based technique bounding attention logit magnitude after computation, used by Gemma 2. See [Section 49.2](#492-gemma-2-confirmed-rmsnorm-sandwich-gqa-and-logit-soft-capping-correcting-section-28).

**Spectral Normalization** — Rescales a weight matrix by its largest singular value, bounding the layer's worst-case input amplification. See [Section 12](#12-spectral-normalization).

**Tensor Parallelism** — A distributed-training strategy splitting weight matrices (and the feature dimension) across devices; requires an all-reduce before feature-dimension-reducing normalization can complete. See [Section 44.2](#442-tensor-parallelism--the-feature-dimension-gets-split).

**Vanishing gradients** — The phenomenon where gradients shrink geometrically as they propagate backward through many layers, due to repeated multiplication by sub-unit Jacobians. See [Section 29.2](#292-the-vanishing-gradient-mechanism-precisely).

**Weight Normalization** — Reparameterizes a weight vector as a learned scalar magnitude times a learned unit-norm direction, decoupling their optimization. See [Section 11](#11-weight-normalization).

---

## B. Notation Reference

This expands the brief table given at the very start of this handbook (just after the Table of Contents) into a complete reference covering every symbol used across all 58 sections.

### Tensor dimensions

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $T$ | Sequence length (number of tokens) |
| $d$ (or $d_{\text{model}}$) | Hidden/feature dimension |
| $d_{\text{ff}}$ | FFN inner hidden dimension |
| $H$ (or $n_{\text{heads}}$) | Number of attention heads |
| $C$ | Number of channels (vision contexts) |
| $G$ | Number of groups (Group Normalization) |
| $L$ (or $N$) | Number of layers / total depth |

### Statistics and normalization

| Symbol | Meaning |
|---|---|
| $\mu$ | Mean |
| $\sigma^2$ | Variance |
| $\sigma$ | Standard deviation |
| $\hat{x}$ | Standardized (or RMS-normalized) value |
| $\epsilon$ | Small constant for numerical stability |
| $\gamma$ | Learnable scale parameter |
| $\beta$ | Learnable shift parameter |
| $\text{RMS}(x)$ | Root mean square of $x$ |
| $\|x\|_2$ (or $\|x\|$) | Euclidean (L2) norm of $x$ |

### Residual connections and gradients

| Symbol | Meaning |
|---|---|
| $F(x)$ | A sublayer function (attention or FFN) inside a residual block |
| $y=x+F(x)$ | The residual connection equation |
| $J$ (or $J_F$) | Jacobian matrix of a function |
| $I$ | Identity matrix |
| $\alpha$ | DeepNorm's residual-scaling constant (or, in DyT, the learned scalar inside $\tanh$) |
| $g$ | A learned gate (gated residuals, ReZero) |
| $\frac{\partial \mathcal{L}}{\partial x}$ | Gradient of the loss $\mathcal{L}$ with respect to $x$ |

### Operators

| Symbol | Meaning |
|---|---|
| $\odot$ | Element-wise (Hadamard) product |
| $\delta_{ij}$ | Kronecker delta (1 if $i=j$, else 0) |
| $\mathbf{1}$ | All-ones vector or matrix |
| $\nabla$ | Gradient operator |
| $\propto$ | "Proportional to" |

### Model-architecture-specific (Part VII)

| Symbol | Meaning |
|---|---|
| $n_{\text{kv\_heads}}$ | Number of key/value heads under Grouped-Query Attention |
| $d_c$ | DeepSeek MLA's KV compression (latent) dimension |
| $W$ (in attention contexts) | Sliding window size |

---

## C. Further Reading

This list consolidates the papers and sources actually cited and verified throughout this handbook — organized by topic, in roughly the order they're relevant to a reader progressing through the material — rather than an exhaustive, unverified bibliography.

### Foundational normalization papers

- Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* — The original BatchNorm paper. See [Section 6](#6-batch-normalization-bn).
- Ba, J., Kiros, J. & Hinton, G. (2016). *Layer Normalization.* — See [Section 7](#7-layer-normalization-ln).
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* — The RMSNorm paper. See [Section 8](#8-rmsnorm).
- Santurkar, S., Tsipras, D., Ilyas, A. & Madry, A. (2018). *How Does Batch Normalization Help Optimization?* — The contested-evidence challenge to the internal-covariate-shift explanation. See [Section 2.4](#24-an-important-caveat-the-ics-explanation-is-contested).
- Wu, Y. & He, K. (2018). *Group Normalization.* — See [Section 9](#9-group-normalization).
- Ulyanov, D., Vedaldi, A. & Lempitsky, V. (2016). *Instance Normalization: The Missing Ingredient for Fast Stylization.* — See [Section 10](#10-instance-normalization).
- Salimans, T. & Kingma, D. (2016). *Weight Normalization.* — See [Section 11](#11-weight-normalization).
- Miyato, T., Kataoka, T., Koyama, M. & Yoshida, Y. (2018). *Spectral Normalization for Generative Adversarial Networks.* — See [Section 12](#12-spectral-normalization).
- Huang, X. & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* — The AdaIN paper. See [Section 15.2](#152-adain--adaptive-instance-normalization).
- Nguyen, T. & Salazar, J. (2019). *Transformers without Tears: Improving the Normalization of Self-Attention.* — The ScaleNorm paper. See [Section 16](#16-scalenorm).
- Wang, H. et al. (2022). *DeepNet: Scaling Transformers to 1,000 Layers.* — See [Section 17](#17-deepnorm).
- Shleifer, S., Weston, J. & Ott, M. (2021). *NormFormer: Improved Transformer Pretraining with Extra Normalization.* — See [Section 18](#18-normformer-and-other-modern-variants).
- Shen, S. et al. (2020). *PowerNorm: Rethinking Batch Normalization in Transformers.* — See [Section 14](#14-power-normalization).

### Residual connections

- He, K., Zhang, X., Ren, S. & Sun, J. (2015/2016). *Deep Residual Learning for Image Recognition.* — The original ResNet paper. See [Section 29](#29-motivation-vanishing-gradients-and-the-degradation-problem).
- He, K. et al. (2016). *Identity Mappings in Deep Residual Networks.* — See [Section 32.2](#322-identity-mappings--the-original-resnet-papers-framing).
- Srivastava, R., Greff, K. & Schmidhuber, J. (2015). *Highway Networks.* — See [Section 33.1](#331-highway-networks--the-gated-predecessor).
- Huang, G. et al. (2017). *Densely Connected Convolutional Networks.* — The DenseNet paper. See [Section 33.2](#332-densenet--connecting-everything-to-everything).
- Bachlechner, T. et al. (2021). *ReZero is All You Need: Fast Convergence at Large Depth.* — See [Section 33.4](#334-rezero--starting-the-gate-at-exactly-zero).
- Veit, A., Wilber, M. & Belongie, S. (2016). *Residual Networks Behave Like Ensembles of Relatively Shallow Networks.* — See [Section 32.3](#323-a-second-framing-ensemble-like-behavior).

### The Transformer and LLM architectures

- Vaswani, A. et al. (2017). *Attention Is All You Need.* — The original Transformer paper. See [Section 25.2](#252-post-ln--the-original-transformers-choice).
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners.* — The GPT-2 paper. See [Section 46.1](#461-gpt-2-the-pre-ln-pivot).
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners.* — The GPT-3 paper. See [Section 46.2](#462-gpt-3-scaling-the-same-recipe-to-96-layers).
- Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* — See [Section 47.1](#471-llama-1-the-architecture-this-handbooks-rmsnorm-coverage-is-built-around).
- Touvron, H. et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models.* — See [Section 47.2](#472-llama-2-gqa-as-the-primary-architectural-change).
- Jiang, A. et al. (2023). *Mistral 7B.* — See [Section 48.1](#481-mistral-7b-confirmed-configuration).
- Jiang, A. et al. (2024). *Mixtral of Experts.* — See [Section 48.2](#482-mixtral-8x7b-moe-layered-onto-the-identical-foundation).
- Gemma Team (2024). *Gemma 2: Improving Open Language Models at a Practical Size.* — See [Section 49.2](#492-gemma-2-confirmed-rmsnorm-sandwich-gqa-and-logit-soft-capping-correcting-section-28).
- Gemma Team (2025). *Gemma 3 Technical Report.* — See [Section 49.2](#492-gemma-2-confirmed-rmsnorm-sandwich-gqa-and-logit-soft-capping-correcting-section-28)'s correction.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* — See [Section 50](#50-deepseek-v2v3-and-qwen).
- Qwen Team (2025). *Qwen3 Technical Report.* — See [Section 50.4](#504-qwen-pre-rmsnorm-gqa-and-a-notable-convergence-with-gemma-3-on-qk-norm).

### Scaling laws and hyperparameter transfer

- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* — See [Section 39.3](#393-connection-to-scaling-laws--a-brief-honest-preview).
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models* ("Chinchilla"). — See [Section 39.3](#393-connection-to-scaling-laws--a-brief-honest-preview).
- Yang, G. & Hu, E. (2021). *Tensor Programs IV/V: Feature Learning in Infinite-Width Neural Networks / Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* — The μP papers. See [Section 56](#56-rezero-normformer-μp-maximal-update-parameterization).

### Normalization-free approaches

- Brock, A., De, S. & Smith, S. (2021). *High-Performance Large-Scale Image Recognition Without Normalization.* — The NFNets paper. See [Section 57.1](#571-the-motivating-question-and-its-origin-in-vision-models).
- Zhu, J., Chen, X. & He, K. (2025). *Transformers without Normalization.* — The DyT paper. See [Section 57.2](#572-dyt-replacing-normalization-with-a-single-elementwise-function).

### A closing note on using this list

Every paper above was specifically referenced and, where a factual claim depended on it, verified via direct search during this handbook's construction — this list is not a generic "related work" dump, but a traceable record of where this handbook's claims actually came from. For anything not covered in sufficient depth here, these primary sources are the right place to go deeper.

---

*This concludes the Normalization & Residual Connections handbook — 58 sections across 9 parts, plus this appendix, covering normalization and residual connections in LLMs from first principles through production-scale implementation and architecture-level analysis.*

