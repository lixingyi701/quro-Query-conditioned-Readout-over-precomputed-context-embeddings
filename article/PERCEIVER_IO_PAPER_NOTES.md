# Perceiver IO 论文阅读笔记（聚焦 Query-as-Q 与 QURO）

> **论文**：Andrew Jaegle et al., *Perceiver IO: A General Architecture for Structured Inputs & Outputs*, ICLR 2022  
> **arXiv**：2107.14795v3  
> **本笔记的阅读目标**：不是全面复述论文所有实验，而是回答一个与 QURO 直接相关的问题：
>
> **Perceiver IO 如何把“我要什么输出”的信息构造成 decoder 的 query，并让它作为 cross-attention 的 Q 去读取 latent？**
>
> 这篇论文对 QURO 最重要的不是光流、StarCraft 或 ImageNet 的具体指标，而是：
>
> 1. latent 可以与原始输入/输出结构解耦；
> 2. decoder 用一个独立的 **Output Query Array** 去读取 latent；
> 3. output query 可以由 position、task、modality、input feature 等信息组合得到；
> 4. **query 数量决定输出长度**；
> 5. 论文并没有解决“自然语言 information need 如何变成一组 query vectors”——这恰恰是 QURO 后续需要继续设计的地方。

---

## 1. 一句话概括 Perceiver IO

Perceiver IO 把网络统一成一个 **Read → Process → Write** 结构：

```text
Input array
   │
   ▼
[Encode: cross-attention]
   │
   ▼
Latent array
   │
   ▼
[Process: latent self-attention × L]
   │
   ▼
Processed latent
   │
   ▼
[Decode: output queries cross-attend to latent]
   │
   ▼
Output array
```

数学上：

\[
x \in \mathbb{R}^{M\times C}
\rightarrow
z \in \mathbb{R}^{N\times D}
\rightarrow
y \in \mathbb{R}^{O\times E}
\]

其中：

- \(M\)：输入元素数；
- \(C\)：输入特征维度；
- \(N\)：latent 数量；
- \(D\)：latent 维度；
- \(O\)：输出元素数；
- \(E\)：输出特征维度。

真正关键的是：**大量计算发生在固定大小的 latent space 中，而不是一直在原始输入空间里做 self-attention。**

---

# 2. Section 3.1：Encoding, Processing, and Decoding

## 2.1 Encode：Input → Latent

Perceiver IO 首先用 cross-attention 把大输入数组映射到较小 latent array：

\[
x\in\mathbb{R}^{M\times C}
\rightarrow
z\in\mathbb{R}^{N\times D}
\]

在 encoder cross-attention 中：

```text
Q  ← latent array
K,V← input array
```

也就是 latent slots 主动去“读取”输入。

如果 \(N\ll M\)，那么原始输入就被投影到了一个较短的 latent representation 中。

对 QURO 来说，这一部分最值得保留的思想是：

> **文档/context 可以先被编码为一个 query-independent latent memory。**

如果 encoder 完全不依赖用户 query，那么这部分表示就天然具有 **precompute/cache** 的潜力。

---

## 2.2 Process：只在 Latent Space 中做深层计算

随后模型在 latent 上进行多层 self-attention：

```text
Q,K,V ← latent array
```

即：

\[
z\rightarrow z'\rightarrow z''\rightarrow\cdots
\]

这里最重要的不是具体层数，而是：

> **网络深度与原始输入长度解耦。**

Transformer 如果始终对 \(M\) 个输入元素做 self-attention，会出现 \(O(M^2)\) 级别的代价；Perceiver 则把深层处理限制在 \(N\) 个 latent 上。

---

## 2.3 Decode：Output Query → Cross-Attention → Output

这是本文对 QURO 最关键的部分。

Decoder 同样是一个 cross-attention，但方向与 encoder 不同：

```text
Q   ← Output Query Array
K,V ← Processed Latent Array
```

因此：

\[
X_Q \rightarrow Q
\]

\[
Z \rightarrow K,V
\]

然后：

\[
\operatorname{Attention}(Q,K,V)
\]

输出的 index dimension 与 **query input 的 index dimension 相同**。

也就是说：

\[
\boxed{
\#\text{Output Queries}=\#\text{Outputs}
}
\]

这是后面固定输出 \(P\) 个 compressed soft tokens 时最重要的理论基础之一。

---

# 3. Output Query 和 Attention 里的 Q 不是同一个概念

这是阅读这篇论文时最容易混淆的地方。

论文中的 **Output Query Array** 是 attention 的 **query input**，记作：

\[
X_Q
\]

真正进入 scaled dot-product attention 的 Q，还需要经过一个线性投影：

\[
Q=f_Q(X_Q)
\]

而 latent 作为 key-value input：

\[
K=f_K(X_{KV}),\qquad V=f_V(X_{KV})
\]

其中 decoder 中通常：

\[
X_{KV}=Z
\]

完整过程可以写成：

\[
Q=f_Q(X_Q),\qquad
K=f_K(Z),\qquad
V=f_V(Z)
\]

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{F}}\right)
\]

\[
Y=A V
\]

因此：

```text
Output Query Array X_Q
        │
        ▼
      W_Q
        │
        ▼
Attention Query Q
```

后面我们说 QURO 的“query-as-Q”时，最好始终区分这两层：

1. **自然语言 query / information need**；
2. **由它构造出来的 decoder query input \(X_Q\)**；
3. **线性投影后的 attention Q**。

---

# 4. Section 3.2：Decoding the Latent Representation with a Query Array

这一节是本文最核心的部分。

作者的问题是：

> 已经有了 \(N\times D\) 的 latent representation，怎样生成一个任意结构的 \(O\times E\) 输出？

答案是：

> 构造一个 index dimension 为 \(O\) 的 query array，每个 query vector 描述一个 desired output 的语义。

因此：

\[
X_Q=
\begin{bmatrix}
q_1\\
q_2\\
\vdots\\
q_O
\end{bmatrix}
\]

其中每个：

\[
q_i
=
\text{information relevant for desired output }i
\]

这就是 Perceiver IO 的核心抽象。

---

# 5. Query 到底怎么构造？——论文给出的统一原则

论文原文最值得记住的一句话是：

> Queries are constructed by **concatenating or adding** a set of vectors so that each query vector contains the information relevant to one desired output.

也就是：

\[
q_i=f(\text{output-specific information})
\]

这个 \(f\) 并没有被限制为某一种固定形式。

Query 可以是：

- hand-designed feature；
- learned embedding；
- input 的简单函数；
- position encoding；
- task embedding；
- modality embedding；
- 某个位置对应的 input feature；
- 上述多个向量的 concat / add。

因此 Perceiver IO 的关键思想不是“放一个 learned token”，而是：

\[
\boxed{
\text{Output Query 是“我要什么输出”的语义描述器}
}
\]

---

# 6. Figure 3：六类 Output Query 的真正含义

Figure 3 不需要记住所有具体任务细节，真正要看的是作者如何根据“输出语义”选择 query feature。

---

## 6.1 Masked Language Modeling：Position as Query

当不同输出点之间主要区别只是 **位置** 时，query 只需要告诉模型：

> “我要哪个位置的输出？”

因此可以使用：

\[
q_i=PE(i)
\]

例如要预测 mask 位于位置 37：

\[
q_{37}=PE(37)
\]

如果有多个 masked positions：

\[
X_Q=[PE(i_1),PE(i_2),\ldots]
\]

论文语言实验中也明确使用了 **learnable position-dependent vectors** 去 query 最终 latent。

关键理解：

> 这里的 query 不是自然语言“问题”，而是“我要第几个位置”。

---

## 6.2 Classification：Single Learned Query

如果任务只需要一个固定语义的输出，例如分类：

\[
O=1
\]

那么一个 learned query 就足够：

\[
X_Q=[q_{cls}]
\]

可以把它理解为：

> “请从 latent 中读出完成 classification 所需要的信息。”

因为这个“问题”对所有样本都是一样的，所以 query 可以跨样本复用并直接学习。

---

## 6.3 Multi-task Classification：Task Identity as Query

多任务场景下，区别不再是位置，而是：

\[
\text{task identity}
\]

因此可以学习：

\[
q_{task_1},q_{task_2},\ldots,q_{task_T}
\]

例如不同 GLUE task 分别有自己的 task query。

这里有一个对 QURO 很重要的思想升级：

> query 不一定表示“位置”，它也可以表示一种 **semantic role / task identity**。

也就是说，Output Query 本质是在告诉 latent：

> “当前我要从你这里读出哪一种语义的信息？”

---

## 6.4 StarCraft II：Entity / Input Feature as Query

在 StarCraft II 中，每一个输出对应具体 entity/unit，因此可以直接让 query 携带 unit feature。

抽象成：

\[
q_i=g(x_i)
\]

它证明了：

> **Output Query 不一定是静态 learned parameter，也可以依赖当前输入样本。**

对于 QURO，这比具体游戏任务本身更重要。

---

## 6.5 Optical Flow：Position + Input Feature

光流任务中，仅仅告诉模型“我要位置 \((x,y)\)”还不够，因此作者把：

- 位置特征；
- 该位置对应的 input feature

组合进一个 query：

\[
q_{x,y}
=
[PE(x,y);F(x,y)]
\]

这个例子说明：

> Query 可以是一个**复合结构**，多个信息源可以 concat/add 到一个 query vector 中。

具体光流结果本身不是我们关注的重点。

---

## 6.6 Multimodal Autoencoding：Position + Modality

当输出来自不同模态时，仅有 position 不够，因为同一个 index 在不同 modality 中代表的语义不同。

因此：

\[
q_i=[PE(i);E_{modality}]
\]

例如：

\[
q^{video}_{x,y,t}=[PE(x,y,t);E_{video}]
\]

\[
q^{audio}_{t}=[PE(t);E_{audio}]
\]

而 label 则可以只使用：

\[
q_{label}=E_{label}
\]

这里最重要的不是视频/音频，而是：

> **不同 semantic factors 可以共同构成一个 query。**

---

# 7. Figure 3 可以压缩成一个统一公式

所有例子都可以写成：

\[
\boxed{
q_i=f(\text{features specifying desired output }i)
}
\]

具体 feature 根据任务不同而变化：

| 场景 | Output Query 携带的信息 |
|---|---|
| MLM | position |
| Classification | learned task query |
| Multi-task | task identity |
| Entity output | entity/input feature |
| Dense spatial output | position + input feature |
| Multimodal | position + modality |

因此 Figure 3 真正想表达的不是“有六种固定 query 模板”，而是：

> **Output Query 的构造应该由“希望这个输出代表什么”决定。**

---

# 8. 为什么这种 Decoder 设计重要？

## 8.1 Query 数量直接决定输出数量

如果：

\[
X_Q\in\mathbb{R}^{O\times d}
\]

那么 decoder 输出就是：

\[
Y\in\mathbb{R}^{O\times E}
\]

所以：

\[
O=\#\text{queries}
\]

这是未来把输出长度解释为 compression budget 的直接理论依据：

\[
P\text{ readout queries}
\rightarrow
P\text{ compressed representations}
\]

---

## 8.2 每个 Output Point 可以独立解码

每个输出：

\[
y_i=f(q_i,Z)
\]

只依赖：

- 自己的 query；
- 同一份 latent array。

因此所有 output points 可以并行计算，也可以训练时只采样部分 query。

这意味着 decoder 并不是 autoregressive 的。

---

## 8.3 Latent 不需要显式保存原始空间结构

Perceiver IO 的 latent 不需要保持：

- 原文本 token 位置；
- 图像二维坐标；
- 各模态原来的数组结构。

想读出什么信息，由 decoder query 指定。

这形成了一个非常重要的抽象：

```text
latent = shared information memory
query  = readout specification
```

---

# 9. 与 QURO 的核心映射

Perceiver IO 给 QURO 的价值，不是让我们直接照搬它的某个下游实验，而是给出一个非常干净的 **precompute → query-conditioned readout** 骨架。

可以映射成：

```text
Long Context / Document
        │
        ▼
Query-independent Encoder
        │
        ▼
Cached Latents Z
        │
        │  natural-language query q
        │            │
        │            ▼
        │      Query Constructor
        │            │
        │            ▼
        │      X_Q ∈ R^(P×d)
        │            │
        └────────────┼─────────────┐
                     ▼             │
        CrossAttention(Q=X_Q, K,V=Z)
                     │
                     ▼
               E ∈ R^(P×d)
                     │
                     ▼
              Projector / LLM
```

这里：

\[
Z=\text{query-independent cached context representation}
\]

而：

\[
X_Q=f(q)
\]

才是 query-conditioned 部分。

---

# 10. Perceiver IO 已经解决了什么？

对 QURO 来说，它已经回答了三个核心问题。

## 10.1 如何让 query 去读取 latent？

答案非常明确：

\[
Q\leftarrow X_Q,\qquad K,V\leftarrow Z
\]

通过 decoder cross-attention：

\[
E=\operatorname{CrossAttention}(X_Q,Z)
\]

---

## 10.2 Output Query 能不能带结构化信息？

可以。

论文明确展示了：

- learned embedding；
- position；
- task id；
- modality；
- input feature；
- 以及它们的 concat/add。

因此 \(X_Q\) 完全可以是一个经过设计的 structured representation。

---

## 10.3 Output Query 数量能不能控制输出长度？

可以。

\[
P\text{ queries}\rightarrow P\text{ outputs}
\]

这正好支持固定 soft-token budget：

\[
P\in\{4,8,16,32,\ldots\}
\]

---

# 11. Perceiver IO 没有解决什么？——真正留给 QURO 的问题

这是本次阅读最重要的结论。

Perceiver IO 中的 “query” **并不是任意自然语言用户问题**。

它主要来自：

- position；
- task identity；
- modality identity；
- input/entity feature；
- learned vectors。

论文没有系统回答：

\[
\boxed{
\text{Natural-language information need}
\rightarrow
P\text{ decoder query vectors}
}
\]

例如：

```text
“What caused the revenue decline in 2024?”
```

怎样转成：

\[
X_Q=
[q_1,q_2,\ldots,q_P]
\]

并使这 \(P\) 个 vectors 真正承担不同的信息读取角色？

Perceiver IO 没有给出完整答案。

因此不能把论文的贡献夸大成：

> “Perceiver IO 已经解决了自然语言 query-as-Q。”

更准确的说法是：

> **Perceiver IO 提供了 query-conditioned readout 的机制与设计原则，但没有解决 natural-language query 到 structured readout queries 的表示学习问题。**

---

# 12. 对 QURO 最值得继续探索的 Query Constructor

从 Perceiver IO 出发，下一步真正需要研究的是：

\[
H_q=E_q(q)
\]

如何得到：

\[
X_Q\in\mathbb{R}^{P\times d}
\]

下面这些应该作为后续方法设计 / ablation，而不是混为 Perceiver IO 原论文的内容。

---

## 12.1 Baseline A：Single pooled query

\[
h_q=\operatorname{Pool}(H_q)
\]

\[
X_Q=h_qW
\]

只生成一个 query。

优点：简单。  
问题：信息瓶颈非常强，不适合需要多个 compressed tokens 的设置。

---

## 12.2 Baseline B：Natural-language tokens directly as queries

\[
X_Q=H_qW
\]

于是 query token 数量等于自然语言 query 长度：

\[
P=L_q
\]

优点：保留 token-level semantics。  
问题：输出 budget 与自然语言 query 长度绑定，不符合固定压缩预算目标。

---

## 12.3 Candidate C：Learned slots + query conditioning

先定义：

\[
S=[s_1,\ldots,s_P]
\]

然后让自然语言 query 调制这些 slots：

\[
x_i=s_i+g_i(H_q)
\]

于是：

\[
X_Q=[x_1,\ldots,x_P]
\]

直觉：

- slot identity 决定不同 readout role；
- natural-language query 决定当前要读取的内容。

---

## 12.4 Candidate D：Query Perceiver / semantic slots

先用 \(P\) 个 learned slots 去读取自然语言 query：

\[
X_Q=\operatorname{CrossAttention}(S,H_q)
\]

得到：

\[
X_Q\in\mathbb{R}^{P\times d}
\]

然后再用这些 query vectors 读取 cached document latent：

\[
E=\operatorname{CrossAttention}(X_Q,Z)
\]

即两级 cross-attention：

```text
Natural-language query
        │
        ▼
Query encoder H_q
        │
        ▼
P semantic query slots
        │
        ▼
X_Q
        │
        ▼
Cross-attend cached context latents Z
        │
        ▼
P compressed readout tokens
```

这是目前最值得继续探索的方向之一，但它是 **我们基于 Perceiver IO 的延伸设计**，不是原论文已有方案。

---

# 13. 固定预算与 Adaptive Budget

Perceiver IO 的一个非常自然的性质是：

\[
\#\text{output queries}=\#\text{outputs}
\]

因此 QURO 可以直接把：

\[
P=\#\text{readout queries}
\]

解释成：

\[
P=\text{soft compression budget}
\]

固定预算实验：

\[
P\in\{4,8,16,32\}
\]

进一步可以考虑：

\[
P=f(q)
\]

即：

- 简单 query → 少量 readout slots；
- 复杂 / multi-hop query → 更多 readout slots。

Perceiver IO 为“输出长度由 query 数量控制”提供了基础，但 **如何根据自然语言 query 自适应决定 \(P\)** 仍然是新的问题。

---

# 14. 对论文 Claim 边界的提醒

后续写 QURO 论文时需要特别谨慎。

不宜直接 claim：

> “We introduce conditioning cross-attention queries on the input query.”

因为 Perceiver IO 已经明确允许 query 是：

- learned；
- hand-designed；
- input-derived；
- 多种 feature 的组合。

更合理的创新空间应落在：

> **How a natural-language information need is transformed into a fixed-budget set of structured readout queries over precomputed context latents.**

中文可以理解为：

> **如何把自然语言信息需求转化为固定预算的一组结构化 readout queries，并用它们从预计算 context latent 中读取任务相关信息。**

这比单纯说“query-conditioned cross-attention”更准确，也更接近我们真正需要证明的新东西。

---

# 15. 与 QURO 最相关的消融方向

基于这篇论文，后续最直接的实验轴可以整理为：

| 维度 | 可比较方案 |
|---|---|
| Query representation | mean/CLS pooling / token-level / learned slots / semantic slots |
| Query conditioning | none / global query vector / per-slot conditioning |
| Slot count | 4 / 8 / 16 / 32 |
| Slot identity | shared / learned positional-slot embedding |
| Query constructor | MLP / attention pooling / Perceiver-style cross-attention |
| Context latent | fixed precomputed / jointly updated |
| Budget | fixed P / adaptive P |

这组实验比继续复现 Perceiver IO 的光流或游戏结果更直接服务于 QURO。

---

# 16. 这篇论文对 QURO 的最终结论

阅读到这里，Perceiver IO 对本项目的价值已经基本清楚。

### 可以直接继承的部分

1. **Read-Process-Write 框架**；
2. **query-independent latent memory** 的思路；
3. decoder 中：
   \[
   Q\leftarrow X_Q,\qquad K,V\leftarrow Z
   \]
4. **Output Query 可以是 structured / input-dependent representation**；
5. **Query 数量控制输出长度**；
6. latent 与原始输入结构可以解耦。

### 不能直接从论文得到的部分

1. arbitrary natural-language query 怎样变成 \(P\) 个 readout queries；
2. 多个 query slots 是否会自动产生语义分工；
3. 如何避免多个 slots collapse 到相同 attention pattern；
4. 如何让固定预算 \(P\) 最大化保留 query-relevant information；
5. 如何设计 adaptive budget；
6. 如何与冻结生成 LLM 的 `inputs_embeds` 接口联合训练；
7. 如何在“context 可缓存”的约束下进行有效 query conditioning。

因此，这篇论文在当前阅读链条中的定位可以概括为：

\[
\boxed{
\text{Perceiver IO gives the readout mechanism, not the natural-language query constructor.}
}
\]

或者更具体：

> **它解决了“有了一个 query vector，如何让它作为 Q 去读取 latent”；但没有解决“自然语言 query 应该如何被提炼成一组有结构、有预算、有分工的 Q”。**

这正是下一阶段需要继续向 RRK / SeleCom / Tree Cross Attention 等工作以及我们自己的 Query Constructor 设计中探索的问题。

---

## 17. 当前阅读结论（供后续快速引用）

**最值得记忆的 5 句话：**

1. `Output Query Array` 是 decoder cross-attention 的 query input，不等于 attention 公式里投影后的 `Q`。
2. 每个 output query 的本质是：**描述“我要什么输出”**。
3. Query 可以通过 position / task / modality / input feature / learned embedding 的 concat 或 add 构造。
4. `P` 个 output queries 会产生 `P` 个输出，因此 `P` 可以自然解释为 readout / compression budget。
5. Perceiver IO 没有解决 arbitrary natural-language query → fixed-budget structured query slots；这仍是 QURO 的核心研究问题。
