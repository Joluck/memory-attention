分享一下近期的工作：Normalized Low-Rank Adaptation (NoRA)

如何缩小 Low-Rank Matrix 与 Full-Rank Matrix 在优化和性能上的差距，并获得更高的 rank efficiency，是高效大模型训练中的一个重要问题。相比继续设计更复杂的 LoRA 变体，我们尝试换个角度思考：一个好的低秩投影本身应该满足怎样的性质？

受 MLA 中 latent normalization 启发，我们发现，直接归一化 Norm(Ax) 虽然能改善优化，但会破坏线性与 mergeability。于是我们提出 NoRA，将 normalization 转移到 down-projection A 本身，对其沿 rank 维度施加归一化约束，得到 Norm(A)x。

NoRA能够有效稳定低秩投影的数值尺度，并使 early gradient dynamics 更接近 Full-Rank Matrix；同时，这一约束仅在初始化阶段施加，也能保留大部分性能收益。

在 Pre-training、SFT 和 RL 中，NoRA 均带来更快收敛、更好性能与稳定性，同时不增加额外参数和推理开销，训练后仍可直接 merge。

相同的归一化原则应用到 DoRA 上同样有效，这表明 NoRA 可能是缩小 Low-Rank Matrix 与 Full-Rank Matrix gap 的一个更普适方向。

Paper: https://huggingface.co/papers/2608.31036