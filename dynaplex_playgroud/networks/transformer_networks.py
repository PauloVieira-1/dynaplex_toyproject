"""Transformer-based policy and value networks.

Policy net architecture (cross-data with action-group pooling):
    1. Per-type linear projections: raw features → d_model
    2. Learned type embeddings added to projected features
    3. Context tokens (non-actions) processed with self-data encoder
    4. All action-group tokens (root + children) receive **intra-group
       positional embeddings** so the model can distinguish token order
       within each action (e.g. stop sequence in a route)
    5. Action tokens cross-attend to (context + action tokens) so they see
       context, sibling tokens inside the same action, and other actions
    6. Per-action-group mean-pool → one embedding per action
    7. Action embeddings → MLP → per-action logits

    For flat actions (group size 1) positional embeddings are zero and the
    behaviour collapses to the previous version.

Value net architecture (self-data over all tokens):
    1. Same per-type projections + type embeddings
    2. Self-data over ALL tokens
    3. Mean-pool → MLP → scalar value
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn


# ============================================================================
# Shared helper: build embeddings from sequence dict
# ============================================================================

def _build_embeddings(
    input_projs: nn.ModuleDict,
    type_embeddings: nn.Embedding,
    type_features: Dict[str, Tensor],
    type_ids: Tensor,
    indices_by_type: Dict[str, List[int]],
    d_model: int,
) -> Tensor:
    """Project raw features and add type embeddings.  Returns [seq_len, d_model]."""
    seq_len = type_ids.shape[0]
    device = type_ids.device
    embeddings = torch.zeros(seq_len, d_model, device=device)
    for tname, proj in input_projs.items():
        if tname not in indices_by_type:
            continue
        idxs = indices_by_type[tname]
        feats = type_features[tname]
        projected = proj(feats)
        for local_i, global_i in enumerate(idxs):
            embeddings[global_i] = projected[local_i]
    # Clamp type_ids in case a late-discovered type has an id beyond
    # the embedding table size (it will share the last type embedding).
    safe_type_ids = type_ids.clamp(max=type_embeddings.num_embeddings - 1)
    embeddings = embeddings + type_embeddings(safe_type_ids)
    return embeddings


# ============================================================================
# Cross-data decoder layer
# ============================================================================

class CrossAttentionLayer(nn.Module):
    """One layer of: action self-data → cross-data to (context + actions) → FFN.

    Actions attend to:
    - Other action-group tokens (via self-data)
    - Context tokens and themselves (via cross-data to combined set)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, action_emb: Tensor, context_emb: Tensor) -> Tensor:
        """
        Args:
            action_emb:  [1, num_action_tokens, d_model]
            context_emb: [1, num_context, d_model]
        Returns:
            updated action_emb: [1, num_action_tokens, d_model]
        """
        # Action-group self-data
        x = action_emb
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + sa_out)

        # Cross-data to context + updated action tokens
        combined_kv = torch.cat([context_emb, x], dim=1)
        ca_out, _ = self.cross_attn(x, combined_kv, combined_kv)
        x = self.norm2(x + ca_out)

        # FFN
        x = self.norm3(x + self.ffn(x))
        return x


# ============================================================================
# Action-group pooling helper
# ============================================================================

def _pool_action_groups(
    action_emb: Tensor,
    action_group_ids_for_actions: Tensor,
    num_groups: int,
) -> Tensor:
    """Mean-pool action-token embeddings per action group.

    Args:
        action_emb: [num_action_tokens, d_model]
        action_group_ids_for_actions: [num_action_tokens] with values in [0, num_groups)
        num_groups: number of distinct action groups

    Returns:
        pooled: [num_groups, d_model]
    """
    d_model = action_emb.shape[-1]
    device = action_emb.device
    pooled = torch.zeros(num_groups, d_model, device=device)
    counts = torch.zeros(num_groups, 1, device=device)

    for i in range(num_groups):
        mask = action_group_ids_for_actions == i
        if mask.any():
            pooled[i] = action_emb[mask].mean(dim=0)
            counts[i] = mask.sum()

    return pooled


# ============================================================================
# Policy net (cross-data with action-group pooling)
# ============================================================================

class TransformerPolicyNet(nn.Module):
    """Cross-data policy network with action-group support.

    All tokens belonging to an action's sub-tree (root Action + nested children)
    are treated as "action tokens".  After cross-data processing, tokens in
    each action group are mean-pooled to produce one embedding per action, which
    is then scored to produce per-action logits.

    For flat actions (group size 1) this is identical to the previous behaviour.
    """

    def __init__(
        self,
        raw_dims: Dict[str, int],
        type_names: List[str],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        max_group_size: int = 32,
    ):
        super().__init__()
        self.type_names = list(type_names)
        self.d_model = d_model

        # Per-type input projections
        self.input_projs = nn.ModuleDict()
        for tname in type_names:
            self.input_projs[tname] = nn.Linear(raw_dims[tname], d_model)

        # Learned type embeddings
        self.type_embeddings = nn.Embedding(len(type_names), d_model)

        # Learned intra-group positional embeddings (order-sensitive actions)
        # Position 0 = root, 1..max_group_size-1 = children in order
        self.intra_group_pos_emb = nn.Embedding(max_group_size, d_model)

        # Context encoder: self-data over non-action tokens
        ctx_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(ctx_layer, num_layers=num_layers)

        # Cross-data decoder: action tokens attend to context + actions
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Action scoring head (operates on pooled per-action embeddings)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward_logits(
        self,
        type_features: Dict[str, Tensor],
        type_ids: Tensor,
        action_mask: Tensor,
        indices_by_type: Dict[str, List[int]],
        action_group_ids: Optional[Tensor] = None,
        num_action_groups: Optional[int] = None,
        intra_group_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute per-action logits via cross-data + group pooling.

        Parameters
        ----------
        action_group_ids : optional LongTensor[seq_len]
            -1 for context tokens, ≥ 0 for action-group membership.
            When ``None``, falls back to treating each action token as its
            own group (backward compatible).
        num_action_groups : optional int
            Total number of action groups.  Required when ``action_group_ids``
            is provided.
        intra_group_pos : optional LongTensor[seq_len]
            Position of each token within its action group (0 = root,
            1+ = children in order).  When provided, a learned positional
            embedding is added to action tokens so the model can distinguish
            token ordering within each action (e.g. stop sequence).

        Returns
        -------
        logits : Tensor[num_actions]
            One logit per action (i.e. per action group).
        """
        embeddings = _build_embeddings(
            self.input_projs, self.type_embeddings,
            type_features, type_ids, indices_by_type, self.d_model,
        )

        # ----------------------------------------------------------
        # Add intra-group positional embeddings to action tokens
        # ----------------------------------------------------------
        if intra_group_pos is not None:
            action_positions = action_mask.nonzero(as_tuple=True)[0]
            if action_positions.numel() > 0:
                pos_ids = intra_group_pos[action_positions]
                # Clamp to max embedding size
                pos_ids = pos_ids.clamp(max=self.intra_group_pos_emb.num_embeddings - 1)
                embeddings[action_positions] = (
                    embeddings[action_positions] + self.intra_group_pos_emb(pos_ids)
                )

        # ----------------------------------------------------------
        # Split into context vs action-group tokens
        # ----------------------------------------------------------
        action_idx = action_mask.nonzero(as_tuple=True)[0]
        context_idx = (~action_mask).nonzero(as_tuple=True)[0]

        action_emb = embeddings[action_idx].unsqueeze(0)    # [1, A_total, d]
        context_emb = embeddings[context_idx].unsqueeze(0)   # [1, C, d]

        # Enrich context with self-data
        if context_emb.shape[1] > 0:
            context_emb = self.context_encoder(context_emb)

        # Action tokens cross-attend to context + themselves
        for layer in self.cross_layers:
            action_emb = layer(action_emb, context_emb)      # [1, A_total, d]

        action_emb = action_emb.squeeze(0)                   # [A_total, d]

        # ----------------------------------------------------------
        # Pool per action group → one embedding per action
        # ----------------------------------------------------------
        if action_group_ids is not None and num_action_groups is not None:
            # Extract group ids for the action tokens only
            group_ids_for_actions = action_group_ids[action_idx]  # [A_total]
            pooled = _pool_action_groups(
                action_emb, group_ids_for_actions, num_action_groups,
            )  # [num_groups, d]
        else:
            # Backward compatible: each action token is its own "group"
            pooled = action_emb  # [A, d]

        logits = self.action_head(pooled).squeeze(-1)        # [num_actions]
        return logits


# ============================================================================
# Value net (self-data over all tokens)
# ============================================================================

class TransformerValueNet(nn.Module):
    """Value network using self-data over ALL tokens.

    Unlike the policy net which uses cross-data (actions query context),
    the value net needs to see everything — including action tokens which
    may carry essential state information (e.g. bin weights in bin packing).
    """

    def __init__(
        self,
        raw_dims: Dict[str, int],
        type_names: List[str],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.type_names = list(type_names)
        self.d_model = d_model

        self.input_projs = nn.ModuleDict()
        for tname in type_names:
            self.input_projs[tname] = nn.Linear(raw_dims[tname], d_model)

        self.type_embeddings = nn.Embedding(len(type_names), d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        type_features: Dict[str, Tensor],
        type_ids: Tensor,
        indices_by_type: Dict[str, List[int]],
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute state value from ALL tokens (full self-data)."""
        embeddings = _build_embeddings(
            self.input_projs, self.type_embeddings,
            type_features, type_ids, indices_by_type, self.d_model,
        )
        x = self.encoder(embeddings.unsqueeze(0)).squeeze(0)
        pooled = x.mean(dim=0, keepdim=True)
        value = self.value_head(pooled).squeeze(-1)
        return value


# ============================================================================
# Factory functions
# ============================================================================

def build_transformer_policy_net(
    raw_dims: Dict[str, int],
    type_names: List[str],
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
) -> TransformerPolicyNet:
    """Create a Transformer policy network."""
    return TransformerPolicyNet(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )


def build_transformer_value_net(
    raw_dims: Dict[str, int],
    type_names: List[str],
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
) -> TransformerValueNet:
    """Create a Transformer value network."""
    return TransformerValueNet(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )

