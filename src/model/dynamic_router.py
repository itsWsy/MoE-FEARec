import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicIntentRouter(nn.Module):
    """Dynamically route a sequential context to latent intent experts."""

    def __init__(self, hidden_size):
        super(DynamicIntentRouter, self).__init__()
        self.router = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, seq_emb, intent_vectors):
        if seq_emb.dim() != 2:
            raise ValueError("seq_emb must have shape [B, D]")
        if intent_vectors.dim() != 3:
            raise ValueError("intent_vectors must have shape [B, K, D]")
        if (
            seq_emb.size(0) != intent_vectors.size(0)
            or seq_emb.size(1) != intent_vectors.size(2)
        ):
            raise ValueError("seq_emb and intent_vectors have incompatible shapes")

        num_intents = intent_vectors.size(1)
        expanded_seq_emb = seq_emb.unsqueeze(1).expand(-1, num_intents, -1)
        router_input = torch.cat([expanded_seq_emb, intent_vectors], dim=-1)
        router_scores = self.router(router_input).squeeze(-1)
        routing_weights = torch.softmax(router_scores, dim=-1)

        user_emb = torch.sum(
            routing_weights.unsqueeze(-1) * intent_vectors, dim=1
        )

        mean_alpha = routing_weights.mean(dim=0)
        uniform = torch.full_like(mean_alpha, 1.0 / num_intents)
        eps = torch.finfo(mean_alpha.dtype).eps
        # Follow the requested MoE load-balancing form. ``sum`` keeps the
        # regularizer scale independent of the number of intent experts.
        balance_loss = F.kl_div(
            torch.log(mean_alpha.clamp_min(eps)),
            uniform,
            reduction="sum",
        )
        return user_emb, routing_weights, balance_loss
