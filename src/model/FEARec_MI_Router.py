import torch

from .dynamic_router import DynamicIntentRouter
from .FEARec_MI import FEARecMIModel


class FEARecMIRouterModel(FEARecMIModel):
    """FEARec + MultiIntentExtractor + Dynamic MoE Router."""

    def __init__(self, args):
        super(FEARecMIRouterModel, self).__init__(args)
        if args.lambda_balance < 0:
            raise ValueError("lambda_balance must be non-negative")

        self.lambda_balance = args.lambda_balance
        self.dynamic_router = DynamicIntentRouter(args.hidden_size)
        self.dynamic_router.apply(self.init_weights)

        # This v2 model replaces v1's static attention fusion with the router.
        # Keep the v1 module for checkpoint/class compatibility, but do not train it.
        self.multi_intent_extractor.interest_fusion.requires_grad_(False)

        self.last_balance_loss = None
        self._capture_original_balance = False
        self._original_balance_loss = None

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        hidden_states = self.encode(input_ids)
        _, intent_vectors, intent_attention, _ = self.multi_intent_extractor(
            hidden_states, self.intent_embedding, input_ids.ne(0)
        )

        seq_emb = hidden_states[:, -1, :]
        user_embedding, routing_weights, balance_loss = self.dynamic_router(
            seq_emb, intent_vectors
        )
        output = torch.cat(
            [hidden_states[:, :-1, :], user_embedding.unsqueeze(1)], dim=1
        )

        self.last_intent_attention = intent_attention.detach()
        self.last_intent_weights = routing_weights.detach()
        self.last_balance_loss = balance_loss.detach()

        if self._capture_original_balance and self._original_balance_loss is None:
            self._original_balance_loss = balance_loss
        return output

    def get_intent_visualization(self, user_index=0):
        visualization = super(FEARecMIRouterModel, self).get_intent_visualization(
            user_index
        )
        if visualization is not None:
            visualization["router_weights"] = visualization["weights"]
            visualization["balance_loss"] = self.last_balance_loss.cpu()
        return visualization

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        self._capture_original_balance = True
        self._original_balance_loss = None
        try:
            original_loss = super(FEARecMIRouterModel, self).calculate_loss(
                input_ids, answers, neg_answers, same_target, user_ids
            )
            self.last_balance_loss = self._original_balance_loss.detach()
            return original_loss + self.lambda_balance * self._original_balance_loss
        finally:
            self._capture_original_balance = False


MOE_FEARecV2Model = FEARecMIRouterModel
