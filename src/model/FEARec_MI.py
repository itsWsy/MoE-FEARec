import torch
import torch.nn as nn

from .fearec import FEARecModel


class MultiIntentExtractor(nn.Module):
    """Extract and fuse multiple latent interests from FEARec hidden states."""

    def __init__(self, hidden_size, num_intents):
        super(MultiIntentExtractor, self).__init__()
        self.num_intents = num_intents
        self.interest_fusion = nn.Linear(hidden_size, num_intents)

    def forward(self, hidden_states, intent_embedding, input_mask=None):
        # [B, L, D] x [K, D] -> [B, K, L]
        attention_scores = torch.matmul(
            hidden_states, intent_embedding.transpose(0, 1)
        ).transpose(1, 2)

        if input_mask is not None:
            valid_mask = input_mask.bool()
            # Very short training prefixes can contain no history. In that case,
            # use the final encoder position instead of producing an all-masked softmax.
            empty_sequences = ~valid_mask.any(dim=1)
            if empty_sequences.any():
                valid_mask = valid_mask.clone()
                valid_mask[empty_sequences, -1] = True
            attention_scores = attention_scores.masked_fill(
                ~valid_mask.unsqueeze(1), torch.finfo(attention_scores.dtype).min
            )

        intent_attention = torch.softmax(attention_scores, dim=-1)
        intent_representations = torch.matmul(intent_attention, hidden_states)

        # The latest FEARec state summarizes the current sequential context and
        # determines how strongly each extracted interest contributes.
        fusion_weights = torch.softmax(
            self.interest_fusion(hidden_states[:, -1, :]), dim=-1
        )
        user_embedding = torch.sum(
            fusion_weights.unsqueeze(-1) * intent_representations, dim=1
        )
        return user_embedding, intent_representations, intent_attention, fusion_weights


class FEARecMIModel(FEARecModel):
    """FEARec with a MultiIntentExtractor prediction representation."""

    def __init__(self, args):
        super(FEARecMIModel, self).__init__(args)
        if args.num_intents < 1:
            raise ValueError("num_intents must be at least 1")
        self.num_intents = args.num_intents
        self.intent_embedding = nn.Parameter(
            torch.empty(args.num_intents, args.hidden_size)
        )
        self.multi_intent_extractor = MultiIntentExtractor(
            args.hidden_size, args.num_intents
        )

        nn.init.normal_(
            self.intent_embedding, mean=0.0, std=args.initializer_range
        )
        self.multi_intent_extractor.apply(self.init_weights)

        self.last_intent_attention = None
        self.last_intent_weights = None

    def encode(self, input_ids, all_sequence_output=False):
        """Run the unchanged FEARec embedding and encoder stack."""
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        return encoded_layers if all_sequence_output else encoded_layers[-1]

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        hidden_states = self.encode(input_ids)
        user_embedding, _, intent_attention, fusion_weights = (
            self.multi_intent_extractor(
                hidden_states, self.intent_embedding, input_ids.ne(0)
            )
        )

        # Keep Trainer/SequentialRecModel's [B, L, D] prediction contract. The
        # final position is now the fused multi-interest user representation.
        output = torch.cat(
            [hidden_states[:, :-1, :], user_embedding.unsqueeze(1)], dim=1
        )

        self.last_intent_attention = intent_attention.detach()
        self.last_intent_weights = fusion_weights.detach()
        return output

    def get_intent_visualization(self, user_index=0):
        """Return the most recent user's sequence attention and fusion weights."""
        if self.last_intent_weights is None:
            return None
        return {
            "attention": self.last_intent_attention[user_index].cpu(),
            "weights": self.last_intent_weights[user_index].cpu(),
        }

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        # Recommendation loss and all native FEARec objectives are retained;
        # no multi-intent-specific auxiliary loss is introduced.
        seq_output = self.forward(input_ids)
        original_intent_attention = self.last_intent_attention
        original_intent_weights = self.last_intent_weights
        user_embedding = seq_output[:, -1, :]

        logits = torch.matmul(
            user_embedding, self.item_embeddings.weight.transpose(0, 1)
        )
        loss = nn.CrossEntropyLoss()(logits, answers)

        aug_seq_output = None
        sem_aug_seq_output = None

        if self.ssl in ["us", "un"]:
            aug_seq_output = self.forward(input_ids)
            nce_logits, nce_labels = self.info_nce(
                seq_output,
                aug_seq_output,
                temp=self.tau,
                batch_size=input_ids.shape[0],
                sim=self.sim,
            )
            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        if self.ssl in ["us", "su"]:
            sem_aug_seq_output = self.forward(same_target)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=input_ids.shape[0],
                sim=self.sim,
            )
            loss += self.lmd_sem * self.aug_nce_fct(
                sem_nce_logits, sem_nce_labels
            )

        if self.ssl == "us_x":
            aug_seq_output = self.forward(input_ids)
            sem_aug_seq_output = self.forward(same_target)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                aug_seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=input_ids.shape[0],
                sim=self.sim,
            )
            loss += self.lmd_sem * self.aug_nce_fct(
                sem_nce_logits, sem_nce_labels
            )

        if self.fredom:
            seq_output_f = torch.fft.rfft(user_embedding, dim=1, norm="ortho")

            if self.fredom_type in ["us", "un"] and aug_seq_output is not None:
                aug_output_f = torch.fft.rfft(
                    aug_seq_output[:, -1, :], dim=1, norm="ortho"
                )
                loss += 0.1 * abs(seq_output_f - aug_output_f).flatten().mean()

            if self.fredom_type in ["us", "su"] and sem_aug_seq_output is not None:
                sem_output_f = torch.fft.rfft(
                    sem_aug_seq_output[:, -1, :], dim=1, norm="ortho"
                )
                loss += 0.1 * abs(seq_output_f - sem_output_f).flatten().mean()

            if (
                self.fredom_type == "us_x"
                and aug_seq_output is not None
                and sem_aug_seq_output is not None
            ):
                aug_output_f = torch.fft.rfft(
                    aug_seq_output[:, -1, :], dim=1, norm="ortho"
                )
                sem_output_f = torch.fft.rfft(
                    sem_aug_seq_output[:, -1, :], dim=1, norm="ortho"
                )
                loss += 0.1 * abs(aug_output_f - sem_output_f).flatten().mean()

        # Visualization should describe the original sequence, not an augmentation.
        self.last_intent_attention = original_intent_attention
        self.last_intent_weights = original_intent_weights
        return loss


# Convenient alias matching the experiment/model name used in the requirement.
MOE_FEARecModel = FEARecMIModel
