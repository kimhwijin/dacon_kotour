import torch

def _get_attention_score(attn_matrix, device):
    #attn_matrix : N Layer  x Batch x Head x Seq x Seq

    # N Layer x Batch x Seq x Seq
    attn_matrix = torch.mean(attn_matrix, dim=2)

    # Seq x Seq
    residual_attn = torch.eye(attn_matrix.size(2)).to(device)
    # N Layer x Batch x Seq x Seq
    aug_attn = attn_matrix + residual_attn
    # N Layer x Batch x Seq x Seq
    aug_attn = aug_attn / aug_attn.sum(dim=-1).unsqueeze(-1)
    # N Layer x Batch x Seq x Seq
    joint_attn = torch.zeros(aug_attn.size()).to(device)
    joint_attn[0] = aug_attn[0]

    for n in range(1, aug_attn.size(0)):
        joint_attn[n] = torch.bmm(aug_attn[n], joint_attn[n-1])
    
    #Batch x Seq x Seq
    attn_scores = joint_attn[-1]
    #Batch x Seq - 1 ( cls token score )
    attn_scores = attn_scores[:, 0, 1:].softmax(-1)
    return attn_scores


@torch.no_grad()
def attention_guied_dropout_mask(attention_masks, all_self_attentions, dropout, threshold, device):
    # attention_masks : Batch x Seq
    # all_self_attentions : Tuple ( N Layers, )
    # all_self_attentions : Tensor ( Batch x Head x Seq x Seq )
    
    # N layer x Batch x Head x Seq x Seq
    all_self_attentions = torch.stack(all_self_attentions).to(device)

    # Batch x Seq - 1 ( except cls token )
    # Score : 0 ~ 1
    attention_scores = _get_attention_score(all_self_attentions, device)

    # Batch x Seq - 1
    apply_dropout_indices = attention_scores < threshold
    # Batch x Seq - 1
    indices = (torch.Tensor(apply_dropout_indices.size()).uniform_(0, 1) < dropout) * apply_dropout_indices
    attention_masks[:, 1:][indices] = 1
    return attention_masks