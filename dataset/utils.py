from sklearn.model_selection import KFold, StratifiedKFold
import torch, gc

def _stratified_kfold(config, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for train_index, test_index in skf.split(X, y):
        return train_index, test_index

@torch.no_grad()
def _step(model, images, input_ids, attn_masks):
    img_embed = model.img_embedding(images)
    txt_embed = model.txt_embedding(input_ids)
    x = torch.cat((img_embed, txt_embed), axis=1)
    attn_masks = model.get_attn_mask(attn_masks, attn_masks.shape, attn_masks.device)
    attention_maps = model.encoder(x, attn_masks, output_attentions=True).attentions
    return attention_maps

@torch.no_grad()
def _get_score(att_mat, device, max_seq):
    #att_mat : (Layer Attn x Heads x Seq x Seq)
    att_mat = torch.mean(att_mat, dim=1)
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
    v = joint_attentions[-1][:max_seq, :max_seq]
    # mask : (196, )
    mask = v[0, 1:]
    arg_sorted = torch.argsort(mask)
    return arg_sorted.detach().cpu().numpy()