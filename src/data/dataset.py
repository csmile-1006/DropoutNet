import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.utils.utils import gather_nd


class XINGDataset(Dataset):
    def __init__(self, u_latent, u_indices, device):
        self.u_latent = torch.Tensor(u_latent).to(device)
        self.u_indices = u_indices

    def __len__(self):
        return len(self.u_indices)

    def __getitem__(self, index):
        idx = self.u_indices[index]
        u_latent = self.u_latent[idx]
        return idx, u_latent


class Collate(object):
    def __init__(
        self, v_latent, n_scores_user, num_users, data_batch_size, p_dropout, device
    ):
        self.v_latent = torch.Tensor(v_latent).to(device)
        self.n_scores_user = n_scores_user
        self.num_users = num_users
        self.data_batch_size = data_batch_size
        self.p_dropout = p_dropout
        self.device = device

    def __call__(self, batch):
        idx, u_latent = map(torch.stack, zip(*batch))

        target_users = idx.repeat_interleave(self.n_scores_user)
        target_users_rand = (
            torch.arange(len(idx)).repeat_interleave(self.n_scores_user).to(self.device)
        )
        target_items_rand = torch.LongTensor(
            [
                np.random.choice(self.v_latent.shape[0] - 1, self.n_scores_user)
                for _ in idx
            ]
        )
        target_items_rand = target_items_rand.flatten().to(self.device)
        target_ui_rand = torch.vstack([target_users_rand, target_items_rand]).t()

        preds_pref = torch.mm(u_latent, self.v_latent.t())

        target_scores, target_items = torch.topk(
            preds_pref, axis=1, k=self.n_scores_user
        )
        target_scores = target_scores.reshape(-1)
        target_items = target_items.reshape(-1)
        random_scores = gather_nd(preds_pref, target_ui_rand)

        target_scores = torch.cat([target_scores, random_scores])
        target_items = torch.cat([target_items, target_items_rand])
        target_users = torch.cat([target_users, target_users])

        n_targets = len(target_scores)
        perm = np.random.permutation(n_targets)
        n_targets = min(n_targets, self.n_scores_user * idx.shape[0])
        data_batch = [
            (n, min(n + self.data_batch_size, n_targets))
            for n in range(0, n_targets, self.data_batch_size)
        ]
        batches = []
        for (start, stop) in data_batch:
            batch_perm = perm[start:stop]
            batch_users = target_users[batch_perm]
            batch_items = target_items[batch_perm]
            if self.p_dropout != 0:
                n_to_drop = int(np.floor(self.p_dropout * len(batch_perm)))
                perm_user = np.random.permutation(len(batch_perm))[:n_to_drop]
                perm_item = np.random.permutation(len(batch_perm))[:n_to_drop]
                batch_v_pref = torch.clone(batch_items)
                batch_u_pref = torch.clone(batch_users)
                batch_v_pref[perm_user] = self.v_latent.shape[0]
                batch_u_pref[perm_item] = self.num_users
            else:
                batch_v_pref = batch_items
                batch_u_pref = batch_users
            batches.append(
                [
                    batch_u_pref,
                    batch_v_pref,
                    batch_users,
                    batch_items,
                    target_scores[batch_perm],
                ]
            )

        return batches
