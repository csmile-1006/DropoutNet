import os
import math

import fire
import torch
import scipy
import numpy as np
import pandas as pd
import ujson as json
from torch import nn
from tqdm import tqdm
from dotmap import DotMap

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model.deepCF import DeepCF
from src.data.dataset import XINGDataset, Collate
from src.utils.utils import get_logger, prep_standardize, convert_svmlight_to_torch


class Trainer(object):
    def __init__(self, conf_fname):
        self.logger = get_logger()
        with open(conf_fname, "r") as f:
            config = json.load(f)
        self.config = DotMap(config)
        self.logger.info(json.dumps(config, indent=2))
        self.device = (
            torch.device("cuda") if self.config.use_gpu else torch.device("cpu")
        )
        self.model = DeepCF(**self.config.model).to(self.device)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.SGD(
            self.model.parameters(), lr=self.config.train.lr, momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=self.config.train._decay_lr_every,
            gamma=self.config.train._lr_decay,
        )
        self._data_loaded = False
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def _get_path(self, attr):
        return os.path.join(
            self.config.path.base, self.__get_multiple_attr(self.config.path, attr)
        )

    def __get_multiple_attr(self, origin, attrs):
        attr_list = attrs.split(".")
        res = origin
        for at in attr_list:
            res = getattr(res, at)
        return res

    def load_train_data(self):
        # load preference data
        self.u_latent = np.fromfile(
            self._get_path("train.u_file"), dtype=np.float32
        ).reshape(-1, 200)
        self.v_latent = np.fromfile(
            self._get_path("train.v_file"), dtype=np.float32
        ).reshape(-1, 200)

        # pre-process
        _, u_latent_scaled = prep_standardize(self.u_latent)
        _, v_latent_scaled = prep_standardize(self.v_latent)

        # append pref factors for faster dropout
        u_latent_expanded = np.vstack(
            [u_latent_scaled, np.zeros_like(u_latent_scaled[0, :])]
        )
        self.u_latent_expanded = torch.Tensor(u_latent_expanded).to(self.device)
        v_latent_expanded = np.vstack(
            [v_latent_scaled, np.zeros_like(v_latent_scaled[0, :])]
        )
        self.v_latent_expanded = torch.Tensor(v_latent_expanded).to(self.device)

        # load content data
        self.u_content = convert_svmlight_to_torch(
            self._get_path("train.u_content_file")
        ).to(self.device)
        self.v_content = convert_svmlight_to_torch(
            self._get_path("train.v_content_file")
        ).to(self.device)

        # load split
        self.train_data = (
            pd.read_csv(
                self._get_path("train.train_file"),
                delimiter=",",
                header=None,
                dtype=np.int32,
            )
            .values.ravel()
            .view(
                dtype=[
                    ("uid", np.int32),
                    ("iid", np.int32),
                    ("inter", np.int32),
                    ("date", np.int32),
                ]
            )
        )
        self.u_indices = torch.from_numpy(
            np.fromfile(self._get_path("train.u_indices"), dtype=np.int64)
        ).to(self.device)

    def load_dataloader(self):
        base_path = "../data/recsys2017.pub/eval"
        self.ds = XINGDataset(
            u_latent=self.u_latent, u_indices=self.u_indices, device=self.device
        )

        self.collate = Collate(
            v_latent=self.v_latent,
            num_users=self.u_latent.shape[0],
            p_dropout=self.config.train.p_dropout,
            n_scores_user=self.config.train.n_scores_user,
            data_batch_size=self.config.train.data_batch_size,
            device=self.device,
        )

        self.dl = DataLoader(
            self.ds,
            shuffle=True,
            collate_fn=self.collate,
            batch_size=self.config.train.user_batch_size,
        )

    def _get_eval_data(self, file_path, iid_path, is_cold=False):
        with open(iid_path, "r") as f:
            test_item_ids = [int(line) for line in f]
        test_data = pd.read_csv(
            file_path, delimiter=",", header=None, dtype=np.int32
        ).values.ravel()
        test_data = test_data.view(
            dtype=[
                ("uid", np.int32),
                ("iid", np.int32),
                ("inter", np.int32),
                ("date", np.int32),
            ]
        )
        test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}
        _test_ij_for_inf = [
            (t[0], t[1]) for t in test_data if t[1] in test_item_ids_map
        ]

        test_user_ids = np.unique(test_data["uid"])
        test_user_ids_map = {uid: i for i, uid in enumerate(test_user_ids)}
        _test_i_for_inf = [test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
        _test_j_for_inf = [test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]

        R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)), (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(test_user_ids), len(test_item_ids)],
        ).tolil(copy=False)

        train_ij_for_inf = [
            (test_user_ids_map[_t[0]], test_item_ids_map[_t[1]])
            for _t in self.train_data
            if _t[1] in test_item_ids_map and _t[0] in test_user_ids_map
        ]
        if is_cold and len(train_ij_for_inf) != 0:
            raise Exception("using cold dataset, but data is not cold!")
        if not is_cold and len(train_ij_for_inf) == 0:
            raise Exception("using warm datset, but data is not warm!")

        R_train_inf = (
            None
            if is_cold
            else scipy.sparse.coo_matrix(
                (np.ones(len(train_ij_for_inf)), zip(*train_ij_for_inf)),
                shape=R_test_inf.shape,
            ).tolil(copy=False)
        )

        dat = {}
        dat["R_test_inf"] = R_test_inf

        dat["u_latent_test"] = self.u_latent_expanded[test_user_ids, :]
        dat["v_latent_test"] = self.v_latent_expanded[test_item_ids, :]
        dat["u_content_test"] = self.u_content[test_user_ids, :]
        dat["v_content_test"] = self.v_content[test_item_ids, :]

        eval_l = R_test_inf.shape[0]
        dat["eval_batch"] = [
            (x, min(x + self.config.eval.batch_size, eval_l))
            for x in range(0, eval_l, self.config.eval.batch_size)
        ]

        if not is_cold:
            dat["eval_train"] = []
            for (eval_start, eval_finish) in dat["eval_batch"]:
                _ui = R_train_inf[eval_start:eval_finish, :].tocoo()
                dat["eval_train"].append(
                    torch.sparse_coo_tensor(
                        indices=(_ui.row, _ui.col),
                        values=torch.full(_ui.data.shape, -100000),
                        size=[eval_finish - eval_start, R_train_inf.shape[1]],
                    ).to(self.device)
                )

        return dat

    def load_eval_data(self):
        self.eval_data = {}
        for tp in ["warm", "cold_user", "cold_item"]:
            is_cold = False if tp == "warm" else True
            dat = self._get_eval_data(
                self._get_path(f"test.{tp}_file"),
                self._get_path(f"test.{tp}_iid_file"),
                is_cold=is_cold,
            )
            self.eval_data[tp] = dat

    def _evaluate(self, dat):
        self.model.eval()
        preds_batch = []
        with torch.no_grad():
            v_latent = dat["v_latent_test"]
            v_content = dat["v_content_test"]
            for (batch, (eval_start, eval_stop)) in enumerate(dat["eval_batch"]):
                u_latent = dat["u_latent_test"][eval_start:eval_stop, :]
                u_content = dat["u_content_test"][eval_start:eval_stop, :]
                u_emb, i_emb = self.model(u_latent, v_latent, u_content, v_content)
                preds = torch.matmul(u_emb, i_emb.t())
                if dat.get("eval_train"):
                    preds = preds.add(dat["eval_train"][batch])
                _, topk_inds = torch.topk(
                    preds, axis=1, k=self.config.eval.recall_at[-1]
                )
                preds_batch.append(topk_inds.cpu().numpy())

        eval_preds = np.concatenate(preds_batch)

        # filter non-zero targets
        y_nz = [len(x) > 0 for x in dat["R_test_inf"].rows]
        y_nz = np.arange(len(dat["R_test_inf"].rows))[y_nz]
        y = dat["R_test_inf"][y_nz, :]

        preds_all = eval_preds[y_nz, :]

        recall = []
        for at_k in self.config.eval.recall_at:
            preds_k = preds_all[:, :at_k]
            x = scipy.sparse.lil_matrix(y.shape)
            x[:, preds_k] = 1
            z = y.multiply(x)
            recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
        return recall

    def evaluate(self):
        result = []
        for tp, data in self.eval_data.items():
            ret = self._evaluate(data)
            result.append(ret)
        return result

    def load(self):
        self.logger.info("load train data.")
        self.load_train_data()
        self.logger.info("load dataloader.")
        self.load_dataloader()
        self.logger.info("load eval data.")
        self.load_eval_data()
        self._data_loaded = True

    def train(self):
        self.logger.info("train start.")
        best_warm = []
        best_cold_user = []
        best_cold_item = []
        best_step = 0
        n_batch_trained = 0
        for epoch in range(self.config.num_epochs):
            for n_step, batch in enumerate(self.dl, start=1):
                f_batch = 0
                for idx, data_batch in enumerate(batch, start=1):
                    self.optim.zero_grad()
                    (
                        batch_u_pref,
                        batch_v_pref,
                        batch_users,
                        batch_items,
                        target_scores,
                    ) = data_batch
                    u_latent = self.u_latent_expanded[batch_u_pref, :]
                    v_latent = self.v_latent_expanded[batch_v_pref, :]
                    u_content = self.u_content[batch_users, :]
                    v_content = self.v_content[batch_items, :]

                    u_emb, i_emb = self.model(u_latent, v_latent, u_content, v_content)
                    preds = torch.mul(u_emb, i_emb).sum(axis=1)
                    loss = self.loss(preds, target_scores)
                    f_batch += loss.item()
                    if not torch.isfinite(loss):
                        self.logger.error("WARNING: non-finite loss, ending training ")
                        return
                    loss.backward()
                    self.optim.step()

                # scheduler is changed according to user batch.
                self.scheduler.step()

                n_batch_trained += len(batch)
                if n_step % self.config.train.eval_every == 0:
                    recall_warm, recall_cold_user, recall_cold_item = self.evaluate()
                    if np.sum(
                        recall_warm + recall_cold_user + recall_cold_item
                    ) > np.sum(best_warm + best_cold_user + best_cold_item):
                        best_cold_user = recall_cold_user
                        best_cold_item = recall_cold_item
                        best_warm = recall_warm
                        best_step = n_step
                    self.logger.info(
                        f"{epoch} epoch {n_step} step [{len(batch)}]b [{n_batch_trained}]tot f={f_batch:.2f} best[{best_step}]"
                    )
                    self.logger.info(
                        "\t\t"
                        + " ".join(
                            [
                                ("@" + str(i)).ljust(6)
                                for i in self.config.eval.recall_at
                            ]
                        )
                    )
                    self.logger.info(f"warm start\t{' '.join([f'{i:.4f}' for i in recall_warm])}")
                    self.logger.info(f"cold user\t{' '.join([f'{i:.4f}' for i in recall_cold_user])}")
                    self.logger.info(f"cold item\t{' '.join([f'{i:.4f}' for i in recall_cold_item])}")
                    self.logger.info("==============================================================")

                    # write tensorboard
                    for i, recall in enumerate(self.config.eval.recall_at):
                        self.writer.add_scalar(
                            f"recall@{recall}-warm", recall_warm[i], n_batch_trained
                        )
                        self.writer.add_scalar(
                            f"recall@{recall}-cold-user",
                            recall_cold_user[i],
                            n_batch_trained,
                        )
                        self.writer.add_scalar(
                            f"recall@{recall}-cold-item",
                            recall_cold_item[i],
                            n_batch_trained,
                        )

        return (best_warm, best_cold_user, best_cold_item, best_step)

    def run(self):
        self.load()
        self.train()


if __name__ == "__main__":
    fire.Fire(Trainer)
