import os
import pickle
import logging

import torch
from torchdrug import core, data, utils
from torchdrug.utils import comm
from torch import nn
from torch.utils import data as torch_data
logger = logging.getLogger(__name__)
import joblib

class DiffusionEngine(core.Engine):
    def process_batch(self, batch):
        true_proteins = batch["graph"].clone()
        pred_proteins = self.model.generate(batch)["graph"]
        evaluation_metric = self.model.get_metric(pred_proteins, true_proteins, {})
        return evaluation_metric

    def generate(self, test_set, path):
        if self.device.type == "cuda":
            self.batch_size = 64  # Adjust the batch size for GPU processing
            dataloader = data.DataLoader(test_set, self.batch_size, shuffle=False)
        else:
            self.batch_size = 1
            dataloader = data.DataLoader(test_set, self.batch_size, shuffle=False)

        if comm.get_rank() == 0:
            logger.warning(f"Test on {test_set}")
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        logger.warning(path)

        model = self.model
        model.eval()
        id = 0
        data_dict = {}

        if self.device.type == "cuda":
            for batch in dataloader:
                batch = utils.cuda(batch, device=self.device)
                evaluation_metric = self.process_batch(batch)
                self.print_evaluation_metrics(evaluation_metric)
                self.save_proteins(batch, path, data_dict, id)
                id += 1
        else:
            results = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.process_batch)(batch) for batch in dataloader)
            for batch_id, result in enumerate(results):
                self.print_evaluation_metrics(result)
                self.save_proteins(dataloader[batch_id], path, data_dict, id)
                id += 1

    def print_evaluation_metrics(self, evaluation_metric):
        print(f"atom_rmsd_per_residue: {evaluation_metric['atom_rmsd_per_residue'].mean():<20}"
              f"chi_0_mae_deg: {evaluation_metric['chi_0_ae_deg'].mean():<20}"
              f"chi_1_mae_deg: {evaluation_metric['chi_1_ae_deg'].mean():<20}"
              f"chi_2_mae_deg: {evaluation_metric['chi_2_ae_deg'].mean():<20}"
              f"chi_3_mae_deg: {evaluation_metric['chi_3_ae_deg'].mean():<20}")

    def save_proteins(self, batch, path, data_dict, id):
        for p in batch["graph"].unpack():
            pdb_file = os.path.basename(batch["graph"].pdb_files[id])
            protein = p.cpu()
            protein.to_pdb(os.path.join(path, pdb_file))
            data_dict[pdb_file] = p.cpu()
