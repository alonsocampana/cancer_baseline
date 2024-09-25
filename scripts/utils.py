from torch.utils.data import Dataset
from torch import Tensor
import numpy as np



class OmicsDataset(Dataset):
    def __init__(self, omic_dict, drug_dict, data):
        self.omic_dict = omic_dict
        self.drug_dict = drug_dict
        self.cell_mapped_ids = {key:i for i, key in enumerate(self.omic_dict.keys())}
        self.drug_mapped_ids = {key:i for i, key in enumerate(self.drug_dict.keys())}
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        cell_id = instance.iloc[0]
        drug_id = instance.iloc[1]
        target = instance.iloc[2]
        return (self.omic_dict[cell_id],
                self.drug_dict[drug_id],
                Tensor([target]),
                Tensor([self.cell_mapped_ids[cell_id]]),
                Tensor([self.drug_mapped_ids[drug_id]]))
    
    
import rdkit
from rdkit.Chem import AllChem
class FingerprintFeaturizer():
    def __init__(self,
                 fingerprint = "morgan",
                 R=2, 
                 fp_kwargs = {},
                 transform = Tensor):
        """
        Get a fingerprint from a list of molecules.
        Available fingerprints: MACCS, morgan, topological_torsion
        R is only used for morgan fingerprint.
        fp_kwards passes the arguments to the rdkit fingerprint functions:
        GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint, GetTopologicalTorsionFingerprint
        """
        self.R = R
        self.fp_kwargs = fp_kwargs
        self.fingerprint = fingerprint
        if fingerprint == "morgan":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x, self.R, **fp_kwargs)
        elif fingerprint == "MACCS":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(x, **fp_kwargs)
        elif fingerprint == "topological_torsion":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(x, **fp_kwargs)
        self.transform = transform
    def __call__(self, smiles_list, drugs = None):
        drug_dict = {}
        if drugs is None:
            drugs = np.arange(len(smiles_list))
        for i in range(len(smiles_list)):
            try:
                smiles = smiles_list[i]
                molecule = AllChem.MolFromSmiles(smiles)
                feature_list = self.f(molecule)
                f = np.array(feature_list)
                if self.transform is not None:
                    f = self.transform(f)
                drug_dict[drugs[i]] = f
            except:
                drug_dict[drugs[i]] = None
        return drug_dict
    def __str__(self):
        """
        returns a description of the featurization
        """
        return f"{self.fingerprint}Fingerprint_R{self.R}_{str(self.fp_kwargs)}"