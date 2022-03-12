import torch
from torch import nn
import pandas as pd
#from student_submission_folder.model import Model


class Model(nn.Module):
    r"""
    This is a dummy model just for illustrtation. Your own model should have an 'inference' function as defined below. 
    The 'inference' function should do all necessary data pre-processing and the prediction process of your NN model. 
    When grading, we will call the 'inference' function of your own model.
    You do not need a GPU to train your model. When grading, however, we might use a GPU to make a faster work.
    """
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        # TODO: define your modules
        pass

    def forward(self, x):   
        x = x.to(self.device)
        # TODO: Implement your own forward function
        pass

    def inference(self, PID, centroid, LIDs, ligands):
        r"""
        Your own model should have this 'inference' function, which does all necessary data pre-processing and the prediction process of your NN model. 
        We will call this function to run your model and grade its performance. Please note that the input to this funciton is strictly defined as follows.
        Args:
            PID: str, one single protein ID, e.g., '112M'.
            centroid: float tuple, the x-y-z binding location of protein PID, e.g., (34.8922, 7.174, 12.4984).
            LIDs: str list, a list of ligand IDs, e.g., ['3', '3421']. You can regard len(LIDs) as the batch size during inference.
            ligands: str list, a list of SIMLEs formulas of the ligands in LIDs, e.g., ['NCCCCCCNCCCCCCN', 'C1CC1(N)P(=O)(O)O']
        Return:
            A Torch Tensor in the shape of (len(LIDs), 1), representing the predicted binding score (or likelihood) for the protein PID and each ligand in LIDs.

        About GPU:
            Again, you do not need a GPU to train your model. However, We might use GPU to accelerate out grading work. 
            So please send all your processed inputs to self.device.
            If you define any object that is not a torch.nn module, you should also explicitly send this object to self.device.
        """
        # TODO: Implement the inference function
        return torch.rand(len(LIDs), 1)


if __name__ == '__main__':
    CENTROIDS_DIR = './project_test_data/centroids.csv'
    PDBS_DIR = './project_test_data/pdbs'
    LIGAND_DIR = './project_test_data/ligand.csv'
    GT_PAIR_DIR = './project_test_data/pair.csv'

    #read centroids.csv
    centroids = {}
    df = pd.read_csv(CENTROIDS_DIR)
    for i in range(len(df)):
        centroids[str(df.PID[i])] = (float(df.x[i]), float(df.y[i]), float(df.z[i]))

    #read ligand.csv
    ligands = {}
    df = pd.read_csv(LIGAND_DIR)
    for i in range(len(df)):
        ligands[str(df.LID[i])] = (str(df.Smiles[i]))

    #read groundtruth pair.csv for grading
    gt_pairs = {}
    df = pd.read_csv(GT_PAIR_DIR)
    for i in range(len(df)):
        gt_pairs[str(df.PID[i])] = (str(df.LID[i]))

    
    BS = 100                        #Batch size for inference
    TOPK = 10                       #Set top-10 accuracy
    DEVICE = 'cpu'                  #You do not necessarily need a GPU. Of course, you are free to use a GPU if you have one.

    model = Model(device=DEVICE)    #When grading, we will import and call your own model
    model.to(DEVICE)

    #inference
    prediction_correctness = []
    for PID in centroids:
        binding_scores = torch.empty(0, 1)
        LIDs =[LID for LID in ligands]
        #check the binding score of each ligand to one specific protein
        for i in range(0, len(LIDs)-BS+1, BS):
            batch_pred = model.inference(PID, centroids[PID], LIDs[i: i+BS], [ligands[LID] for LID in LIDs[i: i+BS]])
            binding_scores = torch.cat([binding_scores, batch_pred], dim=0)
        if i < len(LIDs)-BS:
            batch_pred = model.inference(PID, centroids[PID], LIDs[i+BS: ], [ligands[LID] for LID in LIDs[i+BS: ]])
            binding_scores = torch.cat([binding_scores, batch_pred], dim=0)

        #transform torch.tensor to list
        binding_scores = binding_scores.squeeze(-1).cpu().detach().numpy().tolist()

        #get top-k scores and corresponding LIDs
        topk_pred = sorted(zip(binding_scores, LIDs), reverse=True)[:TOPK]
        topk_scores, topk_LIDs = zip(*topk_pred)
        #print(topk_LIDs)
        
        #compare with groundtruth
        if str(gt_pairs[PID]) in topk_LIDs:
            prediction_correctness.append(1)
        else:
            prediction_correctness.append(0)

    accuracy = sum(prediction_correctness) / len(prediction_correctness)

    print(f"Inference Prediction Score: {'{:.5f}'.format(accuracy)}.")