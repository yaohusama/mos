from lzma import MF_BT2
import torch
import numpy as np
import os
from utils.reranking import re_ranking
# from ..modeling.baseline import InferenceXtt
from .data_parallel import BalancedDataParallel
class InferenceXtt(torch.nn.Module):
    in_planes = 2048

    def __init__(self):
        super(InferenceXtt, self).__init__()
        # self.qf=qf #.to("cuda")
        # self.Mp=Mp

    def forward(self, qf,x,Mp,ze):
        # qf,x,Mp,ze=batch
        # global_feat = (self.base(x))  # (b, 2048, 1, 1)
        # qf,gf,Mp=x
        # x=x.to("cuda")
        # qf=qf.to("cuda")
        m = qf.shape[0]
        n = x.shape[0] #gf
        qff=qf.unsqueeze(1).expand(m,n,qf.shape[1])
        gff=x.unsqueeze(0).expand(m,n,qf.shape[1])

        # maxf=torch.max(qff,gff)
        
        dist0=torch.min(qff,gff).sum(-1, keepdim=False)/torch.max(qff,gff).sum(-1, keepdim=False)
        #.unsqueeze(0).unsqueeze(0)
        dist0=dist0-0.001*torch.sum(torch.max(ze,torch.reshape(torch.matmul(torch.max(qff,gff).unsqueeze(3),torch.max(qff,gff).unsqueeze(2))-Mp.unsqueeze(0).unsqueeze(0).expand(m,n,qf.shape[1],qf.shape[1]),(m,n,-1))),dim=2) 

#.reshape(qf.shape[0],x.shape[0],-1)
        return dist0 #.to(device="cuda:2")
def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()
def jaccard_distance(qf1, gf1,Mp):
    # m = qf.shape[0]
    # Mp=Mp.to('cpu')
    step=1
    step1=440 #192 496 480   400
    dist=[]
    if torch.cuda.device_count()>1:
        model1=torch.nn.DataParallel(InferenceXtt())
        # model1=BalancedDataParallel(74,InferenceXtt(Mp),dim=0)
    # model1=InferenceXtt(Mp) #,qf1
    model1.eval()
    Mp1=Mp.repeat(4,1)
    for item1 in range(0,len(qf1)-step1,step1):
        dist1=[]

        
        qf=qf1[item1:item1+step1]
        for item in range(0,len(gf1)-step,step):
            gf=gf1[item:item+step]
            # n = gf.shape[0]
            
            # qff=qf.unsqueeze(1).expand(m,n,qf.shape[1])
            # gff=gf.unsqueeze(0).expand(m,n,gf.shape[1])
            # maxf=torch.max(qff,gff)
            
            # dist0=torch.min(qff,gff).sum(-1, keepdim=False)/torch.max(qff,gff).sum(-1, keepdim=False)-0.001*torch.sum(torch.reshape(torch.matmul(torch.max(qff,gff).unsqueeze(3),torch.max(qff,gff).unsqueeze(2))-Mp.unsqueeze(0).unsqueeze(0).expand(m,n,qf.shape[1],qf.shape[1]),(m,n,-1)),dim=2)

            
                # print("right")
            # model.to(device)  
            ze=torch.zeros(qf.shape[0],gf.shape[0],qf.shape[1]*qf.shape[1])
            gf=gf.repeat(4,1)
            
            # dist0=model1(qf,gf)
            res=model1(qf,gf,Mp1,ze).to(device='cuda:2')
            dist1.append(res) #.to(device='cuda:3')qf,gf,Mp,ze
            dist1t=torch.cat(dist1,0)
            # dist.append(dist0.to(device='cuda:3'))
        if len(gf1)%step!=0:
            gf=gf1[len(gf1)-len(gf1)%step:]
            ze=torch.zeros(qf.shape[0],gf.shape[0],qf.shape[1]*qf.shape[1])
            gf=gf.repeat(4,1)
            # Mp1=Mp.repeat(4,1)
            # dist0=model1(qf,gf)
            
            dist1.append(model1(qf,gf,Mp1,ze).to(device='cuda:2')) #.to(device='cuda:3')qf,gf,Mp,ze
            dist1t=torch.cat(dist,0)
        dist.append(dist1t)
    if len(qf1)%step!=0:
        qf=qf1[len(qf1)-len(qf1)%step:]
        for item in range(0,len(gf1)-step,step):
            gf=gf1[item:item+step]
            ze=torch.zeros(qf.shape[0],gf.shape[0],qf.shape[1]*qf.shape[1])
            gf=gf.repeat(4,1)
            # Mp1=Mp.repeat(4,1)
            # dist0=model1(qf,gf)
            
            dist1.append(model1(qf,gf,Mp1,ze).to(device='cuda:2')) #.to(device='cuda:3')
            dist1t=torch.cat(dist,0)
            # dist.append(dist0.to(device='cuda:3'))
        if len(gf1)%step!=0:
            gf=gf1[len(gf1)-len(gf1)%step:]
            ze=torch.zeros(qf.shape[0],gf.shape[0],qf.shape[1]*qf.shape[1])
            gf=gf.repeat(4,1)
            # Mp1=Mp.repeat(4,1)
            # dist0=model1(qf,gf)
            
            dist1.append(model1(qf,gf,Mp1,ze).to(device='cuda:2')) #.to(device='cuda:3')
            dist1t=torch.cat(dist1,0)
        dist.append(dist1t)
    dist=torch.cat(dist,1)
    # dist = [((torch.minimum(qf[i] ,gf[j]))).sum(-1, keepdim=False) * 1.0 / (
    #     (torch.maximum(qf[i] ,gf[j])).sum(-1, keepdim=False))-(0.001*torch.sum(torch.max(torch.zeros(qf.shape[1],qf.shape[1]),torch.exp(torch.maximum(qf[i],gf[j]).unsqueeze(1)@torch.maximum(qf[i],gf[j]).unsqueeze(0)-Mp.to('cpu')-0.1)-1))) for i in range(m) for j in range(n)]
    # dist=torch.from_numpy(np.array(dist).reshape(m, n))
    
    dist_mat = 1 - dist
    dist_mat=dist_mat.to('cuda:0')
    return dist_mat.cpu().numpy()
def jaccard_distance1(qf1, gf1,Mp):
    m = qf1.shape[0]
    Mp=Mp.to('cpu')
    

    
    
            
    
    
    # maxf=torch.max(qff,gff)
    stepj=len(gf1)//10

    dist=[]
    for item in range(0,len(gf1)-stepj,stepj):
        gf=gf1[item:item+stepj]
        
        n = gf.shape[0]
        gff=gf.unsqueeze(0).expand(m,n,gf.shape[1])
        qff=qf1.unsqueeze(1).expand(m,n,qf1.shape[1])
        dist0=torch.min(qff,gff).sum(-1, keepdim=False)/torch.max(qff,gff).sum(-1, keepdim=False)
        dist.append(dist0)
    if len(gf1)%stepj!=0:
        gf=gf1[len(gf1)-len(gf1)%stepj:]
        n = gf.shape[0]
        gff=gf.unsqueeze(0).expand(m,n,gf.shape[1])
        qff=qf1.unsqueeze(1).expand(m,n,qf1.shape[1])
        # gf=gf.repeat(4,1)
        # dist0=model1(qf,gf)
        dist0=torch.min(qff,gff).sum(-1, keepdim=False)/torch.max(qff,gff).sum(-1, keepdim=False)
        dist.append(dist0)
        # dist1t=torch.cat(dist,0)
    dist=torch.cat(dist,1)
       
   
    # dist = [((torch.minimum(qf[i] ,gf[j]))).sum(-1, keepdim=False) * 1.0 / (
    #     (torch.maximum(qf[i] ,gf[j])).sum(-1, keepdim=False))-(0.001*torch.sum(torch.max(torch.zeros(qf.shape[1],qf.shape[1]),torch.exp(torch.maximum(qf[i],gf[j]).unsqueeze(1)@torch.maximum(qf[i],gf[j]).unsqueeze(0)-Mp.to('cpu')-0.1)-1))) for i in range(m) for j in range(n)]
    # dist=torch.from_numpy(np.array(dist).reshape(m, n))
    
    dist_mat = 1 - dist
    dist_mat=dist_mat.to('cuda:0')
    return dist_mat.cpu().numpy()
def cosine_similarity1(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query,Mp, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.Mp=Mp

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with jaccard_distance')
            # distmat = euclidean_distance(qf, gf)
            distmat = jaccard_distance(qf, gf,self.Mp)
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



