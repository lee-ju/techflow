# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from techflow.nx_tech import nx_preps
from techflow.nlp_tech import nlp_preps
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


class pic_preps:

    def __init__(self, apps, apps_date, forws, texts=None):
        self.apps = apps
        self.apps_date = apps_date
        self.forws = forws
        self.texts = texts

    def get_cam(self, num_slice=4, spliter='||'):
        self.num_slice = int(num_slice)
        self.spliter = spliter

        forws_preps = nx_preps(x=self.apps, y=self.forws)
        df_cam = forws_preps.edges(
            obj='forws', num_slice=self.num_slice, spliter=self.spliter)
        from_cam = df_cam['from'].tolist()
        to_cam = df_cam['to'].tolist()

        return from_cam, to_cam

    def get_sam(self, max_features=100, use_ptrain=True, use_weight=True, ptrain_path=None, min_sim=0.5):
        self.max_features = int(max_features)
        self.use_ptrain = use_ptrain
        self.use_weight = use_weight
        self.ptrain_path = ptrain_path
        self.from_sam = []
        self.to_sam = []
        self.min_sim = min_sim

        texts_preps = nlp_preps(self.texts)
        emb_x = texts_preps.dtmx(max_features=self.max_features, use_ptrain=self.use_ptrain,
                                 use_weight=self.use_weight, ptrain_path=self.ptrain_path)

        def cos_sim(A, B):
            val = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
            return val

        N = emb_x.shape[0]
        for i in range(N):
            for j in range((i+1), N):
                if j < N:
                    sim_val = cos_sim(emb_x[i], emb_x[j])
                    if sim_val >= self.min_sim:
                        self.from_sam.append(apps[i])
                        self.to_sam.append(apps[j])
        return from_sam, to_sam

    def get_repo(self, num_slice=4):
        self.num_slice = int(num_slice)
        repo = dict()
        for i in range(len(apps)):
            str_ad = str(apps_date[i])
            len_ad = len(str_ad)
            int_ad = int(str_ad[:(len_ad - self.num_slice)])
            repo[apps[i]] = int_ad
        return repo


class pic_utils:

    def __init__(self, from_cam, to_cam, from_sam, to_sam, repo, direct=True):
        self.from_cam = from_cam
        self.to_cam = to_cam
        self.from_sam = from_sam
        self.to_sam = to_sam
        self.repo = repo
        self.direct = direct

        self.pic_E = []
        self.pic_L = []

        if self.direct:
            self.CS_net = nx.DiGraph()
            
        else:
            self.CS_net = nx.DiGraph()

    def explorer(self, max_date):
        self.max_date = int(max_date)

        for i in tqdm(range(len(self.from_sam))):
            from_date = self.repo[self.from_sam[i]]
            to_date = self.repo[self.to_sam[i]]
            diff_date = from_date - to_date
            
            if abs(diff_date) <= self.max_date:
                if diff_date <= 0:
                    E = self.from_sam[i]
                    L = self.to_sam[i]
                    
                else:
                    E = self.to_sam[i]
                    L = self.from_sam[i]

                idx_E = [e for e, value in enumerate(
                    self.from_cam) if value == E]
                F1 = [self.to_cam[f] for f in idx_E]

                for h in F1:
                    idx_F1 = [e for e, value in enumerate(
                        self.from_cam) if value == h]
                    F2 = [self.to_cam[f] for f in idx_F1]
                    
                    if len(F2) != 0:
                        for k in F2:
                            if k == L:
                                self.pic_E.append(E)
                                self.pic_L.append(L)
            else:
                pass

        return self.pic_E, self.pic_L

    def cs_net(self, pic_E, pic_L, fs=[10, 10], with_labels=True,
               node_size=300, font_size=12, seed=10):
        self.pic_E = pic_E
        self.pic_L = pic_L

        self.fs = fs
        self.with_labels = with_labels
        self.node_size = int(node_size)
        self.font_size = font_size
        self.seed = int(seed)

        for m in range(len(self.from_cam)):
            self.CS_net.add_nodes_from(
                [self.from_cam[m], self.to_cam[m]], color='white')
            self.CS_net.add_edge(
                self.from_cam[m], self.to_cam[m], color='black')
            
        for n in range(len(self.pic_E)):
            self.CS_net.add_nodes_from(
                [self.pic_E[n], self.pic_L[n]], color='#FF1744')
            self.CS_net.add_edge(self.pic_E[n], self.pic_L[n], color='#FF1744')
            
        node_color = nx.get_node_attributes(self.CS_net, 'color').values()
        edge_color = nx.get_edge_attributes(self.CS_net, 'color').values()

        plt.figure(figsize=(self.fs[0], self.fs[1]))
        pos = nx.spring_layout(CS_net, seed=self.seed)

        nx.draw(self.CS_net, pos=pos, with_labels=self.with_labels,
                node_color=node_color, edge_color=edge_color,
                font_size=self.font_size, node_size=self.node_size,
                alpha=0.8, width=0.5)
        
        ax = plt.gca()
        ax.collections[0].set_edgecolor('#000000')
        plt.axis('off')
        plt.show()

        return self.CS_net


# ## Test

if __name__ == '__main__':
    # Read dataset
    sample_pic = pd.read_csv('sample_dataset/sample_pic.csv')
    apps = sample_pic['Reg_id'].tolist()
    forws = sample_pic['Forw_in_id'].tolist()
    apps_date = sample_pic['Reg_date'].tolist()

    # Preprocessing: CAM
    pp = pic_preps(apps=apps, apps_date=apps_date,
                   forws=forws, texts=texts)
    repo = pp.get_repo(num_slice=0)
    from_cam, to_cam = pp.get_cam(num_slice=0, spliter='||')

    # Preprocessing: SAM
    ptrain_path = '/Users/leeju/Dev_dir/eng/GoogleNews-vectors-negative300.bin.gz'
    from_sam, to_sam = pp.get_sam(
        max_features=100, use_ptrain=True, use_weight=True, ptrain_path=ptrain_path, min_sim=0.6)

    # PIC-Explorer
    pu = pic_utils(from_cam, to_cam, from_sam, to_sam, repo, direct=True)
    pic_E, pic_L = pu.explorer(max_date=20)
    pic = {'P_E': pic_E, 'P_L': pic_L}
    df_pic = pd.DataFrame(pic)
    print(df_pic)
    
    # PIC-Visualization
    CS_net = pu.cs_net(pic_E, pic_L, fs=[3, 3], with_labels=True,
                       node_size=300, font_size=12, seed=10)
