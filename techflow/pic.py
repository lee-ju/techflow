from techflow.nx_tech import nx_preps
from techflow.nlp_tech import nlp_preps
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


class pic_preps:
    """
    Read more in the 'github.com/lee-ju/techflow#techflowpic'
    
    Parameters
    ----------
    1. `apps`: Applicant id.
    2. `regs`: Registration id.
    3. `regs_date`: Registration dates.
    4. `forws`: Forward citation patents.
    5. `texts`: Text of documents. (default: None)
    """
    def __init__(self, apps, regs, regs_date, forws, texts=None):
        self.apps = apps
        self.regs = regs
        self.regs_date = regs_date
        self.forws = forws
        self.texts = texts

    def get_cam(self, num_slice=4, spliter='||'):
        """
        Read more in the 'github.com/lee-ju/techflow#techflowpic'
    
        Parameters
        ----------
        1. `num_slice`: An argument to how much to truncate the code from behind. (default: 4)
        2. `spliter`: An arguments to break code. (default: '||')
        """
        self.num_slice = int(num_slice)
        self.spliter = spliter

        forws_preps = nx_preps(x=self.regs, apps=self.apps, forws=self.forws)
        df_cam = forws_preps.edges(
            obj='forws', num_slice=self.num_slice, spliter=self.spliter)
        from_cam = df_cam['from'].tolist()
        to_cam = df_cam['to'].tolist()

        return from_cam, to_cam

    def get_sam(self, max_features=100, use_ptrain=True, use_weight=True, ptrain_path=None, min_sim=0.5):
        """
        Read more in the 'github.com/lee-ju/techflow#techflowpic'
    
        Parameters
        ----------
        1. `max_features`: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. (default: 100)
        2. `use_ptrain`: True if using the pre-trained word embedding model, False if using only tf-idf. (default: True)
        3. `use_weight`: True if embedding values are used as weights, False otherwise. Select only when use_ptrain is True. (default: True)
        4. `ptrain_path`: Path for pre-trained word embedding model. (default: None)
        5. `min_sim`: Minimum value of similarity (threshold value on [PIC](https://doi.org/10.3390/su13020820)). (default: 0.5)
        """
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
                        self.from_sam.append(self.regs[i])
                        self.to_sam.append(self.regs[j])
        return self.from_sam, self.to_sam

    def get_repo(self, num_slice=4):
        """
        Read more in the 'github.com/lee-ju/techflow#techflowpic'
    
        Parameters
        ----------
        1. `num_slice`: An argument to how much to truncate the code from behind. (default: 4)
        """
        self.num_slice = int(num_slice)

        repo = dict()
        for i in range(len(self.regs)):
            str_rd = str(self.regs_date[i])
            len_rd = len(str_rd)
            int_rd = int(str_rd[:(len_rd - self.num_slice)])
            repo[self.regs[i]] = int_rd
        return repo


class pic_utils:
    """
    Read more in the 'github.com/lee-ju/techflow#techflowpic'

    Parameters
    ----------
    1. `from_cam`: In-node lists of Citation Adjacency Matrix.
    2. `to_cam`: Out-node lists of Citation Adjacency Matrix.
    3. `from_sam`: In-node lists of Similarity Adjacency Matrix.
    4. `to_sam`: Out-node lists of Similarity Adjacency Matrix.
    5. `repo`: Dictionary of regs and regs_date.
    6. `direct`: Boolean controlling the DiGraph. (default: True)
    """
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

    def explorer(self, max_date=20):
        """
        Read more in the 'github.com/lee-ju/techflow#techflowpic'
    
        Parameters
        ----------
        1. `max_date`: The maximum value of the time difference between the filing of two patents. (default: 20)
        """
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

                # if (E, L) in CAM, they are not pic
                chk = 0
                for idx_e in idx_E:
                    if self.to_cam[idx_e] == L:
                        chk += 1
                        break
                    else:
                        pass
                if chk == 0:
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
                    # (E, L) in CAM
                    pass
            else:
                # diff_date > max_date
                pass

        d = {'P_E': self.pic_E, 'P_L': self.pic_L}

        return_df = pd.DataFrame(d)
        return_df_dupdrop = return_df.drop_duplicates()
        return_pic_E = return_df_dupdrop['P_E']
        return_pic_L = return_df_dupdrop['P_L']

        return return_pic_E, return_pic_L

    def cs_net(self, pic_E, pic_L, fs=[10, 10], with_labels=True,
               node_size=300, font_size=12, seed=10):
        """
        Read more in the 'github.com/lee-ju/techflow#techflowpic'
    
        Parameters
        ----------
        1. `pic_E`: Output of pic_utils.explorer (CS-Net  on [PIC](https://doi.org/10.3390/su13020820)).
        2. `fs`: List of figsize=[horizontal_size, vertical_size]. (default: [10, 10])
        3. `with_labels`: Boolean controlling the use of node labels. (default: True)
        4. `node_size`: Size of nodes. (default: 100)
        5. `font_size`: Size of labels. (default: 12)
        6. `seed`: Seed for random visualization. (default: 10)
        """
        self.pic_E = pic_E
        self.pic_L = pic_L

        self.fs = fs
        self.with_labels = with_labels
        self.node_size = int(node_size)
        self.font_size = int(font_size)
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
        pos = nx.spring_layout(self.CS_net, seed=self.seed)

        nx.draw(self.CS_net, pos=pos, with_labels=self.with_labels,
                node_color=node_color, edge_color=edge_color,
                font_size=self.font_size, node_size=self.node_size,
                alpha=0.8, width=0.5)
        
        ax = plt.gca()
        ax.collections[0].set_edgecolor('#000000')
        plt.axis('off')
        plt.show()

        return self.CS_net

if __name__ == '__main__':
    ## Read dataset
    sample_pic = pd.read_csv('sample_dataset/sample_pic.csv')
    apps = sample_pic['App_id'].tolist()
    regs = sample_pic['Reg_id'].tolist()
    regs_date = sample_pic['Reg_date'].tolist()
    forws = sample_pic['Forw_in_id'].tolist()
    texts = sample_pic['Text'].tolist()

    ## Preprocessing: CAM
    pp = pic_preps(apps=apps, regs=regs, regs_date=regs_date,
                   forws=forws, texts=texts)
    repo = pp.get_repo(num_slice=0)
    from_cam, to_cam = pp.get_cam(num_slice=0, spliter='||')

    ## Preprocessing: SAM
    #ptrain_path = '.../GoogleNews-vectors-negative300.bin.gz'#Example for ptrain
    from_sam, to_sam = pp.get_sam(
        max_features=100, min_sim=0.6,
        use_ptrain=False, use_weight=False, ptrain_path=None)

    ## PIC-Explorer
    pu = pic_utils(from_cam, to_cam, from_sam, to_sam, repo, direct=True)
    pic_E, pic_L = pu.explorer(max_date=20)
    pic = {'P_E': pic_E, 'P_L': pic_L}
    df_pic = pd.DataFrame(pic)
    print(df_pic)

    ## PIC-Visualization
    CS_net = pu.cs_net(pic_E, pic_L, fs=[3, 3], with_labels=True,
                       node_size=300, font_size=12, seed=10)
