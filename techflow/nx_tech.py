import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class nx_preps:
    """
    Read more in the 'github.com/lee-ju/techflow#techflownx_tech'

    Parameters
    ----------
    1. `x`: The data for social network analysis. On the input will always be list.
    2. 'app': Applicant Number. (default: None)
    3. `fc`: Forward citation list. (default: None)
    """

    def __init__(self, x, apps=None, fc=None):
        self.x = x
        self.apps = apps
        self.fc = fc
        self.len_x = len(self.x)

        self.from_list = []
        self.to_list = []
        self.i_list = []

    def isNan(self, n):
        self.n = n
        return self.n != self.n

    def edges(self, obj='ipcs', num_slice=4, spliter="||"):
        """
        Read more in the 'github.com/lee-ju/techflow#techflownx_tech'

        Parameters
        ----------
        1. `obj`: 'ipcs' for IPC code network, 'forws' for citation network. (default: 'ipcs')
        2. `num_slice`: An argument to how much to truncate the code from behind. (default: 4)
        3. `spliter`: An arguments to break code. (default: '||')
        """
        self.num_slice = int(num_slice)
        self.spliter = spliter
        self.obj = obj

        if self.obj == 'ipcs':
            for i in tqdm(range(self.len_x)):
                if self.isNan(self.x[i]) == False:
                    i_x = self.x[i].split(self.spliter)

                    for j in range(1, len(i_x)):
                        if len(i_x[j]) != 0:
                            self.from_list.append(i_x[0][:self.num_slice])
                            self.to_list.append(i_x[j][:self.num_slice])

                elif self.isNan(self.x[i]) == True:
                    pass

        elif self.obj == 'forws':

            app_reg = dict()
            for a in range(len(self.apps)):
                app_reg[self.apps[a]] = self.x[a]

            for i in tqdm(range(self.len_x)):
                if self.isNan(self.fc[i]) == False:
                    i_fc = self.fc[i].split(self.spliter)

                    for j in range(len(i_fc)):
                        if len(i_fc) != 0:
                            self.from_list.append(self.x[i])
                            len_i_fc = len(i_fc[j])
                            i_fc_j = i_fc[j][:(len_i_fc - self.num_slice)]

                            try:
                                self.to_list.append(app_reg[i_fc_j])
                            except KeyError:
                                self.to_list.append(i_fc_j)
                            # self.to_list.append(i_fc_j)

                elif self.isNan(self.fc[i]) == True:
                    pass

        ft = {'from': self.from_list, 'to': self.to_list}
        df = pd.DataFrame(ft)
        df.drop_duplicates()

        return df


class nx_utils:
    """
    Read more in the 'github.com/lee-ju/techflow#techflownx_tech'
    
    Parameters
    ----------
    1. `df`: Dataframe of edgelist.
    2. `direct`: Boolean controlling the DiGraph. (default: True)
    """
    def __init__(self, df, direct=True):
        self.df = df
        self.direct = direct
        self.name_list = []
        self.degr_list = []
        self.clos_list = []
        self.betw_list = []

        if self.direct:
            self.G = nx.from_pandas_edgelist(
                self.df, 'from', 'to', create_using=nx.DiGraph())
        else:
            self.G = nx.from_pandas_edgelist(
                self.df, 'from', 'to', create_using=nx.Graph())

    def nx_viz(self, fs=[10, 10], with_labels=True, node_size=300, node_color='red', \
               font_size=12, font_color='black', seed=10):
        """
        Read more in the 'github.com/lee-ju/techflow#techflownx_tech'
    
        Parameters
        ----------
        1. `fs`: List of figsize=[horizontal_size, vertical_size]. (default: [10, 10])
        2. `with_labels`: Boolean controlling the use of node labels. (default: True)
        3. `node_size`: Size of nodes. (default: 100)
        4. `node_color`: Node color. (default: 'red')
        5. `font_size`: Size of labels. (default: 12)
        6. `font_color`: Node label. (default: 'black')
        7. `seed`: Seed for random visualization. (default: 10)
        """
        self.fs = fs
        self.with_labels = with_labels
        self.node_size = int(node_size)
        self.node_color = node_color
        self.font_size = int(font_size)
        self.font_color = font_color
        self.seed = int(seed)

        plt.figure(figsize=(self.fs[0], self.fs[1]))
        pos = nx.spring_layout(self.G, seed=self.seed)

        nx.draw(self.G, pos=pos, with_labels=self.with_labels,
                font_size=self.font_size, font_color=self.font_color,
                node_size=self.node_size, node_color=self.node_color,
                alpha=0.5, width=0.5, verticalalignment='top')
        plt.show()

        return self.G

    def nx_centrality(self, top_k=10):
        """
        Read more in the 'github.com/lee-ju/techflow#techflownx_tech'
    
        Parameters
        ----------
        1. `top_k`: Return centrality by top_k. (default: 10)
        """
        self.top_k = int(top_k)

        if self.top_k > self.df.shape[0]:
            print("top_k is greater than the number of data")

        else:
            _degr = nx.degree_centrality(self.G)
            _clos = nx.closeness_centrality(self.G)
            _betw = nx.betweenness_centrality(self.G)

            def find_top(s, k=self.top_k):
                top_s = sorted(s.items(), key=lambda x: x[1], reverse=True)[:k]
                return top_s
            top_degr = find_top(_degr)
            top_clos = find_top(_clos)
            top_betw = find_top(_betw)
            top_name = set([x[0] for x in top_degr + top_clos + top_betw])

            for n in top_name:
                self.name_list.append(n)
                self.degr_list.append(_degr[n])
                self.clos_list.append(_clos[n])
                self.betw_list.append(_betw[n])
            rt = {'id': pd.Series(self.name_list),
                  'Degree': pd.Series(self.degr_list),
                  'Closeness': pd.Series(self.clos_list),
                  'Betweenness': pd.Series(self.betw_list)}
            rt_df = pd.DataFrame(rt)

            rt_df['Centrality'] = rt_df['Degree'] + \
                rt_df['Closeness'] + rt_df['Betweenness']
            rt_df = rt_df.sort_values('Centrality', ascending=False)
            rt_df.set_index('id', inplace=True)

            return_k = min(self.top_k, rt_df.shape[0])
            return_df = rt_df.iloc[:return_k]
            return return_df


if __name__ == '__main__':
    
    sample_ipcs = pd.read_csv('sample_dataset/sample_ipc.csv')
    ipcs = sample_ipcs['all_ipcs'].tolist()

    ipcs_preps = nx_preps(ipcs)
    ipcs_df = ipcs_preps.edges(obj='ipcs', num_slice=4, spliter='||')

    ipcs_utils = nx_utils(ipcs_df, direct=False)
    ipcs_central = ipcs_utils.nx_centrality(top_k=3)
    print(ipcs_central)
    
    ipcs_G = ipcs_utils.nx_viz(fs=[5, 5], with_labels=True,
                           font_size=10, font_color='blue',
                           node_size=100, node_color='red', seed=15)

if __name__ == '__main__':
    
    sample_forws = pd.read_csv('sample_dataset/sample_forw.csv')
    x = sample_forws['Reg_id'].tolist()
    apps = sample_forws['App_id'].tolist()
    forws = sample_forws['Forw_in_id'].tolist()
    
    forws_preps = nx_preps(x=x, apps=apps, fc=forws)
    forws_df = forws_preps.edges(obj='forws', num_slice=0, spliter='||')

    forws_utils = nx_utils(forws_df, direct=True)
    forws_central = forws_utils.nx_centrality(top_k=5)
    print(forws_central)

    forws_G = forws_utils.nx_viz(fs=[5, 5], with_labels=True,
                             font_size=10, font_color='black',
                             node_size=100, node_color='blue', seed=77)
