import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import re
import seaborn as sns

class Utils:
    @staticmethod
    def l2norm(vec1, vec2):
        dist = 0
        for v1,v2 in zip(vec1, vec2):
            dist += (v1-v2)**2
        return dist
    
    @staticmethod
    def plot(data):
        x = np.arange(len(data))
        y = data
        plt.scatter(x, y, color = 'r')
        plt.xlabel('Sentence No.')
        plt.ylabel('Page No.')
        plt.show()

    @staticmethod
    def get_result(strings_list, aligned_list):
        result_dict = defaultdict(list)
        for val1, val2 in aligned_list:
            result_dict[val2].append(strings_list[val1-1])
        return json.dumps(dict(result_dict), ensure_ascii=False, indent=4)
    
    @staticmethod
    def search_format(raw_text_path, length_cutoff = 10):
        with open(raw_text_path, 'r', encoding='cp949') as file:
            text_data = file.read()
        strings = re.findall(r"'String':\s*'([^']*)'", text_data)
        original_strings = strings.copy()
        filtered_strings = [string for string in strings if len(string) >= length_cutoff]
        sig_idx = [idx for idx, string in enumerate(strings) if len(string) > length_cutoff]
        
        return original_strings, filtered_strings, sig_idx

    @staticmethod
    def concat_strings(lst, n):
        result = []
        for i in range(0, len(lst), n):
            result.append(''.join(lst[i:i+n]))
        return result
    
    @staticmethod
    def z_score_normalization(sig_arr):
        normalized_sig_arr = np.zeros_like(sig_arr, dtype=float)
        
        for j in range(sig_arr.shape[1]):
            col = sig_arr[:, j]
            mean = np.mean(col)
            std_dev = np.std(col)
            
            if std_dev > 0:
                normalized_sig_arr[:, j] = (col - mean) / std_dev
            else:
                normalized_sig_arr[:, j] = col

        return normalized_sig_arr

    @staticmethod
    def mean_normalization(sig_arr):
        mean_arr = np.zeros_like(sig_arr, dtype=float)
        for i in range(sig_arr.shape[0]):
            for j in range(sig_arr.shape[1]):
                mean_arr[i][j] = sig_arr[i][j]/(np.mean(sig_arr[:, j])*np.mean(sig_arr[i, :]))

        return mean_arr


    @staticmethod
    def dpalign(sent_vec: list[list], pdf_vec: list[list]) -> list:
        rankmat = []
        for i in range(len(sent_vec)):
            lengthlist = []
            for j in range(len(pdf_vec)):
                lengthlist.append(Utils.l2norm(sent_vec[i], pdf_vec[j]))
            rankmat.append(lengthlist)
        rankmat = Utils.z_score_normalization(np.array(rankmat))
        print(rankmat.shape)
        # rankmat = np.array(rankmat)
        heatmap = np.array(rankmat).T
        sns.heatmap(heatmap, annot=False, cmap='coolwarm')
        plt.gca().invert_yaxis()
        plt.savefig('heatmap.png', format='png')
        plt.clf()
        plt.cla()
        # rankmat = Utils.z_score_normalization(np.array(rankmat))
        prev = rankmat[0]
        curr = [0] * len(pdf_vec)
        oldpath = [[0] for _ in range(len(pdf_vec))]
        newpath = [[] for _ in range(len(pdf_vec))]
        
        for idx in range(1, len(rankmat)):
            for j in range(len(pdf_vec)):
                minval = np.inf
                newvar = 0
                for k in range(j + 1):
                    lf = min(prev[0:k + 1]) + rankmat[idx][j]
                    if lf < minval:
                        minval = lf
                        newvar = k
                curr[j] = minval
                newpath[j] = oldpath[newvar] + [j]
            prev = curr[:]
            oldpath = [row[:] for row in newpath]

        return newpath[-1]

    @staticmethod
    def embedNalign(sig_idx, page_embeddings, sent_embeddings, length_cutoff=10) -> list[(int, int)]:
        return list(zip(sig_idx, Utils.dpalign(sent_embeddings, page_embeddings)))
    
    @staticmethod
    def find_similar_pages(index, sentences, progress_bar = True):

        similar_pages = []

        if progress_bar:
            from tqdm import tqdm
            loop = tqdm(sentences, desc='Finding similar pages...', total=len(sentences))
        else:
            loop = sentences

        for sentence in loop:

            result = index.vectorstore.similarity_search(sentence, k=5)


            if result:
                similar_page = result[0].metadata['page'] 
            else:
                similar_page = "No match found"
            
            similar_pages.append(similar_page)

        return similar_pages