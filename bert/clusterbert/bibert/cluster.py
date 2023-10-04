from functools import reduce
from tqdm import tqdm
from collections import defaultdict, Counter
import pickle
import os
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import pair_confusion_matrix
import umap
from sklearn.manifold import Isomap
import numpy as np


class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
            
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None
        

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=True):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
        

def save_vocab(vocab_path,vocab):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab,f)
# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1,data_type ="protein"):
        print("Building Vocab")
        counter = Counter()

        for line in tqdm(texts):
            if str(line).startswith(">"):
                continue
            if isinstance(line, list):
                words = line
            else:
                if data_type == "protein":
                    line = line.replace("\n", "").replace("\t", "")
                    words = [e for e in line]
                else:
                    words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

        # Parallel(n_jobs=48)(delayed(self.count)(line for line in texts)) # Less efficient than above

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words
    
    def from_seq2freq(self,seq)->list:
        '''
        @msg: 将句子转换为频率向量
        @param:
            seq :word list
        @return:
            list[int] :each word frequency
        '''
        words = seq.strip().split(" ")
        vocab_len = len(self.stoi)-5 
        freq_list = [0 for _ in range(vocab_len)]
        for w in words:
            freq_list[self.stoi[w]-5] += 1
    
        return freq_list

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--data_type",  type=str,default="dna")
    parser.add_argument("-c", "--corpus_path", default =None, type=str)
    parser.add_argument("-o", "--output_path",  default =None, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()
  
    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq,data_type = args.data_type)
        print("VOCAB SIZE:", len(vocab))
        
        vocab.save_vocab(args.output_path)


def load_pk_dict(path):
    with open(path, "rb") as f:
        datadict = pickle.load(f)
    return datadict

class FaFrequency:
    def __init__(self,data,data_path,vocab_path):
        '''
        @msg:计算每个kmer的频率
        @param:
            data:[msg,seq]
            data_path:contig path
        @return:
            self.data[msg,freq_data]
        '''
        vocab = WordVocab.load_vocab(vocab_path=vocab_path)
        self.freqs = []
        self.seq_names = []
        self.data = defaultdict(list)
        if data!=None:
            for msg,seq in data:
                self.seq_names.append(msg)
                freq_list = vocab.from_seq(seq)
                total = sum(freq_list)
                for i,e in enumerate(freq_list):
                    freq_list[i] = round(e/total,4)
                self.freqs.append(freq_list)
        else:
            with open(data_path,"r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line.startswith(">"):
                        self.seq_names.append(line)
                    else:
                        freq_list =vocab.from_seq(line)
                        total = sum(freq_list)
                        for i,e in enumerate(freq_list):
                            freq_list[i] = round(e/total,4)
                        self.freqs.append(freq_list)
        for freq,name in zip(self.freqs,self.seq_names):
            self.data[name] = freq

class Cluster:
    def __init__(self, hidden = 64,data: dict = None, nfiles: int = 111404, data_path: str = None, true_labels:dict=None,file_key: str = "frag_embed",  concat_mode: int = 1, cluster_method: str = "kmeans", partition="gi"):
        '''
        @msg: 在CPU上对以获取的embed数据进行聚类
        @param:
            concat_mode：int 对输入数据所做的后续处理
                0:直接对数据进行聚类
                1：先将fragment embed利用data_concat进行拼接，获取contigs的embed,再聚类

            cluster_method:
                kmeans:kmeans聚类
                iter: vamb 启发式聚类算法
        @return:
        '''
        super(Cluster).__init__()
        assert data != data_path

        self.embed = []
        self.labels = []
        self.seq_names = []
        self.seq_name2label = true_labels
        
        if data:
            self.embed = list(data.values())
            for k in data.keys():
                self.seq_names.append(k.strip().split("|")[-1])
        else:
            
            for i in tqdm(range(nfiles)):
                path = os.path.join(data_path, file_key+str(i)+".pk")
                datadict = load_pk_dict(path)
                for k, v in datadict.items():
                    msg = k.strip().split("|")
                    if msg[3] == partition:
                        self.embed.append(v)
                        self.seq_names.append(msg[-1])
                        if true_labels!=None:
                            self.labels.append(true_labels[msg[-1]])
                        else:
                            self.labels.append(msg[-1])

        self.concat_mode = concat_mode
        self.cluster_method = cluster_method
        self.hidden = hidden

    def data_concat(self,max_fragment_num=10):
        '''
        @msg: 
            先利用已预训练好的模型对数据进行kmer嵌入，获取每个Kmer的嵌入向量，再Kmer的嵌入向量->fragment的嵌入向量->Contig的嵌入向量。
            Kmer嵌入向量（kmers*hidden)-> Fragment嵌入向量(1*hidden)
                由BERTPooler类实现，有多种pooling策略：MAX,CLS,MEAN
            （self.data 为Fragment嵌入向量）
            Fragment嵌入向量-> Contig嵌入向量（关键）
                需要先获取所有的Fragment嵌入向量，再通过拼接策略，拼接为Contig的嵌入向量
                拼接策略：
                    1. 横向拓展策略[Fragment1 Fragment2 ....Fragmentm padding] (1*d)
                        按照Fragment在切割前的顺序做横向拼接，拼接成固定维度d的一维向量，需要确定合适的向量维度d 以至于平衡舍弃的Fragment和padding
                    2. 先纵向拓展再自乘A = [[Fragment1],[Fragment2],...,[Fragmentm]] R = A^T*A(hidden*hidden)
                        按照Fragment在切割前的顺序做纵向拼接，无需padding，需要设计合理的聚类算法对矩阵进行聚类。
                        存在时空消耗过大的问题
                    3. 纵向拓展策略[[Fragment1],[Fragment2],...,[Fragmentm],padding] (d*hidden)
                        按照Fragment在切割前的顺序做纵向拼接，拼接成d*hidden的矩阵，需要确定d,以及设计合理的聚类算法对矩阵进行聚类。
                        存在时空消耗过大的问题
        @param:
            self.embed :list of fragment embeds
            self.seq_names: list of fragment name
            self.labels: list of fragment describe
        @return:
            self.embed: list of contig embeds
            self.labels: list of contig describe
        '''
        #将相同seq_names的序列按照上述策略进行拼接
        contig_embed = defaultdict(list)
        for n,embed in zip(self.seq_names,self.embed):
            embed = list(embed)
            if n in contig_embed.keys():
                if len(contig_embed[n])>=max_fragment_num*self.hidden:
                    continue
                else:
                    contig_embed[n].extend(embed)
            else:
                contig_embed[n] = embed
        self.embed.clear()
        self.labels.clear()
        for n,embed in contig_embed.items():
            if len(contig_embed[n]) < max_fragment_num*self.hidden:
                arr = [0 for _ in range(max_fragment_num*self.hidden - len(contig_embed[n]))]
                embed.extend(arr)
            assert len(embed) == max_fragment_num*self.hidden
            self.embed.append(embed)
            self.labels.append(self.seq_name2label[n])

    def reduction(self, dim=32, method="UMAP"):
        '''
            @msg:利用降维方法对数据进行降维
            @param:
                self.embed
            @return:
                self.reduced_embed
        '''
        npdata = np.array(self.embed)
        samples, ndim = npdata.shape
        npdata = npdata.astype('float32')

        if method == "isomap":
            print("\tUsing Isomap reduction!")
            isomap = Isomap(n_neighbors=20, n_components=dim)
            npdata = isomap.fit_transform(npdata)
            npdata = np.ascontiguousarray(npdata)
            npdata = npdata.astype('float32')
        else:
            print("\tUsing DensMap reduction!")
            reducer = umap.UMAP(densmap=True, n_components=dim)
            npdata = reducer.fit_transform(npdata)
            npdata = np.ascontiguousarray(npdata)
            npdata = npdata.astype('float32')

        # L2 normalization
            row_sums = np.linalg.norm(npdata, axis=1)
            npdata = npdata / row_sums[:, np.newaxis]
        
        self.embed = npdata

    def get_pred2true_label_dict(self,pred_labels):
        pred2true_label_dict = defaultdict(list)
        for p, t in zip(pred_labels, self.labels):
            pred2true_label_dict[p].append(t)
        return pred2true_label_dict

    def cac_precision_recall(self, pred2true_label_dict, n):
        """
        @msg:
            计算每个聚簇的准确率和计算每个聚簇的completeness(recall)
            定义：recall = (预测为1且正确预测的样本数)/ (所有真实情况为1的样本数) 
            定义：precision =（预测为1且正确预测的样本数）/(所有预测为1的样本数)
        @param:
            pred_labels:list，根据self.embed的顺序得到的预测labels
        @return:
            each_cluster_pre_rec:根据每个预测的label（cluster)，得到每个cluster的precision和recall指标
                {
                    key(predict cluster):[precision,recall]
                }

        """
        # 给每个聚类簇分配一个类别，这个类别是在该簇中出现次数最多
        true_counter = dict(Counter(self.labels)) # 每个真实label的样本数量
        for k,v in true_counter.items():
            true_counter[k] = int(v)

        corr = total_pred_sample = total_true_sample = 0

        each_cluster_pre_rec = defaultdict(list)

        for k, v in pred2true_label_dict.items():
            label_counter_ = dict(Counter(v))  # list[(key,cnt)]
    
            sort_label_counter_ = dict(sorted(label_counter_.items(), key=lambda x: x[1], reverse=True))

            index = 0
            # 取Top n作为命中点
            for k_,v_ in sort_label_counter_.items():
                if index < n:
                    corr += v_
                    # 真实label为__[i][0]的所有样本数量和
                    total_true_sample += true_counter[k_]
                total_pred_sample += v_ # 预测的label为k的所有样本数量和

            each_cluster_pre_rec[k] = [
                round(corr/total_pred_sample, 2), round(corr/total_true_sample)]

        return each_cluster_pre_rec

    def cac_NC_genome(self, pred_labels: list = None,top_n:int = 5):
        '''
        @msg:
            NC genome的定义为：(>90% recall and >95% precision的聚簇），计算聚类的NC数目作为最终评判指标
        @param:
        @return:
        '''
        pred2true_label_dict = self.get_pred2true_label_dict(pred_labels)
        each_cluster_pre_rec = self.cac_precision_recall(pred2true_label_dict,top_n)

        NCgene_cnt = 0
        preformance = defaultdict(int)

        for k, nc in each_cluster_pre_rec.items():
            pre, rec = nc[0], nc[1]
            if pre >= 0.95 and rec >= 0.9:
                NCgene_cnt += 1
                preformance["level1"] += 1
            elif pre>0.85 and rec >=0.8:
                preformance["level2"] += 1
            elif pre>0.75 and rec >=0.7:
                preformance["level3"] += 1   
            elif pre>0.65 and rec >=0.6:
                preformance["level3"] += 1   
            elif pre>0.55 and rec >=0.5:
                preformance["level3"] += 1 
            else:
                preformance["level4" ] +=1
            

        return NCgene_cnt,preformance

    def get_rand_index_and_f_measure(self,pred_labels, beta=1.):
        (tn, fp), (fn, tp) = pair_confusion_matrix(self.labels, pred_labels)
        ri = (tp + tn) / (tp + tn + fp + fn)
        ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        p, r = tp / (tp + fp), tp / (tp + fn)
        f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
        purity = tp/(tp+fp)
        recall = tp/(tp+fn)

        return purity,recall,ri, ari, f_beta

    def kmeans(self, k):
        '''
        @msg:
            对embed采用kmeans方法聚类
        @param:
            k:聚簇数目
        @return:
            pred_labels: list 聚类结果
        '''
        X = StandardScaler().fit_transform(self.embed)
        cl = cluster.MiniBatchKMeans(n_clusters=k).fit(X)
        return cl.labels_

    def iter_cluster(self):
        pass

    def cluster_data(self, k: int = 0, top_n: int = 5,dim = 32,reduced_method = 'umap'):
        '''
        @msg:
            按照配置对数据进行聚类
        @param:
        @return:
            NC genome的数量
            每个预测的聚类（genome)的precision和recall指标
        '''
        if self.concat_mode:
            self.data_concat()
            print("\tThere have {} samples".format(len(self.embed)))
      
        ndim = len(self.embed[0])

        if ndim > 32:
            self.reduction(dim=dim,method=reduced_method)
            print("\tThe dim from {} reduced to  {}".format(ndim,self.embed.shape[1]))

        if self.cluster_method == "kmeans":
            pred_labels = self.kmeans(k)

        elif self.cluster_method == "iter":
            pred_labels = self.iter_cluster()
            pass

        else:
            pass

        NCgene_cnt,preformance  = self.cac_NC_genome(
                pred_labels, top_n)
        purity,recall,ri, ari, f_measure  = self.get_rand_index_and_f_measure(pred_labels)
        print("NC genome:", NCgene_cnt)
        print("NC genome rate :",NCgene_cnt/len(pred_labels))
        print("Purity :",purity)
        print("Recall :",recall)
        print("RI :", ri)
        print("ARI ：" ,ari)
        print("F_measure :",f_measure)
        print(preformance)

        return  NCgene_cnt,preformance 
    
def get_true_label_dict(path):
    labels = {}
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            _ = line.split('\t')
            labels[_[0]] = _[1]
    return labels

def test():
    vocab_path = "/workspace/1026.vocab"
    data_path = "/workspace/vamb-data/gi/contigs.fa.4mer"
    ff = FaFrequency(None,data_path,vocab_path)
    label_path = "/workspace/vamb-data/all-reference.tsv"
    
    labels = get_true_label_dict(label_path)

    cl = Cluster(data=ff.data, nfiles=0, data_path=None,true_labels=labels,
                 file_key="frag_embed",  concat_mode=0, cluster_method="kmeans", with_cude=False)

    NCgene_cnt,preformance  = cl.cluster_data(k=250,top_n=1,dim=32,reduced_method='umap')

if __name__ == "__main__":
    test()
