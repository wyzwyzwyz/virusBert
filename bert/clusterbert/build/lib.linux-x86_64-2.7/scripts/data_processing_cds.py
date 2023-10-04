import os
import random
import numpy as np
from scipy import sparse as sp
from tqdm import tqdm
from collections import Counter
import json


def load_data(data_path):
    """
    [[msg, seq],
     [msg, seq],
     ...
     [msg, seq]]
    """
    def del_None_host(genome_seq_):

        new_genome_seq_ = []
        for i, one in enumerate(genome_seq_):
            # get all host lineages
            host_lineages_ = one[0].split('|')[4]

            if host_lineages_=='':
                continue

            new_genome_seq_.append(one)

        return new_genome_seq_

    genome_seq_list_ = []

    with open(data_path + 'virushostdb.formatted.cds.fna', 'r') as f:
        print('----load genome file with not None----')

        lines = f.readlines()

        one_msg = []
        seq_msg = ''

        for line in tqdm(lines):

            if line[0] == '>':

                if seq_msg != '':
                    one_msg.append(seq_msg)  # 若当前seq_msg不为空，则就是已读取完完整序列
                    genome_seq_list_.append(one_msg)

                    seq_msg = ''  # 为下次序列做准备
                    one_msg = []

                id_msg = line.strip('\n')  # 除去首尾回车符号

                # collection
                one_msg.append(id_msg)
            else:
                seq_msg += line.strip('\n')

        # 上面循环无法收集最后一条数据
        one_msg.append(seq_msg)
        genome_seq_list_.append(one_msg)

    dn_ps = del_None_host(genome_seq_list_)
    print('there are', len(dn_ps), 'virus cds of genome seqs !')
    return dn_ps


def load_virus_host_msg(genome_seq_):
    """
    split information for the whole virus

    [[msg, seq],
     [msg, seq],
     ...
     [msg, seq]]

    :param genome_seq_: all cds genome seq
    :return new_genome_seq_: [virus name, virus lineage, host lineage, cds genome]
    :return list(virus_msg_set_): (virus name, virus lineage, host lineage)
    :return cds_num_list_: (virus name, cds) distribution
    """

    def get_host_msg(seq_host_msg_):
        seq_list_ = seq_host_msg_.split('|')
        virus_name_ = ' '.join(seq_list_[0].split(' ')[1:])

        virus_lineage_ = seq_list_[3].split('; ')
        all_virus_lineage_ = ';'.join(virus_lineage_)
        host_lineage_ = seq_list_[4].split('; ')
        all_host_lineage_ = ';'.join(host_lineage_)
        # refseq_id_ = seq_list_[-3]

        return virus_name_, all_virus_lineage_, all_host_lineage_

    print('----load virus & host msg----')
    virus_msg_set_ = set()
    virus_name_list_ = []
    new_genome_seq_ = []

    for ps in genome_seq_:
        msg = get_host_msg(ps[0])   # get msg of one cds
        virus_msg_set_.add(msg)     # while msg
        virus_name_list_.append(msg[0]) # only virus name

        # [virus name, virus lineage, host lineage, cds genome]
        new_genome_seq_.append(list(msg)+[ps[1]])   # [ps[1]] means cds genome sequence

    # 计算各个病毒的CDS个数 (存在个别CDS识别host，与同病毒的其他CDS不同，识别host lineage不够完整or该段CDS就是无法识别完整host)
    # cds片段不是连接在一起，应该按病毒名字统计
    cds_num_list_ = list(Counter(virus_name_list_).items())     # [(virus name: cds num)]
    # cds_num_list_ = sorted(cds_num_list_, key=lambda cds_num: cds_num[1], reverse=True)

    assert len(cds_num_list_) == len(set(virus_name_list_))

    print('there are', len(set(virus_name_list_)), 'different virus seqs !', len(virus_name_list_))
    return deal_same_name(new_genome_seq_), list(virus_msg_set_), cds_num_list_


def deal_same_name(hosts_seqs, host_flag=True):
    """
    find same name host in one lineage, and mark them different

    e.g.
        Anopheles, Anopheles_1, Anopheles_2
    """

    print('----deal same name----')
    new_hosts_seq = []
    same_name_set = set()

    for k, one in enumerate(tqdm(hosts_seqs)):
        if host_flag:
            lineage = one[2]    # host lineage
        else:
            lineage = one[1]    # virus lineage

        members = lineage.split(';')

        # find same name host's index
        index = [j for j, x in enumerate(members) if members.count(x) > 1]

        # change its name
        for i, x in enumerate(index):
            if i > 0:
                # the second same host to be host_1
                members[x] += '_' + str(i)
                same_name_set.add(members[x])

        new_lineage = ';'.join(members)     # restore lineage
        if host_flag:
            new_hosts_seq.append((one[0], one[1], new_lineage, one[3]))
        else:
            new_hosts_seq.append((one[0], new_lineage, one[2], one[3]))

    print('same name num =', len(same_name_set), same_name_set)

    return new_hosts_seq


def find_all_host_type(genome_seq_, save_path_='./', save_flag=True):
    """
    all hosts
    {
        host: host taxonomy type,
        ...
    }
    """

    def get_member_for_encode(hosts_list_):
        """
        add lineages in same list (just need different host name)
        """

        hosts_encode_ = []
        for lineage in hosts_list_:
            members = lineage.split(';')
            hosts_encode_ += members
        return list(set(hosts_encode_))

    # load known host type
    known_host_type_ = {}
    with open('/home/hongweichen/Data/vhp_data/host_lineage_type.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_data = line.strip('\n').split('|')
            if line_data[1] =='no rank':
                continue
            known_host_type_[line_data[0]] = line_data[1]
    print('all known host type num =', len(known_host_type_))

    # get host lineage
    host_list_ = [ps[2] for ps in genome_seq_]
    all_hosts_ = get_member_for_encode(host_list_)
    # print('different host num =', len(all_hosts_))

    # find all host's type
    all_host_type_list = []
    unknown_host_type = []
    for al in all_hosts_:
        if al in known_host_type_.keys():
            all_host_type_list.append((al, known_host_type_[al])) # [(host, host_type),...]
        else:
            unknown_host_type.append(al)

    if len(set(unknown_host_type))>0:
        print('unknown_host_type:', set(unknown_host_type))
        print('unknown_host_type len =:', len(set(unknown_host_type)))
    else:
        print('no unknown')

    all_host_type_set = set(all_host_type_list)
    # print(len(all_host_type_set))

    all_host_type_dict = {}
    for aht in all_host_type_set:
        all_host_type_dict[aht[0]] = aht[1]
    # print(len(all_host_type_dict))

    if save_flag:

        if not os.path.exists(save_path_):
            os.makedirs(save_path_)
        with open(save_path_+'all_host_type.json', 'w') as f:
            str_tree_dict = str(json.dumps(all_host_type_dict))[1:-1]
            value = str_tree_dict.split(', ')

            f.writelines('{\n')
            for i in range(len(value) - 1):
                f.writelines('\t' + value[i] + ',' + '\n')
            f.writelines('\t' + value[-1] + '\n')

            f.writelines('}\n')


def type_host_mapping(host_type_path_, label_map_path_=None, save_path_='./'):

    def save_json(file_name_, data_dict_):
        if not os.path.exists(save_path_):
            os.makedirs(save_path_)

        # save {key: [v1, v2], ...}
        with open(save_path_ + file_name_, 'w') as f:
            str_tree_dict = str(json.dumps(data_dict_))[1:-1]
            value = str_tree_dict.split('], ')

            f.writelines('{\n')
            for i in range(len(value) - 1):
                f.writelines('\t' + value[i] + '],' + '\n')
            f.writelines('\t' + value[-1] + '\n')

            f.writelines('}\n')

    all_host_type_dict_ = load_json(host_type_path_)

    type_host_ = {}
    for v in set(all_host_type_dict_.values()):
        type_host_[v] = []

    for k in list(all_host_type_dict_.keys()):
        type_host_[all_host_type_dict_[k]].append(k)

    save_json('all_type_host.json', type_host_)

    # saving mapping
    if label_map_path_ is not None:
        label_map_dict_ = load_json(label_map_path_)

        all_type_host_mapping_ = {}
        for ath in type_host_.keys():
            all_type_host_mapping_[ath] = []
            for host_ in type_host_[ath]:
                all_type_host_mapping_[ath].append(label_map_dict_[host_])
        save_json('all_type_host_mapping.json', all_type_host_mapping_)


def save_data_by_cds(genome_seq_, save_path_='./', path_='./', using_degenerate_=True):
    """
        e.g.
        virus name|virus lineage|host lineage
        all [CDS]
    """
    degenerate_base_dict = load_json(path_+'dna_code/degenerate_bases.json')

    pre_virus_name_set_ = set()  # get first virus name
    virus_host_lineage_ = []

    for i, gs_ in enumerate(genome_seq_):
        new_seq = gs_[-1]
        if using_degenerate_:
            # 执行简并碱基：随机选取
            new_seq = deal_with_one_seq(degenerate_base_dict, gs_[-1])
        new_seq = new_seq.upper()

        if gs_[0] not in pre_virus_name_set_:
            pre_virus_name_set_.add(gs_[0])     # never see this virus, add it

            cds_seqs_ = [new_seq]       # collect cds
            virus_host_lineage_.append((gs_[0], gs_[1], gs_[2], cds_seqs_))
        else:
            for vhl_ in virus_host_lineage_:
                if vhl_[0] == gs_[0]:
                    vhl_[-1].append(new_seq)

    assert len(virus_host_lineage_) == 10915

    with open(save_path_ + 'all_msg_cds.txt', 'w') as f:
        for vhl_ in virus_host_lineage_:
            f.write('>|' + '|'.join(vhl_[:-1]) + '\n')
            for seq_ in vhl_[-1]:
                f.write(seq_+'\n')

    print('----save done----')


def load_data_by_cds(path_):
    virus_host_lineage_ = []

    print('----loading all_msg_cds----')
    with open(path_ + 'all_msg_cds.txt', 'r') as f:
        lines = f.readlines()

        i = -1
        for line in lines:

            line_data = line.strip('\n')
            if line_data.split('|')[0] == '>':
                split_data = line_data.split('|')[1:]
                virus_host_lineage_.append((split_data[0], split_data[1], split_data[2], []))
                i += 1
            else:
                virus_host_lineage_[i][-1].append(line_data)

    assert len(virus_host_lineage_) == 10915

    return virus_host_lineage_


def load_json(file_name_):
    with open(file_name_, 'r') as f:
        json_ = json.load(f)
    return json_


def deal_with_one_seq(degenerate_base_dict_, seq_one_):
    # 简并碱基
    seq_one_list = list(seq_one_)
    for i, char_ in enumerate(seq_one_list):
        if char_ in degenerate_base_dict_.keys():
            randomly_select = random.sample(degenerate_base_dict_[char_], 1)[0]
            # print(randomly_select)
            seq_one_list[i] = randomly_select
    return ''.join(seq_one_list)


def get_label_by_taxonomy(virus_cds_, type_, save_path_='./', save_flag=False):
    """
    class: Mammalia
        order: Primates, Rodentia, Lagomorpha, Scandentia, Dermoptera,
               Carnivora, Perissodactyla, Artiodactyla, Eulipotyphla, Chiroptera, Pholidota,
               Proboscidea, Sirenia,
               Cingulata, Pilosa,
               Didelphimorphia,
               Peramelemorphia, Diprotodontia

    order: Primates
        family: Hominidae, Hylobatidae
        genus: Homo, Pan
    """
    order_list = ["Cingulata", "Bonnemaisoniales", "Bdellovibrionales", "Mesostigmata", "Procellariiformes", "Rodentia", "Perciformes", "Rhabditida", "Diptera", "Caudovirales", "Veneroida", "Charadriiformes", "Apiales", "Micromonosporales", "Gruiformes", "Perissodactyla", "Didelphimorphia", "Ericales", "Acidithiobacillales", "Psittaciformes", "Phaeocystales", "Lepetellida", "Saccharomycetales", "Chlamydiales", "Polyporales", "Ustilaginales", "Aplysiida", "Peronosporales", "Ophiostomatales", "Streptomycetales", "Acholeplasmatales", "Magnaporthales", "Oxalidales", "Hymenoptera", "Alcyonacea", "Laurales", "Celastrales", "Aeromonadales", "Pezizales", "Esociformes", "Malpighiales", "Rhodobacterales", "Brassicales", "Isochrysidales", "Architaenioglossa", "Zingiberales", "Gentianales", "Euamoebida", "Carcharhiniformes", "Anabantiformes", "Botryosphaeriales", "Anura", "Caudata", "Pedunculata", "Bicosoecida", "Burkholderiales", "Eulipotyphla", "Thermoanaerobacterales", "Sphingomonadales", "Rhizobiales", "Orthoptera", "Chiroptera", "Trichomonadida", "Cantharellales", "Erysipelotrichales", "Calanoida", "Unionida", "Temnopleuroida", "Glomerales", "Prymnesiales", "Lobata", "Eurotiales", "Thysanoptera", "Pectinoida", "Poecilosclerida", "Helicobasidiales", "Araneae", "Bacteroidetes Order II. Incertae sedis", "Rhizosoleniales", "Thelephorales", "Capnodiales", "Littorinimorpha", "Chattonellales", "Rhodospirillales", "Commelinales", "Pinales", "Caulobacterales", "Longamoebia", "Amphipoda", "Stylommatophora", "Hemiptera", "Arecales", "Chlamydomonadales", "Nitrososphaerales", "Liliales", "Cichliformes", "Alismatales", "Haplotaxida", "Sphenisciformes", "Lophiiformes", "Sapindales", "Thermococcales", "Lepidoptera", "Fusobacteriales", "Haloferacales", "Mermithida", "Pilosa", "Mycoplasmatales", "Pelecaniformes", "Entomoplasmatales", "Methanobacteriales", "Carnivora", "Diplomonadida", "Galliformes", "Pristiformes/Rhiniformes group", "Poales", "Neuroptera", "Rosales", "Corynebacteriales", "Siphonaptera", "Xylariales", "Mytiloida", "Halobacteriales", "Thermales", "Blattodea", "Dilleniales", "Falconiformes", "Chlorellales", "Laminariales", "Xanthomonadales", "Bacteroidales", "Thraustochytrida", "Clostridiales", "Anguilliformes", "Pasteurellales", "Hypocreales", "Cypriniformes", "Diprotodontia", "Chroococcales", "Fragilariales", "Flavobacteriales", "Enterobacterales", "Salmoniformes", "Primates", "Sabellida", "Lactobacillales", "Pleuronectiformes", "Diaporthales", "Vitales", "Pucciniales", "Scutigeromorpha", "Ciconiiformes", "Pelagomonadales", "Micrococcales", "Solanales", "Thermoproteales", "Synechococcales", "Peramelemorphia", "Actiniaria", "Phyllodocida", "Lagomorpha", "Myxococcales", "Sulfolobales", "Coleoptera", "Struthioniformes", "Bacillales", "Accipitriformes", "Erysiphales", "Peridiniales", "Ectocarpales", "Siluriformes", "Sirenia", "Testudines", "Pseudomonadales", "Fagales", "Russulales", "Diplostraca", "Dioscoreales", "Oscillatoriales", "Trypanosomatida", "Pelagibacterales", "Glomerellales", "Gadiformes", "Desulfurococcales", "Natrialbales", "Geraniales", "Aquificales", "Helotiales", "Cystofilobasidiales", "Lamiales", "Scandentia", "Cucurbitales", "Enterogona", "Apterygiformes", "Malvales", "Peniculida", "Spariformes", "Myrtales", "Dipsacales", "Nostocales", "Neogastropoda", "Ranunculales", "Neisseriales", "Octopoda", "Chaetocerotales", "Aquifoliales", "Cornales", "Mamiellales", "Oceanospirillales", "Microascales", "Columbiformes", "Carangiformes", "Proboscidea", "Coraciiformes", "Tetraodontiformes", "Ixodida", "Saxifragales", "Vibrionales", "Pleosporales", "Anseriformes", "Polypodiales", "Eucoccidiorida", "Isopoda", "Passeriformes", "Chlorodendrales", "Tricladida", "Actinomycetales", "Squamata", "Artiodactyla", "Propionibacteriales", "Caryophyllales", "Campylobacterales", "Piperales", "Alteromonadales", "Siphonostomatoida", "Pseudonocardiales", "Pythiales", "Tinamiformes", "Diversisporales", "Leptospirales", "Crocodylia", "Asparagales", "Acipenseriformes", "Centrarchiformes", "Ostreoida", "Trichosphaeriales", "Asterales", "Sessilia", "Odonata", "Fabales", "Scolopendromorpha", "Agaricales", "Decapoda"]
    family_list = ["Scarabaeidae", "Peramelidae", "Thymelaeaceae", "Bdellovibrionaceae", "Actinidiaceae", "Tortricidae", "Lauraceae", "Primnoidae", "Crambidae", "Paramuriceidae", "Pythonidae", "Polygalaceae", "Vespertilionidae", "Acetobacteraceae", "Rhynchobatidae", "Viverridae", "Membracidae", "Cyclopteridae", "Nitrososphaeraceae", "Chromobacteriaceae", "Microcoleaceae", "Agelenidae", "Hyacinthaceae", "Cryptobranchidae", "Galagidae", "Acidithiobacillaceae", "Grossulariaceae", "Streptococcaceae", "Uloboridae", "Terapontidae", "Actinomycetaceae", "Parameciidae", "Molossidae", "Berberidaceae", "Equidae", "Cricetidae", "Delphinidae", "Tannerellaceae", "Hesperiidae", "Amaranthaceae", "Channidae", "Procyonidae", "Zosteraceae", "Artamidae", "Alydidae", "Ectobiidae", "Bradybaenidae", "Typhulaceae", "Asparagaceae", "Struthionidae", "Suidae", "Tetraodontidae", "Dilleniaceae", "Mytilidae", "Cacatuidae", "Hepialidae", "Scutigeridae", "Halorubraceae", "Littorinidae", "Urticaceae", "Aurantimonadaceae", "Cheloniidae", "Geometridae", "Cladosporiaceae", "Chironomidae", "Centrarchidae", "Aspergillaceae", "Estrildidae", "Onagraceae", "Notodontidae", "Euphorbiaceae", "Haloferacaceae", "Solanaceae", "Herpestidae", "Lymantriidae", "Mustelidae", "Scolopendridae", "Aeromonadaceae", "Cyperaceae", "Brucellaceae", "Pycnonotidae", "Serranidae", "Hipposideridae", "Sparidae", "Apidae", "Rhinolophidae", "Odobenidae", "Canidae", "Halobacteriaceae", "Sinipercidae", "Stercorariidae", "Ebenaceae", "Phaeocystaceae", "Balanidae", "Ophiocordycipitaceae", "Castoridae", "Pontellidae", "Aplysiidae", "Megascolecidae", "Nesomyidae", "Piperaceae", "Aquifoliaceae", "Camelidae", "Ceratocystidaceae", "Convolvulaceae", "Thermaceae", "Araneidae", "Rhizosoleniaceae", "Amphibolidae", "Moraxellaceae", "Fusobacteriaceae", "Planorbidae", "Philodromidae", "Reduviidae", "Colwelliaceae", "Mermithidae", "Prochloraceae", "Hydrocharitaceae", "Parastacidae", "Musaceae", "Sicyoniidae", "Turritellidae", "Toxopneustidae", "Comamonadaceae", "Bovidae", "Thermoproteaceae", "Paridae", "Apocynaceae", "Dioscoreaceae", "Caryophyllaceae", "Andrenidae", "Physalacriaceae", "Siboglinidae", "Alcaligenaceae", "Araceae", "Corynebacteriaceae", "Heteroderidae", "Sphingomonadaceae", "Ichneumonidae", "Streptomycetaceae", "Esocidae", "Formicariidae", "Thelephoraceae", "Psittacidae", "Aiptasiidae", "Juglandaceae", "Amaryllidaceae", "Bathycoccaceae", "Papilionidae", "Lasiocampidae", "Erwiniaceae", "Indriidae", "Lactobacillaceae", "Mrakiaceae", "Ampullariidae", "Vitaceae", "Accipitridae", "Falconidae", "Bondarzewiaceae", "Tupaiidae", "Enterococcaceae", "Bolinopsidae", "Peptostreptococcaceae", "Xylariaceae", "Lessoniaceae", "Lamiaceae", "Basellaceae", "Pimoidae", "Pseudomonadaceae", "Myocastoridae", "Chenopodiaceae", "Verbenaceae", "Agaricaceae", "Enterobacteriaceae", "Rhodothermaceae", "Spheniscidae", "Sulfolobaceae", "Nototheniidae", "Rutaceae", "Cercopithecidae", "Noctuidae", "Tipulidae", "Giraffidae", "Ambystomatidae", "Spiroplasmataceae", "Shewanellaceae", "Geminiviridae", "Drosophilidae", "Erethizontidae", "Scophthalmidae", "Megalonychidae", "Ericaceae", "Tateidae", "Sclerotiniaceae", "Megachilidae", "Thraustochytriaceae", "Cryphonectriaceae", "Trichechidae", "Myobatrachidae", "Chlamyphoridae", "Caviidae", "Apterygidae", "Ursidae", "Mesodesmatidae", "Psittaculidae", "Antilocapridae", "Nectriaceae", "Cebidae", "Poaceae", "Ranidae", "Moraceae", "Melanthiaceae", "Hypericaceae", "Heterocapsaceae", "Thripidae", "Blattidae", "Ligiidae", "Pipidae", "Leptospiraceae", "Gigasporaceae", "Liliaceae", "Burkholderiaceae", "Alteromonadaceae", "Scincidae", "Aeshnidae", "Ascarididae", "Unionidae", "Betulaceae", "Acanthaceae", "Araliaceae", "Glossinidae", "Malvaceae", "Tinamidae", "Microcystaceae", "Vespidae", "Mycoplasmataceae", "Pteromalidae", "Primulaceae", "Simuliidae", "Acipenseridae", "Braconidae", "Chaetocerotaceae", "Chaoboridae", "Dunaliellaceae", "Liviidae", "Adoxaceae", "Chrysomelidae", "Cordycipitaceae", "Corduliidae", "Propionibacteriaceae", "Phocidae", "Synechococcaceae", "Flavobacteriaceae", "Muricidae", "Ploceidae", "Pteropodidae", "Gordoniaceae", "Lorisidae", "Pyriculariaceae", "Eimeriidae", "Botryosphaeriaceae", "Nephilidae", "Bonnemaisoniaceae", "Saturniidae", "Nostocaceae", "Libellulidae", "Polemoniaceae", "Carangidae", "Oscillatoriaceae", "Plectosphaerellaceae", "Rhinonycteridae", "Leptosphaeriaceae", "Micromonosporaceae", "Ranunculaceae", "Acholeplasmataceae", "Pseudonocardiaceae", "Pyralidae", "Cichlidae", "Otariidae", "Glomerellaceae", "Trichomonadidae", "Salmonidae", "Lycaenidae", "Gliridae", "Procellariidae", "Veneridae", "Mycobacteriaceae", "Cimicidae", "Helicobasidiaceae", "Cryptosporidiidae", "Prymnesiaceae", "Haliotidae", "Rhodobacteraceae", "Dryopteridaceae", "Emballonuridae", "Bacteroidaceae", "Gruidae", "Asteraceae", "Armadillidiidae", "Dicroglossidae", "Anguillidae", "Rosaceae", "Tettigoniidae", "Cafeteriaceae", "Sphingidae", "Ostreidae", "Phyllanthaceae", "Methanobacteriaceae", "Pleurotaceae", "Gentianaceae", "Tellinidae", "Agamidae", "Gammaridae", "Orobanchaceae", "Leptolyngbyaceae", "Polygonaceae", "Soleidae", "Gnomoniaceae", "Scalpellidae", "Hafniaceae", "Echimyidae", "Oxalidaceae", "Miridae", "Bromeliaceae", "Gadidae", "Erysiphaceae", "Bombycidae", "Acrididae", "Bacillaceae", "Mormoopidae", "Formicidae", "Vibrionaceae", "Potoroidae", "Viperidae", "Ictaluridae", "Saccharomycetaceae", "Chlorellaceae", "Debaryomycetaceae", "Chattonellaceae", "Delphacidae", "Elapidae", "Bufonidae", "Siluridae", "Cactaceae", "Didelphidae", "Varunidae", "Staphylococcaceae", "Aotidae", "Portunidae", "Boidae", "Capparaceae", "Sturnidae", "Ixodidae", "Octopodidae", "Nymphalidae", "Geraniaceae", "Cybaeidae", "Muscidae", "Acanthizidae", "Mycosphaerellaceae", "Cucurbitaceae", "Bradypodidae", "Pectobacteriaceae", "Theridiidae", "Oleaceae", "Zingiberaceae", "Cleomaceae", "Haloarculaceae", "Turdidae", "Phytolaccaceae", "Diogenidae", "Soricidae", "Valsaceae", "Aleyrodidae", "Alstroemeriaceae", "Curculionidae", "Catostomidae", "Sapindaceae", "Papaveraceae", "Commelinaceae", "Oceanospirillaceae", "Tricholomataceae", "Theaceae", "Proscylliidae", "Crassulaceae", "Corvidae", "Caricaceae", "Limacodidae", "Limeaceae", "Pasteurellaceae", "Orchidaceae", "Pseudeurotiaceae", "Clostridiaceae", "Trichosphaeriaceae", "Chlamydiaceae", "Coelopidae", "Leporidae", "Talpidae", "Ciconiidae", "Pleuronectidae", "Campanulaceae", "Muridae", "Ectocarpaceae", "Juncaceae", "Phascolarctidae", "Fringillidae", "Halomonadaceae", "Rhabditidae", "Moridae", "Sciaenidae", "Morganellaceae", "Cannaceae", "Phyllobacteriaceae", "Myoviridae", "Iridaceae", "Noelaerhabdaceae", "Acinetosporaceae", "Nyctaginaceae", "Thermococcaceae", "Aquificaceae", "Gelechiidae", "Amoebidae", "Petroicidae", "Chlorodendraceae", "Phocoenidae", "Trionychidae", "Paralichthyidae", "Phalangeridae", "Didemnidae", "Cervidae", "Microcionidae", "Sciuridae", "Elephantidae", "Plumbaginaceae", "Tsukamurellaceae", "Rhamnaceae", "Teiidae", "Ustilaginaceae", "Erinaceidae", "Tropaeolaceae", "Campylobacteraceae", "Didymosphaeriaceae", "Pentatomidae", "Pelagibacteraceae", "Hirundinidae", "Tetragnathidae", "Percidae", "Ailuridae", "Helicobacteraceae", "Felidae", "Arecaceae", "Muscicapidae", "Antennariidae", "Tuberaceae", "Adenoviridae", "Rhodospirillaceae", "Acanthamoebidae", "Idiomarinaceae", "Coraciidae", "Columbidae", "Micrococcaceae", "Ophiostomataceae", "Crocodylidae", "Hyaenidae", "Scrophulariaceae", "Diprionidae", "Pieridae", "Listeriaceae", "Celastraceae", "Pinaceae", "Clavicipitaceae", "Pleosporaceae", "Intrasporangiaceae", "Varroidae", "Aphididae", "Penaeidae", "Caligidae", "Pulicidae", "Leuconostocaceae", "Rallidae", "Microbacteriaceae", "Helodermatidae", "Thermoanaerobacterales Family III. Incertae Sedis", "Fragilariaceae", "Ocypodidae", "Isochrysidaceae", "Gekkonidae", "Linyphiidae", "Cyprinidae", "Erysipelotrichaceae", "Laridae", "Brassicaceae", "Hypocreaceae", "Pharidae", "Caulobacteraceae", "Trypanosomatidae", "Sphaeriidae", "Acartiidae", "Desulfurococcaceae", "Hexamitidae", "Salamandridae", "Xanthomonadaceae", "Sesarmidae", "Helotiaceae", "Figitidae", "Psychodidae", "Schizophyllaceae", "Calliphoridae", "Phasianidae", "Segestriidae", "Rubiaceae", "Mamiellaceae", "Nocardiaceae", "Fabaceae", "Rhizobiaceae", "Anatidae", "Pseudococcidae", "Natrialbaceae", "Mephitidae", "Testudinidae", "Hyriidae", "Lemuridae", "Mecicobothriidae", "Pseudoalteromonadaceae", "Ceratopogonidae", "Phyllostomidae", "Monodontidae", "Coenagrionidae", "Yersiniaceae", "Cicadellidae", "Diaporthaceae", "Gryllidae", "Dugesiidae", "Plantaginaceae", "Culicidae", "Hydrangeaceae", "Balsaminaceae", "Cronartiaceae", "Apiaceae", "Cannabaceae", "Phanerochaetaceae", "Nycteribiidae", "Argasidae", "Paenibacillaceae", "Pythiaceae", "Palaemonidae", "Myxococcaceae", "Pectinidae", "Peronosporaceae", "Glomeraceae", "Ceratobasidiaceae", "Hominidae", "Caprifoliaceae", "Atelidae", "Sporolactobacillaceae", "Plutellidae", "Passifloraceae", "Scolopacidae", "Centropomidae"]
    genus_list = ["Providencia", "Kobus", "Capsicum", "Delphinapterus", "Paralichthys", "Andrias", "Steatoda", "Datura", "Poecile", "Mesocricetus", "Torulaspora", "Papio", "Ecklonia", "Echyridella", "Taphozous", "Cyrtanthus", "Francolinus", "Agaricus", "Acerodon", "Artemisina", "Opuntia", "Enterococcus", "Salvelinus", "Tuber", "Bathycoccus", "Piper", "Dunaliella", "Dysaphis", "Heterosigma", "Vaccinium", "Passalora", "Myotis", "Homalodisca", "Artibeus", "Betula", "Salinibacter", "Lama", "Cydia", "Verasper", "Pouzolzia", "Asclepias", "Cygnus", "Osmia", "Chrysotila", "Mastadenovirus", "Rhododendron", "Sodiomyces", "Tylonycteris", "Tramea", "Micromys", "Alternaria", "Capparis", "Delftia", "Sapajus", "Pusa", "Cryphonectria", "Erethizon", "Lactuca", "Triatoma", "Sogatella", "Oxalis", "Arctocephalus", "Meleagris", "Scophthalmus", "Rubus", "Adelphocoris", "Phlebiopsis", "Pongo", "Chaetocoelopa", "Lygus", "Petrochirus", "Yersinia", "Bdellovibrio", "Sorex", "Acidithiobacillus", "Saccharum", "Tupaia", "Phlebotomus", "Delisea", "Parasesarma", "Eupatorium", "Festuca", "Tagetes", "Bison", "Dinocampus", "Holcus", "Sechium", "Gerbilliscus", "Leptopilina", "Neoscona", "Agrotis", "Hylocereus", "Aliivibrio", "Adoxophyes", "Riparia", "Sclerotinia", "Saimiri", "Cimex", "Reithrodontomys", "Cercocebus", "Struthio", "Periplaneta", "Ovis", "Bos", "Seriola", "Actinidia", "Girardia", "Procordulia", "Dama", "Eragrostis", "Blicca", "Atkinsonella", "Petroica", "Vermamoeba", "Testudo", "Teleogryllus", "Varecia", "Beta", "Roseobacter", "Rhizobium", "Haemophilus", "Colocasia", "Leptospira", "Rudbeckia", "Vicia", "Spathiphyllum", "Tadarida", "Clitocybe", "Flammulina", "Artemisia", "Scylla", "Marmota", "Epinephelus", "Helicobasidium", "Neurotrichus", "Dracaena", "Gordonia", "Leucauge", "Malvastrum", "Aeropyrum", "Grammomys", "Ficus", "Plantago", "Myotomys", "Hoplobatrachus", "Hippeastrum", "Sorghum", "Petasites", "Parabacteroides", "Echinocystis", "Areca", "Sooretamys", "Trichosurus", "Thalassotalea", "Dimocarpus", "Prochlorococcus", "Varroa", "Oncorhynchus", "Cellulophaga", "Nicotiana", "Helleborus", "Botryosphaeria", "Beauveria", "Sciurus", "Hypericum", "Clitoria", "Croceibacter", "Impatiens", "Juncus", "Oligoryzomys", "Aureococcus", "Imperata", "Planktothrix", "Hyacinthus", "Arabidopsis", "Kluyvera", "Rhodobacter", "Gossypium", "Cyrtomium", "Phytolacca", "Tradescantia", "Centrosema", "Rhipicephalus", "Prunus", "Acheta", "Diuris", "Limnodynastes", "Mizuhopecten", "Diplacodes", "Dendrolimus", "Golovinomyces", "Phodopus", "Medicago", "Chaoborus", "Sulfolobus", "Ailurus", "Clostera", "Eunectes", "Sparus", "Arthrobacter", "Cronobacter", "Bandicota", "Heterodera", "Musculium", "Scrophularia", "Atheris", "Paracoccus", "Helicoverpa", "Gallus", "Notemigonus", "Cronartium", "Garrulus", "Rhinella", "Sinapis", "Burkholderia", "Penicillium", "Pleurotus", "Erythrodiplax", "Kitasatospora", "Sodalis", "Mastigeulota", "Leptomonas", "Synechococcus", "Pycnonotus", "Cordyceps", "Larus", "Sesbania", "Uraeginthus", "Castor", "Apteryx", "Urochloa", "Ceratobasidium", "Equus", "Catopsilia", "Capraria", "Didemnum", "Inachis", "Gluconobacter", "Paenibacillus", "Setothosea", "Rangifer", "Cannabis", "Escherichia", "Oryza", "Coccinia", "Trichechus", "Zea", "Erysiphe", "Somateria", "Daphnis", "Panicum", "Suncus", "Campoletis", "Erigeron", "Chelonia", "Peduovirus", "Cotesia", "Drosophila", "Psophocarpus", "Actinoplanes", "Tiliqua", "Vibrio", "Phaffia", "Antilocapra", "Zostera", "Pinus", "Mycobacteroides", "Indri", "Oeciacus", "Palaemon", "Cucurbita", "Distimake", "Antonospora", "Cafeteria", "Callistephus", "Neodiprion", "Carya", "Nyctereutes", "Pygoscelis", "Mustela", "Aplysia", "Arachis", "Alces", "Spinacia", "Myxococcus", "Phocoena", "Proechimys", "Malachra", "Cracticus", "Morus", "Endoconidiophora", "Enterobacter", "Rhizophagus", "Armadillidium", "Vicugna", "Morelia", "Otolemur", "Digitaria", "Eliurus", "Urbanus", "Psittacus", "Talpa", "Pseudoalteromonas", "Fusobacterium", "Otospermophilus", "Culex", "Haloarcula", "Chrysomya", "Ploceus", "Jacquemontia", "Eucampsipoda", "Salmonella", "Thalassia", "Brucella", "Linepithema", "Philodromus", "Crocidura", "Fritillaria", "Tetrasphaera", "Glaesserella", "Panthera", "Isoodon", "Takifugu", "Ocimum", "Pseudomonas", "Capsella", "Paramecium", "Neofusicoccum", "Clavibacter", "Erysipelothrix", "Ara", "Sigesbeckia", "Campylobacter", "Liatris", "Lophuromys", "Hoya", "Tolypocladium", "Vasconcellea", "Caenorhabditis", "Pudu", "Carassius", "Primula", "Halogeometricum", "Papilio", "Gallinago", "Eretmapodites", "Molossus", "Zalophus", "Rupicapra", "Lens", "Myodes", "Crematogaster", "Phoca", "Rumex", "Cucumis", "Achromobacter", "Hyalomma", "Manihot", "Croton", "Rhynchosia", "Python", "Pyricularia", "Elaeis", "Choristoneura", "Coturnix", "Heloderma", "Bordetella", "Corvus", "Shewanella", "Hydrobates", "Mythimna", "Actinomyces", "Armigeres", "Progne", "Rana", "Malus", "Leucas", "Halobacterium", "Trifolium", "Urocitellus", "Anredera", "Phalaenopsis", "Lespedeza", "Blattella", "Stenotrophomonas", "Leptosphaeria", "Thermoanaerobacterium", "Grus", "Lonchura", "Phlox", "Ateles", "Cryptosporidium", "Candidatus Hamiltonella", "Tetraselmis", "Anethum", "Parthenium", "Callinectes", "Rattus", "Chaerephon", "Serinus", "Coracias", "Psittacula", "Taeniopygia", "Diplodia", "Nitratiruptor", "Allamanda", "Verticillium", "Paraphaeosphaeria", "Capreolus", "Lamium", "Crocuta", "Geobacillus", "Paspalum", "Pleione", "Campanula", "Lynx", "Colletotrichum", "Callosciurus", "Lepus", "Scortum", "Otomops", "Sigmodon", "Brochothrix", "Benincasa", "Nyssomyia", "Pogona", "Gadus", "Exomis", "Siniperca", "Leonurus", "Mirabilis", "Rosellinia", "Gonimbrasia", "Acanthamoeba", "Pimoa", "Diopsittaca", "Catostomus", "Agave", "Hippotragus", "Canna", "Phytomonas", "Osedax", "Delphinus", "Amphibalanus", "Herpestes", "Urocyon", "Phormidium", "Fulmarus", "Estrilda", "Flavobacterium", "Petunia", "Chlamydia", "Capra", "Mandrillus", "Poicephalus", "Alopecurus", "Sabethes", "Euonymus", "Pteromalus", "Berkeleyomyces", "Mirounga", "Trichormus", "Brassica", "Indotestudo", "Culicoides", "Cercis", "Ursus", "Acyrthosiphon", "Columba", "Coquillettidia", "Nocardia", "Littorina", "Dinoroseobacter", "Hordeum", "Choloepus", "Streptopus", "Otostigmus", "Diaphorina", "Lactobacillus", "Anomala", "Acidianus", "Raphanus", "Trematomus", "Urtica", "Bemisia", "Cynodon", "Microbacterium", "Atherigona", "Meles", "Ctenopharyngodon", "Halyomorpha", "Gremmeniella", "Musa", "Samia", "Lemur", "Segestria", "Diachasmimorpha", "Botrytis", "Halorubrum", "Thermoproteus", "Cacatua", "Cyprinus", "Carollia", "Spermophilus", "Xanthocnemis", "Trichomonas", "Aulacorthum", "Orgyia", "Streptomyces", "Sicyonia", "Cavia", "Erythrura", "Esox", "Damaliscus", "Hibbertia", "Colobus", "Carnegiea", "Lagothrix", "Chlorocebus", "Thysanoplusia", "Caulobacter", "Laodelphax", "Tursiops", "Hyptiotes", "Clostridioides", "Biomphalaria", "Neodon", "Phaseolus", "Trichoderma", "Cladosporium", "Alouatta", "Hibiscus", "Sophora", "Gymnocalycium", "Ustilaginoidea", "Pelophylax", "Diabrotica", "Cicer", "Schmidtea", "Copsychus", "Lelliottia", "Aquamicrobium", "Pantala", "Circus", "Caladenia", "Riptortus", "Perca", "Potamochoerus", "Opsiphanes", "Potamopyrgus", "Rusa", "Sphingomonas", "Nylanderia", "Rhinolophus", "Rhizosolenia", "Colobopsis", "Stemphylium", "Paris", "Urotrichus", "Phthorimaea", "Nigrospora", "Bidens", "Thunbergia", "Microcystis", "Apodemus", "Tetragnatha", "Camponotus", "Paguma", "Nilaparvata", "Helicobacter", "Philantomba", "Rhodococcus", "Abutilon", "Thermus", "Megaskepasma", "Allium", "Labidocera", "Scutigera", "Bradypus", "Emilia", "Cestrum", "Hyporthodus", "Serratia", "Mnemiopsis", "Saccharolobus", "Amphibola", "Cercopithecus", "Pteropus", "Olea", "Psychrobacter", "Sclerotium", "Pipistrellus", "Spissistilus", "Hemidesmus", "Salmo", "Weissella", "Condylorrhiza", "Eratigena", "Ia", "Sulfitobacter", "Nodularia", "Eustoma", "Triticum", "Gorilla", "Bubalus", "Ilex", "Perina", "Cyanthillium", "Cocos", "Phleum", "Malva", "Uroderma", "Homo", "Pyrobaculum", "Sida", "Eucharis", "Andrena", "Narcissus", "Ageratum", "Blechomonas", "Aotus", "Pantoea", "Dendroctonus", "Charybdis", "Vanilla", "Gentiana", "Luffa", "Senna", "Humulus", "Abelmoschus", "Atropa", "Chenopodium", "Sinorhizobium", "Tipula", "Coleura", "Erinaceus", "Cnidoscolus", "Heteronychus", "Pythium", "Synedrella", "Rhynchobatus", "Scotophilus", "Solanum", "Euproctis", "Aspergillus", "Chlorella", "Eptesicus", "Antennarius", "Amblyomma", "Apis", "Arctopus", "Azumapecten", "Camelus", "Propionibacterium", "Pelodiscus", "Callithrix", "Diadromus", "Turdus", "Trichonephila", "Sturnus", "Syringa", "Cynopterus", "Verrallina", "Micaelamys", "Macrosiphum", "Lagenorhynchus", "Chrysodeixis", "Symphysodon", "Planococcus", "Civettictis", "Proscyllium", "Erwinia", "Discula", "Procyon", "Atractylodes", "Sturnira", "Elettaria", "Sigmoidotropis", "Mus", "Ruellia", "Taraxacum", "Velarifictorus", "Dysphania", "Spodoptera", "Vespertilio", "Theobroma", "Ligia", "Piliocolobus", "Odocoileus", "Axonopus", "Anser", "Tinca", "Sonchus", "Leptonychotes", "Tulipa", "Celeribacter", "Amaranthus", "Dickeya", "Eidolon", "Spheniscus", "Rhabdomys", "Saccharomyces", "Erechtites", "Dahlia", "Rousettus", "Oryctes", "Echinothrips", "Bettongia", "Oryzomys", "Parus", "Jatropha", "Alkekengi", "Pleospora", "Ambystoma", "Morganella", "Glis", "Aspidites", "Sambucus", "Miathyria", "Perisesarma", "Phomopsis", "Telfairia", "Saccharomonospora", "Murina", "Euscelidius", "Momordica", "Perigonia", "Haemaphysalis", "Lambdina", "Crotalus", "Mytilus", "Halomonas", "Perameles", "Anticarsia", "Mephitis", "Xanthomonas", "Wisteria", "Elliptio", "Oenococcus", "Ensis", "Blainvillea", "Portunus", "Xylella", "Pelargonium", "Nanorana", "Etheostoma", "Cebus", "Lacanobia", "Erinnyis", "Klebsiella", "Passiflora", "Staphylococcus", "Rhionaeschna", "Mucuna", "Citrobacter", "Caretta", "Lycianthes", "Premna", "Alcea", "Agrobacterium", "Graminella", "Eclipta", "Asterionellopsis", "Bombyx", "Clerodendrum", "Plutella", "Amazona", "Boerhavia", "Lagenaria", "Daphne", "Poa", "Bacillus", "Sander", "Aratinga", "Plumeria", "Alternanthera", "Phascolarctos", "Niviventer", "Microtus", "Centropristis", "Aggregatibacter", "Trichoplusia", "Cricetomys", "Panax", "Libellula", "Begomovirus", "Corallus", "Pimephales", "Haptolina", "Macroptilium", "Ectropis", "Ralstonia", "Eimeria", "Pyrus", "Nitrososphaera", "Giardia", "Neacomys", "Brugmansia", "Erythemis", "Papaver", "Lytechinus", "Miniopterus", "Melopsittacus", "Muscina", "Macrotyloma", "Acidovorax", "Ascaris", "Plautia", "Gallinula", "Pectobacterium", "Vigna", "Phytophthora", "Thaumetopoea", "Mastomys", "Ailuropoda", "Pisum", "Hylaeamys", "Gammarus", "Methanothermobacter", "Mikania", "Acinetobacter", "Operophtera", "Rehmannia", "Uroplatus", "Chironomus", "Lablab", "Solenopsis", "Pasteurella", "Cervus", "Hydrangea", "Ischnura", "Cricetulus", "Lolium", "Nyctalus", "Lupinus", "Physalis", "Haliotis", "Gigaspora", "Dioscorea", "Epichloe", "Cutibacterium", "Puma", "Chondrostereum", "Jasminum", "Rhodothermus", "Proteus", "Bufo", "Conepatus", "Streptococcus", "Azospirillum", "Anthoxanthum", "Heliconius", "Micromonas", "Aedeomyia", "Candidatus Pelagibacter", "Danaus", "Plecotus", "Natrialba", "Orthetrum", "Leishmania", "Penaeus", "Brevibacillus", "Salinivibrio", "Cherax", "Listeria", "Lissotriton", "Camellia", "Enhydra", "Scheffersomyces", "Eriocheir", "Silurus", "Phragmites", "Nycticebus", "Myocastor", "Euphorbia", "Larimichthys", "Felis", "Psammotettix", "Wiseana", "Triaenops", "Sauropus", "Anourosorex", "Bacteroides", "Bothrops", "Corynorhinus", "Ornithodoros", "Heterocapsa", "Smallanthus", "Epirrita", "Sclerophthora", "Lepomis", "Dianthus", "Hedyotis", "Acartia", "Aeromonas", "Ctenocephalides", "Fallopia", "Solea", "Pyrococcus", "Bougainvillea", "Armoracia", "Parabuteo", "Aedes", "Ludwigia", "Coriandrum", "Calidris", "Lymantria", "Cynara", "Corynebacterium", "Clostridium", "Tholymis", "Zygodontomys", "Macrobrachium", "Halogranum", "Elephas", "Mycolicibacterium", "Gelasimus", "Pseudogymnoascus", "Liberibacter", "Cyanoramphus", "Loxodonta", "Ameiva", "Psathyromyia", "Vulpes", "Hexura", "Channa", "Lilium", "Catharanthus", "Melanoplus", "Canis", "Senecio", "Falco", "Pieris", "Glycine", "Tropaeolum", "Buteo", "Ruditapes", "Gasteracantha", "Curvularia", "Pueraria", "Lactococcus", "Rosa", "Ustilago", "Petrochelidon", "Mops", "Persea", "Corchorus", "Galleria", "Budorcas", "Musca", "Sus", "Stachytarpheta", "Arrhenatherum", "Antheraea", "Trisetum", "Daucus", "Cleome", "Capitulum", "Hyalopterus", "Anthocercis", "Odobenus", "Fusarium", "Mansonia", "Boa", "Nasua", "Hipposideros", "Kalanchoe", "Macaca", "Oryctolagus", "Diospyros", "Stercorarius", "Ligustrum", "Mycobacterium", "Rhodoferax", "Desmodus", "Formicarius", "Metaphire", "Pyrrhula", "Dobsonia", "Xenopus", "Asystasia", "Heterobasidion", "Mannheimia", "Desmodium", "Ameiurus", "Ananas", "Apium", "Paphies", "Anguilla", "Shigella", "Acipenser", "Paramuricea", "Zinnia", "Diaporthe", "Wissadula", "Leuconostoc", "Andrographis", "Diatraea", "Mesorhizobium", "Spiroplasma", "Aurantimonas", "Scalopus", "Eutrema", "Cairina", "Cnaphalocrocis", "Angulus", "Arracacia", "Culiseta", "Primnoa", "Iris", "Neotoma", "Lonomia", "Eothenomys", "Cordyline", "Hardenbergia", "Thelephora", "Citrus", "Martes", "Chelonus", "Phaeocystis", "Gymnanthemum", "Avena", "Brevicoryne", "Mycoplasma", "Microplitis", "Ribes", "Ipomoea", "Lonicera", "Salisaeta", "Dendrobium", "Acholeplasma", "Cymbidium", "Pteronotus", "Austrovenus", "Angelica", "Vitis", "Rhinopithecus", "Caligus", "Ochlerotatus", "Ciconia", "Carios", "Helianthus", "Ruegeria", "Crocus", "Formica", "Epinotia", "Polygala", "Sphingobium", "Gryllus", "Tsukamurella", "Rhizoctonia", "Myzus", "Boehmeria", "Brachiaria", "Vespula", "Neovison", "Styphnolobium", "Macrophomina", "Fragaria", "Bipolaris", "Dermacentor", "Ostreococcus", "Alteromonas", "Aselliscus", "Ixeridium", "Riemerella", "Edwardsiella", "Cyclopterus", "Ceratorhiza", "Nesidiocoris", "Eudromia", "Alpinia", "Limeum", "Phaius", "Cytospora", "Crassostrea", "Spilanthes", "Hylomyscus", "Pan", "Diadegma", "Anas", "Gerygone", "Branta", "Lates", "Crassocephalum", "Duranta", "Telosma", "Salvia", "Ectocarpus", "Pomacea", "Carica", "Mentha", "Angelonia", "Calomys", "Ophiostoma", "Citrullus", "Argas", "Chaetoceros", "Emiliania", "Peromyscus", "Miscanthus", "Giraffa", "Cajanus", "Fringilla", "Bromus", "Bombus", "Ixodes", "Anopheles", "Iodobacter", "Lutzomyia", "Feldmannia", "Deinbollia", "Akodon", "Pasiphila", "Rhodovulum", "Plodia", "Cyamopsis", "Eleocharis", "Micractinium", "Neoaliturus", "Candidatus Puniceispirillum", "Aphis", "Crotalaria"]

    superorder_list = ['Euarchontoglires', 'Laurasiatheria']
    Primates_family_list = ['Hominidae', 'Hylobatidae']  # NO Hylobatidae sample
    Hominidae_genus_list = ['Homo', 'Pan']
    Mammalia_list = ['Primates', 'Rodentia', 'Lagomorpha', 'Scandentia', 'Dermoptera',
                     'Carnivora', 'Perissodactyla', 'Artiodactyla', 'Eulipotyphla', 'Chiroptera', 'Pholidota',
                     'Proboscidea', 'Sirenia', 'Cingulata', 'Pilosa', 'Didelphimorphia', 'Peramelemorphia', 'Diprotodontia']

    virus_host_seq_ = []
    virus_Mammalia_seq_ = []
    virus_Primates_seq_ = []
    virus_Primates_family_seq_ = []
    virus_Hominidae_seq_ = []
    virus_Hominidae_genus_seq_ = []
    virus_Homo_seq_ = []
    virus_no_Homo_seq_ = []
    only_Mammalia_list = []

    count_Mammalia = 0

    '''find Mammalia'''
    for vc_ in virus_cds_:
        host_lineage = vc_[2]
        lineages_ = host_lineage.split(';')
        if len(lineages_) > 1:  # phylum 至少在第二位之后
            for host_ in lineages_:
                if host_ == 'Mammalia':    # this host lineage has phylum level host
                    virus_host_seq_.append(vc_)  # add phylum msg
                    count_Mammalia += 1
                    break
    print('Mammalia host of virus sample =', count_Mammalia)

    '''classifier class and order in virus_host_seq_'''
    for vc_ in virus_host_seq_:
        host_lineage = vc_[2]
        lineages_ = host_lineage.split(';')
        only_Mammalia_flag = True
        if len(lineages_) > 1:  # phylum 至少在第二位之后
            for host_ in lineages_:
                if host_ in order_list:
                    virus_Mammalia_seq_.append((vc_[0], vc_[1], vc_[2], host_, vc_[3]))  # add order msg
                    only_Mammalia_flag = False
                    break
            if only_Mammalia_flag:
                if lineages_[-1] in superorder_list:
                    virus_Mammalia_seq_.append((vc_[0], vc_[1], vc_[2], lineages_[-1], vc_[3]))  # add superorder msg
                else:
                    virus_Mammalia_seq_.append((vc_[0], vc_[1], vc_[2], 'Mammalia', vc_[3]))  # add Mammalia msg
                # only_Mammalia_list.append(lineages_)

    host_ = [ht_[3] for ht_ in virus_Mammalia_seq_]
    host_distribution_ = list(Counter(host_).items())
    host_distribution_ = sorted(host_distribution_, key=lambda hd: hd[1], reverse=True)
    print('classifier class and order:', len(host_distribution_), host_distribution_)

    '''classifier family in virus_Mammalia_seq_'''
    for vc_ in virus_Mammalia_seq_:
        host_lineage = vc_[2]
        lineages_ = host_lineage.split(';')
        only_Hominidae_flag = True
        if len(lineages_) > 1:  # phylum 至少在第二位之后
            for host_ in lineages_:
                if host_ == 'Primates':
                    virus_Primates_seq_.append((vc_[0], vc_[1], vc_[2], host_, vc_[3]))  # add order msg
                    break

    for vc_ in virus_Primates_seq_:
        host_lineage = vc_[2]
        lineages_ = host_lineage.split(';')
        only_Primates_flag = True
        if len(lineages_) > 1:  # phylum 至少在第二位之后
            for host_ in lineages_:
                if host_ in family_list:
                    virus_Primates_family_seq_.append((vc_[0], vc_[1], vc_[2], host_, vc_[3]))  # add family msg
                    only_Primates_flag = False
                    break
            # if only_Primates_flag:
            #     only_Mammalia_list.append(lineages_)

    host_ = [ht_[3] for ht_ in virus_Primates_family_seq_]
    host_distribution_ = list(Counter(host_).items())
    host_distribution_ = sorted(host_distribution_, key=lambda hd: hd[1], reverse=True)
    print('classifier family:', len(host_distribution_), host_distribution_)

    '''classifier genus in virus_Primates_family_seq_'''
    for vc_ in virus_Primates_family_seq_:
        host_lineage = vc_[2]
        lineages_ = host_lineage.split(';')
        if len(lineages_) > 1:  # phylum 至少在第二位之后
            for host_ in lineages_:
                if host_ == 'Hominidae':
                    virus_Hominidae_seq_.append((vc_[0], vc_[1], vc_[2], host_, vc_[3]))  # add order msg
                    break

    for vc_ in virus_Hominidae_seq_:
        host_lineage = vc_[2]
        lineages_ = host_lineage.split(';')
        only_Hominidae_flag = True
        if len(lineages_) > 1:  # phylum 至少在第二位之后
            for host_ in lineages_:
                if host_ in genus_list:
                    virus_Hominidae_genus_seq_.append((vc_[0], vc_[1], vc_[2], host_, vc_[3]))  # add family msg
                    only_Hominidae_flag = False
                    break
            if only_Hominidae_flag:
                only_Mammalia_list.append(lineages_)

    host_ = [ht_[3] for ht_ in virus_Hominidae_genus_seq_]
    host_distribution_ = list(Counter(host_).items())
    host_distribution_ = sorted(host_distribution_, key=lambda hd: hd[1], reverse=True)
    print('classifier genus:', len(host_distribution_), host_distribution_)

    '''find Homo in Mammalia'''
    for vc_ in virus_Mammalia_seq_:
        host_lineage = vc_[2]
        lineages_ = host_lineage.split(';')
        homo_flag = False
        if len(lineages_) > 1:  # phylum 至少在第二位之后
            for host_ in lineages_:
                if host_ == 'Homo':
                    virus_Homo_seq_.append((vc_[0], vc_[1], vc_[2], 'Homo', vc_[-1]))  # add Homo msg
                    homo_flag = True
                    break
            if not homo_flag:
                virus_no_Homo_seq_.append((vc_[0], vc_[1], vc_[2], 'no_Homo', vc_[-1]))  # add Homo msg

    homo_index = np.random.choice(range(len(virus_Homo_seq_)), 1100)
    no_homo_index = np.random.choice(range(len(virus_no_Homo_seq_)), 1600)
    virus_Homo_seq = [virus_Homo_seq_[i] for i in homo_index] + [virus_no_Homo_seq_[i] for i in no_homo_index]

    print('Homo host of virus sample =', len(virus_Homo_seq))

    host_ = [ht_[3] for ht_ in virus_Homo_seq]
    host_distribution_ = list(Counter(host_).items())
    host_distribution_ = sorted(host_distribution_, key=lambda hd: hd[1], reverse=True)
    print('classifier Homo:', len(host_distribution_), host_distribution_)

    file_path_ = save_path + type_ + '_genome/'
    if not os.path.exists(file_path_):
        os.makedirs(file_path_)

    with open(file_path_ + 'X.txt', 'w') as f:
        for vhs_ in virus_Homo_seq:
            f.write('|'.join(vhs_[-1]) + '\n')  # cds 1|cds 2|...|cds i|...

    with open(file_path_ + 'Y.txt', 'w') as f:
        for vhs_ in virus_Homo_seq:
            f.write(vhs_[-2] + '\n')            # label (different taxonomy level)

    with open(file_path_ + 'virus_name.txt', 'w') as f:
        for vhs_ in virus_Homo_seq:
            f.write(vhs_[-2] + '|' + vhs_[0] + '\n')             # virus name

    if save_flag:

        file_path_ = save_path + 'host_distribution/'

        if not os.path.exists(file_path_):
            os.makedirs(file_path_)

        with open(file_path_+type_+'_distribution.txt', 'w') as f:
            for hd_ in host_distribution_:
                f.write(str(hd_[0]) + '\t' + str(hd_[1]) + '\n')


if __name__ == '__main__':
    path = '/home/hongweichen/Data/vhp_data/'
    save_path = path + 'data_0531_2021_mammalia/'

    '''
    genome_seq: get virus msg and cds genome seq 
    (host have not None, and host lineages have not same name)
    genome CDS length: [12, 40671]
    
    cds_num_list: all virus cds distribution
    '''
    genome_seq, _, cds_num_list = load_virus_host_msg(load_data(path))

    '''{host: type} and {type: host}'''
    # find_all_host_type(genome_seq, save_path, True)
    # type_host_mapping(save_path+'all_host_type.json', save_path_=save_path)

    '''save data by cds, (one virus, all [CDS])'''
    # save_data_by_cds(genome_seq, save_path, path, using_degenerate_=True)

    # virus_cds = load_data_by_cds(save_path)
    # virus_dict = [(vhl[0], len(vhl[-1])) for vhl in virus_cds]
    # assert virus_dict == cds_num_list

    '''split label by taxonomy'''
    # type_list = ['phylum', 'class', 'order', 'family', 'genus']
    # get_label_by_taxonomy(virus_cds, 'Homo1', save_path, save_flag=False)

