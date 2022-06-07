import json

entity_dictionary = dict()
entitylist = list()
nodes = list()
edges = list()
data_sample = {"nodes": [], "edges": []}

def process():
    with open("save_files/tmp/origin_data.txt", "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        triples = list()
        for ls in lines:
            ls = ls.replace("[", "")
            ls = ls.replace("]", "")
            ls = ls.replace("(", "")
            ls = ls.replace(")", "")
            ls = ls.replace("\'", "")
            ls = ls.replace("\n", "")
            ls = ls.split(",")
            print(ls)
            i =0
            while(i < len(ls)):
                e1 = ls[i]
                e2 = ls[i+1]
                r = ls[i+2]
                i = i + 3
                triple = (e1, e2, r)
                if triple not in triples:
                    triples.append(triple)
        print(triples)
        k = 0
        for t in triples:
            entity1, entity2, relation = t
            edge_sample = {"source": "", "target": "", "label": ""}
            if entity1 not in entitylist:
                node_sample = {"id": "", "label": ""}
                entitylist.append(entity1)
                index1 = str(k)
                entity_dictionary[entity1] = index1
                node_sample["id"] = index1
                node_sample["label"] = entity1
                nodes.append(node_sample)
                k = k + 1
            else:
                index1 = entity_dictionary[entity1]
            if entity2 not in entitylist:
                node_sample = {"id": "", "label": ""}
                entitylist.append(entity2)
                index2 = str(k)
                entity_dictionary[entity2] = index2
                node_sample["id"] = index2
                node_sample["label"] = entity2
                nodes.append(node_sample)
                k = k + 1
            else:
                index2 = entity_dictionary[entity2]
            edge_sample["source"] = index1
            edge_sample["target"] = index2
            edge_sample["label"] = relation
            edges.append(edge_sample)
        data_sample["nodes"] = nodes
        data_sample["edges"] = edges
        print(data_sample)
        with open("save_files/final/data.json", "w", encoding="utf-8") as fj:
            json.dump(data_sample, fj, indent=2, ensure_ascii=False)


