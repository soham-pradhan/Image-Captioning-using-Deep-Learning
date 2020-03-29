def load_doc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text



def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split("\t",1)
        if len(line)<2:
            continue
        image_id, image_desc = tokens[0],tokens[1]
        image_id = image_id.split('.')[0]
        if image_id not in mapping:
            mapping[image_id]=list()
        mapping[image_id].append(image_desc)
    return mapping
        


import string

def clean_descriptions(descriptions):
    table = str.maketrans('','',string.punctuation)
    #print(table)
    for key, desc_list in descriptions.items():
            #print(desc_list)
            for i in range(len(desc_list)):
                desc = desc_list[i]

                #print(desc)
                desc = desc.split()
                desc = [word.lower() for word in desc]
                desc = [w.translate(table) for w in desc]
                
                desc = [word for word in desc if len(word)>1]
                
                desc = [word for word in desc if word.isalpha()]
                
                desc_list[i] = ' '.join(desc)



def to_vocab(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc



def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


def main():
    filename = "Flicker8k_text\Flickr8k.token.txt"
    doc = load_doc(filename)
    #print(doc[:300])

    descriptions = load_descriptions(doc)
    #print('Loaded: {}'.format(len(descriptions)))
    #print(descriptions['1000268201_693b08cb0e'])

    clean_descriptions(descriptions)
    #print(descriptions['1000268201_693b08cb0e'])
    #print(descriptions['1001773457_577c3a7d70'])

    vocabulary = to_vocab(descriptions)
    #print(vocabulary)
    print("Original Vocab Size {}".format(len(vocabulary)))

    save_descriptions(descriptions, 'descriptions.txt')

if __name__ == "__main__":
    main()
