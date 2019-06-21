import re


with open("tst2012.from") as f:
    x = f.readlines()

with open("tst2012.to") as f:
    y = f.readlines()

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

x = [[normalizeString(s) for s in str(l).split('\t')] for l in x]
y = [[normalizeString(s) for s in str(l).split('\t')] for l in y]

def tagger(decoder_input_sentence):
    sos = "<SOS> "
    eos = " <EOS>"
    final_target = [sos + text + eos for i in decoder_input_sentence]
    return final_target

decoder_inputs = tagger(y)
