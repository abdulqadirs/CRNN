import torch

class LabelConverter:
    def __init__(self, char_set):
        char = ['*'] + sorted(set(''.join(char_set)))
        self.vocab_size = len(char)
        self.int2char = dict(enumerate(char))
        self.char2int = {char: ind for ind, char in self.int2char.items()}

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, texts):
        'represents each character by a unique id'
        text_length = []
        for t in texts:
            text_length.append(len(t))

        encoded_texts = []
        for t in texts:
            for c in t.lower():
                encoded_texts.append(self.char2int.get(c))

        return torch.tensor(encoded_texts), torch.tensor(text_length)

    def decode(self, encoded_text):
        # decode
        text = []
        for i in encoded_text:
            text.append(self.int2char.get(i.item()))

        # remove duplicate
        decoded_text = ''
        for i, t in enumerate(text):
            if t == '*':
                continue
            if i > 0 and t == text[i-1]:
                continue
            decoded_text = decoded_text + t

        return decoded_text
