SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

class ByteTokenizer:
    def __init__(self):
        self.num_special = len(SPECIAL_TOKENS)
        self.vocab_size = self.num_special + 256  # 0-255 bytes + specials
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    @classmethod
    def load(cls, save_dir=None):
        # kept just for API compatibility
        return cls()

    def encode(self, text, add_special_tokens=True):
        byte_ids = list(text.encode("utf-8"))  # each 0–255
        ids = [b + self.num_special for b in byte_ids]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        bytes_list = []
        for i in ids:
            if i < self.num_special:
                if skip_special_tokens:
                    continue
                # could map specials to something if you want
            else:
                bytes_list.append(i - self.num_special)
        return bytes(bytes_list).decode("utf-8", errors="ignore")
