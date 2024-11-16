import torch

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Decoder(metaclass=Singleton):
    def __init__(self):
        self.tokenizer = None
        self.tokenizer_draft = None
        self.tokenizer_target = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def set_tokenizer(self, tokenizer_draft, tokenizer_target):
        self.tokenizer = tokenizer_draft
        self.tokenizer_draft = tokenizer_draft
        self.tokenizer_target = tokenizer_target

    def encode(self, s: str, return_tensors='pt') -> torch.Tensor:
        return self.tokenizer.encode(s, return_tensors=return_tensors)
    
    def encode_draft(self, s: str, return_tensors='pt') -> torch.Tensor:
        return self.tokenizer_draft.encode(s, return_tensors=return_tensors)
    
    def encode_target(self, s: str, return_tensors='pt') -> torch.Tensor:
        return self.tokenizer_target.encode(s, return_tensors=return_tensors)
    
    def decode(self, t: torch.Tensor) -> str:
        return self.tokenizer.decode(t[0], skip_special_tokens=True)
    
    def decode_draft(self, t: torch.Tensor) -> str:
        return self.tokenizer_draft.decode(t[0], skip_special_tokens=True)
    
    def decode_target(self, t: torch.Tensor) -> str:
        return self.tokenizer_target.decode(t[0], skip_special_tokens=True)