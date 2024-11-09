SPECIAL_TOKENS = {
    'xlm-r': {
        'bos': '<s>',
        'eos': '</s>',
    },
    'mistral': {
        'bos': '<s>',
        'eos': '</s>',
        'pad': '</s>',
        'mask': '<unk>',
    },
    'llama': {
        'bos': '<|begin_of_text|>',
        'eos': '<|end_of_text|>',
        'pad': '<|finetune_right_pad_id|>',
        'mask': "<|reserved_special_token_0|>",
    },
    'nvidia/NV-Embed-v2': {
        'bos': '<s>',
        'eos': '</s>',
        'pad': '</s>',
        'mask': '<unk>',
    },
    'qwen2': {
        'bos': "<|im_start|>",
        'eos': '<|im_end|>',
        'pad': '<|endoftext|>',
        'mask': "<|object_ref_start|>",
    },
}