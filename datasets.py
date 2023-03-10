import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 syntax_nl=None,
                 syntax_code=None
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.syntax_nl = syntax_nl
        self.syntax_code = syntax_code

def read_examples(filename, type=None):
    examples = []
    data = pd.read_csv(filename)
    nls = data['nl'].tolist()
    codes = data['code'].tolist()
    if(type!='train'):
        for idx in range(len(nls)):
            examples.append(
                Example(
                    idx=idx,
                    source=nls[idx].replace('Generate origin code: ', ''),
                    target=codes[idx],
                )
            )
    else:
        syntax_nl = data['syntax_nl'].tolist()
        syntax_code = data['syntax_code'].tolist()
        for idx in range(len(nls)):
            examples.append(
                Example(
                    idx=idx,
                    source=nls[idx].replace('Generate origin code: ', ''),
                    target=codes[idx],
                    syntax_nl=syntax_nl[idx].replace('Generate syntax code: ', ''),
                    syntax_code=syntax_code[idx]
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 source_mask,
                 target_ids,
                 target_mask,
                 syntax_nl_ids=None,
                 syntax_nl_mask=None,
                 syntax_code_ids=None,
                 syntax_code_mask=None,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask

        self.syntax_nl_ids = syntax_nl_ids
        self.syntax_nl_mask = syntax_nl_mask
        self.syntax_code_ids = syntax_code_ids
        self.syntax_code_mask = syntax_code_mask


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    if stage != "train":
        for example_index, example in enumerate(examples):
            # source
            source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 2]
            source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = max_source_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length
            # target
            if stage == "test":
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] * len(target_ids)
            padding_length = max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length
            target_mask += [0] * padding_length

            features.append(
                InputFeatures(
                    example_index,
                    source_ids,
                    source_mask,
                    target_ids,
                    target_mask,
                )
            )
    else:
        for example_index, example in enumerate(examples):
            # source
            source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 2]
            source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = max_source_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length
            # target
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] * len(target_ids)
            padding_length = max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length
            target_mask += [0] * padding_length

            # syntax_nl
            syntax_nl_tokens = tokenizer.tokenize(example.syntax_nl)[:max_source_length - 2]
            syntax_nl_tokens = [tokenizer.cls_token] + syntax_nl_tokens + [tokenizer.sep_token]
            syntax_nl_ids = tokenizer.convert_tokens_to_ids(syntax_nl_tokens)
            syntax_nl_mask = [1] * (len(syntax_nl_tokens))
            padding_length = max_source_length - len(syntax_nl_ids)
            syntax_nl_ids += [tokenizer.pad_token_id] * padding_length
            syntax_nl_mask += [0] * padding_length
            # syntax_code
            syntax_code_tokens = tokenizer.tokenize(example.syntax_code)[:max_target_length - 2]
            syntax_code_tokens = [tokenizer.cls_token] + syntax_code_tokens + [tokenizer.sep_token]
            syntax_code_ids = tokenizer.convert_tokens_to_ids(syntax_code_tokens)
            syntax_code_mask = [1] * len(syntax_code_ids)
            padding_length = max_target_length - len(syntax_code_ids)
            syntax_code_ids += [tokenizer.pad_token_id] * padding_length
            syntax_code_mask += [0] * padding_length

            features.append(
                InputFeatures(
                    example_index,
                    source_ids,
                    source_mask,
                    target_ids,
                    target_mask,
                    syntax_nl_ids,
                    syntax_nl_mask,
                    syntax_code_ids,
                    syntax_code_mask
                )
            )
    return features
