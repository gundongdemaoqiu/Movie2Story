from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from videollava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

# ID介绍
#     2. **词汇表（Vocabulary）**：
#    - 分词器有一个预定义的词汇表，每个词块在词汇表中都有一个唯一的ID。词汇表是一个从词块到ID的映射表。例如，词汇表中可能包含以下映射：
#      ```
#      {"hello": 1, "world": 2, "good": 3, "morning": 4, "[UNK]": 5}
#      ```
#      这里，"hello" 的ID是1，"world" 的ID是2，等等。

# 3. **ID计算**：
#    - 分词器根据词汇表将分词后的词块转换为ID。例如，输入文本 "hello world" 会被分词为 ["hello", "world"]，然后被转换为ID [1, 2]。
# 1. **嵌入层（Embedding Layer）**：
#    - 在大多数NLP模型中，输入的ID序列会首先通过一个嵌入层（embedding layer）。嵌入层是一个查找表，将每个ID映射到一个高维向量，这个向量叫做词向量（word vector）


    ### 处理步骤：
# 1. **拆分提示**：函数通过`<image>`标记拆分提示字符串，得到多个文本块。
# 2. **分词文本块**：使用分词器将每个文本块转换为一系列标记ID。
# 3. **插入分隔符**：函数在分词后的文本块之间插入图像标记ID。
# 4. **处理特殊标记**：如果第一个文本块的第一个标记是序列开始（BOS）标记，则在最终标记序列的开头保留它。

# ### 生成最终标记序列：
# - 函数通过组合文本标记和图像标记ID来构建最终的标记ID序列。
# - 如果提供了`return_tensors`参数并设置为`'pt'`，函数将把标记ID列表转换为PyTorch张量。否则，它返回标记ID列表。

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:   #`bos` 是 `beginning of sequence` 的缩写，表示序列的起始标记（token）
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    # **插入分隔符**：
    # - 使用`insert_separator`函数将图像标记ID插入到分词后的文本块之间：


    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])


#     ### 输出结果：

# - 如果`return_tensors=None`，输出将是一个整数列表：
#   ```python
#   [36825, 26159, 19968, 24102, 26377, 102, 21644, 26356, 22810, 25991, 102, 30340, 35828, 12290]
#   ```

# - 如果`return_tensors='pt'`，输出将是一个PyTorch张量：
#   ```python
#   tensor([36825, 26159, 19968, 24102, 26377, 102, 21644, 26356, 22810, 25991, 102, 30340, 35828, 12290])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    # 在使用自然语言生成模型时，`StoppingCriteria` 类的作用是定义何时停止生成文本的标准。
    # 这个类允许你根据特定的条件来终止生成过程，而不是仅仅依赖于模型生成的最大长度参数。例如，你可以定义一些关键字，当生成的文本包含这些关键字时就停止生成。
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        # 一个检测方式是对比 token ID，另一个是对比分词器解码后的文本。
#         1. **固定长度的 ID 检测**：
#     - **位置**：通常只检测生成序列的最后几个 token。
#     - **适用场景**：这种方法适用于需要在生成的最后部分匹配特定的 token 序列的情况，例如检测特殊的终止标记（如 `sep` 或 `eos`）。
#     - **示例**：
#       ```python
#       keyword_id = [7592, 2088]
#       if output_ids[-len(keyword_id):] == keyword_id:
#           print("Detected keyword at the end")
#       ```

# 2. **文本全局检测**：
#     - **位置**：可以在整个生成的文本中进行全局搜索，不限定位置。
#     - **适用场景**：这种方法适用于需要在整个文本中查找特定关键词的情况，特别是当关键词可能出现在任意位置时。
#     - **示例**：
#       ```python
#       decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
#       if "hello world" in decoded_text:
#           print("Detected keyword in the text")
#       ```
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
