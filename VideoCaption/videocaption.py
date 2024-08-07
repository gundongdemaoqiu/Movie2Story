import os
import sys
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def videocaption(video):
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 将这个路径添加到 sys.path
    sys.path.append(current_dir)
    
    disable_torch_init()
    # video = 'video/1.mp4'
    inp = 'Gnerate the caption for this video.'
    # 模型权重加载为vepfs地址
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = '/vepfs/fs_users/lkn/video_llava'
    device = 'cuda:0'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    # 其中model是llava模型，文本处理模型。  对于image和video的处理是通过processor，这里似乎没有实现视频映射文本空间，只是单纯提取了视频或者image的基础信息，详情可以见load_and_transform_video函数
    # 因此model里添加了一些多模态数据的处理函数
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    # 这里是对于视频进行处理，还有image但是这里没有用到
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    # 这是conv_templates[conv_mode]的例子：
# conv_llava_v1 = Conversation(
#     system="A chat between a curious human and an artificial intelligence assistant. "
#            "The assistant gives helpful, detailed, and polite answers to the human's questions.",
#     roles=("USER", "ASSISTANT"),
#     version="v1",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="</s>",
# )

# 1. **USER**：
#    - 代表提问者或用户，也就是与模型进行交互的人。
#    - 用户通常提出问题、提供输入或请求信息。
#    - 例如，在你的代码中，用户输入了一条消息，要求生成视频的字幕。

# 2. **ASSISTANT**：
#    - 代表人工智能助手或模型本身。
#    - 助手的角色是根据用户的输入做出回应，提供答案或执行请求的任务。
#    - 在你的代码中，ASSISTANT角色的职责是根据USER的请求生成视频的字幕。
   
    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        # tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        tensor = [vt.to(device, dtype=torch.float16) for vt in video_tensor]
    else:
        # tensor = video_tensor.to(model.device, dtype=torch.float16)
        tensor = video_tensor.to(device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    # inp的例子：
    # [DEFAULT_IMAGE_TOKEN][DEFAULT_IMAGE_TOKEN][DEFAULT_IMAGE_TOKEN][DEFAULT_IMAGE_TOKEN]。。。
# Generate the caption for this video.

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    # 这例子的结果就是每次user/assitant两个信息成对出现
    
    prompt = conv.get_prompt()
    # prompt的结果实例：
    # A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: [DEFAULT_IMAGE_TOKEN] [DEFAULT_IMAGE_TOKEN] ... [DEFAULT_IMAGE_TOKEN] Generate the caption for this video. ASSISTANT:</s>


    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    # tokenizer_image_token介绍，请看函数内部定义
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    # 上面就是将seperator作为切分要求
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


    # class LlavaMetaForCausalLM(ABC)里面定义了所有的模态数据处理函数，包含了将image变成feature等信息

    # ！！！！！！！！！！！关键函数：class LlavaMetaForCausalLM(ABC)：prepare_inputs_labels_for_multimodal！！！！！！
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    outputs = outputs.rstrip("</s>")  # 移除字符串末尾的 "</s>"
    return outputs

if __name__ == '__main__':
    videocaption = videocaption('/root/code/MMAD/Example/Video/demo.mp4')
    print(videocaption)