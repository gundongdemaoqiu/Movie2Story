# VideoCaption处理流程

## 整体workflow处理
    利用一些预定义的transform方法对image，video进行基本特征提取，比如pixel values提取（可见LanguageBindVideoProcessor类），然后通过自己定义的 prepare_inputs_labels_for_multimodal 来将视频tensor 结合text tensot输入到llava文本模型生成文本。
## 利用模型：
    1. LLAVA等基础模型和causal预训练模型。eg:LlavaLlamaForCausalLM类。 但是这里只能应对文本模型，其中定义了新的forward等其他函数，为了针对多模态数据。
    2. 对于video 的processor， 利用了 video-tower和image-tower等一些模型处理，但是注意，这里并不是实现了image-》text的特征空间映射，而是通过了一些tramsform的方法进行裁剪等操作，提取视频图像的基本feature。eg：LanguageBindVideoProcessor(ProcessorMixin)类中的load_and_transform_video，对三种不同的方法处理视频数据，提取pixel_values。
    3. 真正结合image到文本模型的是 LlavaMetaForCausalLM中的 prepare_inputs_labels_for_multimodal 函数。这里定义了怎么把image的tensor 结合到text空间  （具体函数思路待日后补充！！！）
    4. 利用generate函数（应该是预定义的文本生成函数）结合列所有的多模态输入最终经 prepare_inputs_labels_for_multimodal 预处理后进行文本生成。


## 其他信息补充
    1. 通过预定义一些Seperator的关键词进行user/assistant的对话对构建和prompt生成。  详情见 get_prompt 函数，其中根据config的不同，有不同的sep风格： One, Two....
    2. 检测keyword来实现文本生成长度限制，eg: KeywordsStoppingCriteria类
