# VideoCaption处理流程

## 整体workflow处理
    利用一些预定义的transform方法对image，video进行基本特征提取，比如pixel values提取（可见LanguageBindVideoProcessor类），然后通过自己定义的 prepare_inputs_labels_for_multimodal 来将视频tensor 结合text tensot输入到llava文本模型生成文本。
## 利用模型：
    1. LLAVA等基础模型和causal预训练模型。eg:LlavaLlamaForCausalLM类。 但是这里只能应对文本模型，其中定义了新的forward等其他函数，为了针对多模态数据。
    2. 对于video 的processor， 利用了 video-tower和image-tower等一些模型处理，但是注意，这里并不是实现了image-》text的特征空间映射，eg：LanguageBindVideoProcessor(ProcessorMixin)类中的load_and_transform_video，对三种不同的方法处理视频数据，提取pixel_values。
    3. 真正结合image到文本模型的是 LlavaMetaForCausalLM中的 prepare_inputs_labels_for_multimodal 函数。这里定义了怎么把image的tensor 结合到text空间  ：
        a. 这个函数的整体流程就是先利用image tower进行简单的图像变换和裁剪，提取基本feature，然后利用mm-projector自定义linear/MLP 模型进行特征空间的映射，可以通过load_state_dict实现预训练模型的加载。  （详情见 LlavaMetaModel类的 build_vision_projector(config, delay_load=False, **kwargs) ）
    4. 利用generate函数（应该是预定义的文本生成函数）结合列所有的多模态输入最终经 prepare_inputs_labels_for_multimodal 预处理后进行文本生成。

    5. /root/code/MMAD/VideoCaption/videollava/model/multimodal_encoder/languagebind 里面有所有的encoder和tower模型设置
    

## LanguageBindVideoTower模型：
    模型加载和管理：LanguageBindVideoTower 类负责加载和管理 LanguageBindVideo 模型，并在前向传播中提取视频特征。LanguageBindVideoProcessor负责load并且transform视频
    特征选择：可以根据配置选择不同层的特征。
    前向传播：支持对单个视频或视频列表进行处理，提取和返回视频特征。
    设备和数据类型：通过 dtype 和 device 属性管理模型的设备和数据类型。
    不是简单地对视频进行像素级别的提取。相反，它利用了预训练的 LanguageBindVideo 模型来进行更高级的特征提取

    真正的video_tower: CLIPVisionTransformer()，
    根本模型：  LanguageBindVideo 类结合了文本和视觉模型，使用 CLIP 架构处理文本和图像之间的关系。它包括文本和视觉模型的投影、位置嵌入调整、以及将特征投影到相同的维度。该模型支持通过文本和图像输入计算相似度得分，并在训练时可以计算损失

    CLIPTextTransformer
    功能:
    CLIPTextTransformer 主要用于处理和理解文本输入。它将文本数据（如句子或词组）转换为嵌入表示，以便与图像嵌入进行比较，从而实现多模态的对比学习。

    架构:

    模型类型: Transformer
    组件:
    Embedding层: 将输入的token转换为固定维度的向量。
    Transformer Encoder层: 处理文本序列，通过自注意力机制捕捉文本中的长程依赖关系。
    Pooler层: 对Transformer的输出进行池化（通常是取[CLS] token的输出），以生成句子级别的嵌入。
    输入:

    input_ids: 文本的token id。
    attention_mask: 指示哪些tokens是padding的掩码。
    position_ids: 位置编码，帮助模型理解tokens的顺序。
    输出:

    文本嵌入: 一个固定维度的向量，表示文本输入的特征。

    CLIPVisionTransformer
    功能:
    CLIPVisionTransformer 负责处理和理解图像输入。它将图像数据转换为嵌入表示，与文本嵌入进行比较以进行多模态学习。

    架构:

    CLIPVisionTransformer 是 CLIP 模型中用于处理视觉输入的变换器（Transformer）模块。它利用了 CLIP 视觉模型的预训练组件，同时也包括了一些自定义的层和功能。以下是它的预训练组件和自定义部分的详细介绍：

        预训练组件
        CLIPVisionEmbeddings:

        功能：这一层负责将输入的像素值映射到嵌入空间。它通常包括卷积层和位置嵌入，用于将输入图像转换为适合 Transformer 编码器处理的嵌入表示。
        预训练：CLIPVisionEmbeddings 在 CLIP 模型的预训练过程中已经训练好，用于提取和处理图像特征。
        CLIPEncoder:

            CLIPEncoderLayer 的特点和修改：
            标准组件:

            自注意力机制（Self-Attention）: self_attn 是一个标准的自注意力机制，用于捕捉输入序列中元素之间的依赖关系。这与传统 Transformer 的做法一致。
            前馈网络（MLP）: mlp 处理经过自注意力层之后的隐藏状态。这也是 Transformer 编码器的标准组件。
            层归一化（LayerNorm）: 使用 layer_norm1 和 layer_norm2 对隐藏状态进行归一化，这是传统 Transformer 的常见做法。
            时间注意力（Temporal Attention）:

            时间嵌入（Temporal Embedding）: temporal_embedding 是一个额外的组件，用于为时间维度添加嵌入。这是对标准 Transformer 结构的一个修改，使其能够处理具有时间序列信息的输入数据。
            时间自注意力（Temporal Self-Attention）: temporal_attn 是一个用于时间维度的自注意力机制。它允许模型在时间维度上进行注意力计算，这对处理视频或时间序列数据特别重要。
            修改与标准 Transformer 的不同之处:

            时间维度处理: 在传统 Transformer 中，时间序列的处理通常不是内置的。CLIPEncoderLayer 通过引入时间嵌入和时间自注意力机制，显著增强了对时间序列的处理能力。
            时间嵌入的初始化: 时间嵌入 temporal_embedding 通过正态分布进行初始化，标准 Transformer 通常没有这种特定的时间嵌入处理。
            位置嵌入:

            在传统 Transformer 中，位置嵌入（positional embeddings）是用于为序列中的位置提供信息的一部分。CLIPEncoderLayer 没有显式地使用位置嵌入，而是通过 temporal_embedding 和时间注意力处理时间维度。

            整体 CLIPEncoder 的设计：
            层堆叠: CLIPEncoder 由多个 CLIPEncoderLayer 堆叠而成，形成深层网络结构。这与标准的 Transformer 编码器设计是一致的。
            自定义功能: CLIPEncoder 主要在以下方面进行了自定义：
            时间处理: 在 CLIPEncoderLayer 中加入了时间注意力和时间嵌入，使模型能够处理具有时间序列特征的输入数据。这是标准 Transformer 编码器不具备的功能。
            梯度检查点: 通过 gradient_checkpointing 提供内存优化功能，以便在训练时处理较大的模型。

        PatchDropout:

        功能：这是一个自定义的层，用于对图像补丁应用 dropout。它可以帮助增加模型的鲁棒性和防止过拟合。
        自定义：PatchDropout 是根据配置 config.force_patch_dropout 自定义的，增加了模型的灵活性。
        pre_layrnorm 和 post_layernorm:

        功能：这两个 LayerNorm 层分别在编码器之前和之后应用。它们用于标准化层的输入和输出，有助于稳定训练过程。
        自定义：这些标准化层的设置和配置（例如 epsilon 值）可以根据需要进行调整。
        forward 方法:

        功能：定义了模型的前向传播过程。包括图像的预处理、嵌入计算、Dropout 应用、编码器前后的标准化以及最后的池化操作。
        自定义：在 forward 方法中，处理输入数据的形状和维度，以及对 pixel_values 的重新排列和转换是自定义的。此外，池化操作和隐藏状态的重新排列也是根据模型的需求进行的自定义操作。
        BaseModelOutputWithPooling:

        功能：这是模型的输出格式，包括最后的隐藏状态、池化后的输出、隐藏状态的列表和注意力权重。
        自定义：根据需要选择是否返回详细的模型输出，并对隐藏状态进行适当的重塑和处理。
        总结
        CLIPVisionTransformer 结合了 CLIP 模型的预训练组件（如视觉嵌入层和编码器）与自定义的功能（如 PatchDropout 和标准化层）。这种设计允许模型在利用预训练知识的同时，进行特定的定制和优化，以适应不同的任务或数据集。
    输入:

    pixel_values: 图像的像素值，通常是经过预处理的图像数据。
    输出:

    图像嵌入: 一个固定维度的向量，表示图像输入的特征。
    总结
    CLIPTextTransformer 负责将文本转换为嵌入，通过处理文本数据中的词汇和语法信息。
    CLIPVisionTransformer 负责将图像转换为嵌入，通过处理图像数据中的空间特征和内容信息。
    这两个模型的目标是将文本和图像映射到一个共享的嵌入空间中，以便能够进行对比学习。通过对比学习，CLIP 模型能够实现文本和图像之间的语义对齐，从而进行多模态检索和分类等任务。
    

## 其他信息补充
    1. 通过预定义一些Seperator的关键词进行user/assistant的对话对构建和prompt生成。  详情见 get_prompt 函数，其中根据config的不同，有不同的sep风格： One, Two....
    2. 检测keyword来实现文本生成长度限制，eg: KeywordsStoppingCriteria类
    3. person_search文件夹里有一个关于box的model和demo
    4. ActorTracking.py 里面定义了而怎么跟踪acotr的名字的函数，可以（待定之后细看！！！）
    5. 选用LLaVa模型原因：开源优势:

        可访问性: LLava 模型是开源的，这使得研究人员和开发者可以访问和修改其源代码，进行实验和自定义。
        社区支持: 开源模型通常有一个活跃的社区，可以提供支持和共享经验，帮助解决使用过程中遇到的问题。
        视觉处理能力:

        多模态处理: LLava 是一个多模态模型，旨在处理视觉和文本数据的融合。它不仅可以处理图像，还能处理图像和文本的组合任务，如图像描述生成、视觉问答等。
        视觉专长: 相较于 GPT 模型，LLava 专门设计用于处理视觉数据，可能在视觉任务上表现更好。GPT 模型主要处理文本数据，尽管它可以与图像模型结合，但本身并不处理图像
    6. LlavaLlamaModel 的定义，它结合了 LlavaMetaModel 和 LlamaModel。具体来说：

        LlavaMetaModel:

        多模态处理: LlavaMetaModel 可能专注于处理多模态数据，如图像和文本的结合。它负责将不同模态的数据（如图像和文本）进行融合，生成嵌入向量。
        自定义功能: LlavaMetaModel 可以包括用于处理图像或其他非文本数据的特定层和机制，例如视觉嵌入层和图像处理模块。
        LlamaModel:

        文本生成: LlamaModel 是处理文本数据的模型，专注于生成文本和处理文本嵌入。它可以从 LlavaMetaModel 生成的嵌入向量中进一步处理，并生成最终的文本输出。
        LlavaLlamaModel 的结合:

        组合能力: LlavaLlamaModel 结合了 LlavaMetaModel 的多模态处理能力和 LlamaModel 的文本处理能力。这样可以实现从多模态数据（如图像和文本的组合）生成文本的任务。
        初始化: 在 __init__ 方法中，LlavaLlamaModel 初始化了两个模型，并可能利用 LlamaModel 处理从 LlavaMetaModel 生成的嵌入向量。

    7. ！！！！！！理论上 LlamaModel 部分可以被其他文本语言模型替换，只要 Llava 部分能够有效地处理图像数据并生成合适的嵌入向量。这里的核心思想是：！！！！（日后的可能修改之处）

        图像处理和嵌入生成: Llava 模块（或 LlavaMetaModel）负责从图像和其他模态数据中提取特征和生成嵌入向量。只要这个部分能够有效地处理图像数据并生成有意义的嵌入，就可以替换不同的文本处理模块。

        文本生成和处理: 生成文本或处理文本的任务可以交给任何适合的文本语言模型。这意味着你可以用其他强大的语言模型（如 GPT、T5、BERT 等）替换 LlamaModel 部分，只要这些模型能够接受 Llava 生成的嵌入向量作为输入并生成合理的文本。

        替换的考虑因素
        嵌入向量的兼容性: 替换文本模型时，需要确保新的文本模型能够接受和有效利用 Llava 生成的嵌入向量。这通常涉及到输入和输出维度的匹配，以及模型的训练和优化方式。
        任务需求: 根据具体任务的需求（如问答、生成描述、对话等），选择合适的文本语言模型来处理生成的嵌入向量。