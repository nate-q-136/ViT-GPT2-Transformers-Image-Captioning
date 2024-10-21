from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel

def get_tokenizer(name:str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer

def get_feature_extractor(name:str):
    return ViTFeatureExtractor.from_pretrained(name)

def get_vit_gpt2_model(feature_extractor_name:str, tokenizer_name:str, pretrained:bool, model_pretrained_name:str)->VisionEncoderDecoderModel:
    if pretrained:
        model = VisionEncoderDecoderModel.from_pretrained(model_pretrained_name)
    else:
        # feature_extractor = get_feature_extractor(feature_extractor_name)
        tokenizer = get_tokenizer(tokenizer_name)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(feature_extractor_name, tokenizer_name)
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        # make sure vocab size is set correctly
        model.config.vocab_size = model.config.decoder.vocab_size
        # set beam search parameters
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.max_length = 128
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
    return model