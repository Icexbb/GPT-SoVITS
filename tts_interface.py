import logging
import os
import re

import LangSegment
import librosa
import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BertForMaskedLM,
)

from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.my_utils import load_audio
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.ERROR)

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = True

cnhubert.cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

SPLIT_CHAR = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

DICT_LANG = {
    "中文": "all_zh",
    "英文": "en",
    "日文": "all_ja",
    "中英混合": "zh",
    "日英混合": "ja",
    "多语种混合": "auto",
}
dtype = torch.float16 if is_half else torch.float32


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class BertFunc:
    tokenizer: PreTrainedTokenizerFast
    bert_model: BertForMaskedLM
    ssl_model: cnhubert.CNHubert

    def __init__(self):
        bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half:
            bert_model = bert_model.half()
        self.bert_model = bert_model.to(device)

        ssl_model = cnhubert.get_model()
        if is_half:
            ssl_model = ssl_model.half()
        self.ssl_model = ssl_model.to(device)

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros((1024, len(phones)), dtype=dtype).to(device)
        return bert

    def nonen_get_bert_inf(self, text, language):
        if language != "auto":
            textlist, langlist = TextFunc.split_en_inf(text, language)
        else:
            textlist = []
            langlist = []
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        bert_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = TextFunc.clean_text_inf(textlist[i], lang)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)

        return bert

    def get_bert_final(self, phones, word2ph, text, language, device):
        if language == "en":
            bert = self.get_bert_inf(phones, word2ph, text, language)
        elif language in {"zh", "ja", "auto"}:
            bert = self.nonen_get_bert_inf(text, language)
        elif language == "all_zh":
            bert = self.get_bert_feature(text, word2ph).to(device)
        else:
            bert = torch.zeros((1024, len(phones))).to(device)
        return bert


class TextFunc:
    @staticmethod
    def split_en_inf(sentence: str, language: str) -> tuple[list[str], list[str]]:
        pattern = re.compile(r"[a-zA-Z ]+")
        textlist = []
        langlist = []
        pos = 0
        for match in pattern.finditer(sentence):
            start, end = match.span()
            if start > pos:
                textlist.append(sentence[pos:start])
                langlist.append(language)
            textlist.append(sentence[start:end])
            langlist.append("en")
            pos = end
        if pos < len(sentence):
            textlist.append(sentence[pos:])
            langlist.append(language)
        # Merge punctuation into previous word
        for i in range(len(textlist) - 1, 0, -1):
            if re.match(r"^[\W_]+$", textlist[i]):
                textlist[i - 1] += textlist[i]
                del textlist[i]
                del langlist[i]
        # Merge consecutive words with the same language tag
        i = 0
        while i < len(langlist) - 1:
            if langlist[i] == langlist[i + 1]:
                textlist[i] += textlist[i + 1]
                del textlist[i + 1]
                del langlist[i + 1]
            else:
                i += 1

        return textlist, langlist

    @staticmethod
    def clean_text_inf(text, language):
        formatted_text = ""
        language = language.replace("all_", "")
        for tmp in LangSegment.getTexts(text):
            if language == "ja":
                if tmp["lang"] == language or tmp["lang"] == "zh":
                    formatted_text += tmp["text"] + " "
                continue
            if tmp["lang"] == language:
                formatted_text += tmp["text"] + " "
        while "  " in formatted_text:
            formatted_text = formatted_text.replace("  ", " ")
        phones, word2ph, norm_text = clean_text(formatted_text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    @staticmethod
    def nonen_clean_text_inf(text, language):
        if language != "auto":
            textlist, langlist = TextFunc.split_en_inf(text, language)
        else:
            textlist = []
            langlist = []
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        phones_list = []
        word2ph_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = TextFunc.clean_text_inf(textlist[i], lang)
            phones_list.append(phones)
            if lang == "zh":
                word2ph_list.append(word2ph)
            norm_text_list.append(norm_text)
        phones = sum(phones_list, [])
        word2ph = sum(word2ph_list, [])
        norm_text = " ".join(norm_text_list)

        return phones, word2ph, norm_text

    @staticmethod
    def get_first(text):
        pattern = "[" + "".join(re.escape(sep) for sep in SPLIT_CHAR) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    @staticmethod
    def get_cleaned_text_final(text, language):
        if language in {"en", "all_zh", "all_ja"}:
            phones, word2ph, norm_text = TextFunc.clean_text_inf(text, language)
        else:  # elif language in {"zh", "ja", "auto"}:
            phones, word2ph, norm_text = TextFunc.nonen_clean_text_inf(text, language)
        return phones, word2ph, norm_text

    @staticmethod
    def merge_short_text_in_array(texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    @staticmethod
    def cut_text(inp: str, text_language: str, ponds=None) -> str:
        if ponds is None:
            ponds = SPLIT_CHAR
        if inp[-1] not in SPLIT_CHAR:
            if text_language == "en" or text_language == "auto":
                inp += "."
            else:
                inp += "。"
        inp = inp.strip("\n")
        punds = rf"[{''.join([c for c in ponds])}]"
        items = re.split(f"({punds})", inp)
        items = ["".join(group) for group in zip(items[::2], items[1::2])]
        opt = "\n".join(items)
        return opt


class TTS:
    vq_model: SynthesizerTrn
    hps: DictToAttrRecursive
    hz: int
    max_sec: any
    t2s_model: Text2SemanticLightningModule
    config: any

    prompt_path: str
    prompt_text: str
    prompt_lang: str

    top_k: int
    top_p: float
    temperature: float

    bert_functions: BertFunc

    def __init__(self):
        self.bert_functions = BertFunc()

    def set_model_from_dataset(self, dataset: str):
        with open(dataset, "r", encoding="utf-8") as f:
            dataset = f.read().split("\n")
        dataset = [line.strip().split("|") for line in dataset if "|" in line]
        self.set_model(dataset[0][1])
        for line in dataset:
            wav16k, _ = librosa.load(line[0], sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                continue
            else:
                self.set_prompt(line[0], line[-1], "ja")
                break

    def set_model(self, model_name: str):
        sov_list = os.listdir("SoVITS_weights")
        gpt_list = os.listdir("GPT_weights")
        sov_model = ""
        gpt_model = ""
        for sov in sov_list:
            if model_name in sov:
                sov_model = sov
        for gpt in gpt_list:
            if model_name in gpt:
                gpt_model = gpt

        if sov_model == "" or gpt_model == "":
            raise FileNotFoundError("Model not found")

        sovits_path = os.path.join("SoVITS_weights", sov_model)
        gpt_path = os.path.join("GPT_weights", gpt_model)
        self.set_model_path(sovits_path, gpt_path)
        self.set_model_args()

    def set_model_path(self, sovits_path: str, gpt_path: str):
        self.change_sovits_weights(sovits_path)
        self.change_gpt_weights(gpt_path)
        print("模型加载完成！")
        print("SoVITS Model:", sovits_path)
        print("GPT Model:", gpt_path)

    def change_sovits_weights(self, sovits_path: str):
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        if "pretrained" not in sovits_path:
            del vq_model.enc_q
        if is_half:
            vq_model = vq_model.half().to(device)
        else:
            vq_model = vq_model.to(device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)

        self.vq_model = vq_model
        self.hps = hps

    def change_gpt_weights(self, gpt_path: str):
        hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if is_half:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(device)
        t2s_model.eval()

        self.hz = hz
        self.max_sec = max_sec
        self.t2s_model = t2s_model
        self.config = config

    def set_prompt(self, path: str, text: str, lang: str):
        self.prompt_path = path
        self.prompt_text = text.strip("\n")
        if lang in DICT_LANG.values():
            self.prompt_lang = lang
        else:
            self.prompt_lang = DICT_LANG.get(lang, "auto")

        if self.prompt_text[-1] not in SPLIT_CHAR:
            self.prompt_text += "。" if self.prompt_lang != "en" else "."

        print("参考音频:", self.prompt_path)
        print("参考文本:", self.prompt_text)

    def set_model_args(
        self, top_k: int = 20, top_p: float = 0.6, temperature: float = 0.6
    ):
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def get_tts_wav(self, process_text, process_lang):
        with torch.no_grad():
            wav16k, _ = librosa.load(self.prompt_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError("参考音频在3~10秒范围外，请更换！")

            zero_wav = np.zeros(
                int(self.hps.data.sampling_rate * 0.3),
                dtype=np.float16 if is_half else np.float32,
            )
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)

            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.bert_functions.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)

            prompt_semantic = codes[0, 0]
        phones1, word2ph1, norm_text1 = TextFunc.get_cleaned_text_final(
            self.prompt_text, self.prompt_lang
        )
        bert1 = self.bert_functions.get_bert_final(
            phones1, word2ph1, norm_text1, self.prompt_lang, device
        ).to(dtype)
        print("前端处理后的参考文本:%s" % norm_text1)
        prompt = prompt_semantic.unsqueeze(0).to(device)

        process_text = process_text.strip("\n")
        if (
            process_text[0] not in SPLIT_CHAR
            and len(TextFunc.get_first(process_text)) < 4
        ):
            process_text = (
                "。" + process_text if process_lang != "en" else "." + process_text
            )
        print("实际输入的目标文本:", process_text)
        process_text = TextFunc.cut_text(
            process_text,
            process_lang,
            ["。", "？", "！", ".", "?", "!", "~", ":", "：", "—", "…"],
        )

        while "\n\n" in process_text:
            process_text = process_text.replace("\n\n", "\n")
        print("实际输入的目标文本(切句后):", process_text)
        texts = process_text.split("\n")
        texts = TextFunc.merge_short_text_in_array(texts, 5)

        audio_opt = []

        for process_text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if len(process_text.strip()) == 0:
                continue
            if process_text[-1] not in SPLIT_CHAR:
                process_text += "。" if process_lang != "en" else "."
            print("实际输入的目标文本(每句):", process_text)
            phones2, word2ph2, norm_text2 = TextFunc.get_cleaned_text_final(
                process_text, process_lang
            )
            print("前端处理后的文本(每句):", norm_text2)
            bert2 = self.bert_functions.get_bert_final(
                phones2, word2ph2, norm_text2, process_lang, device
            ).to(dtype)

            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = (
                torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            )

            bert = bert.to(device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = self.get_spec()
            if is_half:
                refer = refer.half()
            refer = refer.to(device)

            audio = (
                self.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(device).unsqueeze(0),
                    refer,
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )  # 试试重建不带上prompt部分
            max_audio = np.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1:
                audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
        yield self.hps.data.sampling_rate, (
            np.concatenate(audio_opt, 0) * 32768
        ).astype(np.int16)

    def get_spec(self):
        audio = load_audio(self.prompt_path, int(self.hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return spec

    def get_voice(self, text: str, lang: str) -> tuple[int, np.ndarray]:
        text = text.replace("...", "…")
        if lang not in DICT_LANG.values():
            lang = DICT_LANG.get(lang, "auto")

        res = self.get_tts_wav(text, lang)
        sr, wav = next(res)
        return sr, wav


if __name__ == "__main__":
    tts = TTS()
    dataset_list = ""
    tts.set_model_from_dataset(dataset_list)
    tts.get_voice("你好", "中文")
