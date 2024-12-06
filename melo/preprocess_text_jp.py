import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
import os
import torch
from text.symbols import symbols, num_languages, num_tones

@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--pretrained-config-path", default="/src/quantz/processing-server/tts4/pretrained_checkpoints/JP_config.json")
@click.option("--pretrained-checkpoint-path", default="/src/quantz/processing-server/tts4/pretrained_checkpoints/JP_checkpoint.pth")
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    pretrained_config_path: str,
    pretrained_checkpoint_path: str,
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        new_symbols = []
        for line in tqdm(open(metadata, encoding="utf-8").readlines()):
            try:
                audio_path, spk, language, text = line.strip().split("|")

                # Extract transcript for jsut, moraspeech
                print('\n')
                print(audio_path)
                if spk == "JP-jsut":
                    # speaker JP-jsut
                    #   audio_path: /disk1/jsut/{jsut_category}/wav/{jsut_category}_{i}.wav
                    #   transcript_path: /disk1/jsut/{jsut_category}/transcripts/{jsut_category}_{i}.txt
                    #   (ex)
                    #       audio_path: /disk1/jsut/basic5000/wav/BASIC5000_0001.wav
                    #       transcript_path: /disk1/jsut/basic5000/transcripts/BASIC5000_0001.txt
                    jsut_category, file_id = audio_path.split("/")[-3], audio_path.split("/")[-1].split("_")[1].split(".")[0]
                    print('-----------------')
                    print(jsut_category)
                    print(file_id)
                    print('-----------------')
                    transcript_path = f"/disk1/jsut/{jsut_category}/transcripts/{jsut_category.upper()}_{file_id}.txt"
                elif spk == "JP-moraspeech":
                    # speaker JP-moraspeech
                    # jsut
                    #   audio_path: /disk1/moraspeech/wavs/jsut/JSUT_{i}.wav
                    #   transcript_path: /disk1/jsut/basic5000/BASIC5000_{i}.txt
                    # original
                    #   audio_path: /disk1/moraspeech/wavs/original/{i}.wav
                    #   transcript_path: /disk1/moraspeech/kana_transcripts/original/{i}.txt
                    if "jsut" in audio_path:
                        file_id = audio_path.split("/")[-1].split("_")[1].split(".")[0]
                        transcript_path = f"/disk1/jsut/basic5000/BASIC5000_{file_id}.txt"
                    else:
                        file_id = audio_path.split("/")[-1].split(".")[0]
                        transcript_path = f"/disk1/moraspeech/kana_transcripts/original/{file_id}.txt"
                    print('-----------------')
                    print(file_id)
                    print(transcript_path)
                    print('-----------------')
 

                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()

                print(f"Transcript: {transcript} for audio path {audio_path}")

                norm_text, phones, tones, word2ph, bert = clean_text_bert(transcript, language, device='cuda:0')
                for ph in phones:
                    if ph not in symbols and ph not in new_symbols:
                        new_symbols.append(ph)
                        print('update!, now symbols:')
                        print(new_symbols)
                        with open(f'{language}_symbol.txt', 'w') as f:
                            f.write(f'{new_symbols}')

                assert len(phones) == len(tones)
                assert len(phones) == sum(word2ph)
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        audio_path,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
                bert_path = audio_path.replace(".wav", ".bert.pt")
                os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                torch.save(bert.cpu(), bert_path)
            except Exception as error:
                print("err!", line, error)

        out_file.close()

        metadata = cleaned_path

    spk_utt_map = defaultdict(list)

    # Load existing spk2id mapping
    with open(pretrained_config_path, 'r', encoding='utf-8') as f:
        pretrained_config = json.load(f)
    existing_n_speakers = pretrained_config['data']['n_speakers']
    existing_spk2id = pretrained_config['data']['spk2id']
    current_sid = existing_n_speakers
    spk_id_map = existing_spk2id.copy()
    print(f'existing speakers {existing_n_speakers}')
    print(f'existing spk2id {existing_spk2id}')


    # Process new metadata
    new_n_speakers = 0
    with open(cleaned_path, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map:
                print(f'added new speaker {spk} as index [{current_sid}]')
                spk_id_map[spk] = current_sid
                new_n_speakers += 1
                current_sid += 1

    total_n_speakers = existing_n_speakers + new_n_speakers
    print(f'total speaker num {total_n_speakers}')

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(pretrained_config_path, encoding="utf-8"))
    config["data"]["n_speakers"] = total_n_speakers
    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["pretrained_checkpoint_path"] = pretrained_checkpoint_path
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    # Load default config and update missing parameters
    default_config = json.load(open(config_path, encoding='utf-8'))
    for section in ['train', 'data']:
        for key, value in default_config[section].items():
            if key not in config[section]:
                config[section][key] = value

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
