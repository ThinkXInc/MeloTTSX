import os
import click
from melo.api import TTS

@click.command()
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
@click.option('--config_path', '-c', type=str, default=None, help="Path to the config file")
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output directory")
@click.option('--speaker', '-s', type=str, default=None, help="Specify the speaker (e.g., EN-US)")
def main(ckpt_path, config_path, text, language, output_dir, speaker):
    if ckpt_path is None:
        #raise ValueError("The model_path must be specified")
        model_id = ""
    else:
        model_id = ckpt_path.split('/')[-1]
    
    if config_path is None:
        # Optionally infer the config path
        pass
    
    if text is None:
        raise ValueError("Text must be specified")
    
    # Initialize TTS model
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    
    if speaker:
        # Use specified speaker
        if speaker not in model.hps.data.spk2id:
            raise ValueError(f"Speaker '{speaker}' not found in model. Available speakers: {list(model.hps.data.spk2id.keys())}")
        spk_id = model.hps.data.spk2id[speaker]
        save_path = f'{output_dir}/{speaker}/{text}_{model_id}_{speaker}.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spk_id, save_path)
        #tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        print(f"Saved TTS output for speaker '{speaker}' to {save_path}")
    else:
        # Process all speakers if no specific speaker is provided
        for spk_name, spk_id in model.hps.data.spk2id.items():
            save_path = f'{output_dir}/{spk_name}/{text}_{model_id}.wav'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.tts_to_file(text, spk_id, save_path)
            print(f"Saved TTS output for speaker '{spk_name}' to {save_path}")

if __name__ == "__main__":
    main()
