import os
from mutagen.mp3 import MP3

def calculate_total_duration(dataset_path):
    total_duration = 0
    audio_extensions = ['.mp3']
    file_count = 0
    error_count = 0
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(root, file)
                try:
                    audio = MP3(file_path)
                    total_duration += audio.info.length
                    file_count += 1
                    if file_count % 100 == 0:
                        print(f"Processados {file_count} arquivos...")
                except Exception as e:
                    error_count += 1
                    print(f"Erro ao processar {file_path}: {e}")

    hours = total_duration / 3600
    return hours, total_duration, file_count, error_count

dataset_path = "../data/validated-audios"
hours, seconds, file_count, error_count = calculate_total_duration(dataset_path)
print(f"\n{'='*50}")
print(f"Duração total: {hours:.2f} horas ({int(hours)} horas e {int((hours % 1) * 60)} minutos)")
print(f"Duração em segundos: {seconds:.2f}s")
print(f"Total de arquivos processados: {file_count}")
print(f"Arquivos com erro: {error_count}")
print(f"{'='*50}")