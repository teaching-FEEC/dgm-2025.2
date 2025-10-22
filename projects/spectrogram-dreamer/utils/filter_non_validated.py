import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

def filter_and_copy_validated(
    metadata_file: str,
    clips_dir: str,
    output_dir: str,
    min_votes: int = 2,
    copy_files: bool = True
):
    df = pd.read_csv(metadata_file, sep='\t')
    
    print(f"\nEstatísticas do dataset:")
    print(f"Total de clips: {len(df)}")
    
    # Filtrar validados
    # Common Voice considera validado quando votes_up > votes_down
    if 'votes_up' in df.columns and 'votes_down' in df.columns:
        validated_clips = df[
            (df['votes_up'] > df['votes_down']) & 
            (df['votes_up'] >= min_votes)
        ]
    else:
        # Se não tem votos, assume que arquivo validated.tsv já está filtrado
        validated_clips = df
    
    print(f"Clips validados: {len(validated_clips)}")
    print(f"Taxa de validação: {len(validated_clips)/len(df)*100:.1f}%")
    
    # Estatísticas de duração
    if 'duration' in validated_clips.columns:
        total_duration = validated_clips['duration'].sum()
        hours = total_duration / 3600
        print(f"\nDuração total validada: {hours:.2f} horas")
        print(f"Duração média por clip: {validated_clips['duration'].mean():.2f} segundos")
    
    # Criar pasta de destino
    clips_path = Path(clips_dir)
    output_path = Path(output_dir)
    
    if copy_files:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copiar arquivos validados
        print(f"\nCopiando arquivos para {output_dir}...")
        
        copied = 0
        not_found = 0
        
        for _, row in tqdm(validated_clips.iterrows(), total=len(validated_clips)):
            filename = row['path']
            source = clips_path / filename
            destination = output_path / filename
            
            if source.exists():
                shutil.copy2(source, destination)
                copied += 1
            else:
                not_found += 1
        
        print(f"\n✅ Copiados: {copied} arquivos")
        if not_found > 0:
            print(f"⚠️ Não encontrados: {not_found} arquivos")
    
    # Salvar metadata filtrado
    metadata_output = output_path / 'validated_metadata.tsv'
    validated_clips.to_csv(metadata_output, sep='\t', index=False)
    print(f"\n📝 Metadata salvo em: {metadata_output}")
    
    return validated_clips

parser = argparse.ArgumentParser(
    description='Filtrar e copiar clips validados do Common Voice'
)

parser.add_argument(
    '--metadata',
    type=str,
    required=True,
    help='Caminho para validated.tsv ou clips.tsv'
)

parser.add_argument(
    '--clips-dir',
    type=str,
    required=True,
    help='Pasta com os arquivos .mp3 originais'
)

parser.add_argument(
    '--output-dir',
    type=str,
    required=True,
    help='Pasta de destino para clips validados'
)

parser.add_argument(
    '--min-votes',
    type=int,
    default=2,
    help='Número mínimo de votos positivos (padrão: 2)'
)

parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Apenas mostra estatísticas sem copiar arquivos'
)

args = parser.parse_args()

filter_and_copy_validated(
    metadata_file=args.metadata,
    clips_dir=args.clips_dir,
    output_dir=args.output_dir,
    min_votes=args.min_votes,
    copy_files=not args.dry_run
)