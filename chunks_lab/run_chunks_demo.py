from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Permite importar chunks.py mesmo executando o script dentro de chunks_lab
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunks import Chunks


def main() -> None:
    sample_text = (
        "Os drones estão revolucionando diversas indústrias, desde o monitoramento "
        "ambiental até entregas de última milha.\n\n"
        "Entretanto, o uso massivo desses dispositivos exige novas soluções de "
        "detecção para garantir segurança e privacidade.\n\n"
        "Métodos tradicionais não acompanham a velocidade dessa evolução, "
        "tornando essencial o uso de abordagens modernas."
    )

    chunks = Chunks(chunk_size=120, overlap_size=30)

    print("== Informações iniciais do chunker ==")
    pprint(chunks.get_chunk_info())
    print()

    basic_chunks = chunks.create_chunks(sample_text)
    print("== Chunks gerados ==")
    for idx, chunk_text in enumerate(basic_chunks, start=1):
        print(f"Chunk {idx}:\n{chunk_text}\n")

    metadata_chunks = chunks.create_chunks_with_metadata(
        sample_text,
        source_info={"document": "demo", "author": "team"},
    )
    print("== Chunks com metadados ==")
    pprint(metadata_chunks)
    print()

    print("== Atualizando parâmetros ==")
    chunks.update_settings(chunk_size=80, overlap_size=20)
    pprint(chunks.get_chunk_info())
    print()

    updated_chunks = chunks.create_chunks(sample_text)
    print("== Novos chunks após atualização ==")
    for idx, chunk_text in enumerate(updated_chunks, start=1):
        print(f"Chunk {idx}:\n{chunk_text}\n")


if __name__ == "__main__":
    main()