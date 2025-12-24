import re
from pathlib import Path

def get_config_val(file, var_name):
    content = Path(file).read_text()
    match = re.search(f"{var_name} = (.*)", content)
    return match.group(1) if match else "Unknown"

doc_content = f"""
# System Status: LLM Vault
**Last Updated:** Automatically via GitHub Actions

## Current Configurations
- **Embedding Model:** {get_config_val('build_kb.py', 'EMBED_MODEL')}
- **Chunk Size:** {get_config_val('build_kb.py', 'CHUNK_SIZE')}
- **Similarity Threshold:** {get_config_val('vault_query.py', 'SIM_THRESHOLD')}

## Architecture Notes
- Uses **KNN Search** (Looping through JSON).
- Uses **Cosine Similarity** for ranking.
"""
Path("SYSTEM_OVERVIEW.md").write_text(doc_content)
