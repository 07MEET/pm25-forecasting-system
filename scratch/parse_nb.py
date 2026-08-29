import json

with open("visuals.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/nb_parsed.txt", "w", encoding="utf-8") as out:
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        out.write(f"=== Cell {idx} [{cell_type}] ===\n")
        out.write(source + "\n")
        out.write("\n" + "-"*50 + "\n\n")

print("Done parsing notebook.")
