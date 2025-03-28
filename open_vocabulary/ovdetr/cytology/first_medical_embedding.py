import json

# Your lesion classes
lesion_classes = [
    "ascus", "asch", "lsil", "hsil", "scc",
    "agc", "trichomonas", "candida", "flora",
    "herps", "actinomyces"
]

# 1. Create split_classes.json (manual split example)
split_config = {
    "seen": [
        "agc", "asch", "hsil",
        "trichomonas", "candida", "flora",
        "scc", "actinomyces"
    ],
    "unseen": ["lsil", "ascus", "ascus"],
    "all": lesion_classes
}

# 2. Create medical_embeddings.json with clinical descriptions
medical_embeddings = {
    "ascus": "atypical squamous cells of undetermined significance with mild nuclear abnormalities",
    "asch": "atypical squamous cells where high-grade lesion cannot be excluded",
    "lsil": "low-grade squamous intraepithelial lesion with koilocytic changes",
    "hsil": "high-grade squamous intraepithelial lesion showing marked nuclear atypia",
    "scc": "squamous cell carcinoma with malignant cellular features and invasion",
    "agc": "atypical glandular cells indicating potential neoplastic changes",
    "trichomonas": "flagellated protozoan organisms with associated inflammatory changes",
    "candida": "fungal organisms with pseudohyphae and budding yeast forms",
    "flora": "shift in vaginal microbiota with predominance of bacterial organisms",
    "herps": "viral cytopathic effects with multinucleated giant cells and molding",
    "actinomyces": "filamentous bacteria with sulfur granule formation"
}

# Save files
with open("split_classes.json", "w") as f:
    json.dump(split_config, f, indent=2)

with open("medical_embeddings.json", "w") as f:
    json.dump(medical_embeddings, f, indent=2)

print("Files created successfully!")