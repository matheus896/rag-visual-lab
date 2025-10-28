class DatasetsProvider:
    def __init__(self):
        self.datasets = [
            {
                "dataset": "synthetic_dataset_papers",
                "description": "A construção, utilização ou detecção usando datasets sintéticos",
                "locale": "en",
            },
            {
                "dataset": "direito_constitucional",
                "description": "Se a consulta envolver direito, leis, processos ou jurisprudência",
                "locale": "pt-br",
            },
        ]

    def get_datasets(self):
        return self.datasets

    def get_dataset_description(self):
        description = ""
        for dataset in self.datasets:
            description += f"- {dataset['description']} escreva -> {dataset['dataset']}. Dataset Locale: {dataset['locale']}\n"
        return description

