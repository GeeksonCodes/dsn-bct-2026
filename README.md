# DSN-BCT LLM Agent

An LLM-based agent system for data science tasks, split into two main domains:
- **Task A**: User Modeling (Teammate's domain)
- **Task B**: Recommendation System (Your domain)

## Project Structure

```
dsn-bct-llm-agent/
├── data/                  # Datasets
│   ├── raw/              # Original data
│   ├── processed/        # Processed data
│   └── README.md
├── notebooks/            # Jupyter notebooks
│   ├── eda/              # Exploratory data analysis
│   └── experiments/      # Quick experiments
├── src/                  # Source code
│   ├── shared/           # Shared utilities and Nigerian layer
│   ├── task_a/           # User modeling modules
│   └── task_b/           # Recommendation modules
├── app/                  # Containerized applications
│   ├── task_a/           # Task A app
│   └── task_b/           # Task B app
├── paper/                # LaTeX files (synced from Overleaf)
├── docker-compose.yml    # Docker composition for both tasks
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose (for containerized apps)
- git (for Overleaf sync)

### Installation

1. Clone the repository
```bash
git clone https://github.com/your-org/dsn-bct-llm-agent.git
cd dsn-bct-llm-agent
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Local Development
```bash
# Task A
python -m src.task_a.main

# Task B
python -m src.task_b.main
```

### Docker Containers
```bash
# Build and run both services
docker-compose up --build

# Run specific service
docker-compose up task_a
docker-compose up task_b
```

## Development

### Notebooks
- **EDA**: Exploratory data analysis notebooks in `notebooks/eda/`
- **Experiments**: Quick prototyping and experimentation in `notebooks/experiments/`

### Code Organization
- **Shared**: Common utilities and the Nigerian layer are in `src/shared/`
- **Task A**: User modeling code in `src/task_a/`
- **Task B**: Recommendation system code in `src/task_b/`

### Testing
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint
flake8 src/
```

## Paper

LaTeX source files for the research paper are in the `paper/` directory. These are synced with Overleaf for collaborative editing.

## Contributing

1. Create a feature branch
2. Commit changes
3. Push to origin
4. Open a pull request

## License

[Add your license here]

## Authors

- Task A Lead: [Teammate Name]
- Task B Lead: [Your Name]
