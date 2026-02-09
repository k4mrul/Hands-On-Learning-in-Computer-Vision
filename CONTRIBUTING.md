# Contributing to Hands-On Learning in Computer Vision

First off, thank you for considering contributing to this repository! Whether you're fixing a bug, adding a new use case, or suggesting ideas for the community, your help is greatly appreciated.

---

## How to Contribute

### 1. Submit a Pull Request for an Interesting Use Case

Have a cool computer vision use case you've been working on? We'd love to see it! Here's how:

1. **Fork** this repository.
2. **Create a new branch** from `main`:
   ```bash
   git checkout -b your-use-case-name
   ```
3. **Add your notebook or script** to the appropriate folder (see the folder structure below).
4. Make sure your contribution:
   - Includes a clear, well-documented Jupyter notebook (`.ipynb`) or Python script.
   - Has a descriptive title and markdown cells explaining each step.
   - Lists all required dependencies at the top of the notebook.
   - Runs end-to-end without errors (preferably tested on Google Colab or Kaggle).
5. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "Add: <short description of your use case>"
   git push origin your-use-case-name
   ```
6. **Open a Pull Request** against the `main` branch with:
   - A clear title describing the use case.
   - A brief summary of what the notebook does and why it's useful.
   - Links to any datasets, papers, or references used.

### 2. Add a New Folder for a New Domain

Don't see a folder that fits your use case? Feel free to create one!

- The repository is organized by **industry/domain** (e.g., `Healthcare/`, `Automobile/`, `Manufacturing/`, `Security/`, `Retail/`, `Sports/`, `Robotics/`, etc.).
- If your use case belongs to a new domain, simply **create a new folder** at the root level with a clear, descriptive name (e.g., `Agriculture/`, `Education/`, `Energy/`).
- Place your notebook(s) inside the new folder and submit a PR as described above.

### 3. Open an Issue to Request Notebooks from the Community

Want to see a notebook on a specific topic but don't have time to build it yourself? Open an issue!

1. Go to the [Issues](../../issues) tab.
2. Click **New Issue**.
3. Use a clear, descriptive title, for example:
   - "Request: Fine-tune YOLO for underwater object detection"
   - "Idea: Notebook for document layout analysis using Florence 2"
4. In the issue body, describe:
   - **What** the notebook should cover.
   - **Why** it would be valuable to the community.
   - Any **references** (papers, datasets, blog posts) that could help.
5. Add the label `notebook-request` if available, or just tag it clearly in the title.

The community (and maintainers) can then pick up these issues and submit PRs!

---

## Repository Structure

```
Hands-On-Learning-in-Computer-Vision/
├── Automobile/                          # Automotive use cases
├── Build your AI Agents/               # AI agent tutorials
├── Construction/                        # Construction industry
├── fine-tune YOLO for various use cases/ # YOLO fine-tuning notebooks
├── Healthcare/                          # Healthcare & medical imaging
├── Life Sciences and Biotechnology/     # Life sciences
├── Manufacturing/                       # Manufacturing & QA
├── Model Notebooks/                     # Model-specific tutorials
├── Retail/                              # Retail & e-commerce
├── Robotics/                            # Robotics applications
├── SDK Tutorials/                       # Labellerr SDK guides
├── Security/                            # Security & surveillance
├── Sports/                              # Sports analytics
├── CONTRIBUTING.md                      # This file
├── README.md                            # Project overview
└── SECURITY.md                          # Security policy
```

---

## Guidelines for Notebooks

To keep things consistent and useful for everyone, please follow these guidelines:

- **One notebook per use case** -- keep notebooks focused on a single problem.
- **Use clear markdown headings** to structure the notebook (Introduction, Setup, Data, Training, Evaluation, Results).
- **Include a "Requirements" or "Setup" section** at the top with all `pip install` commands.
- **Add comments** in your code to explain key steps.
- **Use publicly available datasets** whenever possible (or provide clear instructions on how to obtain the data).
- **Include sample outputs** (images, metrics, logs) so readers can see expected results without running the notebook.
- **Test on Google Colab** to ensure broad accessibility.

---

## Code of Conduct

Please be respectful and constructive in all interactions. We're building a community of learners and practitioners -- let's keep it welcoming for everyone.

---

## Questions?

If you have any questions about contributing, feel free to open an issue or reach out via the links in the [README](README.md).

Happy building!
