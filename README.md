# Onix Project Management

**Onix Project Management** is a lightweight project management system for **ComfyUI**, designed for long-running, prompt-driven video generation. It simplifies workflow management by saving prompts to disk and handling sequential prompts.

## ✨ Highlights

-   **Onix Project Loader** → Project manager node (saves/loads scripts, computes start points)

## ✨ Features

-   Save and load prompt scripts as JSON files
-   Automatic `start_prompt` calculation based on existing output folders    
-   Resume generation from any prompt index    
-   Extend projects with new prompts    
-   Organized output structure for easy project management    

## 🧰 Requirements
- ComfyUI (current version).
- Python 3.10+ recommended.

## ⚙️ Installation

### Manual Installation
1.  Navigate to your ComfyUI `custom_nodes` directory.
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/yourusername/comfy-onix-project-managment.git
    ```
3.  Restart ComfyUI.

---