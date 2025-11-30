# Instructions for Code Agent - C. elegans Trajectory Analysis

You are a specialized code agent for implementing C. elegans behavioral analysis methodologies. Your task is to implement analysis methodologies to **distinguish different treatment effects within the same strain** using trajectory data.

## Context
- **Data Type**: C. elegans trajectory data consisting of (x,y) position coordinates by frame
- **Goal**: Detect behavioral differences caused by different treatments/drugs applied to the same strain
- **Testing Strain**: Use strain n2 for all implementations and testing
- **Methodologies**: The `proposed_methodology.md` contains 5 distinct methodologies to implement

## Work Process

1. **FIRST STEP - METHODOLOGY ANALYSIS:**
   - Read and fully analyze the `proposed_methodology.md` file
   - Identify all 5 methodologies: TDA, Multiview Tracking, ML Screening, CNN Posture Classification, and Integrative Modeling
   - Examine the `generate_trajectory_visualizations.py` file (previously `run_batch_analysis.py`) to understand data loading structure
   - Understand that each methodology should differentiate between treatment conditions in the same strain

2. **SECOND STEP - IMPLEMENTATION:**
   - Create ONE separate Python file for EACH of the 5 methodologies
   - Name files descriptively: `methodology_[descriptive_name].py`
   - Each file must focus on **treatment comparison** for the same strain
   - Each file must be independent and functional

3. **THIRD STEP - ITERATIVE TESTING & CORRECTION:**
   - After creating all files, execute them ONE BY ONE on strain n2 data
   - Test and debug each methodology individually
   - Fix any issues found before moving to the next file

## Technical Requirements

- **Data loading:** Replicate exactly the data loading mechanism from `generate_trajectory_visualizations.py` (previously `run_batch_analysis.py`)
- **Data format:** Work with trajectory data containing (x,y) coordinates by frame for C. elegans
- **Implementation focus:** Each methodology must be designed to distinguish between different treatment effects
- **Implementation detail:s** Each methodology at `proposed_methodologies.md` came from a paper at `Comportamiento/References`. Read them on demand to get details like hyperparams used or architectures and so on. Remember to use this in the context of this data.
- **Treatment comparison:** Implement classification/analysis to differentiate control vs. treated conditions, initially filtering by strain n2
- **Structure:** One file per methodology (expect 5 total files), no cross-dependencies
- **Naming:** Use clear and descriptive file names based on methodology purpose
- **Testing scope:** Verify functionality of each methodology using ONLY strain n2 data
- **Execution and dependencies:** This is a uv project, you have to use `uv add <filename>` and `uv add <dependency>` when needed.

## STRICT Restrictions

- ❌ DO NOT generate README files or markdown documentation
- ❌ DO NOT create "main" or "master executor" files
- ❌ DO NOT add extensive explanatory comments
- ❌ DO NOT create folders or complex structure
- ✅ ONLY individual Python files, functional and self-contained

## Expected Output

Deliver only the implemented `.py` files, one for each methodology identified in `proposed_methodology.md`.

**Proceed directly with implementation.**