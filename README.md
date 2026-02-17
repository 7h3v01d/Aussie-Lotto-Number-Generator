# Aussie Lotto Number Generator

A Python-based desktop application for generating and suggesting lottery numbers for Australian lotteries: **Gold Lotto (Queensland)** and **Oz Lotto**. Built with PyQt6, it uses advanced tools like `ProbabilityCore`, `KnowledgeCore`, and `OptimizationCore` to analyze past draws, generate random tickets, and suggest "optimal" tickets based on frequency and Monte Carlo simulations. **Note**: Lotteries are inherently random; this tool is for entertainment purposes only and does not guarantee wins.

## Features
- **Supported Games**:
  - **Gold Lotto (QLD)**: 6 numbers from 1–45 + 2 supplementary numbers. Odds: ~1 in 8.1 million for Division 1.
  - **Oz Lotto**: 7 numbers from 1–47 + 2 supplementary numbers. Odds: ~1 in 62.9 million for Division 1.
- **Random Ticket Generation**: Generate 1–10 random tickets for either game, ensuring unique numbers.
- **Optimal Ticket Suggestion**: Uses frequency analysis (via `KnowledgeCore`) and Monte Carlo optimization (via `OptimizationCore`) to suggest tickets based on recent 2025 draws.
- **Frequency Visualization**: Displays a histogram of number frequencies using Matplotlib.
- **Historical Data**: Includes real 2025 draw data (e.g., Gold Lotto Draw 4609: 05-10-25-26-27-38 +15-34).
- **Disclaimer**: Clearly states that suggestions are for fun, as lottery outcomes are random.

## Prerequisites
- **Python**: Version 3.8–3.11 recommended.
- **Virtual Environment**: Optional but recommended for dependency management.
- **Dependencies**:
  ```bash
  pip install PyQt6 numpy scipy matplotlib networkx pulp python-constraint joblib
  ```

## Installation
1. **Clone or Download**:
   Clone the repository or download the source files (`lotto_generator.py`, `probability_core.py`, `knowledge_core.py`, `optimization_core.py`) to a directory (e.g., `E:\Projects\Lotto\src`).
   ```bash
   git clone <repository-url>
   cd lotto
   ```

2. **Set Up Virtual Environment** (optional):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install PyQt6 numpy scipy matplotlib networkx pulp python-constraint joblib
   ```

4. **Verify Files**:
   Ensure all required Python files (`lotto_generator.py`, `probability_core.py`, `knowledge_core.py`, `optimization_core.py`) are in the same directory.

## Usage
1. **Run the Application**:
   ```bash
   python lotto_generator.py
   ```
   This opens a GUI window with the following controls:
   - **Select Aussie Lotto Game**: Choose "Gold Lotto (QLD)" or "Oz Lotto".
   - **Number of Tickets (1-10)**: Set how many tickets to generate (default: 1).
   - **Generate Random Tickets**: Creates random, valid tickets for the selected game.
   - **Suggest Optimal Ticket**: Analyzes past draws and runs Monte Carlo simulations to suggest a ticket based on frequent ("hot") numbers.
   - **Show Number Frequency**: Displays a histogram of generated or historical numbers.
   - **Output Area**: Shows generated tickets, suggested tickets, frequency stats, and Monte Carlo results.
   - **Disclaimer**: A red label and popup remind users that lotteries are random.

2. **Sample Output** (for Gold Lotto, 1 ticket):
   ```
   Gold Lotto (QLD) Optimal Ticket (Based on Recent 2025 Draws):
   Ticket: 05, 10, 25, 26, 38, 42 + Supps: 15, 34

   Top 10 Frequent Numbers: 25:3, 38:3, 10:2, 16:2, 23:2, 1:1, 3:1, 5:1, 6:1, 9:1

   Monte Carlo Avg Matches: 1.20
   Stats - Mean: 24.5, Std Dev: 14.3

   Note: Lottery draws are random; no strategy guarantees a win!
   ```

3. **Notes**:
   - The app uses real 2025 draws (e.g., Gold Lotto Draw 4609 from Sep 20, 2025). Check The Lott for the latest results.
   - Supplementary numbers are generated randomly but could be enhanced with frequency analysis.
   - Monte Carlo simulations may take a few seconds; reduce `num_simulations` in `lotto_generator.py` if slow.

## Project Structure
- `lotto_generator.py`: Main application with PyQt6 GUI and logic to integrate cores.
- `probability_core.py`: Handles random number generation and odds calculations.
- `knowledge_core.py`: Manages a knowledge graph of past draws for frequency analysis.
- `optimization_core.py`: Runs Monte Carlo simulations to optimize ticket suggestions.

## Technical Details
- **GUI**: Built with PyQt6 for a responsive desktop interface.
- **Data**: Stores 5 recent 2025 draws per game in `KnowledgeCore` (e.g., Gold Lotto: 5 draws, Oz Lotto: 5 draws).
- **Optimization**: Uses `OptimizationCore` for Monte Carlo planning (500 simulations) to balance hot, cold, and random numbers.
- **Error Handling**: Logs errors to the console and falls back to frequency-based suggestions if Monte Carlo fails.
- **Dependencies**:
  - `PyQt6`: GUI framework.
  - `numpy`, `scipy`: Numerical computations for probability and optimization.
  - `matplotlib`: Plotting frequency histograms.
  - `networkx`: Knowledge graph management.
  - `pulp`, `python-constraint`: Constraint solvers for optimization.
  - `joblib`: Parallel processing for Monte Carlo simulations.

## Limitations
- **Randomness**: Lotteries are independent and random; no strategy can predict outcomes. The app analyzes past draws for fun patterns only.
- **Data**: Limited to 10 hardcoded draws (5 per game). Expand by fetching more from The Lott or adding a file loader.
- **Performance**: Monte Carlo simulations may be slow on low-end systems; adjustable via `num_simulations`.
- **Supplementary Numbers**: Currently random; could use frequency analysis for better suggestions.

## Future Enhancements
- **More Draws**: Fetch live draws from The Lott API or load from CSV/JSON.
- **Additional Games**: Add Monday/Wednesday Lotto or Powerball (Australian version).
- **Ticket Checker**: Input your ticket to compare against historical draws.
- **Web Interface**: Port to a web app using Flask or React for broader access.
- **Rule Learning**: Use `LogicCore` to detect patterns (e.g., "if 25 appears, 38 often follows").

## Contributing
Feel free to fork the repository, submit pull requests, or report issues. To add features:
1. Update `lottery_formats` in `lotto_generator.py` for new games.
2. Expand `load_past_draws` with more data or dynamic fetching.
3. Enhance `suggest_optimal` with `LogicCore` or other strategies.

## Disclaimer
This tool is for entertainment purposes only. Lottery draws are random, and no analysis can guarantee wins. Play responsibly! Check official results at [The Lott](https://www.thelott.com/) or the app.

## License
MIT License. See `LICENSE` file (if included) for details.
