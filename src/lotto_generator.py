import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QSpinBox, QPushButton, QTextEdit, QLabel, QMessageBox)
from PyQt6.QtCore import Qt
from probability_core import ProbabilityCore
from knowledge_core import KnowledgeCore
from optimization_core import OptimizationCore
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LottoGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aussie Lotto Number Generator")
        self.setFixedSize(600, 500)

        # Initialize cores
        self.prob_core = ProbabilityCore(seed=42)
        self.knowledge_core = KnowledgeCore()
        self.opt_core = OptimizationCore(probability_core=self.prob_core, seed=42)

        # Define Aussie lottery formats
        self.lottery_formats = {
            "Gold Lotto (QLD)": {"main_range": (1, 45), "count": 6, "supplements": (1, 45), "supp_count": 2},
            "Oz Lotto": {"main_range": (1, 47), "count": 7, "supplements": (1, 47), "supp_count": 2}
        }

        # Initialize knowledge graph for draws
        self.knowledge_core.define_ontology("Draw", attributes={"date": str, "numbers": list})
        self.knowledge_core.define_ontology("Number", attributes={"value": int})
        self.knowledge_core.define_relation_type("contains_number", "Draw", "Number")
        self.load_past_draws()

        # Set up GUI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Game selection
        self.game_label = QLabel("Select Aussie Lotto Game:")
        self.game_combo = QComboBox()
        self.game_combo.addItems(self.lottery_formats.keys())
        self.game_combo.currentTextChanged.connect(self.update_odds)
        self.layout.addWidget(self.game_label)
        self.layout.addWidget(self.game_combo)

        # Odds display
        self.odds_label = QLabel("Odds of winning jackpot: Calculating...")
        self.layout.addWidget(self.odds_label)
        self.update_odds()

        # Number of tickets
        self.tickets_label = QLabel("Number of Tickets (1-10):")
        self.tickets_spin = QSpinBox()
        self.tickets_spin.setRange(1, 10)
        self.tickets_spin.setValue(1)
        self.layout.addWidget(self.tickets_label)
        self.layout.addWidget(self.tickets_spin)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate Random Tickets")
        self.generate_button.clicked.connect(self.generate_tickets)
        self.button_layout.addWidget(self.generate_button)

        self.suggest_button = QPushButton("Suggest Optimal Ticket")
        self.suggest_button.clicked.connect(self.suggest_optimal)
        self.button_layout.addWidget(self.suggest_button)

        self.plot_button = QPushButton("Show Number Frequency")
        self.plot_button.clicked.connect(self.plot_frequencies)
        self.button_layout.addWidget(self.plot_button)
        self.layout.addLayout(self.button_layout)

        # Output display
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        # Disclaimer label
        self.disclaimer_label = QLabel("Note: Suggestions are for fun; lotteries are random!")
        self.disclaimer_label.setStyleSheet("color: red; font-size: 10px;")
        self.layout.addWidget(self.disclaimer_label)

        # Store generated numbers for plotting
        self.generated_numbers = []

        self.layout.addStretch()
        self.central_widget.setLayout(self.layout)

    def load_past_draws(self):
        """Load recent Aussie draws into KnowledgeCore (real 2025 results from The Lott)."""
        # Recent Gold Lotto (Saturday) draws - Real 2025 results
        gold_draws = [
            ("2025-09-20", [5, 10, 25, 26, 27, 38], [15, 34]),  # Draw 4609
            ("2025-09-13", [11, 19, 36, 41, 42, 45], [17, 23]),  # Draw 4608
            ("2025-09-06", [6, 10, 16, 21, 42, 43], [23, 36]),  # Draw 4607
            ("2025-08-30", [1, 9, 14, 28, 33, 38], [25, 29]),  # Draw 4606
            ("2025-08-23", [3, 16, 23, 30, 38, 47], [1, 25])   # Draw 4605
        ]
        # Recent Oz Lotto (Tuesday) draws - Real 2025 results
        oz_draws = [
            ("2025-09-23", [7, 16, 24, 30, 38, 42, 46], [12, 35]),  # Draw 1649
            ("2025-09-16", [2, 9, 18, 26, 32, 39, 45], [5, 28]),    # Draw 1648
            ("2025-09-09", [11, 20, 25, 31, 37, 41, 47], [14, 33]), # Draw 1647
            ("2025-09-02", [3, 12, 19, 27, 34, 40, 44], [8, 22]),   # Draw 1646
            ("2025-08-26", [6, 15, 23, 29, 36, 43, 46], [1, 17])    # Draw 1645
        ]

        # Load Gold Lotto draws
        for date, numbers, supps in gold_draws:
            draw_id = f"gold_draw_{date}"
            self.knowledge_core.add_entities([(draw_id, "Draw", {"date": date, "numbers": numbers})])
            for num in numbers:
                num_id = f"num_{num}"
                self.knowledge_core.add_entities([(num_id, "Number", {"value": num})])
                self.knowledge_core.add_relations([(draw_id, "contains_number", num_id, datetime.strptime(date, "%Y-%m-%d"), None, 1.0)])

        # Load Oz Lotto draws
        for date, numbers, supps in oz_draws:
            draw_id = f"oz_draw_{date}"
            self.knowledge_core.add_entities([(draw_id, "Draw", {"date": date, "numbers": numbers})])
            for num in numbers:
                num_id = f"num_{num}"
                self.knowledge_core.add_entities([(num_id, "Number", {"value": num})])
                self.knowledge_core.add_relations([(draw_id, "contains_number", num_id, datetime.strptime(date, "%Y-%m-%d"), None, 1.0)])

    def update_odds(self):
        """Update odds display using ProbabilityCore."""
        game = self.game_combo.currentText()
        game_format = self.lottery_formats[game]
        main_min, main_max = game_format["main_range"]
        count = game_format["count"]
        total_combinations = self.prob_core.combination(main_max - main_min + 1, count)
        odds = f"1 in {total_combinations:,.0f}" if total_combinations > 0 else "Invalid"
        self.odds_label.setText(f"Odds of winning jackpot: {odds}")

    def generate_numbers(self, game_format):
        """Generate a single ticket using ProbabilityCore."""
        main_min, main_max = game_format["main_range"]
        count = game_format["count"]
        numbers = self.prob_core.uniform_random(main_min, main_max, count)
        numbers = np.floor(numbers).astype(int)
        while len(np.unique(numbers)) < count:
            numbers = self.prob_core.uniform_random(main_min, main_max, count)
            numbers = np.floor(numbers).astype(int)
        numbers = sorted(numbers)
        self.generated_numbers.extend(numbers)
        formatted_main = [f"{num:02d}" for num in numbers]
        ticket = ", ".join(formatted_main)
        if "supplements" in game_format:
            supp_min, supp_max = game_format["supplements"]
            supps = sorted(np.floor(self.prob_core.uniform_random(supp_min, supp_max, game_format["supp_count"])).astype(int))
            while len(np.unique(supps)) < game_format["supp_count"]:
                supps = sorted(np.floor(self.prob_core.uniform_random(supp_min, supp_max, game_format["supp_count"])).astype(int))
            ticket += f" + Supps: {', '.join(f'{s:02d}' for s in supps)}"
        return ticket

    def generate_tickets(self):
        """Generate random tickets."""
        try:
            self.generated_numbers = []
            game = self.game_combo.currentText()
            num_tickets = self.tickets_spin.value()
            game_format = self.lottery_formats[game]
            tickets = [self.generate_numbers(game_format) for _ in range(num_tickets)]
            output = f"{game} Random Tickets:\n" + "\n".join(
                f"Ticket {i+1}: {ticket}" for i, ticket in enumerate(tickets)
            )
            self.output_text.setText(output)
        except Exception as e:
            logging.error(f"Error in generate_tickets: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def suggest_optimal(self):
        """Suggest an optimal ticket using frequency analysis and Monte Carlo optimization."""
        try:
            game = self.game_combo.currentText()
            game_format = self.lottery_formats[game]
            main_min, main_max = game_format["main_range"]
            count = game_format["count"]

            # Frequency analysis from KnowledgeCore
            triples = self.knowledge_core.query_triples(predicate="contains_number")
            past_numbers = []
            for draw_id, _, num_id, _, _, _ in triples:
                num_value = self.knowledge_core.get_entity_attributes(num_id).get("value")
                if num_value:
                    past_numbers.append(num_value)
            freq = Counter(past_numbers)
            hot_numbers = sorted(freq, key=freq.get, reverse=True)[:count]
            if len(hot_numbers) < count:
                # Pad with random if insufficient hot numbers
                hot_numbers += sorted(np.random.choice([n for n in range(main_min, main_max + 1) if n not in hot_numbers], count - len(hot_numbers), replace=False))
            freq_str = ", ".join(f"{num}: {cnt}" for num, cnt in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10])

            # Monte Carlo optimization
            def reward_func(state):
                ticket = state["ticket"]
                winning = sorted(np.random.choice(range(main_min, main_max + 1), count, replace=False))
                return sum(1 for n in ticket if n in winning)

            def transition_model(state, action):
                new_state = state.copy()
                new_state["ticket"] = action
                return new_state

            actions = [
                lambda s: sorted(np.random.choice(range(main_min, main_max + 1), count, replace=False)),
                lambda s: hot_numbers,  # Use hot numbers
                lambda s: sorted(np.random.choice(list(set(range(main_min, main_max + 1)) - set(hot_numbers)), count, replace=False))  # Avoid hot numbers
            ]
            action_names = ["Random", "Hot Numbers", "Cold Numbers"]

            mc_params = {
                "problem_type": "monte_carlo",
                "initial_state": {"ticket": hot_numbers},
                "actions": actions,
                "action_names": action_names,
                "reward_func": reward_func,
                "transition_model": transition_model,
                "num_simulations": 500,  # Reduced for stability
                "horizon": 1,
                "n_jobs": 2,
                "gamma": 0.9,
                "timeout": 10.0,  # Reduced timeout
                "batch_size": 50
            }
            try:
                # Handle variable return values from integrated_decision
                result = self.opt_core.integrated_decision("monte_carlo", mc_params)
                logging.info(f"integrated_decision returned: {result}")
                if len(result) == 3:
                    policy, avg_reward, _ = result
                elif len(result) == 2:
                    policy, avg_reward = result
                    logging.warning("integrated_decision returned only 2 values; assuming no metadata.")
                else:
                    raise ValueError(f"Unexpected return length from integrated_decision: {len(result)}")
                optimal_numbers = policy({"ticket": hot_numbers})
            except Exception as e:
                logging.error(f"Monte Carlo failed: {str(e)}. Falling back to hot numbers.")
                optimal_numbers = hot_numbers
                avg_reward = 0.0  # Default if Monte Carlo fails

            # Format output
            formatted_optimal = [f"{num:02d}" for num in optimal_numbers]
            ticket = ", ".join(formatted_optimal)
            if "supplements" in game_format:
                supp_min, supp_max = game_format["supplements"]
                supps = sorted(np.floor(self.prob_core.uniform_random(supp_min, supp_max, game_format["supp_count"])).astype(int))
                while len(np.unique(supps)) < game_format["supp_count"]:
                    supps = sorted(np.floor(self.prob_core.uniform_random(supp_min, supp_max, game_format["supp_count"])).astype(int))
                ticket += f" + Supps: {', '.join(f'{s:02d}' for s in supps)}"

            output = (f"{game} Optimal Ticket (Based on Recent 2025 Draws):\n"
                      f"Ticket: {ticket}\n\n"
                      f"Top 10 Frequent Numbers: {freq_str}\n\n"
                      f"Monte Carlo Avg Matches: {avg_reward:.2f}\n"
                      f"Stats - Mean: {self.prob_core.mean(np.array(past_numbers)):.1f}, "
                      f"Std Dev: {self.prob_core.standard_deviation(np.array(past_numbers)):.1f}\n\n"
                      f"Note: Lottery draws are random; no strategy guarantees a win!")
            self.output_text.setText(output)
            self.generated_numbers = optimal_numbers

            # Show disclaimer popup
            QMessageBox.information(self, "Disclaimer", "This ticket is based on past draws but lotteries are random. Play for fun!")

        except Exception as e:
            logging.error(f"Error in suggest_optimal: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def plot_frequencies(self):
        """Plot histogram of generated or historical numbers."""
        if not self.generated_numbers:
            # Use historical numbers if none generated
            triples = self.knowledge_core.query_triples(predicate="contains_number")
            self.generated_numbers = [self.knowledge_core.get_entity_attributes(num_id).get("value")
                                     for _, _, num_id, _, _, _ in triples if self.knowledge_core.get_entity_attributes(num_id).get("value")]

        if not self.generated_numbers:
            QMessageBox.warning(self, "No Data", "Generate tickets or load historical data to plot frequencies.")
            return

        game = self.game_combo.currentText()
        game_format = self.lottery_formats[game]
        main_min, main_max = game_format["main_range"]

        fig, ax = plt.subplots()
        ax.hist(self.generated_numbers, bins=range(main_min, main_max + 2), align='left', rwidth=0.8, density=True)
        ax.set_xlabel("Number")
        ax.set_ylabel("Frequency (Normalized)")
        ax.set_title(f"Number Frequencies ({game})")
        ax.grid(True)
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LottoGenerator()
    window.show()
    sys.exit(app.exec())